"""
backend.py — FastAPI backend: BLIP-2 (GPU) + Grounding DINO (CPU)

Run with:
    uvicorn backend:app --host 0.0.0.0 --port 8000

Two models:
  • BLIP-2 opt-2.7b  — answers the question  (CUDA, float16)
  • Grounding DINO   — draws bounding boxes   (CPU, avoids VRAM pressure)

Pipeline:
  1. BLIP-2  → natural-language answer
  2. Extract noun phrase from answer
  3. Grounding DINO → boxes around those nouns in the image
  4. Draw annotated image, return as base64 PNG
"""

import io
import os
import re
import time
import base64
import warnings

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import logging as hf_logging

# ── silence noise ─────────────────────────────────────────────────────────────
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ── Grounding DINO ────────────────────────────────────────────────────────────
import groundingdino
from groundingdino.util.inference import load_model, predict
from groundingdino.datasets import transforms as GDT

GDINO_CFG  = os.path.join(
    os.path.dirname(groundingdino.__file__),
    "config", "GroundingDINO_SwinT_OGC.py"
)
# Checkpoint auto-downloaded to local cache on first run
GDINO_CKPT = os.path.join(
    os.path.expanduser("~"), ".cache", "gdino", "groundingdino_swint_ogc.pth"
)
GDINO_URL  = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)

# ── Colours for boxes (cycle through) ────────────────────────────────────────
BOX_PALETTE = [
    "#FF3B30", "#FF9500", "#FFCC00", "#34C759",
    "#00C7BE", "#30B0C7", "#007AFF", "#AF52DE",
]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="VisionAsk API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Global model state ────────────────────────────────────────────────────────
_blip_processor = None
_blip_model     = None
_blip_device    = None
_gdino_model    = None


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_blip():
    global _blip_processor, _blip_model, _blip_device

    if _blip_processor is None:
        print("[BLIP-2] Loading processor …")
        _blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)

    if _blip_model is None:
        print("[BLIP-2] Loading model (float16, CUDA) …")
        _blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _blip_model.eval()
        _blip_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[BLIP-2] Ready on {_blip_device}.")

    return _blip_processor, _blip_model, _blip_device


def get_gdino():
    global _gdino_model

    if _gdino_model is None:
        # Download checkpoint if missing
        if not os.path.exists(GDINO_CKPT):
            print("[GDino] Downloading checkpoint (~170 MB, once only) …")
            import urllib.request
            os.makedirs(os.path.dirname(GDINO_CKPT), exist_ok=True)
            urllib.request.urlretrieve(GDINO_URL, GDINO_CKPT)
            print("[GDino] Download complete.")

        print("[GDino] Loading model on CPU …")
        _gdino_model = load_model(GDINO_CFG, GDINO_CKPT, device="cpu")
        _gdino_model.eval()
        print("[GDino] Ready on CPU.")

    return _gdino_model


# ─────────────────────────────────────────────────────────────────────────────
#  Startup
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    get_blip()
    get_gdino()


# ─────────────────────────────────────────────────────────────────────────────
#  Noun extraction from BLIP-2 answer
# ─────────────────────────────────────────────────────────────────────────────

# Stop-words we don't want as DINO captions
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "there", "this", "that", "these", "those", "some", "any",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "and",
    "or", "but", "not", "no", "yes", "have", "has", "had", "do", "does",
    "did", "will", "would", "can", "could", "should", "may", "might",
    "what", "which", "who", "how", "when", "where", "visible", "image",
    "picture", "photo", "shown", "showing", "shows", "appears", "appear",
    "present", "scene", "background", "foreground",
}

def extract_caption(answer: str) -> str:
    """
    Pull meaningful noun/adjective phrases from BLIP-2's answer to use
    as Grounding DINO caption.

    Strategy:
      1. Strip common filler prefixes ("there is", "i see", etc.)
      2. Remove stop-words
      3. Deduplicate, keep order
      4. Join with " . " (DINO's multi-phrase separator)
      5. Fall back to cleaned answer if result is too short
    """
    text = answer.lower().strip()

    # Strip leading filler phrases
    fillers = [
        r"^there (is|are) ", r"^i (see|can see|notice) ",
        r"^the image (shows?|depicts?|contains?) ",
        r"^this (is|shows?) (a |an )?", r"^it (is|appears? to be) (a |an )?",
        r"^in this (image|picture|photo)[,.]? ",
    ]
    for pat in fillers:
        text = re.sub(pat, "", text)

    # Tokenise and filter
    tokens = re.findall(r"[a-z]+", text)
    seen, phrases = set(), []
    for t in tokens:
        if t not in _STOP and len(t) > 2 and t not in seen:
            seen.add(t)
            phrases.append(t)

    if len(phrases) == 0:
        return answer.strip()

    # Group into chunks of ≤3 words separated by " . " for DINO
    chunks = []
    for i in range(0, len(phrases), 3):
        chunks.append(" ".join(phrases[i:i+3]))

    caption = " . ".join(chunks[:4])   # cap at 4 phrases
    return caption if caption else answer.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Image → DINO tensor transform
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_gdino_tensor(pil_img: Image.Image) -> torch.Tensor:
    transform = GDT.Compose([
        GDT.RandomResize([800], max_size=1333),
        GDT.ToTensor(),
        GDT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor, _ = transform(pil_img, None)
    return img_tensor


# ─────────────────────────────────────────────────────────────────────────────
#  Draw bounding boxes on PIL image
# ─────────────────────────────────────────────────────────────────────────────

def draw_boxes(
    pil_img: Image.Image,
    boxes:   torch.Tensor,   # (N,4) cx cy w h normalised
    scores:  torch.Tensor,   # (N,)
    labels:  List[str],      # (N,)
) -> Image.Image:
    """
    Draw clean, sharp bounding boxes with a label+confidence tag above each box.
    Style: 2-px solid border, opaque tag background, white text.
    No semi-transparent fill — keeps the image readable.
    """
    annotated = pil_img.copy().convert("RGB")
    draw      = ImageDraw.Draw(annotated)
    W, H      = pil_img.size

    # Font — try system Arial, fall back to PIL default
    tag_font_size = max(13, H // 45)
    try:
        font = ImageFont.truetype("arial.ttf", tag_font_size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", tag_font_size)
        except Exception:
            font = ImageFont.load_default()

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        cx, cy, bw, bh = box.tolist()
        x0 = int((cx - bw / 2) * W)
        y0 = int((cy - bh / 2) * H)
        x1 = int((cx + bw / 2) * W)
        y1 = int((cy + bh / 2) * H)

        # clamp to image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W - 1, x1), min(H - 1, y1)

        # colour from palette
        hex_col = BOX_PALETTE[i % len(BOX_PALETTE)]
        r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
        rgb     = (r, g, b)

        # ── bounding box — 2 px thick border ──
        for t in range(2):
            draw.rectangle(
                [x0 - t, y0 - t, x1 + t, y1 + t],
                outline=rgb,
            )

        # ── label tag ──
        tag  = f" {label}  {score:.0%} "
        try:
            bbox = draw.textbbox((0, 0), tag, font=font)
            tw   = bbox[2] - bbox[0]
            th   = bbox[3] - bbox[1]
        except AttributeError:           # older Pillow
            tw, th = draw.textsize(tag, font=font)

        pad    = 3
        tag_x0 = x0
        tag_y0 = max(0, y0 - th - pad * 2)
        tag_x1 = x0 + tw
        tag_y1 = tag_y0 + th + pad * 2

        # solid tag background
        draw.rectangle([tag_x0, tag_y0, tag_x1, tag_y1], fill=rgb)
        # white text
        draw.text(
            (tag_x0, tag_y0 + pad),
            tag,
            fill=(255, 255, 255),
            font=font,
        )

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
#  PIL → base64
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
#  Response schema
# ─────────────────────────────────────────────────────────────────────────────

class BoxInfo(BaseModel):
    label:      str
    confidence: float
    x0: int; y0: int; x1: int; y1: int

class VQAResponse(BaseModel):
    answer:           str
    confidence:       float
    inference_ms:     int
    device:           str
    grounding_ran:    bool
    grounding_ms:     Optional[int]
    annotated_image:  Optional[str]   # base64 PNG, None if grounding skipped
    boxes:            List[BoxInfo]
    gdino_caption:    Optional[str]   # what we sent to Grounding DINO


# ─────────────────────────────────────────────────────────────────────────────
#  Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    mem = {}
    if torch.cuda.is_available():
        mem = {
            "vram_used_mb":  round(torch.cuda.memory_allocated() / 1e6, 1),
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6, 1),
        }
    return {"status": "ok", "blip_device": _blip_device or "not loaded", **mem}


# ─────────────────────────────────────────────────────────────────────────────
#  Main endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=VQAResponse)
async def ask(
    image:        UploadFile = File(...),
    question:     str        = Form(...),
    question_type: str       = Form(default="Custom"),
    run_grounding: str       = Form(default="auto"),
    # "auto"  → run only for VISUAL_TYPES
    # "yes"   → always run
    # "no"    → never run
):
    if not question.strip():
        raise HTTPException(400, "Question must not be empty.")

    raw = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image.")

    # ── 1. BLIP-2 inference ───────────────────────────────────────────────
    processor, blip_model, blip_device = get_blip()
    prompt = f"Question: {question.strip()} Answer:"
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(blip_device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    t0 = time.perf_counter()
    with torch.no_grad():
        output = blip_model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    blip_ms = int((time.perf_counter() - t0) * 1000)

    raw_text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
    lower = raw_text.lower()
    if "answer:" in lower:
        raw_text = raw_text[lower.index("answer:") + len("answer:"):].strip()
    answer = raw_text or "No answer generated."

    # confidence
    confidence = 0.0
    try:
        if output.scores:
            probs = [torch.softmax(s[0], dim=-1).max().item() for s in output.scores]
            confidence = float(np.mean(probs))
    except Exception:
        pass

    # ── 2. Grounding DINO — always runs ──────────────────────────────────────
    caption    = extract_caption(answer)
    gdino      = get_gdino()
    img_tensor = pil_to_gdino_tensor(pil_image)

    t1 = time.perf_counter()
    with torch.no_grad():
        boxes, scores, labels = predict(
            model          = gdino,
            image          = img_tensor,
            caption        = caption,
            box_threshold  = 0.30,
            text_threshold = 0.25,
            device         = "cpu",
        )
    gdino_ms = int((time.perf_counter() - t1) * 1000)

    # ── 4. Annotate image ─────────────────────────────────────────────────
    W, H = pil_image.size
    box_infos: List[BoxInfo] = []

    if len(boxes) > 0:
        annotated = draw_boxes(pil_image, boxes, scores, labels)
        for box, score, label in zip(boxes, scores, labels):
            cx, cy, w, h = box.tolist()
            box_infos.append(BoxInfo(
                label=label,
                confidence=round(float(score), 3),
                x0=int((cx - w/2) * W), y0=int((cy - h/2) * H),
                x1=int((cx + w/2) * W), y1=int((cy + h/2) * H),
            ))
        annotated_b64 = pil_to_b64(annotated)
    else:
        # No boxes found — return original image with a subtle border
        annotated_b64 = pil_to_b64(pil_image)

    return VQAResponse(
        answer           = answer,
        confidence       = round(confidence, 4),
        inference_ms     = blip_ms,
        device           = blip_device,
        grounding_ran    = True,
        grounding_ms     = gdino_ms,
        annotated_image  = annotated_b64,
        boxes            = box_infos,
        gdino_caption    = caption,
    )