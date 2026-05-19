# """
# app.py — Streamlit frontend for BLIP-2 + Grounding DINO VQA
# Run with:  streamlit run app.py
# Requires:  uvicorn backend:app --host 0.0.0.0 --port 8000
# """

# import io
# import base64
# import requests
# import streamlit as st
# from PIL import Image
# from datetime import datetime

# BACKEND_URL = "http://localhost:8000"

# # ─────────────────────────────────────────────────────────────────────────────
# #  Page config
# # ─────────────────────────────────────────────────────────────────────────────

# st.set_page_config(
#     page_title="VisionAsk · BLIP-2 + GDino",
#     page_icon="⚡",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# # ─────────────────────────────────────────────────────────────────────────────
# #  Data
# # ─────────────────────────────────────────────────────────────────────────────

# QUESTION_TYPES = {
#     "Custom":                     "",
#     "Counting":                   "How many objects are visible in this image?",
#     "Object Recognition":         "What are the main objects present in this image?",
#     "Object Presence":            "Is there a person in this image? What other objects are present?",
#     "Color":                      "What are the dominant colors in this image?",
#     "Scene Understanding":        "Describe the overall scene and setting of this image.",
#     "Animal Recognition":         "What animal or animals are visible in this image? Describe their appearance.",
#     "Activity Recognition":       "What activity or action is being performed in this image?",
#     "Sports Recognition":         "What sport or athletic activity is shown in this image?",
#     "Transportation Recognition": "What mode of transportation is visible in this image?",
#     "Text Recognition":           "What text, signs, or written words are visible in this image?",
#     "Food Recognition":           "What food or dish is shown in this image? Describe it.",
# }

# TYPE_COLORS = {
#     "Custom":                     "#6b7280",
#     "Counting":                   "#6366f1",
#     "Object Recognition":         "#8b5cf6",
#     "Object Presence":            "#10b981",
#     "Color":                      "#f59e0b",
#     "Scene Understanding":        "#06b6d4",
#     "Animal Recognition":         "#84cc16",
#     "Activity Recognition":       "#ec4899",
#     "Sports Recognition":         "#14b8a6",
#     "Transportation Recognition": "#f97316",
#     "Text Recognition":           "#94a3b8",
#     "Food Recognition":           "#ef4444",
# }

# # ─────────────────────────────────────────────────────────────────────────────
# #  CSS
# # ─────────────────────────────────────────────────────────────────────────────

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

# html, body, [class*="css"] {
#     font-family: 'JetBrains Mono', monospace;
#     background: #080a0f;
#     color: #c9cdd8;
# }
# #MainMenu, footer, header { visibility: hidden; }
# .block-container {
#     padding: 0 1rem 2rem 1rem !important;
#     max-width: 100% !important;
# }
# section[data-testid="stSidebar"] { display: none; }
# [data-testid="stHorizontalBlock"] {
#     gap: 1.5rem !important;
#     align-items: flex-start !important;
# }
# [data-testid="stHorizontalBlock"] > div {
#     overflow: visible !important;
#     min-width: 0 !important;
# }

# /* ── topbar ── */
# .topbar {
#     position: sticky; top: 0; z-index: 100;
#     background: rgba(8,10,15,0.94);
#     backdrop-filter: blur(14px);
#     border-bottom: 1px solid #1a1f2e;
#     padding: 0 2rem; height: 56px;
#     display: flex; align-items: center; justify-content: space-between;
# }
# .topbar-logo {
#     font-family: 'Instrument Serif', serif;
#     font-size: 1.4rem; color: #f0f2f8; letter-spacing: -0.02em;
# }
# .topbar-sub {
#     font-size: 0.58rem; letter-spacing: 0.18em;
#     text-transform: uppercase; color: #3d4460; margin-left: 0.8rem;
# }
# .status-dot {
#     width: 7px; height: 7px; border-radius: 50%;
#     background: #10b981; box-shadow: 0 0 7px #10b981; display: inline-block;
# }
# .status-dot.off { background: #ef4444; box-shadow: 0 0 7px #ef4444; }
# .status-label {
#     font-size: 0.6rem; letter-spacing: 0.12em;
#     text-transform: uppercase; color: #3d4460; margin-right: 0.5rem;
# }

# /* ── section labels ── */
# .sec-label {
#     font-size: 0.58rem; letter-spacing: 0.2em; text-transform: uppercase;
#     color: #3d4460; margin-bottom: 0.55rem;
#     display: flex; align-items: center; gap: 0.5rem;
# }
# .sec-label::after { content:''; flex:1; height:1px; background:#1a1f2e; }

# /* ── uploader ── */
# [data-testid="stFileUploader"] > div {
#     background: #0d1017 !important; border: 1.5px dashed #1e2438 !important;
#     border-radius: 10px !important; transition: border-color .25s, background .25s !important;
# }
# [data-testid="stFileUploader"] > div:hover {
#     border-color: #4f6ef7 !important; background: #0c1020 !important;
# }
# [data-testid="stFileUploader"] label { display: none !important; }

# /* ── image ── */
# [data-testid="stImage"] img {
#     border-radius: 10px !important; border: 1px solid #1a1f2e !important;
#     width: 100% !important; max-height: 380px !important; object-fit: contain !important;
# }

# /* ── selectbox ── */
# [data-testid="stSelectbox"] > div > div {
#     background: #0d1017 !important; border: 1px solid #1e2438 !important;
#     border-radius: 7px !important; color: #c9cdd8 !important;
#     font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important;
# }

# /* ── textarea ── */
# textarea {
#     background: #0d1017 !important; border: 1px solid #1e2438 !important;
#     border-radius: 7px !important; color: #c9cdd8 !important;
#     font-family: 'JetBrains Mono', monospace !important;
#     font-size: 0.82rem !important; resize: vertical !important;
# }
# textarea:focus {
#     border-color: #4f6ef7 !important;
#     box-shadow: 0 0 0 2px rgba(79,110,247,0.1) !important;
# }

# /* ── toggle / checkbox ── */
# [data-testid="stCheckbox"] label {
#     font-family: 'JetBrains Mono', monospace !important;
#     font-size: 0.75rem !important; color: #7a8098 !important;
#     letter-spacing: 0.04em !important;
# }

# /* ── buttons ── */
# .stButton > button {
#     width: 100%;
#     background: linear-gradient(135deg, #4f6ef7, #7c3aed) !important;
#     color: #fff !important; font-family: 'JetBrains Mono', monospace !important;
#     font-weight: 600 !important; font-size: 0.82rem !important;
#     letter-spacing: 0.06em !important; border: none !important;
#     border-radius: 8px !important; padding: 0.72rem 0 !important;
#     transition: opacity .2s, transform .1s !important;
# }
# .stButton > button:hover { opacity: .86 !important; transform: translateY(-1px) !important; }

# /* ── answer card ── */
# .answer-card {
#     background: #0b0e18; border: 1px solid #1e2438;
#     border-radius: 12px; padding: 1.4rem 1.6rem;
#     position: relative; overflow: hidden;
# }
# .answer-card::before {
#     content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
#     background: linear-gradient(90deg, #4f6ef7, #7c3aed, #10b981);
# }
# .answer-main {
#     font-family: 'Instrument Serif', serif;
#     font-size: 1.55rem; color: #f0f2f8; line-height: 1.3; margin-bottom: 1rem;
# }
# .meta-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-bottom: 0.9rem; }
# .meta-chip {
#     font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase;
#     padding: 4px 10px; border-radius: 4px;
#     background: #12151f; border: 1px solid #1e2438; color: #5a6080;
# }
# .meta-chip b { color: #c9cdd8; font-weight: 500; }
# .conf-bar-label {
#     display: flex; justify-content: space-between;
#     font-size: 0.58rem; letter-spacing: 0.12em;
#     text-transform: uppercase; color: #3d4460; margin-bottom: 0.35rem;
# }
# .conf-bar-track { height: 4px; background: #1a1f2e; border-radius: 2px; overflow: hidden; }
# .conf-bar-fill  { height: 100%; border-radius: 2px; }

# /* ── grounding section ── */
# .ground-header {
#     display: flex; align-items: center; gap: 0.7rem;
#     margin-bottom: 0.8rem;
# }
# .ground-badge {
#     font-size: 0.58rem; letter-spacing: 0.14em; text-transform: uppercase;
#     padding: 3px 10px; border-radius: 3px;
#     background: #0f2318; color: #34c759; border: 1px solid #1e4a2e;
# }
# .ground-caption {
#     font-size: 0.68rem; color: #4a5068; letter-spacing: 0.06em;
#     font-style: italic; margin-bottom: 0.7rem;
# }
# .box-list { display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.8rem; }
# .box-row {
#     display: flex; align-items: center; gap: 0.7rem;
#     padding: 0.4rem 0.7rem; border-radius: 6px;
#     background: #0d1017; border: 1px solid #1a1f2e;
#     font-size: 0.68rem;
# }
# .box-dot { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }
# .box-label { color: #c9cdd8; flex: 1; }
# .box-conf  { color: #4a5068; }
# .box-coords { color: #2e3448; font-size: 0.58rem; }
# .no-boxes {
#     font-size: 0.68rem; color: #3d4460; letter-spacing: 0.08em;
#     text-transform: uppercase; padding: 0.8rem 0;
# }

# /* ── history ── */
# .hist-item {
#     background: #0b0e18; border: 1px solid #1a1f2e;
#     border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem;
# }
# .hist-item:hover { border-color: #252a3e; }
# .hist-q  { font-size: 0.72rem; color: #5a6080; margin-bottom: 0.28rem; line-height: 1.4; }
# .hist-a  { font-family: 'Instrument Serif', serif; font-size: 1.05rem; color: #e2e4ea; margin-bottom: 0.45rem; }
# .hist-footer { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.3rem; }
# .hist-badge  { font-size: 0.56rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 7px; border-radius: 3px; }
# .hist-meta   { font-size: 0.58rem; color: #3d4460; }
# .hist-ground { font-size: 0.58rem; color: #34c759; }

# /* ── empty states ── */
# .empty-box {
#     display: flex; flex-direction: column; align-items: center;
#     justify-content: center; min-height: 180px; gap: 0.6rem; color: #2a2f42;
# }
# .empty-box .ei { font-size: 2rem; opacity: .4; }
# .empty-box .et { font-size: 0.66rem; letter-spacing: 0.1em; text-transform: uppercase; }
# .img-ph {
#     height: 180px; background: #0d1017; border: 1.5px dashed #1a1f2e;
#     border-radius: 10px; display: flex; flex-direction: column;
#     align-items: center; justify-content: center; gap: 0.5rem; color: #2a2f42;
# }
# .img-ph .pi { font-size: 1.8rem; }
# .img-ph .pt { font-size: 0.6rem; letter-spacing: 0.12em; text-transform: uppercase; }

# ::-webkit-scrollbar { width: 4px; }
# ::-webkit-scrollbar-track { background: transparent; }
# ::-webkit-scrollbar-thumb { background: #1e2438; border-radius: 2px; }
# </style>
# """, unsafe_allow_html=True)

# # ─────────────────────────────────────────────────────────────────────────────
# #  Session state
# # ─────────────────────────────────────────────────────────────────────────────

# DEFAULTS = {
#     "question_text": "", "selected_type": "Custom",
#     "last_preset": "",   "history": [],
#     "last_result": None, "grounding_override": False,
# }
# for k, v in DEFAULTS.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# # ─────────────────────────────────────────────────────────────────────────────
# #  Backend health
# # ─────────────────────────────────────────────────────────────────────────────

# def check_backend():
#     try:
#         r = requests.get(f"{BACKEND_URL}/health", timeout=3)
#         if r.status_code == 200:
#             return True, r.json()
#     except Exception:
#         pass
#     return False, {}

# backend_ok, health_data = check_backend()

# # ─────────────────────────────────────────────────────────────────────────────
# #  Callbacks
# # ─────────────────────────────────────────────────────────────────────────────

# def on_type_change():
#     chosen = st.session_state.type_selector
#     st.session_state.selected_type = chosen
#     preset = QUESTION_TYPES[chosen]
#     st.session_state.question_text = preset
#     st.session_state.last_preset   = preset
#     st.session_state.last_result   = None

# def on_question_change():
#     cur = st.session_state.question_input
#     st.session_state.question_text = cur
#     if cur != st.session_state.last_preset:
#         st.session_state.selected_type = "Custom"
#     st.session_state.last_result = None

# # ─────────────────────────────────────────────────────────────────────────────
# #  Helpers
# # ─────────────────────────────────────────────────────────────────────────────

# BOX_PALETTE = [
#     "#FF3B30","#FF9500","#FFCC00","#34C759",
#     "#00C7BE","#30B0C7","#007AFF","#AF52DE",
# ]

# def conf_color(c):
#     return "#10b981" if c >= 0.75 else "#f59e0b" if c >= 0.5 else "#ef4444"

# def conf_label(c):
#     return "High" if c >= 0.75 else "Medium" if c >= 0.5 else "Low"

# def b64_to_pil(b64str: str) -> Image.Image:
#     return Image.open(io.BytesIO(base64.b64decode(b64str)))

# # ─────────────────────────────────────────────────────────────────────────────
# #  Topbar
# # ─────────────────────────────────────────────────────────────────────────────

# dot    = "status-dot" if backend_ok else "status-dot off"
# slabel = "Online" if backend_ok else "Offline"
# vram   = ""
# if backend_ok and "vram_used_mb" in health_data:
#     vram = (f'<span class="status-label">'
#             f'VRAM {health_data["vram_used_mb"]:.0f}/'
#             f'{health_data["vram_total_mb"]:.0f} MB</span>')

# st.markdown(f"""
# <div class="topbar">
#   <div style="display:flex;align-items:baseline;gap:.7rem">
#     <span class="topbar-logo">VisionAsk</span>
#     <span class="topbar-sub">BLIP-2 · Grounding DINO · Visual QA</span>
#   </div>
#   <div style="display:flex;align-items:center;gap:.6rem">
#     {vram}
#     <span class="status-label">Backend {slabel}</span>
#     <span class="{dot}"></span>
#   </div>
# </div>
# """, unsafe_allow_html=True)

# if not backend_ok:
#     st.error("⚠️  Backend offline. Run: `uvicorn backend:app --host 0.0.0.0 --port 8000`")

# st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# # ─────────────────────────────────────────────────────────────────────────────
# #  Layout — 3 columns: controls | image | results
# # ─────────────────────────────────────────────────────────────────────────────

# col_ctrl, col_img, col_res = st.columns([4, 5, 5], gap="medium")

# # ══════════════════════════════════════════════════════════════
# #  LEFT — controls
# # ══════════════════════════════════════════════════════════════
# with col_ctrl:
#     st.markdown('<div style="padding:0 0.4rem">', unsafe_allow_html=True)

#     # upload
#     st.markdown('<div class="sec-label">01 · Image</div>', unsafe_allow_html=True)
#     uploaded = st.file_uploader(
#         "img", type=["jpg","jpeg","png","webp","bmp"],
#         label_visibility="collapsed"
#     )
#     pil_image = None
#     img_bytes  = None

#     if not uploaded:
#         st.markdown("""
#         <div class="img-ph">
#           <span class="pi">🖼</span>
#           <span class="pt">Drag & drop or click above</span>
#         </div>""", unsafe_allow_html=True)
#     else:
#         img_bytes = uploaded.read()
#         pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#         w, h = pil_image.size
#         st.markdown(f"""
#         <div style="display:flex;gap:.35rem;margin-bottom:.4rem;flex-wrap:wrap">
#           <span style="font-size:.58rem;background:#12151f;border:1px solid #1e2438;
#                        color:#5a6080;padding:2px 7px;border-radius:3px">{uploaded.name}</span>
#           <span style="font-size:.58rem;background:#12151f;border:1px solid #1e2438;
#                        color:#5a6080;padding:2px 7px;border-radius:3px">{w}×{h}</span>
#         </div>""", unsafe_allow_html=True)

#     st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

#     # question type
#     st.markdown('<div class="sec-label">02 · Question Type</div>', unsafe_allow_html=True)
#     type_list = list(QUESTION_TYPES.keys())
#     cur_idx   = type_list.index(st.session_state.selected_type)
#     tcolor    = TYPE_COLORS.get(st.session_state.selected_type, "#6b7280")

#     st.markdown(f"""
#     <div style="margin-bottom:.4rem;display:flex;gap:.4rem;align-items:center">
#       <span style="font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;
#                    padding:3px 9px;border-radius:3px;
#                    background:{tcolor}18;color:{tcolor};border:1px solid {tcolor}33">
#         {st.session_state.selected_type}
#       </span>
#       <span style="font-size:.56rem;background:#0f2318;color:#34c759;border:1px solid #1e4a2e;padding:2px 7px;border-radius:3px;letter-spacing:.08em">⚡ GROUNDING ON</span>
#     </div>""", unsafe_allow_html=True)

#     st.selectbox("qt", type_list, index=cur_idx,
#                  key="type_selector", on_change=on_type_change,
#                  label_visibility="collapsed")

#     st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

#     # question
#     st.markdown('<div class="sec-label">03 · Question</div>', unsafe_allow_html=True)
#     st.text_area("question",
#                  value=st.session_state.question_text,
#                  height=100,
#                  placeholder="Type a question, or pick a type above …",
#                  key="question_input",
#                  on_change=on_question_change,
#                  label_visibility="collapsed")

#     st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

#     run = st.button("⚡  Ask BLIP-2", use_container_width=True, disabled=not backend_ok)

#     st.markdown("""
#     <div style="font-size:.58rem;color:#3d4460;letter-spacing:.06em;margin-top:.4rem;text-align:center">
#       Grounding DINO runs automatically on every question
#     </div>""", unsafe_allow_html=True)

#     # ── inference ──
#     if run:
#         q = st.session_state.question_text.strip()
#         if pil_image is None:
#             st.warning("Upload an image first.")
#         elif not q:
#             st.warning("Enter or select a question.")
#         else:
#             with st.spinner("Running BLIP-2 + Grounding DINO …"):
#                 try:
#                     resp = requests.post(
#                         f"{BACKEND_URL}/ask",
#                         files={"image": (uploaded.name, img_bytes, uploaded.type)},
#                         data={
#                             "question":      q,
#                             "question_type": st.session_state.selected_type,
#                             "run_grounding": "yes",
#                         },
#                         timeout=180,
#                     )
#                     if resp.status_code == 200:
#                         d = resp.json()
#                         result = {
#                             "answer":          d["answer"],
#                             "confidence":      d["confidence"],
#                             "inference_ms":    d["inference_ms"],
#                             "device":          d["device"],
#                             "grounding_ran":   d["grounding_ran"],
#                             "grounding_ms":    d.get("grounding_ms"),
#                             "annotated_image": d.get("annotated_image"),
#                             "boxes":           d.get("boxes", []),
#                             "gdino_caption":   d.get("gdino_caption"),
#                             "question":        q,
#                             "qtype":           st.session_state.selected_type,
#                             "image_name":      uploaded.name,
#                             "timestamp":       datetime.now().strftime("%H:%M:%S"),
#                             "original_image":  img_bytes,
#                         }
#                         st.session_state.last_result = result
#                         st.session_state.history.insert(0, result)
#                         st.rerun()
#                     else:
#                         st.error(f"Backend {resp.status_code}: {resp.text}")
#                 except requests.exceptions.Timeout:
#                     st.error("Timed out. Grounding DINO on CPU takes ~10–20 s. Try again.")
#                 except Exception as e:
#                     st.error(f"Request failed: {e}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ══════════════════════════════════════════════════════════════
# #  MIDDLE — image viewer (original / annotated)
# # ══════════════════════════════════════════════════════════════
# with col_img:
#     st.markdown('<div style="padding:0 0.3rem">', unsafe_allow_html=True)
#     r = st.session_state.last_result

#     if r and r.get("annotated_image"):
#         # Show tabs: annotated | original
#         tab_ann, tab_orig = st.tabs(["🎯 Annotated", "🖼 Original"])
#         with tab_ann:
#             annotated_pil = b64_to_pil(r["annotated_image"])
#             st.image(annotated_pil, use_column_width=True)
#             if r.get("gdino_caption"):
#                 st.markdown(
#                     f'<div class="ground-caption">Caption sent to DINO: "{r["gdino_caption"]}"</div>',
#                     unsafe_allow_html=True
#                 )
#         with tab_orig:
#             orig_pil = Image.open(io.BytesIO(r["original_image"])).convert("RGB")
#             st.image(orig_pil, use_column_width=True)
#     elif pil_image is not None:
#         st.markdown('<div class="sec-label">Image Preview</div>', unsafe_allow_html=True)
#         st.image(pil_image, use_column_width=True)
#     else:
#         st.markdown("""
#         <div class="empty-box" style="min-height:300px">
#           <span class="ei">🔭</span>
#           <span class="et">Image will appear here</span>
#         </div>""", unsafe_allow_html=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ══════════════════════════════════════════════════════════════
# #  RIGHT — answer + box list + history
# # ══════════════════════════════════════════════════════════════
# with col_res:
#     st.markdown('<div style="padding:0 0.3rem">', unsafe_allow_html=True)

#     # ── latest answer ──
#     st.markdown('<div class="sec-label">Latest Answer</div>', unsafe_allow_html=True)

#     r = st.session_state.last_result
#     if r:
#         c  = r["confidence"]
#         bc = conf_color(c)
#         bw = int(c * 100)

#         total_ms = r["inference_ms"] + (r.get("grounding_ms") or 0)

#         st.markdown(f"""
#         <div class="answer-card">
#           <div class="answer-main">{r['answer']}</div>
#           <div class="meta-row">
#             <div class="meta-chip">Confidence <b>{c*100:.1f}% — {conf_label(c)}</b></div>
#             <div class="meta-chip">BLIP-2 <b>{r['inference_ms']} ms</b></div>
#             {'<div class="meta-chip">GDino <b>' + str(r["grounding_ms"]) + ' ms</b></div>' if r.get("grounding_ms") else ''}
#             <div class="meta-chip">Device <b>{r['device'].upper()}</b></div>
#           </div>
#           <div class="conf-bar-label">
#             <span>Token Confidence</span>
#             <span style="color:{bc}">{c*100:.1f}%</span>
#           </div>
#           <div class="conf-bar-track">
#             <div class="conf-bar-fill" style="width:{bw}%;background:{bc}"></div>
#           </div>
#         </div>
#         """, unsafe_allow_html=True)

#         # ── bounding box list ──
#         if r.get("grounding_ran"):
#             boxes = r.get("boxes", [])
#             st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
#             st.markdown(f"""
#             <div class="ground-header">
#               <span class="ground-badge">Grounding DINO</span>
#               <span style="font-size:.62rem;color:#4a5068">{len(boxes)} object{'s' if len(boxes)!=1 else ''} detected</span>
#             </div>""", unsafe_allow_html=True)

#             if boxes:
#                 box_html = '<div class="box-list">'
#                 for i, box in enumerate(boxes):
#                     col = BOX_PALETTE[i % len(BOX_PALETTE)]
#                     box_html += f"""
#                     <div class="box-row">
#                       <div class="box-dot" style="background:{col}"></div>
#                       <span class="box-label">{box['label']}</span>
#                       <span class="box-conf" style="color:{col}">{box['confidence']*100:.0f}%</span>
#                       <span class="box-coords">[{box['x0']},{box['y0']}→{box['x1']},{box['y1']}]</span>
#                     </div>"""
#                 box_html += '</div>'
#                 st.markdown(box_html, unsafe_allow_html=True)
#             else:
#                 st.markdown(
#                     '<div class="no-boxes">No objects detected above confidence threshold</div>',
#                     unsafe_allow_html=True
#                 )
#     else:
#         st.markdown("""
#         <div class="empty-box">
#           <span class="ei">💬</span>
#           <span class="et">Ask a question to see the answer here</span>
#         </div>""", unsafe_allow_html=True)

#     st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

#     # ── session history ──
#     hist_count = len(st.session_state.history)
#     hc1, hc2 = st.columns([6, 1])
#     with hc1:
#         st.markdown(
#             f'<div class="sec-label">History <span style="color:#4f6ef7;margin-left:.3rem">({hist_count})</span></div>',
#             unsafe_allow_html=True)
#     with hc2:
#         if hist_count > 0 and st.button("Clear", key="clr"):
#             st.session_state.history    = []
#             st.session_state.last_result = None
#             st.rerun()

#     if not st.session_state.history:
#         st.markdown("""
#         <div style="color:#2a2f42;font-size:.66rem;letter-spacing:.08em;
#                     text-transform:uppercase;padding:.8rem 0">
#             Q&amp;A history will appear here
#         </div>""", unsafe_allow_html=True)
#     else:
#         for h in st.session_state.history:
#             c  = h["confidence"]
#             tc = TYPE_COLORS.get(h["qtype"], "#6b7280")
#             bc = conf_color(c)
#             ground_tag = ""
#             if h.get("grounding_ran"):
#                 nb = len(h.get("boxes", []))
#                 ground_tag = f'<span class="hist-ground">🎯 {nb} box{"es" if nb!=1 else ""}</span>'
#             st.markdown(f"""
#             <div class="hist-item">
#               <div class="hist-q">❓ {h['question']}</div>
#               <div class="hist-a">{h['answer']}</div>
#               <div class="hist-footer">
#                 <span class="hist-badge"
#                       style="background:{tc}18;color:{tc};border:1px solid {tc}33">
#                   {h['qtype']}
#                 </span>
#                 {ground_tag}
#                 <span class="hist-meta" style="color:{bc}">{c*100:.0f}% conf</span>
#                 <span class="hist-meta">{h['timestamp']} · {h['inference_ms']}ms</span>
#               </div>
#             </div>""", unsafe_allow_html=True)

#     st.markdown('</div>', unsafe_allow_html=True)


























































































"""
app.py — Streamlit frontend for BLIP-2 + Grounding DINO VQA
Run with:  streamlit run app.py
Requires:  uvicorn backend:app --host 0.0.0.0 --port 8000
"""

import io
import base64
import requests
import streamlit as st
from PIL import Image
from datetime import datetime

BACKEND_URL = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VisionAsk · BLIP-2 + GDino",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_TYPES = {
    "Custom":                     "",
    "Counting":                   "How many objects are visible in this image?",
    "Object Recognition":         "What are the main objects present in this image?",
    "Object Presence":            "Is there a person in this image? What other objects are present?",
    "Color":                      "What are the dominant colors in this image?",
    "Scene Understanding":        "Describe the overall scene and setting of this image.",
    "Animal Recognition":         "What animal or animals are visible in this image? Describe their appearance.",
    "Activity Recognition":       "What activity or action is being performed in this image?",
    "Sports Recognition":         "What sport or athletic activity is shown in this image?",
    "Transportation Recognition": "What mode of transportation is visible in this image?",
    "Text Recognition":           "What text, signs, or written words are visible in this image?",
    "Food Recognition":           "What food or dish is shown in this image? Describe it.",
}

TYPE_COLORS = {
    "Custom":                     "#6b7280",
    "Counting":                   "#6366f1",
    "Object Recognition":         "#8b5cf6",
    "Object Presence":            "#10b981",
    "Color":                      "#f59e0b",
    "Scene Understanding":        "#06b6d4",
    "Animal Recognition":         "#84cc16",
    "Activity Recognition":       "#ec4899",
    "Sports Recognition":         "#14b8a6",
    "Transportation Recognition": "#f97316",
    "Text Recognition":           "#94a3b8",
    "Food Recognition":           "#ef4444",
}

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background: #f8f9fb;
    color: #2d3142;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 1rem 2rem 1rem !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none; }
[data-testid="stHorizontalBlock"] {
    gap: 1.5rem !important;
    align-items: flex-start !important;
}
[data-testid="stHorizontalBlock"] > div {
    overflow: visible !important;
    min-width: 0 !important;
}

/* ── topbar ── */
.topbar {
    position: sticky; top: 0; z-index: 100;
    background: rgba(248,249,251,0.96);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid #e2e5ed;
    padding: 0 2rem; height: 56px;
    display: flex; align-items: center; justify-content: space-between;
}
.topbar-logo {
    font-family: 'Instrument Serif', serif;
    font-size: 1.4rem; color: #1a1d2e; letter-spacing: -0.02em;
}
.topbar-sub {
    font-size: 0.58rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: #9098b0; margin-left: 0.8rem;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #10b981; display: inline-block;
}
.status-dot.off { background: #ef4444; }
.status-label {
    font-size: 0.6rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #9098b0; margin-right: 0.5rem;
}

/* ── section labels ── */
.sec-label {
    font-size: 0.58rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: #9098b0; margin-bottom: 0.55rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.sec-label::after { content:''; flex:1; height:1px; background:#e2e5ed; }

/* ── uploader ── */
[data-testid="stFileUploader"] > div {
    background: #ffffff !important; border: 1.5px dashed #d1d5e0 !important;
    border-radius: 10px !important; transition: border-color .25s, background .25s !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: #4f6ef7 !important; background: #f4f6ff !important;
}
[data-testid="stFileUploader"] label { display: none !important; }

/* ── image ── */
[data-testid="stImage"] img {
    border-radius: 10px !important; border: 1px solid #e2e5ed !important;
    width: 100% !important; max-height: 380px !important; object-fit: contain !important;
}

/* ── selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important; border: 1px solid #d1d5e0 !important;
    border-radius: 7px !important; color: #2d3142 !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important;
}

/* ── textarea ── */
textarea {
    background: #ffffff !important; border: 1px solid #d1d5e0 !important;
    border-radius: 7px !important; color: #2d3142 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important; resize: vertical !important;
}
textarea:focus {
    border-color: #4f6ef7 !important;
    box-shadow: 0 0 0 2px rgba(79,110,247,0.1) !important;
}

/* ── toggle / checkbox ── */
[data-testid="stCheckbox"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important; color: #6b7280 !important;
    letter-spacing: 0.04em !important;
}

/* ── buttons ── */
.stButton > button {
    width: 100%;
    background: #4f6ef7 !important;
    color: #fff !important; font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important; font-size: 0.82rem !important;
    letter-spacing: 0.06em !important; border: none !important;
    border-radius: 8px !important; padding: 0.72rem 0 !important;
    transition: opacity .2s, transform .1s !important;
}
.stButton > button:hover { opacity: .88 !important; transform: translateY(-1px) !important; }

/* ── answer card ── */
.answer-card {
    background: #ffffff; border: 1px solid #e2e5ed;
    border-radius: 12px; padding: 1.4rem 1.6rem;
    position: relative; overflow: hidden;
}
.answer-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: #4f6ef7;
}
.answer-main {
    font-family: 'Instrument Serif', serif;
    font-size: 1.55rem; color: #1a1d2e; line-height: 1.3; margin-bottom: 1rem;
}
.meta-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-bottom: 0.9rem; }
.meta-chip {
    font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 4px 10px; border-radius: 4px;
    background: #f1f3f8; border: 1px solid #e2e5ed; color: #6b7280;
}
.meta-chip b { color: #2d3142; font-weight: 500; }
.conf-bar-label {
    display: flex; justify-content: space-between;
    font-size: 0.58rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #9098b0; margin-bottom: 0.35rem;
}
.conf-bar-track { height: 4px; background: #e2e5ed; border-radius: 2px; overflow: hidden; }
.conf-bar-fill  { height: 100%; border-radius: 2px; }

/* ── grounding section ── */
.ground-header {
    display: flex; align-items: center; gap: 0.7rem;
    margin-bottom: 0.8rem;
}
.ground-badge {
    font-size: 0.58rem; letter-spacing: 0.14em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 3px;
    background: #edfbf4; color: #16a35a; border: 1px solid #b6ecd1;
}
.ground-caption {
    font-size: 0.68rem; color: #9098b0; letter-spacing: 0.06em;
    font-style: italic; margin-bottom: 0.7rem;
}
.box-list { display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.8rem; }
.box-row {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.4rem 0.7rem; border-radius: 6px;
    background: #f8f9fb; border: 1px solid #e2e5ed;
    font-size: 0.68rem;
}
.box-dot { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }
.box-label { color: #2d3142; flex: 1; }
.box-conf  { color: #6b7280; }
.box-coords { color: #adb5c8; font-size: 0.58rem; }
.no-boxes {
    font-size: 0.68rem; color: #9098b0; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 0.8rem 0;
}

/* ── history ── */
.hist-item {
    background: #ffffff; border: 1px solid #e2e5ed;
    border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem;
}
.hist-item:hover { border-color: #c5cade; }
.hist-q  { font-size: 0.72rem; color: #6b7280; margin-bottom: 0.28rem; line-height: 1.4; }
.hist-a  { font-family: 'Instrument Serif', serif; font-size: 1.05rem; color: #1a1d2e; margin-bottom: 0.45rem; }
.hist-footer { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.3rem; }
.hist-badge  { font-size: 0.56rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 7px; border-radius: 3px; }
.hist-meta   { font-size: 0.58rem; color: #9098b0; }
.hist-ground { font-size: 0.58rem; color: #16a35a; }

/* ── empty states ── */
.empty-box {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 180px; gap: 0.6rem; color: #c0c6d8;
}
.empty-box .ei { font-size: 2rem; opacity: .5; }
.empty-box .et { font-size: 0.66rem; letter-spacing: 0.1em; text-transform: uppercase; }
.img-ph {
    height: 180px; background: #ffffff; border: 1.5px dashed #d1d5e0;
    border-radius: 10px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 0.5rem; color: #c0c6d8;
}
.img-ph .pi { font-size: 1.8rem; }
.img-ph .pt { font-size: 0.6rem; letter-spacing: 0.12em; text-transform: uppercase; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d1d5e0; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "question_text": "", "selected_type": "Custom",
    "last_preset": "",   "history": [],
    "last_result": None, "grounding_override": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
#  Backend health
# ─────────────────────────────────────────────────────────────────────────────

def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, {}

backend_ok, health_data = check_backend()

# ─────────────────────────────────────────────────────────────────────────────
#  Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def on_type_change():
    chosen = st.session_state.type_selector
    st.session_state.selected_type = chosen
    preset = QUESTION_TYPES[chosen]
    st.session_state.question_text = preset
    st.session_state.last_preset   = preset
    st.session_state.last_result   = None

def on_question_change():
    cur = st.session_state.question_input
    st.session_state.question_text = cur
    if cur != st.session_state.last_preset:
        st.session_state.selected_type = "Custom"
    st.session_state.last_result = None

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

BOX_PALETTE = [
    "#FF3B30","#FF9500","#FFCC00","#34C759",
    "#00C7BE","#30B0C7","#007AFF","#AF52DE",
]

def conf_color(c):
    return "#10b981" if c >= 0.75 else "#f59e0b" if c >= 0.5 else "#ef4444"

def conf_label(c):
    return "High" if c >= 0.75 else "Medium" if c >= 0.5 else "Low"

def b64_to_pil(b64str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64str)))

# ─────────────────────────────────────────────────────────────────────────────
#  Topbar
# ─────────────────────────────────────────────────────────────────────────────

dot    = "status-dot" if backend_ok else "status-dot off"
slabel = "Online" if backend_ok else "Offline"
vram   = ""
if backend_ok and "vram_used_mb" in health_data:
    vram = (f'<span class="status-label">'
            f'VRAM {health_data["vram_used_mb"]:.0f}/'
            f'{health_data["vram_total_mb"]:.0f} MB</span>')

st.markdown(f"""
<div class="topbar">
  <div style="display:flex;align-items:baseline;gap:.7rem">
    <span class="topbar-logo">VisionAsk</span>
    <span class="topbar-sub">BLIP-2 · Grounding DINO · Visual QA</span>
  </div>
  <div style="display:flex;align-items:center;gap:.6rem">
    {vram}
    <span class="status-label">Backend {slabel}</span>
    <span class="{dot}"></span>
  </div>
</div>
""", unsafe_allow_html=True)

if not backend_ok:
    st.error("⚠️  Backend offline. Run: `uvicorn backend:app --host 0.0.0.0 --port 8000`")

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Layout — 3 columns: controls | image | results
# ─────────────────────────────────────────────────────────────────────────────

col_ctrl, col_img, col_res = st.columns([4, 5, 5], gap="medium")

# ══════════════════════════════════════════════════════════════
#  LEFT — controls
# ══════════════════════════════════════════════════════════════
with col_ctrl:
    st.markdown('<div style="padding:0 0.4rem">', unsafe_allow_html=True)

    # upload
    st.markdown('<div class="sec-label">01 · Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "img", type=["jpg","jpeg","png","webp","bmp"],
        label_visibility="collapsed"
    )
    pil_image = None
    img_bytes  = None

    if not uploaded:
        st.markdown("""
        <div class="img-ph">
          <span class="pi">🖼</span>
          <span class="pt">Drag & drop or click above</span>
        </div>""", unsafe_allow_html=True)
    else:
        img_bytes = uploaded.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = pil_image.size
        st.markdown(f"""
        <div style="display:flex;gap:.35rem;margin-bottom:.4rem;flex-wrap:wrap">
          <span style="font-size:.58rem;background:#f1f3f8;border:1px solid #e2e5ed;
                       color:#6b7280;padding:2px 7px;border-radius:3px">{uploaded.name}</span>
          <span style="font-size:.58rem;background:#f1f3f8;border:1px solid #e2e5ed;
                       color:#6b7280;padding:2px 7px;border-radius:3px">{w}×{h}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

    # question type
    st.markdown('<div class="sec-label">02 · Question Type</div>', unsafe_allow_html=True)
    type_list = list(QUESTION_TYPES.keys())
    cur_idx   = type_list.index(st.session_state.selected_type)
    tcolor    = TYPE_COLORS.get(st.session_state.selected_type, "#6b7280")

    st.markdown(f"""
    <div style="margin-bottom:.4rem;display:flex;gap:.4rem;align-items:center">
      <span style="font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;
                   padding:3px 9px;border-radius:3px;
                   background:{tcolor}18;color:{tcolor};border:1px solid {tcolor}33">
        {st.session_state.selected_type}
      </span>
      <span style="font-size:.56rem;background:#edfbf4;color:#16a35a;border:1px solid #b6ecd1;padding:2px 7px;border-radius:3px;letter-spacing:.08em">⚡ GROUNDING ON</span>
    </div>""", unsafe_allow_html=True)

    st.selectbox("qt", type_list, index=cur_idx,
                 key="type_selector", on_change=on_type_change,
                 label_visibility="collapsed")

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

    # question
    st.markdown('<div class="sec-label">03 · Question</div>', unsafe_allow_html=True)
    st.text_area("question",
                 value=st.session_state.question_text,
                 height=100,
                 placeholder="Type a question, or pick a type above …",
                 key="question_input",
                 on_change=on_question_change,
                 label_visibility="collapsed")

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

    run = st.button("⚡  Ask BLIP-2", use_container_width=True, disabled=not backend_ok)

    st.markdown("""
    <div style="font-size:.58rem;color:#9098b0;letter-spacing:.06em;margin-top:.4rem;text-align:center">
      Grounding DINO runs automatically on every question
    </div>""", unsafe_allow_html=True)

    # ── inference ──
    if run:
        q = st.session_state.question_text.strip()
        if pil_image is None:
            st.warning("Upload an image first.")
        elif not q:
            st.warning("Enter or select a question.")
        else:
            with st.spinner("Running BLIP-2 + Grounding DINO …"):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/ask",
                        files={"image": (uploaded.name, img_bytes, uploaded.type)},
                        data={
                            "question":      q,
                            "question_type": st.session_state.selected_type,
                            "run_grounding": "yes",
                        },
                        timeout=180,
                    )
                    if resp.status_code == 200:
                        d = resp.json()
                        result = {
                            "answer":          d["answer"],
                            "confidence":      d["confidence"],
                            "inference_ms":    d["inference_ms"],
                            "device":          d["device"],
                            "grounding_ran":   d["grounding_ran"],
                            "grounding_ms":    d.get("grounding_ms"),
                            "annotated_image": d.get("annotated_image"),
                            "boxes":           d.get("boxes", []),
                            "gdino_caption":   d.get("gdino_caption"),
                            "question":        q,
                            "qtype":           st.session_state.selected_type,
                            "image_name":      uploaded.name,
                            "timestamp":       datetime.now().strftime("%H:%M:%S"),
                            "original_image":  img_bytes,
                        }
                        st.session_state.last_result = result
                        st.session_state.history.insert(0, result)
                        st.rerun()
                    else:
                        st.error(f"Backend {resp.status_code}: {resp.text}")
                except requests.exceptions.Timeout:
                    st.error("Timed out. Grounding DINO on CPU takes ~10–20 s. Try again.")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  MIDDLE — image viewer (original / annotated)
# ══════════════════════════════════════════════════════════════
with col_img:
    st.markdown('<div style="padding:0 0.3rem">', unsafe_allow_html=True)
    r = st.session_state.last_result

    if r and r.get("annotated_image"):
        # Show tabs: annotated | original
        tab_ann, tab_orig = st.tabs(["🎯 Annotated", "🖼 Original"])
        with tab_ann:
            annotated_pil = b64_to_pil(r["annotated_image"])
            st.image(annotated_pil, use_column_width=True)
            if r.get("gdino_caption"):
                st.markdown(
                    f'<div class="ground-caption">Caption sent to DINO: "{r["gdino_caption"]}"</div>',
                    unsafe_allow_html=True
                )
        with tab_orig:
            orig_pil = Image.open(io.BytesIO(r["original_image"])).convert("RGB")
            st.image(orig_pil, use_column_width=True)
    elif pil_image is not None:
        st.markdown('<div class="sec-label">Image Preview</div>', unsafe_allow_html=True)
        st.image(pil_image, use_column_width=True)
    else:
        st.markdown("""
        <div class="empty-box" style="min-height:300px">
          <span class="ei">🔭</span>
          <span class="et">Image will appear here</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  RIGHT — answer + box list + history
# ══════════════════════════════════════════════════════════════
with col_res:
    st.markdown('<div style="padding:0 0.3rem">', unsafe_allow_html=True)

    # ── latest answer ──
    st.markdown('<div class="sec-label">Latest Answer</div>', unsafe_allow_html=True)

    r = st.session_state.last_result
    if r:
        c  = r["confidence"]
        bc = conf_color(c)
        bw = int(c * 100)

        total_ms = r["inference_ms"] + (r.get("grounding_ms") or 0)

        st.markdown(f"""
        <div class="answer-card">
          <div class="answer-main">{r['answer']}</div>
          <div class="meta-row">
            <div class="meta-chip">Confidence <b>{c*100:.1f}% — {conf_label(c)}</b></div>
            <div class="meta-chip">BLIP-2 <b>{r['inference_ms']} ms</b></div>
            {'<div class="meta-chip">GDino <b>' + str(r["grounding_ms"]) + ' ms</b></div>' if r.get("grounding_ms") else ''}
            <div class="meta-chip">Device <b>{r['device'].upper()}</b></div>
          </div>
          <div class="conf-bar-label">
            <span>Token Confidence</span>
            <span style="color:{bc}">{c*100:.1f}%</span>
          </div>
          <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width:{bw}%;background:{bc}"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── bounding box list ──
        if r.get("grounding_ran"):
            boxes = r.get("boxes", [])
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="ground-header">
              <span class="ground-badge">Grounding DINO</span>
              <span style="font-size:.62rem;color:#9098b0">{len(boxes)} object{'s' if len(boxes)!=1 else ''} detected</span>
            </div>""", unsafe_allow_html=True)

            if boxes:
                box_html = '<div class="box-list">'
                for i, box in enumerate(boxes):
                    col = BOX_PALETTE[i % len(BOX_PALETTE)]
                    box_html += f"""
                    <div class="box-row">
                      <div class="box-dot" style="background:{col}"></div>
                      <span class="box-label">{box['label']}</span>
                      <span class="box-conf" style="color:{col}">{box['confidence']*100:.0f}%</span>
                      <span class="box-coords">[{box['x0']},{box['y0']}→{box['x1']},{box['y1']}]</span>
                    </div>"""
                box_html += '</div>'
                st.markdown(box_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="no-boxes">No objects detected above confidence threshold</div>',
                    unsafe_allow_html=True
                )
    else:
        st.markdown("""
        <div class="empty-box">
          <span class="ei">💬</span>
          <span class="et">Ask a question to see the answer here</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── session history ──
    hist_count = len(st.session_state.history)
    hc1, hc2 = st.columns([6, 1])
    with hc1:
        st.markdown(
            f'<div class="sec-label">History <span style="color:#4f6ef7;margin-left:.3rem">({hist_count})</span></div>',
            unsafe_allow_html=True)
    with hc2:
        if hist_count > 0 and st.button("Clear", key="clr"):
            st.session_state.history    = []
            st.session_state.last_result = None
            st.rerun()

    if not st.session_state.history:
        st.markdown("""
        <div style="color:#9098b0;font-size:.66rem;letter-spacing:.08em;
                    text-transform:uppercase;padding:.8rem 0">
            Q&amp;A history will appear here
        </div>""", unsafe_allow_html=True)
    else:
        for h in st.session_state.history:
            c  = h["confidence"]
            tc = TYPE_COLORS.get(h["qtype"], "#6b7280")
            bc = conf_color(c)
            ground_tag = ""
            if h.get("grounding_ran"):
                nb = len(h.get("boxes", []))
                ground_tag = f'<span class="hist-ground">🎯 {nb} box{"es" if nb!=1 else ""}</span>'
            st.markdown(f"""
            <div class="hist-item">
              <div class="hist-q">❓ {h['question']}</div>
              <div class="hist-a">{h['answer']}</div>
              <div class="hist-footer">
                <span class="hist-badge"
                      style="background:{tc}18;color:{tc};border:1px solid {tc}33">
                  {h['qtype']}
                </span>
                {ground_tag}
                <span class="hist-meta" style="color:{bc}">{c*100:.0f}% conf</span>
                <span class="hist-meta">{h['timestamp']} · {h['inference_ms']}ms</span>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)