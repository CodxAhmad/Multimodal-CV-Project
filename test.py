from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import logging
import pandas as pd
import torch
import os
import re

# -----------------------------------
# HIDE HF WARNINGS
# -----------------------------------

logging.set_verbosity_error()

# -----------------------------------
# TEXT NORMALIZATION
# -----------------------------------

number_map = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12"
}

# Words to strip before comparing (articles, filler)
STRIP_WORDS = {"a", "an", "the", "is", "are", "there", "some", "it", "yes,", "no,"}


def normalize_text(text):
    text = str(text).lower().strip()
    text = " ".join(text.split())

    # Convert number words → digits
    for word, digit in number_map.items():
        # Word boundary replacement so "one" doesn't match inside "stone"
        text = re.sub(rf"\b{word}\b", digit, text)

    # Remove punctuation except digits and letters
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())

    return text


def tokenize(text):
    """Return a set of individual word tokens from normalized text."""
    return set(normalize_text(text).split())


# -----------------------------------
# MATCHING LOGIC
# -----------------------------------

def is_match(predicted: str, expected: str) -> bool:
    """
    Multi-strategy matching — handles all the messy real-world cases:

    Strategy 1 — Exact match after normalization
        expected: "2"         predicted: "2"           → TRUE

    Strategy 2 — Predicted contains expected as a substring
        expected: "yellow"    predicted: "it is yellow" → TRUE
        expected: "2 cows"    predicted: "there are 2 cows in the field" → TRUE

    Strategy 3 — Expected contains predicted as a substring
        expected: "a taxi, a truck, and a car"   predicted: "taxi and truck" → TRUE

    Strategy 4 — High token overlap (Jaccard ≥ 0.5)
        expected: "2 cows"    predicted: "cows 2"      → TRUE
        Handles reordering and partial list answers.

    Strategy 5 — Yes / No soft match
        expected: "yes"       predicted: "yes there is a dog" → TRUE
        expected: "no"        predicted: "no there is not"    → TRUE

    Strategy 6 — Number match anywhere in either string
        expected: "3"         predicted: "i see 3 vehicles"  → TRUE
    """

    pred = normalize_text(predicted)
    exp  = normalize_text(expected)

    # Strategy 1: exact
    if pred == exp:
        return True

    # Strategy 2: expected is substring of predicted
    if exp in pred:
        return True

    # Strategy 3: predicted is substring of expected
    if pred in exp:
        return True

    # Strategy 4: token Jaccard overlap
    pred_tokens = tokenize(pred) - STRIP_WORDS
    exp_tokens  = tokenize(exp)  - STRIP_WORDS

    if pred_tokens and exp_tokens:
        intersection = pred_tokens & exp_tokens
        union        = pred_tokens | exp_tokens
        jaccard      = len(intersection) / len(union)

        if jaccard >= 0.5:
            return True

        # Bonus: all expected tokens found in predicted (list-answer case)
        # e.g. expected = "taxi truck car", predicted = "taxi and truck" → 2/3
        if len(exp_tokens) > 0:
            coverage = len(intersection) / len(exp_tokens)
            if coverage >= 0.6:
                return True

    # Strategy 5: yes / no soft match
    for yn in ("yes", "no"):
        if exp == yn and pred.startswith(yn):
            return True

    # Strategy 6: number match
    pred_nums = set(re.findall(r"\b\d+\b", pred))
    exp_nums  = set(re.findall(r"\b\d+\b", exp))

    if exp_nums and exp_nums == pred_nums:
        return True

    # Single number: expected is a digit, predicted contains that digit
    if len(exp_nums) == 1 and exp_nums & pred_nums:
        return True

    return False


# -----------------------------------
# LOAD MODEL
# -----------------------------------

print("Loading BLIP-2 model...")

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------------
# LOAD DATASET
# -----------------------------------

csv_path = "dataset/data.csv"
df = pd.read_csv(csv_path)

print("\nCSV Loaded Successfully")
print("Total Questions:", len(df))

# -----------------------------------
# RESULTS STORAGE
# -----------------------------------

results    = []
correct    = 0
total      = 0
type_stats = {}

# -----------------------------------
# MAIN LOOP
# -----------------------------------

for index, row in df.iterrows():

    image_name = row["image"]
    question   = row["question"]
    expected   = row["expected_answer"]
    qtype      = row["question_type"]

    image_path = os.path.join("images", image_name)

    print("\n-----------------------------------")
    print(f"[{index + 1}/{len(df)}] Image    : {image_name}")
    print(f"Question  : {question}")

    # -----------------------------------
    # LOAD IMAGE
    # -----------------------------------

    image = Image.open(image_path).convert("RGB")

    # -----------------------------------
    # CREATE PROMPT
    # -----------------------------------

    prompt = f"Question: {question} Answer:"

    # -----------------------------------
    # PROCESS INPUTS
    # -----------------------------------

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Cast pixel_values explicitly (model expects float16)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    # -----------------------------------
    # GENERATE ANSWER
    # -----------------------------------

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=4,
            early_stopping=True,
        )

    # -----------------------------------
    # DECODE OUTPUT
    # -----------------------------------

    raw_prediction = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    # Strip any echoed prompt (BLIP-2 sometimes repeats "Question: ... Answer:")
    clean = raw_prediction.lower()
    if "answer:" in clean:
        raw_prediction = raw_prediction[clean.index("answer:") + len("answer:"):].strip()

    print(f"Predicted : {raw_prediction}")
    print(f"Expected  : {expected}")

    # -----------------------------------
    # CHECK CORRECTNESS
    # -----------------------------------

    is_correct = is_match(raw_prediction, str(expected))
    print(f"Match     : {'✓ CORRECT' if is_correct else '✗ WRONG'}")

    # -----------------------------------
    # UPDATE STATS
    # -----------------------------------

    total += 1
    if is_correct:
        correct += 1

    if qtype not in type_stats:
        type_stats[qtype] = {"correct": 0, "total": 0}

    type_stats[qtype]["total"] += 1
    if is_correct:
        type_stats[qtype]["correct"] += 1

    # -----------------------------------
    # STORE RESULTS
    # -----------------------------------

    results.append({
        "image":            image_name,
        "question":         question,
        "expected_answer":  expected,
        "predicted_answer": raw_prediction,
        "question_type":    qtype,
        "correct":          is_correct,
    })

# -----------------------------------
# FINAL ACCURACY
# -----------------------------------

overall_accuracy = (correct / total) * 100 if total > 0 else 0.0

print("\n===================================")
print("FINAL RESULTS")
print("===================================")
print(f"Total Questions : {total}")
print(f"Correct Answers : {correct}")
print(f"Overall Accuracy: {overall_accuracy:.2f}%")

# -----------------------------------
# QUESTION TYPE ACCURACY
# -----------------------------------

print("\nQuestion Type Accuracy:")

# Sort by question type name for clean output
for qtype in sorted(type_stats.keys()):
    stats        = type_stats[qtype]
    type_accuracy = (stats["correct"] / stats["total"]) * 100
    print(
        f"{qtype:<30} "
        f"{type_accuracy:>6.2f}%  "
        f"({stats['correct']}/{stats['total']})"
    )

# -----------------------------------
# SAVE RESULTS CSV
# -----------------------------------

os.makedirs("outputs", exist_ok=True)

results_df  = pd.DataFrame(results)
output_path = "outputs/results.csv"

# Write TRUE/FALSE in caps to match your existing format
results_df["correct"] = results_df["correct"].map({True: "TRUE", False: "FALSE"})

results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")