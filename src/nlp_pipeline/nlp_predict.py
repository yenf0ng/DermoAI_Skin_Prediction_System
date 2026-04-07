"""
nlp_predict.py — BioBERT symptom inference
Loads the saved biobert_skin_model/ and exposes predict_symptoms()
for use in fusion with the image model.
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR    = PROJECT_ROOT / "biobert_skin_model"
LABEL_MAP    = MODEL_DIR / "label_mapping.json"

# ── Constants (must match training) ───────────────────────────
MAX_LEN = 128
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model once at import time ─────────────────────────────
print(f"[INFO] Loading BioBERT from {MODEL_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
nlp_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
nlp_model.to(DEVICE)
nlp_model.eval()

with open(LABEL_MAP) as f:
    # keys are int-as-string e.g. "0", "1" — convert to int
    _label_map = {int(k): v for k, v in json.load(f).items()}

print(f"[INFO] BioBERT ready — {len(_label_map)} classes")


# ── Inference function (call this from fusion.py) ──────────────
@torch.no_grad()
def predict_symptoms(text: str, top_k: int = 3) -> list[dict]:
    """
    Args:
        text  : raw symptom string typed by user
        top_k : number of top predictions to return (default 3)
    Returns:
        list of dicts, e.g.:
        [
          {"rank": 1, "class": "Acne",    "confidence": 0.82},
          {"rank": 2, "class": "Rosacea", "confidence": 0.11},
          {"rank": 3, "class": "Eczema",  "confidence": 0.04},
        ]
    """
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    logits = nlp_model(
        enc["input_ids"].to(DEVICE),
        enc["attention_mask"].to(DEVICE)
    ).logits

    probs       = F.softmax(logits, dim=-1).squeeze(0)
    top_vals, top_idxs = torch.topk(probs, k=top_k)

    return [
        {
            "rank":       rank,
            "class":      _label_map[idx.item()],
            "confidence": conf.item(),
        }
        for rank, (conf, idx) in enumerate(zip(top_vals, top_idxs), start=1)
    ]