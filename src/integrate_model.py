from pathlib import Path
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

NLP_MODEL_DIR = PROJECT_ROOT / "models" / "nlp_model" / "biobert_skin_model"
IMAGE_MODEL_ROOT = PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src" / "image_pipeline"))

from image_predict import load_all_models, predict_image_proba
from gradcam import run_gradcam

# =========================================================
# LOAD MODELS
# =========================================================
print("Loading tokenizer from:", NLP_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(
    NLP_MODEL_DIR.as_posix(), 
    use_fast=True,
    local_files_only=True)

print("Loading NLP model from:", NLP_MODEL_DIR)
nlp_model = AutoModelForSequenceClassification.from_pretrained(
    NLP_MODEL_DIR.as_posix(),
    local_files_only=True)

nlp_model.eval()

label_map_path = NLP_MODEL_DIR / "label_mapping.json"
with open(label_map_path, "r", encoding="utf-8") as f:
    label_map = json.load(f)

print("Loading image models from:", IMAGE_MODEL_ROOT)
image_models = load_all_models()
print("Image models loaded successfully")


# =========================================================
# HELPERS
# =========================================================
def canonical_label(label: str) -> str:
    x = label.strip().lower().replace("-", "_").replace(" ", "_")
    special_map = {
        "drugeruption": "drug_eruption",
        "skincancer": "skin_cancer",
        "sunsunlightdamage": "sun_sunlight_damage",
        "unknownnormal": "unknown_normal",
        "benigntumors": "benign_tumors",
        "actinickeratosis": "actinic_keratosis",
        "seborrhkeratoses": "seborrh_keratoses",
    }
    x_no_underscore = x.replace("_", "")
    return special_map.get(x_no_underscore, x)


def predict_nlp_proba(text: str) -> dict:
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = nlp_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0)

    result = {}
    for i in range(len(probs)):
        if str(i) in label_map:
            label = label_map[str(i)]
        else:
            label = label_map[i]
        result[label] = float(probs[i].item())

    return result


def normalize_prob_dict(prob_dict: dict) -> dict:
    return {canonical_label(label): float(score) for label, score in prob_dict.items()}


def fuse_probs(image_probs: dict, nlp_probs: dict, image_weight: float = 0.65, nlp_weight: float = 0.35, symptoms: str = "") -> dict:
    image_probs = normalize_prob_dict(image_probs)
    nlp_probs = normalize_prob_dict(nlp_probs)

    # If no real symptoms provided, skip NLP entirely and use image model only
    if not symptoms or symptoms.strip() in ["", "skin lesion"]:
        return image_probs

    all_labels = sorted(set(image_probs.keys()) | set(nlp_probs.keys()))
    fused = {}

    for label in all_labels:
        img_score = image_probs.get(label, 0.0)
        txt_score = nlp_probs.get(label, 0.0)
        fused[label] = image_weight * img_score + nlp_weight * txt_score

    return fused


def topk(prob_dict: dict, k: int = 3):
    return sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:k]


def format_topk(prob_dict: dict, k: int = 3):
    ranked = topk(prob_dict, k)
    return [
        {
            "rank": i + 1,
            "class": label,
            "confidence": float(score)
        }
        for i, (label, score) in enumerate(ranked)
    ]


def predict_integrated(image_path: Path, text: str, image_weight: float = 0.65, nlp_weight: float = 0.35) -> dict:
    image_probs = predict_image_proba(image_path, image_models)
    nlp_probs = predict_nlp_proba(text)
    fused_probs = fuse_probs(image_probs, nlp_probs, image_weight=image_weight, nlp_weight=nlp_weight, symptoms=text)

    gradcam_model = "EfficientNetB0"
    gradcam_result = run_gradcam(str(image_path), model_names=[gradcam_model])

    return {
        "input": {
            "image_path": str(image_path),
            "text": text
        },
        "image_topk": format_topk(normalize_prob_dict(image_probs), 3),
        "nlp_topk": format_topk(normalize_prob_dict(nlp_probs), 3),
        "final_topk": format_topk(fused_probs, 3),
        "final_prediction": topk(fused_probs, 1)[0][0],
        "weights": {
            "image_weight": image_weight,
            "nlp_weight": nlp_weight
        },
        "recommended_gradcam_model": gradcam_model,
        "gradcam": {
            "model_used": gradcam_model,
            "output_path": str(gradcam_result["png_path"]) if gradcam_result else None,
            "predictions": gradcam_result["predictions"] if gradcam_result else []
        }
    }


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    test_image_path = PROJECT_ROOT / "test_acne.jpeg"
    text = "itchy red bumps on face for one week"
    result = predict_integrated(test_image_path, text)
    print(json.dumps(result, indent=2))

    from PIL import Image
    gradcam_path = result["gradcam"]["output_path"]
    img = Image.open(gradcam_path)
    img.show()

