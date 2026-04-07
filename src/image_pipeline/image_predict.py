# ──────────────────────────────────────────────────────────────────────────────
# Imports & Constants
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Paths
PROJECT_ROOT   = Path(__file__).resolve().parents[2]   
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR    = PROJECT_ROOT / "reports" / "predictions"
CONFIG_PATH    = PROJECT_ROOT / "config.json"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# Load config 
with open(CONFIG_PATH) as f:
    cfg = json.load(f)

NUM_CLASSES  = cfg["num_classes"]         # 22
CLASS_NAMES  = cfg["class_names"]         # list of 22 names
NORM_MEAN    = cfg["norm_mean"]           # [0.3938, 0.304, 0.2797]
NORM_STD     = cfg["norm_std"]            # [0.3308, 0.2679, 0.2534]

# Weak classes (flag in output)
WEAK_CLASSES = {"Benign_tumors", "Tinea", "Psoriasis", "SkinCancer"}

# Confidence warning threshold 
LOW_CONF_THRESHOLD = 0.60   # warn if top-1 confidence < 60 %

# Supported image extensions 
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


TTA_TRANSFORMS = [
    # View 0 — identity
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]),
    # View 1 — horizontal flip
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]),
    # View 2 — rotate +10°
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(10, 10)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]),
    # View 3 — rotate -10°
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(-10, -10)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]),
    # View 4 — colour jitter
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]),
]


# ──────────────────────────────────────────────────────────────────────────────
# Model Definitions
# ──────────────────────────────────────────────────────────────────────────────

def build_classifier(in_features: int, num_classes: int, dropout: float = 0.4) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )


def build_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = build_classifier(2048, num_classes)
    return model


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier = build_classifier(1280, num_classes)
    return model


def build_mobilenet_v3(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    model.classifier = build_classifier(960, num_classes)
    return model


MODEL_REGISTRY = {
    "ResNet50":        (build_resnet50,       "ResNet50_best.pth"),
    "EfficientNetB0":  (build_efficientnet_b0, "EfficientNetB0_best.pth"),
    "MobileNetV3":     (build_mobilenet_v3,    "MobileNetV3_best.pth"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(name: str) -> nn.Module:
    """Build architecture, load checkpoint, set eval mode."""
    builder, ckpt_file = MODEL_REGISTRY[name]
    ckpt_path = CHECKPOINT_DIR / ckpt_file

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = builder(NUM_CLASSES)
    # weights_only=False suppresses PyTorch 2.6 warnings (known bug #5)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    val_acc = ckpt.get("val_acc", "N/A")
    epoch   = ckpt.get("epoch",   "N/A")
    print(f"  [✓] {name:<18}  epoch={epoch}  val_acc={val_acc}")
    return model


def load_all_models() -> dict:
    """Load all three models and return as a dict."""
    print("\n[INFO] Loading models...")
    models_dict = {}
    for name in MODEL_REGISTRY:
        models_dict[name] = load_model(name)
    return models_dict


# ──────────────────────────────────────────────────────────────────────────────
# TTA Inference for a Single PIL Image
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_single_model_tta(model: nn.Module, pil_img: Image.Image) -> torch.Tensor:
    """
    Run 5-view TTA on a single PIL image with one model.
    Returns averaged softmax probabilities of shape (num_classes,).

    TTA is applied to raw PIL images — NEVER to normalised tensors.
    """
    probs_list = []
    for tfm in TTA_TRANSFORMS:
        tensor = tfm(pil_img).unsqueeze(0).to(DEVICE)   # (1, C, H, W)
        logits = model(tensor)                            # (1, num_classes)
        probs  = F.softmax(logits, dim=1).squeeze(0)     # (num_classes,)
        probs_list.append(probs)

    avg_probs = torch.stack(probs_list).mean(dim=0)      # (num_classes,)
    return avg_probs


@torch.no_grad()
def ensemble_predict(models_dict: dict, pil_img: Image.Image) -> torch.Tensor:
    """
    Run TTA on all models and average their softmax probabilities.
    Returns ensemble probabilities of shape (num_classes,).
    """
    all_probs = []
    for model in models_dict.values():
        probs = predict_single_model_tta(model, pil_img)
        all_probs.append(probs)

    ensemble_probs = torch.stack(all_probs).mean(dim=0)  # average across models
    return ensemble_probs

def predict_image_proba(image_path, models_dict: dict) -> dict:
    """
    Returns {class_name: probability} dict for fusion with NLP model.
    Called by integrate_model.py
    """
    pil_img = Image.open(image_path).convert("RGB")
    probs   = ensemble_predict(models_dict, pil_img)  # tensor (22,)
    return {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))}

# ──────────────────────────────────────────────────────────────────────────────
# Top-3 Predictions & Result Formatting
# ──────────────────────────────────────────────────────────────────────────────

def get_top3(probs: torch.Tensor) -> list[dict]:
    """
    Given ensemble probabilities, return the top-3 predictions.
    Each entry: {"rank": int, "class": str, "confidence": float, "weak": bool}
    """
    top_vals, top_idxs = torch.topk(probs, k=3)
    results = []
    for rank, (conf, idx) in enumerate(zip(top_vals.tolist(), top_idxs.tolist()), start=1):
        class_name = CLASS_NAMES[idx]
        results.append({
            "rank":       rank,
            "class":      class_name,
            "confidence": conf,
            "weak":       class_name in WEAK_CLASSES,
        })
    return results


def format_result(image_path: str | Path, top3: list[dict]) -> str:
    """Format a single image's predictions for terminal output."""
    lines = [f"\n{'─'*60}", f"  Image : {Path(image_path).name}"]

    top1_conf = top3[0]["confidence"]
    if top1_conf < LOW_CONF_THRESHOLD:
        lines.append(f"  ⚠ LOW CONFIDENCE (top-1 = {top1_conf:.1%}) — result may be unreliable")

    for pred in top3:
        weak_tag = "  ← ⚠ WEAK CLASS" if pred["weak"] else ""
        lines.append(
            f"  #{pred['rank']}  {pred['class']:<22}  {pred['confidence']:.1%}{weak_tag}"
        )
    lines.append(f"{'─'*60}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CSV Output
# ──────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "image_path", "filename",
    "top1_class", "top1_conf",
    "top2_class", "top2_conf",
    "top3_class", "top3_conf",
    "low_confidence_warning", "weak_class_flag",
]


def write_csv_header(csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_csv_row(csv_path: Path, image_path: str | Path, top3: list[dict]) -> None:
    top1_conf = top3[0]["confidence"]
    row = {
        "image_path":             str(image_path),
        "filename":               Path(image_path).name,
        "top1_class":             top3[0]["class"],
        "top1_conf":              f"{top3[0]['confidence']:.4f}",
        "top2_class":             top3[1]["class"],
        "top2_conf":              f"{top3[1]['confidence']:.4f}",
        "top3_class":             top3[2]["class"],
        "top3_conf":              f"{top3[2]['confidence']:.4f}",
        "low_confidence_warning": "YES" if top1_conf < LOW_CONF_THRESHOLD else "NO",
        "weak_class_flag":        "YES" if top3[0]["weak"] else "NO",
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


# ──────────────────────────────────────────────────────────────────────────────
# Single Image Prediction
# ──────────────────────────────────────────────────────────────────────────────

def predict_image(image_path: str | Path, models_dict: dict, csv_path: Path) -> None:
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Cannot open image {image_path.name}: {e}")
        return

    probs = ensemble_predict(models_dict, pil_img)
    top3  = get_top3(probs)

    print(format_result(image_path, top3))
    append_csv_row(csv_path, image_path, top3)


# ──────────────────────────────────────────────────────────────────────────────
# Batch Folder Prediction
# ──────────────────────────────────────────────────────────────────────────────

def predict_folder(folder_path: str | Path, models_dict: dict, csv_path: Path) -> None:
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        print(f"[ERROR] Folder not found: {folder_path}")
        return

    image_files = sorted([
        p for p in folder_path.rglob("*")
        if p.suffix.lower() in IMG_EXTENSIONS
    ])

    if not image_files:
        print(f"[WARN] No images found in: {folder_path}")
        return

    print(f"\n[INFO] Found {len(image_files)} image(s) in: {folder_path}")
    print(f"[INFO] Results will be saved to: {csv_path}\n")

    for i, img_path in enumerate(image_files, start=1):
        print(f"  Processing {i}/{len(image_files)}: {img_path.name}", end="\r")
        predict_image(img_path, models_dict, csv_path)

    print(f"\n[INFO] Batch complete. {len(image_files)} images processed.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Skin Disease AI — Ensemble prediction with TTA"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single image file")
    group.add_argument("--folder", type=str, help="Path to a folder of images")
    args = parser.parse_args()

    # CSV output path (timestamped so runs don't overwrite each other)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = REPORTS_DIR / f"results_{timestamp}.csv"

    print("\n" + "═"*60)
    print("  Skin Disease AI — Ensemble Prediction")
    print("  Models : ResNet50 | EfficientNetB0 | MobileNetV3")
    print("  TTA    : 5 views per model  (15 forward passes/image)")
    print(f"  Output : {csv_path}")
    print("═"*60)

    # Load models 
    models_dict = load_all_models()

    # Prepare CSV 
    write_csv_header(csv_path)

    # Run prediction 
    if args.image:
        predict_image(args.image, models_dict, csv_path)
        print(f"\n[INFO] Results saved → {csv_path}")

    elif args.folder:
        predict_folder(args.folder, models_dict, csv_path)
        print(f"[INFO] Results saved → {csv_path}")


if __name__ == "__main__":
    main()
