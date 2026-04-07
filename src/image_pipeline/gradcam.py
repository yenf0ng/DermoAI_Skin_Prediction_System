# ==============================================================================
# Grad-CAM Visualisation — Skin Disease AI
# File   : src/image_pipeline/gradcam.py
#
# Run as script (recommended):
#   python src/imagmatplotlib.usee_pipeline/gradcam.py --image data/split/test/Acne/xxx.jpg
#   python src/image_pipeline/gradcam.py --image data/split/test/Acne/xxx.jpg --model ResNet50
#
# Run in Jupyter Notebook:
#   from src.image_pipeline.gradcam import run_gradcam
#   run_gradcam("data/split/test/Acne/xxx.jpg")
# ==============================================================================

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ------------------------------------------------------------------------------
# Paths — works as both a .py file AND in Jupyter Notebook
# ------------------------------------------------------------------------------
try:
    # Running as a normal .py script
    _THIS_FILE = Path(__file__).resolve()
    PROJECT_ROOT = _THIS_FILE.parents[2]
except NameError:
    # Running inside a Jupyter Notebook (__file__ is not defined)
    # Walk up from cwd until we find config.json
    _cwd = Path(os.getcwd()).resolve()
    PROJECT_ROOT = _cwd
    for _candidate in [_cwd, *_cwd.parents]:
        if (_candidate / "config.json").exists():
            PROJECT_ROOT = _candidate
            break

CONFIG_PATH = PROJECT_ROOT / "config.json"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
GRADCAM_DIR = PROJECT_ROOT / "reports" / "gradcam"

GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device       : {DEVICE}")
print(f"[INFO] Project root : {PROJECT_ROOT}")

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"config.json not found at {CONFIG_PATH}\n"
        f"Make sure you are running from inside the project folder, or that "
        f"PROJECT_ROOT is set correctly."
    )

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

NUM_CLASSES = cfg["num_classes"]          # 22
CLASS_NAMES = cfg["class_names"]          # list of 22 names
NORM_MEAN = cfg["norm_mean"]            # [0.3938, 0.304,  0.2797]
NORM_STD = cfg["norm_std"]             # [0.3308, 0.2679, 0.2534]

print(f"[INFO] num_classes  : {NUM_CLASSES}")
print(f"[INFO] norm_mean    : {NORM_MEAN}")
print(f"[INFO] norm_std     : {NORM_STD}")

# ------------------------------------------------------------------------------
# Transform  — uses project norm stats, NOT ImageNet defaults
# ------------------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

# ------------------------------------------------------------------------------
# Model definitions — MUST match image_predict.py exactly
# ------------------------------------------------------------------------------


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


# Registry: name -> (builder_fn, checkpoint_filename, hook_getter)
MODEL_REGISTRY = {
    "ResNet50":       (build_resnet50,        "ResNet50_best.pth", lambda m: m.layer4[-1]),
    "EfficientNetB0": (build_efficientnet_b0,  "EfficientNetB0_best.pth", lambda m: m.features[-1]),
    "MobileNetV3":    (build_mobilenet_v3,     "MobileNetV3_best.pth", lambda m: m.features[-1]),
}

_MODEL_CACHE = {}
# ------------------------------------------------------------------------------
# Checkpoint loader — weights_only=False matches image_predict.py
# ------------------------------------------------------------------------------


def load_model(name: str) -> nn.Module:
    if name in _MODEL_CACHE:        
        return _MODEL_CACHE[name]

    builder, ckpt_file, _ = MODEL_REGISTRY[name]
    ckpt_path = CHECKPOINT_DIR / ckpt_file

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Expected folder: {CHECKPOINT_DIR}"
        )

    model = builder(NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    _MODEL_CACHE[name] = model

    val_acc = ckpt.get("val_acc", "N/A")
    epoch = ckpt.get("epoch",   "N/A")
    print(f"  [OK] {name:<18}  epoch={epoch}  val_acc={val_acc}")
    return model


# ------------------------------------------------------------------------------
# Grad-CAM class
# ------------------------------------------------------------------------------
class GradCAM:
    """
    Registers forward + backward hooks on a target layer.
    Computes a class-discriminative heatmap (Grad-CAM).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._fwd_hook = target_layer.register_forward_hook(
            self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(
            self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()       # (1, C, H, W)

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # (1, C, H, W)

    def generate(self, input_tensor, class_idx=None):
        """
        Args:
            input_tensor : (1, 3, 224, 224) normalised tensor on DEVICE
                           Must have requires_grad=True
            class_idx    : target class; None -> use predicted class (argmax)
        Returns:
            cam       : np.ndarray (224, 224), values in [0, 1]
            pred_idx  : int   -- predicted class index
            pred_conf : float -- predicted class softmax confidence
        """
        self.model.zero_grad()

        logits = self.model(input_tensor)           # (1, num_classes)
        probs = F.softmax(logits, dim=1)
        pred_idx = int(logits.argmax(dim=1).item())

        if class_idx is None:
            class_idx = pred_idx

        pred_conf = float(probs[0, pred_idx].item())

        # Backward pass on the target class score only
        logits[0, class_idx].backward()

        # Global average pool the gradients -> channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations, then ReLU
        cam = (weights * self.activations).sum(dim=1,
                                               keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to 224x224
        cam = F.interpolate(cam, size=(224, 224),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()   # (224, 224)

        # Normalise to [0, 1]
        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)

        return cam, pred_idx, pred_conf

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ------------------------------------------------------------------------------
# Overlay helper
# ------------------------------------------------------------------------------
def overlay_heatmap(pil_img, cam, alpha=0.45):
    """Blend Grad-CAM heatmap over the original image. Returns RGB PIL Image."""
    img_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    colormap = matplotlib.colormaps["jet"]
    heatmap = colormap(cam)[:, :, :3]                    # drop alpha channel
    blended = np.clip(alpha * heatmap + (1 - alpha) * img_np, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))


# ------------------------------------------------------------------------------
# Main visualisation function (callable from notebook or CLI)
# ------------------------------------------------------------------------------
def run_gradcam(image_path, model_names=None, target_class_idx=None):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return None

    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    # Open image once
    pil_img = Image.open(image_path).convert("RGB")

    n_models = len(model_names)
    _results = []

    fig, axes = plt.subplots(n_models, 3, figsize=(
        12, 4 * n_models), squeeze=False)
    fig.suptitle(f"Grad-CAM -- {image_path.name}",
                 fontsize=14, fontweight="bold", y=1.01)

    print(f"\n[INFO] Running Grad-CAM on : {image_path.name}")
    print(f"[INFO] Models              : {', '.join(model_names)}\n")

    for row_idx, model_name in enumerate(model_names):
        print(f"  Processing {model_name} ...")

        # Load model and attach hooks
        model = load_model(model_name)
        _, _, get_layer = MODEL_REGISTRY[model_name]
        target_layer = get_layer(model)
        gradcam = GradCAM(model, target_layer)

        # Build input tensor with gradients enabled
        tensor = preprocess(pil_img).unsqueeze(0)
        tensor = tensor.requires_grad_(True)
        tensor = tensor.to(DEVICE)

        cam, pred_idx, pred_conf = gradcam.generate(
            tensor, class_idx=target_class_idx)
        gradcam.remove_hooks()
        _results.append((model_name, pred_idx, pred_conf))

        pred_name = CLASS_NAMES[pred_idx]
        overlay = overlay_heatmap(pil_img, cam)

        # Col 0 -- original
        axes[row_idx][0].imshow(pil_img.resize((224, 224)))
        axes[row_idx][0].set_title(f"{model_name}\nOriginal", fontsize=10)
        axes[row_idx][0].axis("off")

        # Col 1 -- heatmap
        axes[row_idx][1].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[row_idx][1].set_title(
            f"Heatmap\nPred: {pred_name}\n({pred_conf:.1%})", fontsize=9
        )
        axes[row_idx][1].axis("off")

        # Col 2 -- overlay
        axes[row_idx][2].imshow(overlay)
        axes[row_idx][2].set_title("Overlay", fontsize=10)
        axes[row_idx][2].axis("off")

        print(f"    Predicted : {pred_name}  ({pred_conf:.1%})")

    plt.tight_layout()

    save_path = GRADCAM_DIR / f"{image_path.stem}_gradcam.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Saved -> {save_path}")
    return {
        "png_path": save_path,
        "predictions": [
            {
                "model":      model_name,
                "class":      CLASS_NAMES[pred_idx],
                "confidence": pred_conf,
            }
            for model_name, pred_idx, pred_conf in _results
        ]
    }

# ------------------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM visualisation for Skin Disease AI"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image (jpg / png / bmp / etc.)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Single model to run (default: all three)"
    )
    args = parser.parse_args()

    model_names = [args.model] if args.model else None
    run_gradcam(args.image, model_names)


if __name__ == "__main__":
    main()
