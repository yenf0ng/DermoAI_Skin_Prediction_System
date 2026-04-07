# =============================================================
# FINAL EVALUATION
# =============================================================

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

# Add project root to path so script runs from anywhere
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

# =============================================================
# PATHS + CONFIG
# =============================================================

SPLIT_DIR   = Path(PROJECT_ROOT) / "data"    / "split"
CONFIG_PATH = Path(PROJECT_ROOT) / "config.json"
CW_PATH     = Path(PROJECT_ROOT) / "reports" / "class_weights.json"
NORM_PATH   = Path(PROJECT_ROOT) / "reports" / "normalization_stats.json"
CKPT_DIR    = Path(PROJECT_ROOT) / "checkpoints"
RESULTS_DIR = Path(PROJECT_ROOT) / "reports" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("SECTION 4 — FINAL EVALUATION")
print("=" * 60)
print(f"Device       : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"VRAM         : "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Project root : {PROJECT_ROOT}")
print(f"Results dir  : {RESULTS_DIR}")

# Verify all required paths 
required = {
    "config.json"              : CONFIG_PATH,
    "class_weights.json"       : CW_PATH,
    "normalization_stats.json" : NORM_PATH,
    "train dir"                : SPLIT_DIR / "train",
    "val dir"                  : SPLIT_DIR / "val",
    "test dir"                 : SPLIT_DIR / "test",
    "ResNet50 checkpoint"      : CKPT_DIR / "ResNet50_best.pth",
    "EfficientNetB0 checkpoint": CKPT_DIR / "EfficientNetB0_best.pth",
    "MobileNetV3 checkpoint"   : CKPT_DIR / "MobileNetV3_best.pth",
}

all_ok = True
print(f"\nPath verification:")
for label, path in required.items():
    ok = path.exists()
    if not ok:
        all_ok = False
    print(f"  {'OK' if ok else 'MISSING'} {label:<30} {path}")

assert all_ok, "\nFix missing paths above before continuing."

# Load configs
with open(CONFIG_PATH) as f:
    config = json.load(f)

with open(NORM_PATH) as f:
    norm_stats = json.load(f)

with open(CW_PATH) as f:
    cw_data = json.load(f)

NUM_CLASSES = config["num_classes"]
CLASS_NAMES = config["class_names"]

# Use mean_list / std_list — already plain lists, as noted in the file
NORM_MEAN = norm_stats["mean_list"]
NORM_STD  = norm_stats["std_list"]

# Class weight tensor — weight_array already index-ordered
class_weight_tensor = torch.tensor(
    cw_data["weight_array"],
    dtype=torch.float32
).to(DEVICE)

assert len(cw_data["weight_array"]) == NUM_CLASSES, \
    "weight_array length mismatch"
assert cw_data["class_names"] == CLASS_NAMES, \
    "class_names order mismatch"

print(f"\nConfig loaded")
print(f"   num_classes  : {NUM_CLASSES}")
print(f"   norm_mean    : {NORM_MEAN}")
print(f"   norm_std     : {NORM_STD}")
print(f"   weight range : {class_weight_tensor.min():.4f} – "
      f"{class_weight_tensor.max():.4f}")

# Checkpoint summary 
print(f"\nCheckpoint details:")
for name in ["ResNet50", "EfficientNetB0", "MobileNetV3"]:
    ckpt    = torch.load(
        CKPT_DIR / f"{name}_best.pth",
        map_location="cpu",
        weights_only=False,
    )
    epoch   = ckpt.get("epoch",   "?")
    val_acc = ckpt.get("val_acc", 0)
    size_mb = (CKPT_DIR / f"{name}_best.pth").stat().st_size / (1024**2)
    print(f"  OK {name:<18} epoch={epoch:<4} "
          f"val_acc={val_acc:.2f}%  ({size_mb:.1f} MB)")

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"{'='*60}")

# =============================================================
# TRANSFORMS + TTA + TEST DATASET
# =============================================================

# Clean test transform — no augmentation
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

# 5 TTA views — applied to raw PIL images from disk
# NOTE: TTA must be applied at the dataset level (not on tensors)
TTA_TRANSFORMS = [
    # View 1: clean baseline
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]),
    # View 2: horizontal flip
    transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]),
    # View 3: clockwise rotation
    transforms.Compose([
        transforms.RandomRotation(degrees=(10, 10)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]),
    # View 4: counter-clockwise rotation
    transforms.Compose([
        transforms.RandomRotation(degrees=(-10, -10)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]),
    # View 5: colour jitter
    transforms.Compose([
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]),
]

# Test dataset 
test_dataset = ImageFolder(
    root=SPLIT_DIR / "test",
    transform=test_transform,
)

assert test_dataset.classes == CLASS_NAMES, (
    f"Class order mismatch!\n"
    f"   ImageFolder : {test_dataset.classes}\n"
    f"   config.json : {CLASS_NAMES}"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,      
    pin_memory=True,
)

# Per-class counts 
test_labels = [label for _, label in test_dataset.samples]
test_counts = Counter(test_labels)

print(f"\nTest set breakdown:")
print(f"  {'Idx':<5} {'Class':<30} {'Count':>6}")
print(f"  {'-'*5} {'-'*30} {'-'*6}")
for idx, name in enumerate(CLASS_NAMES):
    print(f"  {idx:<5} {name:<30} {test_counts[idx]:>6}")
print(f"\n  Total test samples : {len(test_dataset):,}")
print(f"  Total test batches : {len(test_loader)}")
print(f"  TTA views          : {len(TTA_TRANSFORMS)}")

# Batch sanity check 
images, labels = next(iter(test_loader))
print(f"\nBatch sanity check:")
print(f"   Image shape  : {images.shape}")
print(f"   Label shape  : {labels.shape}")
print(f"   Pixel range  : [{images.min():.3f}, {images.max():.3f}]")

assert images.shape[1:] == (3, 224, 224), \
    f"Wrong image shape: {images.shape}"
assert labels.max().item() < NUM_CLASSES, \
    f"Label out of range: {labels.max().item()}"

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"{'='*60}")

# =============================================================
# MODEL LOADER
# =============================================================

def build_classifier(in_features, num_classes, dropout=0.4):
    """Same head architecture used during training."""
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )


def load_model(model_name):
    """Rebuilds backbone architecture and loads best checkpoint weights."""

    if model_name == "ResNet50":
        model    = models.resnet50(weights=None)
        model.fc = build_classifier(model.fc.in_features, NUM_CLASSES)

    elif model_name == "EfficientNetB0":
        model            = models.efficientnet_b0(weights=None)
        in_feat          = model.classifier[1].in_features
        model.classifier = build_classifier(in_feat, NUM_CLASSES)

    elif model_name == "MobileNetV3":
        model            = models.mobilenet_v3_large(weights=None)
        in_feat          = model.classifier[0].in_features
        model.classifier = build_classifier(in_feat, NUM_CLASSES)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(
        CKPT_DIR / f"{model_name}_best.pth",
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)
    model.eval()

    val_acc      = ckpt.get("val_acc", 0)
    epoch        = ckpt.get("epoch",   "?")
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  OK {model_name:<18} "
          f"val_acc={val_acc:.2f}%  "
          f"epoch={epoch:<4} "
          f"params={total_params:,}")
    return model


print(f"\n{'='*60}")
print("LOADING ALL 3 MODELS")
print("=" * 60)

MODELS = {}
for name in ["ResNet50", "EfficientNetB0", "MobileNetV3"]:
    MODELS[name] = load_model(name)

# Forward pass verification
print(f"\nForward pass verification:")
dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
with torch.no_grad():
    for name, model in MODELS.items():
        with torch.amp.autocast('cuda'):
            out = model(dummy)
        assert out.shape == (2, NUM_CLASSES), \
            f"Wrong output shape: {out.shape}"
        print(f"  OK {name:<18} {dummy.shape} → {out.shape}")
del dummy

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"{'='*60}")

# =============================================================
# TEST EVALUATION WITH FULL TTA
# =============================================================
def evaluate_with_tta(model, model_name):
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []
    start      = time.time()

    # One DataLoader per TTA view
    tta_loaders = []
    for t in TTA_TRANSFORMS:
        ds = ImageFolder(root=SPLIT_DIR / "test", transform=t)
        ldr = DataLoader(
            ds,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        tta_loaders.append(ldr)

    total_batches = len(tta_loaders[0])

    with torch.no_grad():
        for batch_idx, batches in enumerate(zip(*tta_loaders)):
            # labels identical across views (shuffle=False, same dataset order)
            labels = batches[0][1].to(DEVICE, non_blocking=True)

            tta_probs = []
            for images, _ in batches:
                images = images.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                tta_probs.append(
                    torch.softmax(logits.float(), dim=1)
                )

            # Average probabilities across all 5 TTA views
            avg_probs = torch.stack(tta_probs).mean(dim=0)
            preds     = avg_probs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(avg_probs.cpu())

            if (batch_idx + 1) % 100 == 0:
                print(f"  {model_name} — "
                      f"[{batch_idx+1}/{total_batches}]")

    elapsed    = time.time() - start
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()
    acc        = accuracy_score(all_labels, all_preds) * 100

    print(f"\n  OK {model_name:<18} "
          f"test acc: {acc:.2f}%  "
          f"({elapsed:.0f}s  {len(all_labels):,} samples)")

    return all_preds, all_labels, all_probs, acc


print(f"\n{'='*60}")
print("RUNNING TEST EVALUATION WITH FULL TTA")
print(f"  Test samples : {len(test_dataset):,}")
print(f"  TTA views    : {len(TTA_TRANSFORMS)}")
print(f"  Batches/view : {len(test_loader)}")
print("=" * 60)

results     = {}
total_start = time.time()

for name, model in MODELS.items():
    print(f"\nEvaluating {name}...")
    preds, labels, probs, acc = evaluate_with_tta(model, name)
    results[name] = {
        "preds"  : preds,
        "labels" : labels,
        "probs"  : probs,
        "acc"    : acc,
    }

total_elapsed = time.time() - total_start

print(f"\n{'='*60}")
print(f"TEST EVALUATION COMPLETE  ({total_elapsed:.0f}s total)")
print(f"{'='*60}")
print(f"\n  {'Model':<18} {'Val Acc':>9} {'Test Acc':>10} {'Diff':>8}")
print(f"  {'-'*18} {'-'*9} {'-'*10} {'-'*8}")

for name in ["ResNet50", "EfficientNetB0", "MobileNetV3"]:
    ckpt     = torch.load(
        CKPT_DIR / f"{name}_best.pth",
        map_location="cpu",
        weights_only=False,
    )
    val_acc  = ckpt.get("val_acc", 0)
    test_acc = results[name]["acc"]
    diff     = test_acc - val_acc
    flag     = "UP" if diff >= 0 else "DOWN"
    print(f"  {name:<18} {val_acc:>8.2f}% "
          f"{test_acc:>9.2f}% "
          f"  {flag} {abs(diff):.2f}%")

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"{'='*60}")

# =============================================================
# CONFUSION MATRIX
# =============================================================

def plot_confusion_matrices(results, class_names, save_path):
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(10 * n_models, 10))
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        "Confusion Matrices — Test Set (with TTA)",
        fontsize=18, fontweight="bold", y=1.01
    )

    for ax, (model_name, res) in zip(axes, results.items()):
        cm      = confusion_matrix(res["labels"], res["preds"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        short   = [c[:12] for c in class_names]

        sns.heatmap(
            cm_norm,
            annot=True, fmt=".2f",
            cmap="Blues",
            xticklabels=short,
            yticklabels=short,
            ax=ax,
            linewidths=0.3,
            linecolor="white",
            vmin=0, vmax=1,
            annot_kws={"size": 7},
        )
        ax.set_title(
            f"{model_name}\nTest Acc: {res['acc']:.2f}%",
            fontsize=13, fontweight="bold", pad=12
        )
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True",      fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  OK Saved → {save_path.name}")


print(f"\n{'='*60}")
print("GENERATING CONFUSION MATRICES")
print("=" * 60)

plot_confusion_matrices(
    results, CLASS_NAMES,
    RESULTS_DIR / "confusion_matrices.png"
)

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"{'='*60}")

# =============================================================
# PER-CLASS ACCURACY REPORT
# =============================================================

def build_per_class_report(results, class_names, save_path):
    weak_classes = {"Benign_tumors", "Tinea", "SkinCancer", "Psoriasis"}
    rows = []

    for cls_idx, cls_name in enumerate(class_names):
        row = {
            "Class": cls_name,
            "Weak" : "WEAK" if cls_name in weak_classes else ""
        }
        for model_name, res in results.items():
            mask    = res["labels"] == cls_idx
            correct = (res["preds"][mask] == cls_idx).sum()
            total   = mask.sum()
            acc     = (correct / total * 100) if total > 0 else 0.0
            row[model_name] = round(float(acc), 1)
        rows.append(row)

    df            = pd.DataFrame(rows)
    model_cols    = list(results.keys())
    df["Average"] = df[model_cols].mean(axis=1).round(1)
    df            = df.sort_values("Average").reset_index(drop=True)

    df.to_csv(save_path, index=False)
    print(f"  OK Saved → {save_path.name}")

    print(f"\n  {'Class':<30} {'ResNet50':>10} "
          f"{'EffNetB0':>10} {'MobileV3':>10} "
          f"{'Average':>10} {'':>6}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

    for _, row in df.iterrows():
        print(f"  {row['Class']:<30} "
              f"{row[model_cols[0]]:>9.1f}% "
              f"{row[model_cols[1]]:>9.1f}% "
              f"{row[model_cols[2]]:>9.1f}% "
              f"{row['Average']:>9.1f}% "
              f"  {row['Weak']}")
    return df


print(f"\n{'='*60}")
print("PER-CLASS ACCURACY REPORT")
print("=" * 60)

per_class_df = build_per_class_report(
    results, CLASS_NAMES,
    RESULTS_DIR / "per_class_accuracy.csv"
)

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"{'='*60}")

# =============================================================
# MODEL COMPARISON TABLE
# =============================================================

def build_model_comparison(results, class_names,
                            save_path_json, save_path_txt):
    model_names       = list(results.keys())
    metrics_collected = {}

    for model_name, res in results.items():
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
            res["labels"], res["preds"],
            average="macro", zero_division=0
        )
        p_wtd, r_wtd, f1_wtd, _ = precision_recall_fscore_support(
            res["labels"], res["preds"],
            average="weighted", zero_division=0
        )
        metrics_collected[model_name] = {
            "test_accuracy"      : round(res["acc"],     2),
            "precision_macro"    : round(p_mac  * 100,   2),
            "recall_macro"       : round(r_mac  * 100,   2),
            "f1_macro"           : round(f1_mac * 100,   2),
            "precision_weighted" : round(p_wtd  * 100,   2),
            "recall_weighted"    : round(r_wtd  * 100,   2),
            "f1_weighted"        : round(f1_wtd * 100,   2),
        }

    metric_labels = [
        ("Test Accuracy (%)",      "test_accuracy"),
        ("Precision Macro (%)",    "precision_macro"),
        ("Recall Macro (%)",       "recall_macro"),
        ("F1 Macro (%)",           "f1_macro"),
        ("Precision Weighted (%)","precision_weighted"),
        ("Recall Weighted (%)",    "recall_weighted"),
        ("F1 Weighted (%)",        "f1_weighted"),
    ]

    header = (f"{'Metric':<25} {'ResNet50':>12} "
              f"{'EfficientNetB0':>16} {'MobileNetV3':>13}")
    sep    = "-" * 70

    report_lines = [
        "MODEL COMPARISON — TEST SET WITH TTA",
        "=" * 70,
        header,
        sep,
    ]

    print(f"\n  {header}")
    print(f"  {sep}")

    for label, key in metric_labels:
        vals = [metrics_collected[m][key] for m in model_names]
        best = max(vals)
        row  = f"  {label:<25}"
        for v in vals:
            marker = " <--" if v == best else "    "
            row   += f" {v:>10.2f}%{marker}"
        print(row)
        report_lines.append(row)

    report_lines.append("=" * 70)
    report_lines.append("<-- = best across models for that metric")

    for model_name, res in results.items():
        report_lines += [
            f"\n\n{'='*70}",
            f"FULL CLASSIFICATION REPORT — {model_name}",
            "=" * 70,
            classification_report(
                res["labels"], res["preds"],
                target_names=class_names,
                zero_division=0
            ),
        ]

    with open(save_path_json, "w") as f:
        json.dump(metrics_collected, f, indent=2)
    print(f"\n  OK Saved → {save_path_json.name}")

    with open(save_path_txt, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  OK Saved → {save_path_txt.name}")

    return metrics_collected


print(f"\n{'='*60}")
print("MODEL COMPARISON TABLE")
print("=" * 60)

comparison = build_model_comparison(
    results, CLASS_NAMES,
    RESULTS_DIR / "model_comparison.json",
    RESULTS_DIR / "classification_report.txt",
)

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"{'='*60}")

# =============================================================
# LEARNING CURVES
# =============================================================

def plot_learning_curves(histories_path, save_path):
    if not histories_path.exists():
        print(f"  WARNING: all_histories.json not found — skipping")
        print(f"           Expected: {histories_path}")
        return

    with open(histories_path) as f:
        histories = json.load(f)

    colors = {
        "ResNet50"      : "#E63946",
        "EfficientNetB0": "#2196F3",
        "MobileNetV3"   : "#4CAF50",
    }

    model_names = list(histories.keys())
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "Learning Curves — All Models",
        fontsize=16, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3)

    for col, model_name in enumerate(model_names):
        h      = histories[model_name]
        epochs = range(1, len(h["train_loss"]) + 1)
        color  = colors.get(model_name, "#666")

        # Loss subplot
        ax_loss = fig.add_subplot(gs[0, col])
        ax_loss.plot(epochs, h["train_loss"],
                     color=color, linewidth=2,
                     label="Train", alpha=0.9)
        ax_loss.plot(epochs, h["val_loss"],
                     color=color, linewidth=2,
                     linestyle="--", label="Val", alpha=0.7)
        ax_loss.set_title(f"{model_name}\nLoss", fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=9)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_xlim(1, len(epochs))

        # Accuracy subplot
        ax_acc = fig.add_subplot(gs[1, col])
        ax_acc.plot(epochs, h["train_acc"],
                    color=color, linewidth=2,
                    label="Train", alpha=0.9)
        ax_acc.plot(epochs, h["val_acc"],
                    color=color, linewidth=2,
                    linestyle="--", label="Val", alpha=0.7)

        best_val = max(h["val_acc"])
        best_ep  = h["val_acc"].index(best_val) + 1
        ax_acc.axvline(x=best_ep, color=color,
                       linewidth=1, linestyle=":", alpha=0.6)
        ax_acc.scatter([best_ep], [best_val],
                       color=color, s=80, zorder=5,
                       label=f"Best: {best_val:.1f}%")

        ax_acc.set_title(f"{model_name}\nAccuracy", fontweight="bold")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.legend(fontsize=9)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_xlim(1, len(epochs))
        ax_acc.set_ylim(0, 100)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  OK Saved → {save_path.name}")


print(f"\n{'='*60}")
print("GENERATING LEARNING CURVES")
print("=" * 60)

plot_learning_curves(
    CKPT_DIR / "all_histories.json",
    RESULTS_DIR / "learning_curves.png",
)

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"{'='*60}")

# =============================================================
# WEAK CLASS DEEP DIVE
# =============================================================

def weak_class_analysis(results, class_names, weak_classes, save_path):
    weak_indices = [
        class_names.index(c) for c in weak_classes
        if c in class_names
    ]

    lines = [
        "WEAK CLASS ANALYSIS",
        "=" * 70,
        f"Classes analysed: {', '.join(weak_classes)}\n",
    ]

    print(f"\n  {'Class':<25} {'ResNet50':>10} "
          f"{'EffNetB0':>10} {'MobileV3':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    for cls_idx in weak_indices:
        cls_name = class_names[cls_idx]
        lines   += [f"\n{'─'*50}", f"Class: {cls_name}"]
        row      = f"  {cls_name:<25}"

        for model_name, res in results.items():
            mask    = res["labels"] == cls_idx
            correct = (res["preds"][mask] == cls_idx).sum()
            total   = mask.sum()
            acc     = (correct / total * 100) if total > 0 else 0.0
            row    += f" {acc:>9.1f}%"

            wrong = res["preds"][mask][res["preds"][mask] != cls_idx]
            if len(wrong) > 0:
                unique, counts = np.unique(wrong, return_counts=True)
                top3           = np.argsort(-counts)[:3]
                confusions     = [
                    f"{class_names[unique[i]]} ({counts[i]})"
                    for i in top3
                ]
                lines.append(
                    f"  {model_name}: {acc:.1f}% correct | "
                    f"confused with: {', '.join(confusions)}"
                )
            else:
                lines.append(f"  {model_name}: {acc:.1f}% correct")

        print(row)

    lines.append("\n" + "=" * 70)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  OK Saved → {save_path.name}")


print(f"\n{'='*60}")
print("WEAK CLASS DEEP DIVE")
print("=" * 60)

weak_class_analysis(
    results, CLASS_NAMES,
    weak_classes=["Benign_tumors", "Tinea", "SkinCancer", "Psoriasis"],
    save_path=RESULTS_DIR / "weak_class_analysis.txt",
)

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"{'='*60}")

# =============================================================
# FINAL SUMMARY
# =============================================================

print(f"\n{'='*60}")
print("SECTION 4 COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"\n  {'Model':<18} {'Val Acc':>9} {'Test Acc':>10} "
      f"{'F1 Macro':>10} {'F1 Weighted':>13}")
print(f"  {'-'*18} {'-'*9} {'-'*10} {'-'*10} {'-'*13}")

for model_name in ["ResNet50", "EfficientNetB0", "MobileNetV3"]:
    ckpt     = torch.load(
        CKPT_DIR / f"{model_name}_best.pth",
        map_location="cpu", weights_only=False,
    )
    val_acc  = ckpt.get("val_acc", 0)
    test_acc = comparison[model_name]["test_accuracy"]
    f1_mac   = comparison[model_name]["f1_macro"]
    f1_wtd   = comparison[model_name]["f1_weighted"]
    print(f"  {model_name:<18} {val_acc:>8.2f}% "
          f"{test_acc:>9.2f}% "
          f"{f1_mac:>9.2f}% "
          f"{f1_wtd:>12.2f}%")

print(f"\nResults saved to: {RESULTS_DIR}")
print(f"   confusion_matrices.png")
print(f"   per_class_accuracy.csv")
print(f"   model_comparison.json")
print(f"   classification_report.txt")
print(f"   learning_curves.png")
print(f"   weak_class_analysis.txt")
print(f"\nALL COMPLETE")
