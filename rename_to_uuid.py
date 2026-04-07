import os
import uuid
import shutil
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────
# Point this to your local split folder
SPLIT_DIR = Path("data/split")          # adjust if needed
SPLITS    = ["train", "val", "test"]
VALID_EXT = {".jpg", ".jpeg", ".png",
             ".bmp", ".tiff", ".webp"}

DRY_RUN   = False   # ← set to False to actually rename
# ─────────────────────────────────────────────────────────────

def rename_images_to_uuid(split_dir, splits, dry_run=True):
    total_renamed = 0
    total_skipped = 0
    errors        = []

    for split in splits:
        split_path = split_dir / split

        if not split_path.exists():
            print(f"⚠️  Split folder not found, skipping: {split_path}")
            continue

        class_folders = sorted([
            d for d in split_path.iterdir() if d.is_dir()
        ])

        print(f"\n📂 {split}/ — {len(class_folders)} class folders")

        for class_folder in class_folders:
            images = sorted([
                f for f in class_folder.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXT
            ])

            renamed_in_class = 0

            for img_path in images:
                new_name    = f"{uuid.uuid4()}.jpg"
                new_path    = class_folder / new_name

                # Collision guard (astronomically unlikely but safe)
                while new_path.exists():
                    new_name = f"{uuid.uuid4()}.jpg"
                    new_path = class_folder / new_name

                if dry_run:
                    print(f"   [DRY RUN] {img_path.name:40s} → {new_name}")
                else:
                    try:
                        img_path.rename(new_path)
                        renamed_in_class += 1
                        total_renamed    += 1
                    except Exception as e:
                        errors.append((str(img_path), str(e)))
                        total_skipped += 1

            if not dry_run:
                print(f"   ✅ {class_folder.name:<30} "
                      f"{renamed_in_class} images renamed")

    if dry_run:
        print("\n" + "─"*60)
        print("DRY RUN COMPLETE — no files were changed.")
        print("If the output looks correct, set DRY_RUN = False and re-run.")
    else:
        print("\n" + "─"*60)
        print(f"✅ Done. {total_renamed} images renamed.")
        if total_skipped:
            print(f"⚠️  {total_skipped} files skipped due to errors:")
            for path, err in errors:
                print(f"   {path} → {err}")

if __name__ == "__main__":
    print(f"{'DRY RUN' if DRY_RUN else 'LIVE RUN'} — "
          f"scanning: {SPLIT_DIR.resolve()}")
    rename_images_to_uuid(SPLIT_DIR, SPLITS, dry_run=DRY_RUN)