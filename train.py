# Refer this to train your own custom model on your own dataset

"""
YOLO11 Training Script — Explicit Content Detection
Dataset: BE explicit v1 (Roboflow), 5 classes, 9273 train / 866 val / 443 test
"""

import shutil
import yaml
import torch
from pathlib import Path
from datetime import datetime


def select_device() -> str:
    """Return 'mps', 'cuda'/'0', or 'cpu' depending on what's available."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"   # first GPU
    return "cpu"

# ── Config ────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.resolve()
DATASET_DIR = ROOT / "BE explicit.v1i.yolov11"
DATA_YAML   = DATASET_DIR / "data.yaml"
MODEL_OUT   = ROOT / "Model" / "best_v11m.pt"

# Training hyperparameters — tweak as needed
CONFIG = {
    "model":     "yolo11m.pt",   # base pretrained weights (auto-downloaded)
    "epochs":    1000,
    "imgsz":     640,
    "batch":     16,             # lower to 8 if GPU OOM
    "patience":  50,             # early stopping (epochs with no improvement)
    "lr0":       0.01,           # initial learning rate
    "lrf":       0.01,           # final lr = lr0 * lrf
    "momentum":  0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "box":       7.5,            # box loss weight
    "cls":       0.5,            # class loss weight
    "dfl":       1.5,            # distribution focal loss weight
    "hsv_h":     0.015,          # hue augmentation
    "hsv_s":     0.7,            # saturation augmentation
    "hsv_v":     0.4,            # value augmentation
    "flipud":    0.0,
    "fliplr":    0.5,
    "mosaic":    1.0,
    "mixup":     0.1,
    "copy_paste": 0.1,
    "project":   str(ROOT / "runs" / "train"),
    "name":      f"explicit_{datetime.now().strftime('%Y%m%d_%H%M')}",
    "exist_ok":  False,
    "pretrained": True,
    "optimizer": "AdamW",
    "verbose":   True,
    "seed":      42,
    "deterministic": True,
    "device":    select_device(),  # auto: mps → cuda → cpu
    "workers":   8,
    "save":      True,
    "save_period": -1,           # save checkpoint every N epochs (-1 = only best/last)
    "plots":     True,           # save training plots (confusion matrix, PR curve, etc.)
    "val":       True,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_abs_data_yaml() -> Path:
    """
    Roboflow exports data.yaml with relative paths (../train/images).
    YOLO needs absolute paths when the working directory differs.
    This writes a patched copy next to the original.
    """
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)

    # Resolve each split path relative to the dataset directory
    for key in ("train", "val", "test"):
        if key in cfg and cfg[key]:
            raw = Path(cfg[key])
            if not raw.is_absolute():
                if raw.parts[0] == "..":
                    raw = Path(*raw.parts[1:])
                cfg[key] = str((DATASET_DIR / raw).resolve())

    out_path = ROOT / "data_abs.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    return out_path


def check_dataset():
    """Validate that expected dataset directories and labels exist."""
    for split, folder in [("train", "train"), ("val", "valid"), ("test", "test")]:
        img_dir = DATASET_DIR / folder / "images"
        lbl_dir = DATASET_DIR / folder / "labels"
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing {split} images directory: {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Missing {split} labels directory: {lbl_dir}")
        n = len(list(img_dir.glob("*.*")))
        print(f"  {split:5s}: {n:,} images")


def copy_best_model(run_dir: Path):
    """Copy the best trained weights to Model/best.pt."""
    best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, MODEL_OUT)
        print(f"\n[✓] Best model saved → {MODEL_OUT}")
    else:
        print(f"\n[!] best.pt not found at {best_src} — check the run directory.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  YOLO11 Training — Explicit Content Detection")
    print("=" * 60)

    # 1. Validate ultralytics
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"\n[✓] Ultralytics {ultralytics.__version__} found")
    except ImportError:
        raise SystemExit(
            "\n[✗] ultralytics not installed.\n"
            "    Run:  pip install -U ultralytics"
        )

    # 2. Validate dataset
    print(f"\n[→] Dataset: {DATASET_DIR}")
    check_dataset()

    # 3. Build absolute-path data.yaml
    abs_yaml = make_abs_data_yaml()
    print(f"\n[✓] Patched data.yaml written → {abs_yaml}")

    # 4. Print config summary
    print(f"\n[→] Training config:")
    print(f"    Base model : {CONFIG['model']}")
    print(f"    Epochs     : {CONFIG['epochs']}  (early stop patience={CONFIG['patience']})")
    print(f"    Image size : {CONFIG['imgsz']}px")
    print(f"    Batch size : {CONFIG['batch']}")
    print(f"    Optimizer  : {CONFIG['optimizer']}")
    print(f"    Device     : {CONFIG['device']}")
    print(f"    Output dir : {CONFIG['project']}/{CONFIG['name']}")

    # 5. Load model and train
    print("\n[→] Starting training...\n")
    model = YOLO(CONFIG["model"])

    results = model.train(
        data=str(abs_yaml),
        **{k: v for k, v in CONFIG.items() if k != "model"},
    )

    # 6. Evaluate on test set
    run_dir = Path(CONFIG["project"]) / CONFIG["name"]
    print("\n[→] Running evaluation on test set...")
    model.val(data=str(abs_yaml), split="test")

    # 7. Copy best weights
    copy_best_model(run_dir)

    print("\n[✓] Training complete.")
    print(f"    Run artifacts : {run_dir}")
    print(f"    Deployed model: {MODEL_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
