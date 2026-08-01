#!/usr/bin/env python3
"""Đóng gói dữ liệu local thành 2 thư mục sẵn sàng `kaggle datasets create`.

Chạy MỘT LẦN sau khi có token. Tạo layout mà các kernel K2/K3/K4 mong đợi:

  kaggle_datasets/pan25-labels/
      train_spans.csv          (từ outputs/train_spans.csv)
      val_spans.csv            (từ outputs/validation_spans.csv  -- đổi tên)
      spot_spans.csv

  kaggle_datasets/pan25-corpus/
      {train,val}/docs/{src,susp}/*.txt

Lưu ý: corpus ~2.4GB -> mặc định CHỈ stage labels (nhẹ). Thêm --corpus để copy
corpus (nặng, một lần). --split để giới hạn (vd chỉ 'val' cho prototype nhanh).

Usage:
    python scripts/kaggle_stage.py                     # chỉ labels
    python scripts/kaggle_stage.py --corpus --split val   # + corpus val (nhỏ, 8k doc)
    python scripts/kaggle_stage.py --corpus               # + corpus train+val (2.4GB)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

KAGGLE_USER = "whaleeatu"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
DS = REPO / "kaggle_datasets"

# Nguồn corpus thật (local) — khớp orchestration/config.yaml
CORPUS_SRC = {
    "train": Path(r"C:/github/PAN2025/pan25-generated-plagiarism-detection-train/01_train/01_train"),
    "val":   Path(r"C:/github/PAN2025/pan25-generated-plagiarism-detection-validation/02_validation/02_validation"),
    "spot":  Path(r"C:/github/PAN2025/00_spot_check/00_spot_check"),   # 50 doc — smoke test
}
LABELS_MAP = {                       # outputs/<src> -> pan25-labels/<dst>
    "train_spans.csv": "train_spans.csv",
    "validation_spans.csv": "val_spans.csv",
    "spot_spans.csv": "spot_spans.csv",
}


def stage_labels() -> None:
    dst = DS / "pan25-labels"
    dst.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in LABELS_MAP.items():
        src = OUTPUTS / src_name
        if not src.exists():
            print(f"  bỏ qua (thiếu): {src}")
            continue
        size_mb = src.stat().st_size / 1e6
        print(f"  copy {src_name} -> {dst_name} ({size_mb:.0f} MB)")
        shutil.copy2(src, dst / dst_name)
    print(f"labels -> {dst}")


def stage_corpus(splits) -> None:
    # Mỗi split = 1 dataset riêng: pan25-corpus-<split>. Wrapper folder = <split>
    # (bị --dir-mode zip drop) -> dataset có docs/ ở root, không đè nhau (gotcha #6).
    for split in splits:
        src_root = CORPUS_SRC[split]
        ds = DS / f"pan25-corpus-{split}"
        ds.mkdir(parents=True, exist_ok=True)
        (ds / "dataset-metadata.json").write_text(json.dumps({
            "title": f"PAN25 Corpus {split}",
            "id": f"{KAGGLE_USER}/pan25-corpus-{split}",
            "licenses": [{"name": "other"}],
        }, indent=2), encoding="utf-8")
        for role in ("src", "susp"):
            src_dir = src_root / role
            dst_dir = ds / split / "docs" / role
            dst_dir.mkdir(parents=True, exist_ok=True)
            files = [p for p in src_dir.glob("*.txt")]
            print(f"  {split}/{role}: copy {len(files)} file -> {dst_dir}")
            for i, p in enumerate(files):
                shutil.copy2(p, dst_dir / p.name)
                if (i + 1) % 10000 == 0:
                    print(f"    ...{i + 1}")
        print(f"corpus split '{split}' -> {ds}  (create: kaggle datasets create -p {ds} --dir-mode zip)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="store_true", help="stage cả corpus (nặng ~2.4GB)")
    ap.add_argument("--split", choices=["train", "val", "spot"], default=None,
                    help="giới hạn corpus 1 split (mặc định train+val)")
    args = ap.parse_args(argv)

    print("== stage labels ==")
    stage_labels()

    if args.corpus:
        splits = [args.split] if args.split else ["train", "val"]
        print(f"== stage corpus {splits} (nặng) ==")
        stage_corpus(splits)
    else:
        print("(bỏ qua corpus — thêm --corpus khi sẵn sàng upload 2.4GB)")

    print("\nTiếp theo: sửa TODO-USERNAME trong dataset-metadata.json, rồi:")
    print("  kaggle datasets create -p kaggle_datasets/pan25-labels")
    print("  kaggle datasets create -p kaggle_datasets/pan25-corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
