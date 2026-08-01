#!/usr/bin/env python3
"""EDA cho PAN 2025 (A0-4) — thống kê từ *_spans.csv (output của parse_labels.py).

Chạy local, KHÔNG tốn quota Kaggle. Sinh:
  - Bảng tóm tắt tidy -> outputs/eda_summary.csv  (metric, category, value)
  - In tóm tắt dễ đọc ra stdout

Thống kê:
  * plagiarism vs altered (đếm + tỉ lệ)
  * phân phối obfuscation (simple/medium/hard) trên span plagiarism
  * phân phối severity (low/medium/high) ở cấp tài liệu
  * phân phối LLM sinh đạo văn (DeepSeek-R1 / Llama-3 / Mistral / ...)
  * độ dài span (this_length): mean/median/p90 theo feature & obfuscation
  * số span / mỗi cặp tài liệu (mức phân mảnh)
  * similarity ở cấp tài liệu

Usage:
    python scripts/eda.py --spans outputs/validation_spans.csv --out outputs/eda_summary.csv
    python scripts/eda.py --spans outputs/train_spans.csv       # tập lớn (~2.7M dòng)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Console Windows mặc định cp1252 — ép UTF-8 để in được tiếng Việt (vô hại trên Kaggle/Linux)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _rows(metric, series_or_dict, kind="count"):
    """Chuẩn hoá một nhóm thống kê thành list dòng (metric, category, value)."""
    out = []
    items = (series_or_dict.items() if hasattr(series_or_dict, "items")
             else series_or_dict)
    for cat, val in items:
        out.append({"metric": metric, "category": str(cat), "value": val})
    return out


def run_eda(spans_csv: Path, out_csv: Path) -> None:
    print(f"Đọc {spans_csv} ...")
    df = pd.read_csv(spans_csv)
    n = len(df)
    print(f"  {n:,} span")

    rows = []

    # --- feature: plagiarism vs altered ---
    feat = df["feature"].value_counts()
    rows += _rows("feature_count", feat)
    rows += _rows("feature_ratio", (feat / n).round(4))

    # --- obfuscation (chỉ trên span plagiarism) ---
    plag = df[df["feature"] == "plagiarism"]
    if "obfuscation" in df.columns:
        obf = plag["obfuscation"].value_counts()
        rows += _rows("obfuscation_count", obf)
        rows += _rows("obfuscation_ratio", (obf / len(plag)).round(4))

    # --- LLM sinh ---
    if "llm" in df.columns:
        rows += _rows("llm_count", df["llm"].value_counts())

    # --- severity ở cấp tài liệu (dedup theo susp doc) ---
    if "severity" in df.columns and "suspicious_reference" in df.columns:
        doc_sev = df.drop_duplicates("suspicious_reference")["severity"].value_counts()
        rows += _rows("severity_doc_count", doc_sev)

    # --- similarity cấp tài liệu ---
    if "similarity" in df.columns:
        doc_sim = df.drop_duplicates("suspicious_reference")["similarity"].dropna()
        if len(doc_sim):
            rows += _rows("similarity_doc", {
                "mean": round(doc_sim.mean(), 4),
                "median": round(doc_sim.median(), 4),
                "min": round(doc_sim.min(), 4),
                "max": round(doc_sim.max(), 4),
            }, kind="stat")

    # --- độ dài span this_length theo feature ---
    for feature in ("plagiarism", "altered"):
        sub = df[df["feature"] == feature]["this_length"]
        if len(sub):
            rows += _rows(f"span_len_{feature}", {
                "mean": round(sub.mean(), 1),
                "median": int(sub.median()),
                "p90": int(sub.quantile(0.90)),
                "max": int(sub.max()),
            }, kind="stat")

    # --- độ dài span theo obfuscation ---
    if "obfuscation" in df.columns:
        for obf_lvl, g in plag.groupby("obfuscation"):
            rows += _rows(f"span_len_obf_{obf_lvl}", {
                "mean": round(g["this_length"].mean(), 1),
                "median": int(g["this_length"].median()),
            }, kind="stat")

    # --- số span / cặp tài liệu (phân mảnh) ---
    if "xml_file" in df.columns:
        per_pair = df.groupby("xml_file").size()
        rows += _rows("spans_per_pair", {
            "mean": round(per_pair.mean(), 2),
            "median": int(per_pair.median()),
            "p90": int(per_pair.quantile(0.90)),
            "max": int(per_pair.max()),
        }, kind="stat")

    # --- ghi + in ---
    out_df = pd.DataFrame(rows, columns=["metric", "category", "value"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nĐã ghi {len(out_df)} dòng -> {out_csv}\n")

    # In gọn vài nhóm chính
    print("=== Tóm tắt ===")
    for m in ("feature_count", "obfuscation_count", "llm_count",
              "severity_doc_count", "span_len_plagiarism", "spans_per_pair"):
        sub = out_df[out_df["metric"] == m]
        if len(sub):
            print(f"\n[{m}]")
            for _, r in sub.iterrows():
                print(f"  {r['category']:<16} {r['value']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spans", required=True, type=Path, help="đường dẫn *_spans.csv")
    ap.add_argument("--out", type=Path, default=Path("outputs/eda_summary.csv"))
    args = ap.parse_args(argv)
    if not args.spans.exists():
        ap.error(f"không thấy file: {args.spans}")
    run_eda(args.spans, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
