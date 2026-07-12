#!/usr/bin/env python3
"""EDA summary generator for the PAN 2025 Generated Plagiarism Detection dataset.

Task A0-4 (Phase 0). Consumes the per-pair JSONL produced by
``scripts/parse_labels.py`` (task A0-3) and emits a single tidy long-format
CSV covering the statistics needed downstream by:

  * Source Retrieval (a RAG-style retriever)      -> doc-type ratio, similarity
    score distribution, full-document length (helps pick embedding/chunk size)
  * Text Alignment (window / threshold choices)   -> span length stats,
    length_ratio (paraphrase compression/expansion), obfuscation distribution
  * Scoring / reporting (stratified evaluation)    -> severity distribution,
    spans-per-document (relates to Granularity/PlagDet)

IMPORTANT terminology (do not conflate):
  * ``severity``    - document-level ("about" feature). Low/Medium/High =
                       the % of the suspicious paragraph that was replaced
                       (20-40% / 40-60% / 70-100%, per the problem statement).
                       This is what the task "muc do dao van Low/Medium/High"
                       refers to.
  * ``obfuscation``  - span-level (only on "plagiarism" features).
                       simple/medium/hard = the paraphrase PROMPT complexity
                       (Simple 60% / Default 30% / Complex 10%). A different
                       axis from severity; both are analysed here.

Why JSONL and not the flat *_spans.csv:
  ``write_csv`` in parse_labels.py only emits one row per SPAN, so documents
  with zero spans (Original documents, ~5% of the corpus) are entirely
  absent from *_spans.csv. Doc-type ratios and "Original" counts MUST come
  from the JSONL (one record per pair, including zero-span ones).

Output schema (tidy / long format, one row per statistic)::

    split, metric_group, group_key, statistic, value, n

Usage
-----
    # one split at a time, then append into the same summary file:
    python scripts/eda_analysis.py \
        --labels-jsonl outputs/spot_labels.jsonl --split spot_check \
        --out-csv outputs/eda_summary.csv

    python scripts/eda_analysis.py \
        --labels-jsonl outputs/train_labels.jsonl --split train \
        --out-csv outputs/eda_summary.csv --append

    python scripts/eda_analysis.py \
        --labels-jsonl outputs/validation_labels.jsonl --split validation \
        --out-csv outputs/eda_summary.csv --append

    # optional: also sample raw .txt files to get full-document length stats
    # (useful for choosing retriever chunk size). Reads a random sample so it
    # stays fast even on the 60k-file train split.
    python scripts/eda_analysis.py \
        --labels-jsonl outputs/train_labels.jsonl --split train \
        --out-csv outputs/eda_summary.csv --append \
        --docs-dir "data/raw/pan25-generated-plagiarism-detection-train/01_train/01_train" \
        --sample-docs 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def iter_records(jsonl_path: Path) -> Iterator[dict]:
    """Stream one pair-record (dict) at a time; keeps memory flat even for
    the ~62k-line train JSONL (spans are nested lists inside each line, but
    we only ever hold one line in memory at once)."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def describe(values: list) -> dict:
    """Descriptive stats for a numeric list. Returns {} if empty."""
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


class Rows:
    """Accumulates tidy output rows."""

    def __init__(self, split: str):
        self.split = split
        self.rows: list = []

    def add(self, metric_group: str, group_key: str, statistic: str,
            value: float, n: Optional[int] = None) -> None:
        self.rows.append({
            "split": self.split,
            "metric_group": metric_group,
            "group_key": group_key,
            "statistic": statistic,
            "value": value,
            "n": n if n is not None else "",
        })

    def add_distribution(self, metric_group: str, counter: dict) -> None:
        total = sum(counter.values())
        if total == 0:
            return
        for key, cnt in sorted(counter.items(), key=lambda kv: -kv[1]):
            self.add(metric_group, str(key), "count", cnt, n=total)
            self.add(metric_group, str(key), "percentage", round(100 * cnt / total, 4), n=total)

    def add_describe(self, metric_group: str, group_key: str, values: list) -> None:
        d = describe(values)
        if not d:
            return
        n = d.pop("count")
        for stat, val in d.items():
            self.add(metric_group, group_key, stat, round(val, 4), n=n)
        self.add(metric_group, group_key, "count", n, n=n)


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def classify_pair(rec: dict) -> str:
    if rec.get("n_plagiarism", 0) > 0:
        return "plagiarism"
    if rec.get("n_altered", 0) > 0:
        return "altered"
    return "original"


def run_analysis(jsonl_path: Path, split: str) -> Rows:
    rows = Rows(split)

    doc_type_pair = defaultdict(int)                       # per pair (per XML)
    doc_best_type = {}                                      # per suspicious_reference (aggregated)
    severity_counter = defaultdict(int)
    llm_counter = defaultdict(int)
    obf_counter = defaultdict(int)

    span_len_by_feature = defaultdict(list)                 # feature -> [this_length,...]
    span_len_by_feature_obf = defaultdict(list)              # "feature|obf" -> [...]
    length_ratio = defaultdict(list)                         # "plagiarism" / "plagiarism|obf" -> [ratio,...]

    similarity_by_type = defaultdict(list)
    n_plag_per_pair = []
    n_alt_per_pair = []
    severity_vs_nplag = defaultdict(list)
    severity_vs_nalt = defaultdict(list)
    severity_vs_similarity = defaultdict(list)

    n_records = 0
    _RANK = {"low": 0, "medium": 1, "high": 2}  # keep a stable priority for doc-level rollup

    for rec in iter_records(jsonl_path):
        n_records += 1
        ptype = classify_pair(rec)
        doc_type_pair[ptype] += 1

        susp = rec.get("suspicious_reference", "")
        # roll up to document level: plagiarism > altered > original priority
        prev = doc_best_type.get(susp)
        if prev is None or _RANK.get(ptype, -1) > _RANK.get(prev, -1) or \
           (ptype == "plagiarism" and prev != "plagiarism"):
            # simple precedence: plagiarism wins over altered wins over original
            precedence = {"plagiarism": 2, "altered": 1, "original": 0}
            if prev is None or precedence[ptype] > precedence.get(prev, -1):
                doc_best_type[susp] = ptype

        sev = rec.get("severity")
        sev_key = sev.lower() if sev else "unknown/none"
        severity_counter[sev_key] += 1

        sim = rec.get("similarity")
        if sim is not None:
            similarity_by_type[ptype].append(sim)
            severity_vs_similarity[sev_key].append(sim)

        n_plag = rec.get("n_plagiarism", 0)
        n_alt = rec.get("n_altered", 0)
        n_plag_per_pair.append(n_plag)
        n_alt_per_pair.append(n_alt)
        severity_vs_nplag[sev_key].append(n_plag)
        severity_vs_nalt[sev_key].append(n_alt)

        for span in rec.get("spans", []):
            feat = span.get("feature")
            length = span.get("this_length")
            if length is not None:
                span_len_by_feature[feat].append(length)

            llm = span.get("llm")
            if llm:
                llm_counter[llm] += 1

            if feat == "plagiarism":
                obf = span.get("obfuscation") or "unknown"
                obf_counter[obf] += 1
                if length is not None:
                    span_len_by_feature_obf[f"plagiarism|{obf}"].append(length)
                src_len = span.get("source_length")
                if length and src_len:
                    ratio = length / src_len
                    length_ratio["plagiarism"].append(ratio)
                    length_ratio[f"plagiarism|{obf}"].append(ratio)
            elif feat == "altered" and length is not None:
                span_len_by_feature_obf[f"altered|n_a"].append(length)

    # ---- write out ----
    rows.add_distribution("doc_type_distribution_pair", doc_type_pair)

    doc_type_doc_counter = defaultdict(int)
    for t in doc_best_type.values():
        doc_type_doc_counter[t] += 1
    rows.add_distribution("doc_type_distribution_doc", doc_type_doc_counter)

    rows.add_distribution("severity_distribution", severity_counter)
    rows.add_distribution("obfuscation_distribution", obf_counter)
    rows.add_distribution("llm_distribution", llm_counter)

    for feat, vals in span_len_by_feature.items():
        rows.add_describe("span_length_stats", feat, vals)
    for key, vals in span_len_by_feature_obf.items():
        rows.add_describe("span_length_by_obfuscation", key, vals)
    for key, vals in length_ratio.items():
        rows.add_describe("length_ratio_stats", key, vals)

    for t, vals in similarity_by_type.items():
        rows.add_describe("similarity_stats", t, vals)

    rows.add_describe("spans_per_pair_stats", "n_plagiarism", n_plag_per_pair)
    rows.add_describe("spans_per_pair_stats", "n_altered", n_alt_per_pair)

    for sev_key, vals in severity_vs_nplag.items():
        rows.add_describe("severity_x_n_plagiarism", sev_key, vals)
    for sev_key, vals in severity_vs_nalt.items():
        rows.add_describe("severity_x_n_altered", sev_key, vals)
    for sev_key, vals in severity_vs_similarity.items():
        rows.add_describe("severity_x_similarity", sev_key, vals)

    rows.add("dataset_size", "n_pairs_parsed", "count", n_records, n=n_records)
    rows.add("dataset_size", "n_unique_suspicious_docs", "count", len(doc_best_type),
             n=len(doc_best_type))

    return rows


def sample_full_doc_lengths(docs_dir: Path, sample_size: int, rows: Rows) -> None:
    """Optional: sample raw susp/src .txt files to get full-document length
    stats (character count). Useful for choosing retriever chunk size."""
    rng = random.Random(42)
    for sub in ("susp", "src"):
        d = docs_dir / sub
        if not d.exists():
            print(f"  WARN: {d} not found, skipping full-doc length sample", file=sys.stderr)
            continue
        files = list(d.glob("*.txt"))
        if not files:
            continue
        sample = rng.sample(files, min(sample_size, len(files)))
        lengths = []
        for fp in sample:
            with open(fp, "r", encoding="utf-8", newline="") as f:
                lengths.append(len(f.read()))
        rows.add_describe("full_document_length_sample", sub, lengths)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-jsonl", required=True, type=Path,
                    help="Per-pair JSONL from scripts/parse_labels.py (--out-jsonl).")
    ap.add_argument("--split", required=True,
                    help="Split name to tag rows with, e.g. train / validation / spot_check.")
    ap.add_argument("--out-csv", required=True, type=Path,
                    help="Output tidy CSV path (e.g. outputs/eda_summary.csv).")
    ap.add_argument("--append", action="store_true",
                    help="Append to an existing eda_summary.csv instead of overwriting "
                         "(use this for the 2nd/3rd split you process).")
    ap.add_argument("--docs-dir", type=Path, default=None,
                    help="Optional: split documents dir (containing susp/, src/) to also "
                         "sample full-document lengths.")
    ap.add_argument("--sample-docs", type=int, default=2000,
                    help="Number of files to sample per susp/src dir if --docs-dir is given.")
    args = ap.parse_args(argv)

    if not args.labels_jsonl.is_file():
        ap.error(f"--labels-jsonl not found: {args.labels_jsonl}\n"
                 f"(generate it first with parse_labels.py --out-jsonl ...)")

    print(f"Analyzing {args.labels_jsonl} (split={args.split}) ...")
    rows = run_analysis(args.labels_jsonl, args.split)

    if args.docs_dir:
        print(f"Sampling up to {args.sample_docs} files/dir from {args.docs_dir} ...")
        sample_full_doc_lengths(args.docs_dir, args.sample_docs, rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and args.out_csv.exists() else "w"
    write_header = not (args.append and args.out_csv.exists())
    with open(args.out_csv, mode, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "metric_group", "group_key",
                                          "statistic", "value", "n"])
        if write_header:
            w.writeheader()
        w.writerows(rows.rows)

    print(f"Wrote {len(rows.rows)} rows -> {args.out_csv} (mode={mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())