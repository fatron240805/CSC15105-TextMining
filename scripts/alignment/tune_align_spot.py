#!/usr/bin/env python3
"""Tune seed-and-extend trên spot (50 pairs) — dùng nguồn GOLD (tách khỏi retrieval).

Embed câu 1 lần/cặp (đắt), rồi sweep threshold + merge_gap (rẻ). Báo P/R/F1/gran/PlagDet
so với trần boundary (0.858) và baseline cả-doc (0.665). Iterate nhanh trước khi lên Kaggle.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.alignment.align_core import sentence_best_sims, spans_from_best
from evaluation.plagdet import Span, plagdet_score

SUSP_DIR = r"C:/github/PAN2025/00_spot_check/00_spot_check/susp"
SRC_DIR = r"C:/github/PAN2025/00_spot_check/00_spot_check/src"
SPANS_CSV = "outputs/spot_spans.csv"
MODEL = "BAAI/bge-small-en-v1.5"


def read(d, name):
    with open(os.path.join(d, name), encoding="utf-8", newline="") as f:
        return f.read()


def main():
    # gold: susp -> spans, và susp -> tập source_reference
    gold_by = defaultdict(list)
    srcs_by = defaultdict(set)
    for r in csv.DictReader(open(SPANS_CSV, encoding="utf-8", newline="")):
        if r["feature"] != "plagiarism":
            continue
        gold_by[r["suspicious_reference"]].append((int(r["this_offset"]), int(r["this_length"])))
        if r["source_reference"]:
            srcs_by[r["suspicious_reference"]].add(r["source_reference"])

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device="cpu")
    embed = lambda texts: model.encode(texts, batch_size=128, normalize_embeddings=True,
                                       show_progress_bar=False)

    print(f"Embed câu {len(gold_by)} cặp susp (nguồn gold)...", flush=True)
    cache, gold = {}, []
    for k, (susp, spans) in enumerate(gold_by.items(), 1):
        susp_text = read(SUSP_DIR, susp)
        src_texts = [read(SRC_DIR, s) for s in srcs_by[susp] if os.path.exists(os.path.join(SRC_DIR, s))]
        cache[susp] = (susp_text, sentence_best_sims(susp_text, src_texts, embed))
        for off, ln in spans:
            gold.append(Span(susp, off, ln))
        if k % 10 == 0:
            print(f"  ...{k}/{len(gold_by)}", flush=True)

    print(f"\n{len(gold)} gold spans | trần=0.858 · baseline cả-doc=0.665\n")
    print(f"{'thr':>5} {'gap':>4} | {'P':>6} {'R':>6} {'F1':>6} {'gran':>6} {'PlagDet':>8}")
    print("-" * 48)
    best_pd = (0, None)
    for gap in (0, 1, 2):
        for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            pred = []
            for susp, (susp_text, (ss, bestsim)) in cache.items():
                for off, ln in spans_from_best(ss, bestsim, thr, merge_gap_sents=gap):   # v1 (neural) — 2-tuple
                    pred.append(Span(susp, off, ln))
            r = plagdet_score(gold, pred)
            print(f"{thr:>5.2f} {gap:>4} | {r.precision:>6.3f} {r.recall:>6.3f} {r.f1:>6.3f} "
                  f"{r.granularity:>6.3f} {r.plagdet:>8.3f}")
            if r.plagdet > best_pd[0]:
                best_pd = (r.plagdet, (thr, gap))
        print("-" * 48)
    print(f"\nTỐT NHẤT: PlagDet={best_pd[0]:.3f} @ threshold={best_pd[1][0]}, gap={best_pd[1][1]}")


if __name__ == "__main__":
    main()
