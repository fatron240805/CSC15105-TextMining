#!/usr/bin/env python3
"""End-to-end pipeline (A4-1): Retrieval -> Alignment -> Scoring, chạy trên 1 split.

Hệ thống THẬT (không dùng gold source): mỗi susp -> TF-IDF truy hồi top-k nguồn ->
align tf-isf với từng nguồn -> gộp span -> chấm PlagDet vs truth. Toàn bộ CPU.

  python orchestration/run_pipeline.py --split val --subset 200 --topk 5
"""
from __future__ import annotations
import argparse, csv, glob, os, sys, time
from collections import defaultdict

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.alignment.align_tfisf import align_pair
from evaluation.plagdet import Span, plagdet_score

VAL = r"C:/github/PAN2025/pan25-generated-plagiarism-detection-validation/02_validation/02_validation"
PATHS = {
    "val":  (os.path.join(VAL, "susp"), os.path.join(VAL, "src"), "outputs/validation_spans.csv"),
    "spot": (r"C:/github/PAN2025/00_spot_check/00_spot_check/susp",
             r"C:/github/PAN2025/00_spot_check/00_spot_check/src", "outputs/spot_spans.csv"),
}
TFIDF_CHARS = 20000


def read(p):
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val", choices=["val", "spot"])
    ap.add_argument("--subset", type=int, default=200)
    ap.add_argument("--topk", type=int, default=1)   # top-1 tốt nhất: distractor (top-k>1) hại precision
    ap.add_argument("--th", type=float, default=0.30)     # seed th1=th2
    ap.add_argument("--th3", type=float, default=0.50)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    susp_dir, src_dir, spans_csv = PATHS[args.split]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # gold: susp -> spans (chấm điểm)
    gold_by = defaultdict(list)
    for r in csv.DictReader(open(spans_csv, encoding="utf-8", newline="")):
        if r["feature"] == "plagiarism":
            gold_by[r["suspicious_reference"]].append((int(r["this_offset"]), int(r["this_length"])))
    subset = list(gold_by)[:args.subset]

    t0 = time.time()
    # ---- Retrieval: TF-IDF index toàn bộ src, top-k cho mỗi susp ----
    src_files = sorted(glob.glob(os.path.join(src_dir, "*.txt")))
    src_ids = [os.path.basename(p) for p in src_files]
    print(f"[retrieval] đọc {len(src_files)} nguồn + fit TF-IDF...", flush=True)
    src_full = [read(p) for p in src_files]
    vec = TfidfVectorizer(max_features=100000, sublinear_tf=True, stop_words="english")
    src_m = vec.fit_transform([t[:TFIDF_CHARS] for t in src_full])

    susp_texts = [read(os.path.join(susp_dir, su)) for su in subset]
    susp_m = vec.transform([t[:TFIDF_CHARS] for t in susp_texts])
    sims = cosine_similarity(susp_m, src_m)
    k = min(args.topk, len(src_ids))
    topk = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    print(f"[retrieval] xong ({time.time()-t0:.0f}s). Align top-{k}...", flush=True)

    # ---- Alignment: align susp vs từng nguồn top-k, gộp span ----
    gold, pred = [], []
    for n, (su, stext, cand) in enumerate(zip(subset, susp_texts, topk), 1):
        for off, ln in gold_by[su]:
            gold.append(Span(su, off, ln))
        cand_sorted = cand[np.argsort(-sims[n - 1, cand])]
        merged = []
        for j in cand_sorted:
            for o, l, *_ in align_pair(stext, src_full[j], args.th, args.th, args.th3, 4):
                merged.append((o, l))
        # gộp span từ nhiều nguồn: dùng luôn (PlagDet xử lý union theo ký tự)
        for o, l in merged:
            pred.append(Span(su, o, l))
        if n % 50 == 0:
            print(f"  align {n}/{len(subset)} ({time.time()-t0:.0f}s)", flush=True)

    # baseline + score
    whole = [Span(su, 0, len(st)) for su, st in zip(subset, susp_texts)]
    r = plagdet_score(gold, pred)
    b = plagdet_score(gold, whole)
    print(f"\n=== END-TO-END ({args.split}, {len(subset)} susp, {len(gold)} gold spans) ===")
    print(f"baseline cả-doc : PlagDet={b.plagdet:.3f}")
    print(f"pipeline        : P={r.precision:.3f} R={r.recall:.3f} F1={r.f1:.3f} "
          f"gran={r.granularity:.3f} PlagDet={r.plagdet:.3f}")
    print(f"(retrieval TF-IDF top-{k} -> align tf-isf th={args.th} th3={args.th3}) · {time.time()-t0:.0f}s")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/pipeline_eval.csv", "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["metric", "value"])
        for kk, vv in [("plagdet", r.plagdet), ("precision", r.precision), ("recall", r.recall),
                       ("f1", r.f1), ("granularity", r.granularity), ("baseline_wholedoc", b.plagdet),
                       ("n_susp", len(subset)), ("topk", k)]:
            w.writerow([kk, round(vv, 4) if isinstance(vv, float) else vv])
    print("-> outputs/pipeline_eval.csv")


if __name__ == "__main__":
    main()
