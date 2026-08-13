#!/usr/bin/env python3
"""So sánh nhiều LLM (FPT) cho verifier #3 trên mẫu val có nhãn gold — quy mô lớn.

Song song 2 tầng: cross-model (mỗi model 1 luồng) + trong-model (nhiều span đồng thời),
có semaphore giới hạn tổng số call đồng thời để tránh rate-limit. Lưu JSON + in bảng.

  python evaluation/compare_models.py --n 1000 --inner 2 --max-concurrent 12
"""
from __future__ import annotations
import argparse, csv, json, os, statistics, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LLM_PROVIDER", "fpt")
from scripts.alignment.align_tfisf import align_pair
from evaluation.plagdet import Span, plagdet_score
from generation.verify import verify_pair

VAL = r"C:/github/PAN2025/pan25-generated-plagiarism-detection-validation/02_validation/02_validation"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generation")
MODELS = ["Llama-3.3-70B-Instruct", "DeepSeek-V4-Flash", "Qwen3.6-27B",
          "gpt-oss-20b", "gpt-oss-120b", "GLM-5.2"]
TH, TH3 = 0.30, 0.50


def rd(p): return open(p, encoding="utf-8", errors="ignore").read()
def overlaps(o, l, golds):
    e = o + l
    return any(not (e <= go or o >= go + gl) for go, gl in golds)


def build_spans(n):
    gold_by, src_of = {}, {}
    for r in csv.DictReader(open("outputs/validation_spans.csv", encoding="utf-8", newline="")):
        if r["feature"] == "plagiarism" and r["source_reference"]:
            gold_by.setdefault(r["suspicious_reference"], []).append((int(r["this_offset"]), int(r["this_length"])))
            src_of.setdefault(r["suspicious_reference"], r["source_reference"])
    spans, gold_spans = [], []
    for su, golds in gold_by.items():
        if len(spans) >= n:
            break
        sp, rp = os.path.join(VAL, "susp", su), os.path.join(VAL, "src", src_of[su])
        if not (os.path.exists(sp) and os.path.exists(rp)):
            continue
        st, rt = rd(sp), rd(rp)
        pred = align_pair(st, rt, TH, TH, TH3, 4)
        if not pred:
            continue
        for go, gl in golds:
            gold_spans.append(Span(su, go, gl))
        for s, l, ss, sl in pred:
            if len(spans) >= n:
                break
            spans.append({"doc": su, "s": s, "l": l, "susp": st[s:s + l], "src": rt[ss:ss + sl],
                          "tp": overlaps(s, l, golds)})
    return spans, gold_spans


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--inner", type=int, default=2, help="số span đồng thời / model")
    ap.add_argument("--max-concurrent", type=int, default=12, help="trần tổng call đồng thời")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Dựng mẫu span (n={args.n})...", flush=True)
    SPANS, gold_spans = build_spans(args.n)
    n_tp = sum(x["tp"] for x in SPANS); n_fp = len(SPANS) - n_tp
    pred_before = [Span(x["doc"], x["s"], x["l"]) for x in SPANS]
    pdb = plagdet_score(gold_spans, pred_before)
    print(f"{len(SPANS)} span ({n_tp} TP, {n_fp} FP) · PlagDet nền {pdb.plagdet:.3f} "
          f"(P={pdb.precision:.3f} R={pdb.recall:.3f})", flush=True)

    gate = threading.Semaphore(args.max_concurrent)     # trần call đồng thời toàn cục
    done = {}

    def verify_one(model, x):
        with gate:
            t = time.time()
            try:
                v = verify_pair(x["susp"], x["src"], model=model)
                return bool(v["is_plagiarism"]), v.get("confidence"), time.time() - t, False
            except Exception:
                return True, None, time.time() - t, True   # lỗi -> giữ (an toàn)

    def run_model(model):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.inner) as ex:
            out = list(ex.map(lambda x: verify_one(model, x), SPANS))
        keeps = [o[0] for o in out]; confs = [o[1] for o in out if isinstance(o[1], (int, float))]
        lats = [o[2] for o in out]; fails = sum(o[3] for o in out)
        keep_tp = sum(1 for x, k in zip(SPANS, keeps) if x["tp"] and k)
        keep_fp = sum(1 for x, k in zip(SPANS, keeps) if not x["tp"] and k)
        after = [Span(x["doc"], x["s"], x["l"]) for x, k in zip(SPANS, keeps) if k]
        a = plagdet_score(gold_spans, after)
        rec = {"model": model,
               "plagdet_before": round(pdb.plagdet, 3), "plagdet_after": round(a.plagdet, 3),
               "delta": round(a.plagdet - pdb.plagdet, 3),
               "fp_reduction": round((n_fp - keep_fp) / n_fp, 3) if n_fp else None,
               "tp_retention": round(keep_tp / n_tp, 3) if n_tp else None,
               "prec_after": round(a.precision, 3),
               "avg_conf": round(statistics.mean(confs), 2) if confs else None,
               "avg_lat_s": round(statistics.mean(lats), 1), "errors": fails}
        done[model] = rec
        print(f"  ✓ {model:24} PlagDet {pdb.plagdet:.3f}->{a.plagdet:.3f} (Δ{rec['delta']:+.3f}) "
              f"fp_red {rec['fp_reduction']} tp_ret {rec['tp_retention']} "
              f"[{time.time()-t0:.0f}s] ({len(done)}/{len(MODELS)} model xong)", flush=True)
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
        res = list(ex.map(run_model, MODELS))
    res.sort(key=lambda r: -r["plagdet_after"])

    summary = {"n_spans": len(SPANS), "n_TP": n_tp, "n_FP": n_fp,
               "plagdet_before": round(pdb.plagdet, 3), "results": res,
               "runtime_sec": round(time.time() - t0, 1), "timestamp": datetime.now().isoformat(timespec="seconds")}
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"model_compare_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n===== SO SÁNH {len(MODELS)} MODEL · {len(SPANS)} span ({n_tp} TP/{n_fp} FP) · "
          f"nền {pdb.plagdet:.3f} · {summary['runtime_sec']:.0f}s =====")
    print(f"{'model':24} {'PlagDet↑':>8} {'Δ':>7} {'fp_red':>7} {'tp_ret':>7} {'prec':>6} {'lat':>6}")
    for r in res:
        print(f"{r['model']:24} {r['plagdet_after']:>8.3f} {r['delta']:>+7.3f} "
              f"{r['fp_reduction']:>7} {r['tp_retention']:>7} {r['prec_after']:>6.3f} {r['avg_lat_s']:>6}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
