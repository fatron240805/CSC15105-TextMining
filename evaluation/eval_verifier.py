#!/usr/bin/env python3
"""Đánh giá vai trò #3 — LLM-verifier khử dương-tính-giả trên CASE BIÊN. KHÁCH QUAN
(nhãn từ gold PAN).

Quy trình (đã rà lại — không còn gọi verifier cho MỌI span, chỉ case biên quanh th3;
xem generation/verify.py cho lý do ranh giới với alignment):
  align susp vs nguồn GOLD (return_sim=True) -> span dự đoán -> gán nhãn TP/FP theo
  giao với gold -> span sim>>th3 (alignment tự tin): GIỮ luôn, không gọi verifier ->
  span sim trong edge band: verifier quyết định GIỮ/BỎ -> đo:
    - ma trận nhầm {giữ,bỏ} × {TP,FP}, FP-reduction, TP-retention (chỉ trên phần được gate)
    - HEADLINE: PlagDet / P / R TRƯỚC vs SAU khi bỏ span verifier reject
    - n_gated: bao nhiêu % span thực sự cần verifier (phần còn lại alignment tự quyết)

  python evaluation/eval_verifier.py --max-spans 20 --sleep 5
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.alignment.align_tfisf import align_pair
from evaluation.plagdet import Span, plagdet_score
from generation.verify import verify_pair, in_edge_band

VAL = r"C:/github/PAN2025/pan25-generated-plagiarism-detection-validation/02_validation/02_validation"
SPANS_CSV = "outputs/validation_spans.csv"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generation")
TH, TH3 = 0.30, 0.50


def read(p):
    with open(p, encoding="utf-8", errors="ignore") as f:
        return f.read()


def overlaps(o, l, golds):
    e = o + l
    return any(not (e <= go or o >= go + gl) for go, gl in golds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-spans", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=5.0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    gold_by, src_of = {}, {}
    for r in csv.DictReader(open(SPANS_CSV, encoding="utf-8", newline="")):
        if r["feature"] == "plagiarism" and r["source_reference"]:
            gold_by.setdefault(r["suspicious_reference"], []).append((int(r["this_offset"]), int(r["this_length"])))
            src_of.setdefault(r["suspicious_reference"], r["source_reference"])

    t0 = time.time()
    rows = []                     # mỗi span: {tp, keep, ...}
    gold_spans, pred_before, pred_after = [], [], []
    fails = 0
    budget = args.max_spans
    for su, golds in gold_by.items():
        if budget <= 0:
            break
        sp, rp = os.path.join(VAL, "susp", su), os.path.join(VAL, "src", src_of[su])
        if not (os.path.exists(sp) and os.path.exists(rp)):
            continue
        stext, rtext = read(sp), read(rp)
        spans = align_pair(stext, rtext, TH, TH, TH3, 4, return_sim=True)
        if not spans:
            continue
        for go, gl in golds:
            gold_spans.append(Span(su, go, gl))
        for s, l, ss, sl, sim in spans:
            if budget <= 0:
                break
            budget -= 1
            tp = overlaps(s, l, golds)
            pred_before.append(Span(su, s, l))
            gated = in_edge_band(sim, TH3)
            if not gated:
                # alignment tự tin (sim ngoài dải biên) -> KHÔNG gọi verifier, tin alignment
                keep, v = True, {"confidence": None}
            else:
                if any(r["gated"] for r in rows):
                    time.sleep(args.sleep)
                try:
                    v = verify_pair(stext[s:s + l], rtext[ss:ss + sl])
                    keep = v["is_plagiarism"]
                except Exception as e:
                    fails += 1
                    keep = True                    # lỗi -> giữ (an toàn), tính vào mẫu
                    v = {"error": str(e)[:80]}
            if keep:
                pred_after.append(Span(su, s, l))
            rows.append({"doc": su, "tp": tp, "keep": keep, "gated": gated, "sim": round(sim, 4),
                        "conf": v.get("confidence")})
            tag = f"{'TP' if tp else 'FP'} -> {'GIỮ' if keep else 'BỎ '}"
            if gated:
                print(f"  {su[:20]} sim={sim:.3f} [gated] {tag} (conf {v.get('confidence')})", flush=True)
            else:
                print(f"  {su[:20]} sim={sim:.3f} [alignment tự quyết] {tag}", flush=True)

    # ma trận nhầm — chỉ tính trên phần THỰC SỰ được verifier xét (gated); phần còn lại
    # là alignment tự quyết, không phải chỗ verifier "khử" gì cả
    gated_rows = [r for r in rows if r["gated"]]
    n_tp = sum(1 for r in gated_rows if r["tp"]); n_fp = len(gated_rows) - n_tp
    keep_tp = sum(1 for r in gated_rows if r["tp"] and r["keep"])
    keep_fp = sum(1 for r in gated_rows if not r["tp"] and r["keep"])
    fp_removed = n_fp - keep_fp
    b = plagdet_score(gold_spans, pred_before)
    a = plagdet_score(gold_spans, pred_after)

    summary = {
        "n_spans": len(rows), "n_gated": len(gated_rows),
        "gated_ratio": round(len(gated_rows) / len(rows), 4) if rows else None,
        "n_TP_gated": n_tp, "n_FP_gated": n_fp, "n_fail": fails,
        "confusion": {"keep_TP": keep_tp, "drop_TP": n_tp - keep_tp,
                      "keep_FP": keep_fp, "drop_FP": fp_removed},
        "fp_reduction": round(fp_removed / n_fp, 4) if n_fp else None,
        "tp_retention": round(keep_tp / n_tp, 4) if n_tp else None,
        "plagdet_before": {"plagdet": round(b.plagdet, 4), "precision": round(b.precision, 4), "recall": round(b.recall, 4)},
        "plagdet_after": {"plagdet": round(a.plagdet, 4), "precision": round(a.precision, 4), "recall": round(a.recall, 4)},
        "runtime_sec": round(time.time() - t0, 1), "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": "nhãn TP/FP suy từ gold PAN (khách quan, không dùng LLM). Cặp gold-source, tách khỏi retrieval. "
                "confusion/fp_reduction/tp_retention chỉ tính trên span 'gated' (case biên quanh th3); "
                "span ngoài dải biên do alignment tự quyết, verifier không đụng tới.",
    }

    print("\n===== EVAL #3 — VERIFIER KHỬ DƯƠNG-TÍNH-GIẢ (case biên) =====")
    print(f"span: {len(rows)} tổng · {len(gated_rows)} gated ({summary['gated_ratio']:.1%}) "
          f"({n_tp} TP, {n_fp} FP trong phần gated) · {fails} lỗi")
    print(f"  FP bị bỏ đúng (fp_reduction): {summary['fp_reduction']}   ({fp_removed}/{n_fp})")
    print(f"  TP được giữ  (tp_retention): {summary['tp_retention']}   ({keep_tp}/{n_tp})")
    print(f"  HEADLINE PlagDet (toàn bộ span, gated+alignment-tự-quyết): {b.plagdet:.3f} -> {a.plagdet:.3f}  |  "
          f"precision {b.precision:.3f} -> {a.precision:.3f}  |  recall {b.recall:.3f} -> {a.recall:.3f}")

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"verifier_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    json.dump({"summary": summary, "spans": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
