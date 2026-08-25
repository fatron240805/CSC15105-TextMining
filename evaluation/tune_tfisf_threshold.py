#!/usr/bin/env python3
"""Sweep th1/th3 của aligner tf-isf trên cặp (susp, nguồn GOLD) — KHÔNG dùng LLM.

Vì sao script này tồn tại (rà soát vai trò sau góp ý thầy, 2026-08-24):
`generation/verify.py` (vai trò #3) báo cáo +0.044 PlagDet (GLM-5.2) khi áp SAU alignment
tại ngưỡng MẶC ĐỊNH cố định th=0.30/th3=0.50 (xem README, evaluation/compare_models.py).
Nhưng chưa ai đo: riêng việc TUNE LẠI ngưỡng th3 (không cần LLM, không tốn API, tái lập
được) có tự đạt PlagDet tương đương không? Nếu có -> verifier đang làm lại việc alignment
lẽ ra nên tự làm bằng ngưỡng (chồng lấn thật). Nếu tune ngưỡng đơn thuần vẫn kém xa PlagDet
sau verifier -> verifier có giá trị thật, không chỉ là bản sao của việc chỉnh ngưỡng.

Cách đọc kết quả: so PlagDet tốt nhất trong bảng sweep này với "plagdet_after" của model
tốt nhất trong evaluation/generation/model_compare_*.json (chạy evaluation/compare_models.py).
  - sweep_best xấp xỉ plagdet_after(best model)  -> verifier dư thừa, nên bỏ khỏi pipeline
    chấm điểm, chỉ tune th3 là đủ.
  - sweep_best thấp hơn rõ rệt                    -> verifier thêm giá trị thật ở case biên,
    giữ lại đúng theo phạm vi in_edge_band() (generation/verify.py).

Dùng cặp GOLD (susp <-> nguồn thật), tách khỏi retrieval — đo đúng aligner, không lẫn
lỗi retrieval. th1=th2 (giữ như mặc định hệ thống); chỉ sweep th3 (ngưỡng similarity mở
rộng — tham số quyết định precision/recall rõ nhất, xem align_tfisf.align_pair).

  python evaluation/tune_tfisf_threshold.py --max-docs 1000
  python evaluation/tune_tfisf_threshold.py --max-docs 1000 --th1 0.25 0.30 0.35
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.alignment.align_tfisf import align_pair
from evaluation.plagdet import Span, plagdet_score

VAL = r"C:/github/PAN2025/pan25-generated-plagiarism-detection-validation/02_validation/02_validation"
SPANS_CSV = "outputs/validation_spans.csv"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generation")
TH3_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def read(p):
    with open(p, encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_gold_pairs(max_docs):
    """susp -> (text, nguồn GOLD text, gold spans) — chỉ tài liệu có source_reference thật."""
    gold_by, src_of = {}, {}
    for r in csv.DictReader(open(SPANS_CSV, encoding="utf-8", newline="")):
        if r["feature"] == "plagiarism" and r["source_reference"]:
            gold_by.setdefault(r["suspicious_reference"], []).append(
                (int(r["this_offset"]), int(r["this_length"])))
            src_of.setdefault(r["suspicious_reference"], r["source_reference"])
    pairs = []
    for su, golds in gold_by.items():
        if len(pairs) >= max_docs:
            break
        sp, rp = os.path.join(VAL, "susp", su), os.path.join(VAL, "src", src_of[su])
        if not (os.path.exists(sp) and os.path.exists(rp)):
            continue
        pairs.append((su, read(sp), read(rp), golds))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-docs", type=int, default=1000,
                    help="số cặp (susp, nguồn gold) — khớp --n của compare_models.py để so sánh công bằng")
    ap.add_argument("--th1", type=float, nargs="+", default=[0.30],
                    help="grid th1=th2 (mặc định hệ thống: 0.30)")
    ap.add_argument("--th3", type=float, nargs="+", default=TH3_GRID, help="grid th3")
    ap.add_argument("--max-gap", type=int, default=4)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    t0 = time.time()
    print(f"Nạp {args.max_docs} cặp (susp, nguồn gold)...", flush=True)
    pairs = load_gold_pairs(args.max_docs)
    gold_spans = [Span(su, o, l) for su, _, _, golds in pairs for o, l in golds]
    print(f"  {len(pairs)} cặp, {len(gold_spans)} gold span ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    best = None
    for th1 in args.th1:
        for th3 in args.th3:
            pred = []
            for su, stext, rtext, _ in pairs:
                for s, l, ss, sl in align_pair(stext, rtext, th1, th1, th3, args.max_gap):
                    pred.append(Span(su, s, l))
            r = plagdet_score(gold_spans, pred)
            row = {"th1": th1, "th3": th3, **r.as_dict()}
            rows.append(row)
            print(f"  th1={th1:.2f} th3={th3:.2f} | P={r.precision:.3f} R={r.recall:.3f} "
                  f"F1={r.f1:.3f} gran={r.granularity:.3f} PlagDet={r.plagdet:.3f}", flush=True)
            if best is None or r.plagdet > best["plagdet"]:
                best = row

    summary = {
        "n_docs": len(pairs), "n_gold_spans": len(gold_spans),
        "th1_grid": args.th1, "th3_grid": args.th3,
        "sweep_best": best,
        "runtime_sec": round(time.time() - t0, 1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": "Chỉ tune ngưỡng, KHÔNG dùng LLM. So plagdet của sweep_best với "
                "plagdet_after (model tốt nhất) trong evaluation/generation/model_compare_*.json "
                "để biết verifier LLM có thêm giá trị ngoài việc tune ngưỡng hay không.",
    }
    print(f"\n===== SWEEP TF-ISF (th3) — {len(pairs)} tài liệu, {len(gold_spans)} gold span =====")
    print(f"TỐT NHẤT: th1={best['th1']:.2f} th3={best['th3']:.2f} -> PlagDet={best['plagdet']:.3f} "
          f"(P={best['precision']:.3f} R={best['recall']:.3f})")

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"tfisf_threshold_sweep_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
