#!/usr/bin/env python3
"""Đánh giá vai trò #1 — phân loại kỹ thuật đạo văn, trên bộ SYNTHETIC có nhãn.

PAN không gán nhãn *kỹ thuật* -> tự sinh: prompt Gemini tạo đạo văn theo kỹ thuật T đã
biết (T = nhãn gold), rồi cho classifier đoán lại. Đo accuracy + macro-F1 + ma trận nhầm.

⚠ Hạn chế: cùng một họ model vừa SINH vừa PHÂN LOẠI -> có thiên lệch; con số là chỉ báo,
   không phải chuẩn vàng (muốn chuẩn: người gán nhãn hoặc model khác để sinh).

  python evaluation/eval_classify.py --n 12 --sleep 5
"""
from __future__ import annotations
import argparse, os, re, sys, json, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation.classify import classify_passage, TECHNIQUES
from generation._gemini import generate_json

VAL = r"C:/github/PAN2025/pan25-generated-plagiarism-detection-validation/02_validation/02_validation"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generation")

GEN_INSTR = {
 "verbatim_copy": "copy it almost word-for-word (change at most 1-2 words).",
 "synonym_substitution": "keep the sentence structure but replace many words with synonyms.",
 "sentence_reordering": "keep the wording but reorder the clauses/sentences.",
 "back_translation": "reword it as if it had been translated to another language and back (natural but different phrasing).",
 "mosaic_patchwork": "stitch fragments of it together, interleaving and reordering short phrases.",
 "idea_plagiarism": "express the same ideas but rewrite the wording almost completely.",
}


def source_paras(n_docs=6):
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(VAL, "src", "*.txt")))[:60]:
        t = open(p, encoding="utf-8", errors="ignore").read()
        for para in re.split(r"\n\s*\n", t):
            para = para.strip().replace("\n", " ")
            if 300 < len(para) < 900 and para.count("\\") < 3 and para.count("{") < 6:
                out.append(para); break
        if len(out) >= n_docs:
            break
    return out


def make_plagiarized(src, technique):
    d = generate_json(
        f"Take the SOURCE passage and {GEN_INSTR[technique]} "
        f"Produce a plagiarized version (English). SOURCE:\n\"\"\"\n{src}\n\"\"\"\n"
        'Return JSON {"text":"<plagiarized passage>"}.',
        temperature=0.9)
    return (d.get("text") or d.get("_raw") or "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12, help="số mẫu (mỗi mẫu = 1 sinh + 1 phân loại)")
    ap.add_argument("--sleep", type=float, default=5.0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    paras = source_paras(max(6, args.n))
    keys = list(TECHNIQUES)
    y_true, y_pred, recs = [], [], []
    fails = 0
    for i in range(args.n):
        src = paras[i % len(paras)]
        gold = keys[i % len(keys)]                 # round-robin qua 6 kỹ thuật
        if recs:
            time.sleep(args.sleep)
        try:
            gen = make_plagiarized(src, gold)
            time.sleep(args.sleep)
            pred = classify_passage(gen, src)["technique"]
        except Exception as e:
            fails += 1
            print(f"  [{i}] LỖI {str(e)[:70]}", flush=True)
            continue
        y_true.append(gold); y_pred.append(pred)
        recs.append({"gold": gold, "pred": pred, "hit": gold == pred})
        print(f"  [{i}] gold={gold:22} pred={pred:22} {'✓' if gold==pred else '✗'}", flush=True)

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc = accuracy_score(y_true, y_pred) if y_true else 0
    macro = f1_score(y_true, y_pred, labels=keys, average="macro", zero_division=0) if y_true else 0
    cm = confusion_matrix(y_true, y_pred, labels=keys).tolist() if y_true else []

    summary = {"n_ok": len(recs), "n_fail": fails, "accuracy": round(acc, 4), "macro_f1": round(macro, 4),
               "labels": keys, "confusion_matrix": cm,
               "timestamp": datetime.now().isoformat(timespec="seconds"),
               "note": "synthetic; cùng model sinh+phân loại -> thiên lệch, chỉ báo tham khảo."}
    print("\n===== EVAL #1 — PHÂN LOẠI KỸ THUẬT (synthetic) =====")
    print(f"mẫu: {len(recs)} OK, {fails} lỗi")
    print(f"  accuracy = {acc:.3f}   macro-F1 = {macro:.3f}")
    print("  ma trận nhầm (hàng=gold, cột=pred):", keys)
    for lab, row in zip(keys, cm):
        print(f"   {lab:22} {row}")

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"classify_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    json.dump({"summary": summary, "cases": recs}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
