#!/usr/bin/env python3
"""Đánh giá tầng GENERATION (viết lại khử đạo văn) — tầng 3 của hệ RAG.

Thiết kế (theo góp ý review):
  HEADLINE — chạy LẠI chính detector trên tài liệu đã viết lại: tỉ lệ ký tự bị
             gắn cờ đạo văn TRƯỚC vs SAU. Đây là bằng chứng end-to-end mạnh nhất
             ("detector của chính hệ không còn bắt"), không tốn API.
  DIAGNOSTIC mỗi đoạn:
     - overlap_after  : 3-gram Jaccard(rewrite, nguồn)  — phải GIẢM so với before
     - fidelity       : cos(embed(nghi vấn), embed(rewrite)) — phải CAO (giữ nghĩa)
     - len_ratio      : len(rewrite)/len(nghi vấn) — gắn cờ nếu ngoài [0.75, 1.25]
                        (chống "viết lại" bằng cách rút gọn/xoá nội dung)
  LLM-as-judge (PHỤ, có thiên lệch — Gemini tự chấm Gemini, nêu rõ trong báo cáo):
     - judge_fidelity 1-5, judge_originality 1-5 (chấm RIÊNG 2 trục)

Dùng cặp GOLD (susp ↔ nguồn thật), KHÔNG qua retrieval — đo generator, không đo
retriever. Rewrite lỗi (sau retry) tính là FAIL trong mẫu, không bỏ âm thầm.

  python evaluation/eval_generation.py --max-spans 15 --judge
"""
from __future__ import annotations
import argparse, csv, glob, json, os, statistics, sys, time
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.alignment.align_tfisf import align_pair
from generation.rewrite import rewrite_passage

VAL = r"C:/github/PAN2025/pan25-generated-plagiarism-detection-validation/02_validation/02_validation"
SPANS_CSV = "outputs/validation_spans.csv"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generation")
TH, TH3 = 0.30, 0.50
LEN_LO, LEN_HI = 0.75, 1.25


def read(p):
    with open(p, encoding="utf-8", errors="ignore") as f:
        return f.read()


def ngrams3(text):
    toks = [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split()]
    return set(zip(toks, toks[1:], toks[2:])) if len(toks) >= 3 else set()


def jaccard3(a, b):
    ga, gb = ngrams3(a), ngrams3(b)
    if not ga and not gb:
        return 0.0
    return len(ga & gb) / max(1, len(ga | gb))


def detected_ratio(susp_text, src_text):
    """Tỉ lệ ký tự susp bị detector gắn cờ đạo văn (so với 1 nguồn)."""
    spans = align_pair(susp_text, src_text, TH, TH, TH3, 4)
    chars = sum(l for _, l, *_ in spans)
    return chars / max(1, len(susp_text)), spans


def splice(susp_text, replacements):
    """Ghép tài liệu mới: thay mỗi [start,start+len) bằng bản viết lại."""
    parts, cur = [], 0
    for start, length, rw in sorted(replacements):
        parts.append(susp_text[cur:start]); parts.append(rw); cur = start + length
    parts.append(susp_text[cur:])
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-spans", type=int, default=15, help="ngân sách đoạn viết lại (giới hạn API)")
    ap.add_argument("--judge", action="store_true", help="thêm LLM-as-judge (Gemini tự chấm)")
    ap.add_argument("--sleep", type=float, default=6.0, help="nghỉ giữa các call (tránh 429 free-tier)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    # gold: susp -> source_reference (chỉ lấy cặp có nguồn thật)
    src_of = {}
    for r in csv.DictReader(open(SPANS_CSV, encoding="utf-8", newline="")):
        if r["feature"] == "plagiarism" and r["source_reference"]:
            src_of.setdefault(r["suspicious_reference"], r["source_reference"])

    print("Nạp mô hình fidelity (e5-base-v2, CPU)...", flush=True)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("intfloat/e5-base-v2", device="cpu")

    def fidelity(a, b):
        v = emb.encode([a, b], normalize_embeddings=True, show_progress_bar=False)
        return float(v[0] @ v[1])

    judge_fn = None
    if args.judge:
        from generation.rewrite import _get_client
        from google.genai import types
        gclient = _get_client()

        def judge_fn(susp, rewrite, src):
            prompt = (f"ORIGINAL (suspicious):\n{susp}\n\nSOURCE:\n{src}\n\nREWRITE:\n{rewrite}\n\n"
                      "Rate the REWRITE on two axes, integers 1-5. "
                      "fidelity = how well it preserves the ORIGINAL's meaning. "
                      "originality = how DIFFERENT its wording/structure is from the SOURCE (5=very different). "
                      'Return JSON {"fidelity":n,"originality":n}.')
            try:
                r = gclient.models.generate_content(
                    model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"), contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
                d = json.loads(r.text)
                return int(d["fidelity"]), int(d["originality"])
            except Exception:
                return None, None

    t0 = time.time()
    per_span, per_doc = [], []
    fails = 0
    budget = args.max_spans
    for su, srcref in src_of.items():
        if budget <= 0:
            break
        sp = os.path.join(VAL, "susp", su); rp = os.path.join(VAL, "src", srcref)
        if not (os.path.exists(sp) and os.path.exists(rp)):
            continue
        susp_text, src_text = read(sp), read(rp)
        before_ratio, spans = detected_ratio(susp_text, src_text)
        if not spans:
            continue
        budget -= len(spans)          # viết lại TRỌN tài liệu (headline mới trung thực)
        replacements = []
        for idx, (s, l, ss, sl) in enumerate(spans):
            if per_span:                     # pace giữa các call API (tránh 429 free-tier)
                time.sleep(args.sleep)
            susp_span = susp_text[s:s + l]
            src_span = src_text[ss:ss + sl]
            try:
                rw = rewrite_passage(susp_span, src_span)["rewritten"]
            except Exception as e:
                fails += 1
                replacements.append((s, l, susp_span))     # giữ nguyên nếu rewrite lỗi
                per_span.append({"doc": su, "fail": True, "error": str(e)[:120]})
                continue
            replacements.append((s, l, rw))
            rec = {"doc": su, "fail": False,
                   "overlap_before": round(jaccard3(susp_span, src_span), 4),
                   "overlap_after": round(jaccard3(rw, src_span), 4),
                   "fidelity": round(fidelity(susp_span, rw), 4),
                   "len_ratio": round(len(rw) / max(1, len(susp_span)), 3)}
            rec["len_flag"] = not (LEN_LO <= rec["len_ratio"] <= LEN_HI)
            if judge_fn:
                jf, jo = judge_fn(susp_span, rw, src_span)
                rec["judge_fidelity"], rec["judge_originality"] = jf, jo
            per_span.append(rec)
            print(f"  [{su[:22]}] before={rec['overlap_before']:.3f} after={rec['overlap_after']:.3f} "
                  f"fid={rec['fidelity']:.3f} len={rec['len_ratio']:.2f}"
                  f"{' ⚠len' if rec['len_flag'] else ''}", flush=True)

        rewritten_doc = splice(susp_text, replacements)
        after_ratio, _ = detected_ratio(rewritten_doc, src_text)
        per_doc.append({"doc": su, "before_ratio": round(before_ratio, 4),
                        "after_ratio": round(after_ratio, 4),
                        "reduction": round(before_ratio - after_ratio, 4)})
        print(f"  == {su[:22]}: detector char-ratio {before_ratio:.3f} -> {after_ratio:.3f}", flush=True)

    ok = [r for r in per_span if not r.get("fail")]
    n = len(ok)

    def agg(key):
        xs = [r[key] for r in ok if r.get(key) is not None]
        if not xs:
            return {}
        return {"mean": round(statistics.mean(xs), 4), "median": round(statistics.median(xs), 4),
                "min": round(min(xs), 4), "max": round(max(xs), 4)}

    doc_before = statistics.mean([d["before_ratio"] for d in per_doc]) if per_doc else 0
    doc_after = statistics.mean([d["after_ratio"] for d in per_doc]) if per_doc else 0
    summary = {
        "n_spans_ok": n, "n_spans_fail": fails, "n_docs": len(per_doc),
        "headline_detector_char_ratio": {"before_mean": round(doc_before, 4),
                                         "after_mean": round(doc_after, 4),
                                         "reduction_mean": round(doc_before - doc_after, 4)},
        "overlap_before": agg("overlap_before"), "overlap_after": agg("overlap_after"),
        "fidelity": agg("fidelity"), "len_ratio": agg("len_ratio"),
        "len_flagged": sum(1 for r in ok if r.get("len_flag")),
        "judge_fidelity": agg("judge_fidelity") if args.judge else None,
        "judge_originality": agg("judge_originality") if args.judge else None,
        "runtime_sec": round(time.time() - t0, 1), "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": "gold-source pairs (không qua retrieval); LLM-judge = Gemini tự chấm Gemini (thiên lệch, chỉ tham khảo)",
    }

    print("\n===== ĐÁNH GIÁ GENERATION (viết lại khử đạo văn) =====")
    h = summary["headline_detector_char_ratio"]
    print(f"HEADLINE — detector char-ratio: {h['before_mean']:.3f} -> {h['after_mean']:.3f} "
          f"(giảm {h['reduction_mean']:.3f}) trên {summary['n_docs']} tài liệu")
    print(f"mẫu: {n} đoạn viết lại OK, {fails} lỗi")
    for k in ("overlap_before", "overlap_after", "fidelity", "len_ratio"):
        a = summary[k]
        if a:
            print(f"  {k:15}: mean={a['mean']:.3f} median={a['median']:.3f} [{a['min']:.3f}, {a['max']:.3f}]")
    print(f"  len_flagged (ngoài [{LEN_LO},{LEN_HI}]): {summary['len_flagged']}/{n}")
    if args.judge and summary["judge_fidelity"]:
        print(f"  judge_fidelity : mean={summary['judge_fidelity']['mean']:.2f}/5  "
              f"judge_originality: mean={summary['judge_originality']['mean']:.2f}/5  "
              f"(⚠ Gemini tự chấm Gemini)")

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"gen_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    json.dump({"summary": summary, "per_doc": per_doc, "per_span": per_span},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
