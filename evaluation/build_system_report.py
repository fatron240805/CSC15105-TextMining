#!/usr/bin/env python3
"""Gom 3 tầng đánh giá (Retrieval + Detection + Generation) -> 1 trang HTML tổng.

Đọc:
  outputs/retrieval_eval.csv                 (tầng 1 — Recall@k, MRR)
  evaluation/results/*.json  (qua eval_store) (tầng 2 — PlagDet)
  evaluation/generation/gen_eval_*.json       (tầng 3 — khử đạo văn)

  python evaluation/build_system_report.py
"""
from __future__ import annotations
import csv, glob, html, json, os, sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.eval_store import load_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_report.html")


def read_retrieval():
    p = os.path.join(ROOT, "outputs", "retrieval_eval.csv")
    if not os.path.exists(p):
        return {}
    d = defaultdict(dict)
    for r in csv.DictReader(open(p, encoding="utf-8")):
        d[r["method"]][r["metric"]] = float(r["value"])
    return d


def read_detection():
    rows = [r for r in load_results() if r.get("split") == "val"]
    # nhóm full-val lớn nhất
    best_sub = max((r.get("subset") or 0) for r in rows) if rows else 0
    grp = [r for r in rows if r.get("subset") == best_sub]
    methods = []
    for r in grp:
        m = r["metrics"]
        methods.append((r["method"], r.get("topk"), r["kind"], m.get("plagdet"),
                        m.get("precision"), m.get("recall")))
    methods.sort(key=lambda x: -(x[3] or 0))
    return best_sub, methods


def read_generation():
    fs = glob.glob(os.path.join(ROOT, "evaluation", "generation", "gen_eval_*.json"))
    if not fs:
        return None
    best = max(fs, key=lambda f: json.load(open(f, encoding="utf-8"))["summary"]["n_spans_ok"])
    return json.load(open(best, encoding="utf-8"))


def esc(x):
    return html.escape(str(x))


def main():
    ret = read_retrieval()
    det_sub, det = read_detection()
    gen = read_generation()
    gen_docs = [d for d in gen["per_doc"] if d["reduction"] > 0] if gen else []
    gs = gen["summary"] if gen else {}

    # ---- tầng 1: retrieval rows ----
    rrows = ""
    for meth, lbl in [("tfidf", "TF-IDF (dùng chính thức)"), ("embedding", "Neural (bge-small)")]:
        m = ret.get(meth, {})
        if not m:
            continue
        top = " top" if meth == "tfidf" else ""
        rrows += (f'<tr class="{top.strip()}"><td>{esc(lbl)}</td>'
                  f'<td class="n">{m.get("recall@1",0):.3f}</td><td class="n">{m.get("recall@5",0):.3f}</td>'
                  f'<td class="n">{m.get("recall@10",0):.3f}</td><td class="n">{m.get("mrr",0):.3f}</td></tr>')

    # ---- tầng 2: detection rows ----
    drows = ""
    best_pd = max((x[3] or 0) for x in det) if det else 0
    for meth, tk, kind, pd, p, r in det:
        if pd is None:
            continue
        top = " top" if pd == best_pd and kind == "method" else ""
        cfg = f"top-{tk}" if tk else ("baseline" if kind == "baseline" else "")
        drows += (f'<tr class="{top.strip()}"><td>{esc(meth)} <span class="cfg">{cfg}</span></td>'
                  f'<td class="n big">{pd:.3f}</td>'
                  f'<td class="n">{p:.3f}</td><td class="n">{r:.3f}</td></tr>')

    # ---- tầng 3: generation ----
    if gen:
        hb = gen_docs[0] if gen_docs else None
        head_before = hb["before_ratio"] if hb else 0
        head_after = hb["after_ratio"] if hb else 0
        ov_b = gs["overlap_before"].get("mean", 0)
        ov_a = gs["overlap_after"].get("mean", 0)
        fid = gs["fidelity"].get("mean", 0)
        lr = gs["len_ratio"].get("mean", 0)
        n_ok, n_fail = gs["n_spans_ok"], gs["n_spans_fail"]
        jf = gs.get("judge_fidelity") or {}
        jo = gs.get("judge_originality") or {}

    gen_block = (f"""
  <section class="card">
   <div class="chd"><h2>Tầng 3 — Generation (viết lại khử đạo văn)</h2>
    <span class="eid">Gemini · {esc(hb['doc']) if gen_docs else '—'} · {n_ok} đoạn viết lại</span></div>
   <div class="statrow">
     <div class="stat"><div class="k">Detector char-ratio (đoạn nghi vấn bị gắn cờ)</div>
       <div class="big2"><span class="pl">{head_before:.3f}</span> → <span class="good">{head_after:.3f}</span></div>
       <div class="sub">chính detector của hệ gắn cờ ít hơn ~{(1-head_after/max(head_before,1e-9))*100:.0f}% sau khi viết lại</div></div>
     <div class="stat"><div class="k">Trùng lặp từ vựng với nguồn (3-gram)</div>
       <div class="big2"><span class="pl">{ov_b:.3f}</span> → <span class="good">{ov_a:.3f}</span></div>
       <div class="sub">giảm ~{(1-ov_a/max(ov_b,1e-9))*100:.0f}%</div></div>
     <div class="stat"><div class="k">Giữ nghĩa (fidelity, embedding cos)</div>
       <div class="big2 good">{fid:.3f}</div><div class="sub">cao = nội dung được bảo toàn</div></div>
     <div class="stat"><div class="k">Tỉ lệ độ dài (chống rút gọn giả)</div>
       <div class="big2">{lr:.2f}</div><div class="sub">quanh 1.0 = không cắt xén nội dung</div></div>
   </div>
   <p class="note">Mẫu: {n_ok} đoạn viết lại thành công, {n_fail} lỗi (hết quota Gemini free-tier).
    LLM-as-judge (⚠ Gemini tự chấm chính nó — chỉ tham khảo): fidelity {jf.get('mean',0):.1f}/5,
    originality {jo.get('mean',0):.1f}/5. Dùng cặp gold-source (không qua retrieval) để đo riêng generator.</p>
  </section>""" if gen else '<section class="card"><h2>Tầng 3 — Generation</h2><p class="note">Chưa có dữ liệu.</p></section>')

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    page = _CSS.replace("__GEN_TIME__", gen_time) \
               .replace("__RRows__", rrows or '<tr><td colspan="5">—</td></tr>') \
               .replace("__DSUB__", str(det_sub)) \
               .replace("__DRows__", drows or '<tr><td colspan="4">—</td></tr>') \
               .replace("__GENBLOCK__", gen_block)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print("->", OUT)


_CSS = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Đánh giá tổng hệ thống — PAN 2025 RAG phát hiện đạo văn</title>
<style>
:root{
 --paper:#f4f5f2;--surface:#fff;--surface-2:#eceee9;--ink:#14181c;--ink-2:#545b61;--ink-3:#7c848a;
 --line:#d7dbd4;--accent:#2b6fb0;--pl:#c05e10;--good:#3b7548;--r:12px;
 --font:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:ui-monospace,"Cascadia Code","JetBrains Mono",Menlo,Consolas,monospace}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --paper:#0f1317;--surface:#171d22;--surface-2:#1e252b;--ink:#e7eae6;--ink-2:#9aa3aa;--ink-3:#6f787f;
 --line:#29313a;--accent:#5a9fe0;--pl:#e08a3c;--good:#6db079}}
:root[data-theme="dark"]{
 --paper:#0f1317;--surface:#171d22;--surface-2:#1e252b;--ink:#e7eae6;--ink-2:#9aa3aa;--ink-3:#6f787f;
 --line:#29313a;--accent:#5a9fe0;--pl:#e08a3c;--good:#6db079}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--font);line-height:1.6}
.top{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:14px;padding:14px clamp(16px,4vw,32px);
 background:color-mix(in srgb,var(--paper) 88%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.top .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}
.top h1{font-size:16px;margin:0;font-weight:640}
.top .meta{font-family:var(--mono);font-size:12px;color:var(--ink-3);margin-left:auto}
.toggle{border:1px solid var(--line);background:var(--surface);color:var(--ink-2);border-radius:999px;
 padding:6px 12px;font:inherit;font-size:12px;cursor:pointer}.toggle:hover{border-color:var(--accent);color:var(--accent)}
.wrap{max-width:1000px;margin:0 auto;padding:clamp(18px,3vw,32px);display:flex;flex-direction:column;gap:20px}
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;font-family:var(--mono);font-size:13px;
 background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:16px 20px}
.flow b{padding:5px 11px;border-radius:7px;background:var(--surface-2)}
.flow b.r{color:var(--accent)}.flow b.d{color:var(--pl)}.flow b.g{color:var(--good)}
.flow span{color:var(--ink-3)}
.card{border:1px solid var(--line);border-radius:var(--r);background:var(--surface);padding:18px 20px}
.chd{display:flex;align-items:baseline;gap:12px;margin-bottom:14px}
.chd h2{font-size:15px;margin:0;font-weight:640}.chd .eid{font-family:var(--mono);font-size:11px;color:var(--ink-3)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-3);font-weight:600}
th:first-child,td:first-child{text-align:left}
td.n{font-variant-numeric:tabular-nums}td.big{font-weight:700}
tr.top td{background:color-mix(in srgb,var(--accent) 7%,transparent)}
tr.top td:first-child{font-weight:640}
.cfg{font-family:var(--mono);font-size:10px;color:var(--ink-3)}
.statrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}
.stat{border:1px solid var(--line);border-radius:10px;padding:14px 16px;background:var(--surface-2)}
.stat .k{font-family:var(--mono);font-size:10.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-3)}
.big2{font-size:24px;font-weight:680;font-variant-numeric:tabular-nums;margin:6px 0 2px}
.big2 .pl{color:var(--pl)}.big2 .good,.big2.good{color:var(--good)}
.stat .sub{font-size:12px;color:var(--ink-2)}
.note{font-size:12.5px;color:var(--ink-2);margin:14px 0 0;line-height:1.65}
.foot{color:var(--ink-3);font-family:var(--mono);font-size:11px;text-align:center;padding:8px}
</style></head><body>
<div class="top">
 <div><div class="eyebrow">PAN 2025 · Text Mining</div><h1>Đánh giá tổng hệ thống RAG phát hiện đạo văn</h1></div>
 <span class="meta">val 5522 susp · dựng __GEN_TIME__</span>
 <button class="toggle" onclick="var r=document.documentElement;r.setAttribute('data-theme',r.getAttribute('data-theme')==='dark'?'light':'dark')">◐</button>
</div>
<div class="wrap">
 <div class="flow"><b class="r">Retrieval</b><span>TF-IDF top-k</span><span>→</span>
  <b class="d">Detection</b><span>tf-isf seed-and-extend + PlagDet</span><span>→</span>
  <b class="g">Generation</b><span>Gemini viết lại khử đạo văn</span></div>

 <section class="card">
  <div class="chd"><h2>Tầng 1 — Retrieval (truy hồi nguồn)</h2><span class="eid">val 5522 susp · Recall@k / MRR</span></div>
  <table><thead><tr><th>Phương pháp</th><th>R@1</th><th>R@5</th><th>R@10</th><th>MRR</th></tr></thead>
  <tbody>__RRows__</tbody></table>
 </section>

 <section class="card">
  <div class="chd"><h2>Tầng 2 — Detection (phát hiện + đối chiếu)</h2><span class="eid">val __DSUB__ susp · PlagDet (PAN 2015)</span></div>
  <table><thead><tr><th>Phương pháp</th><th>PlagDet</th><th>P</th><th>R</th></tr></thead>
  <tbody>__DRows__</tbody></table>
 </section>
__GENBLOCK__
 <div class="foot">Retrieval TF-IDF → Detection tf-isf (PlagDet) → Generation Gemini · đánh giá 3 tầng theo design doc</div>
</div>
</body></html>"""


if __name__ == "__main__":
    main()
