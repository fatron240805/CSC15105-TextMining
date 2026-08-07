#!/usr/bin/env python3
"""Đọc evaluation/results/*.json -> dựng 1 trang HTML so sánh các phương pháp.

Vì trình duyệt chặn fetch() trên file://, ta KHÔNG để HTML tự đọc folder lúc mở;
thay vào đó script này nhúng sẵn dữ liệu và xuất evaluation/leaderboard.html.
Chạy lại sau mỗi lần thêm kết quả (run_pipeline.py gọi tự động cuối mỗi run).

  python evaluation/build_leaderboard.py
"""
from __future__ import annotations
import html
import json
import os
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.eval_store import load_results, RESULTS_DIR

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.html")
METRICS = ["plagdet", "precision", "recall", "f1", "granularity"]


def _latest(recs):
    """Giữ bản mới nhất cho mỗi cấu hình (method, split, subset, topk, params)."""
    best = {}
    for r in recs:
        key = (r.get("method"), r.get("split"), r.get("subset"), r.get("topk"),
               json.dumps(r.get("params") or {}, sort_keys=True))
        if key not in best or r.get("timestamp", "") > best[key].get("timestamp", ""):
            best[key] = r
    return list(best.values())


def _pd(r):
    return r.get("metrics", {}).get("plagdet")


def build(results_dir: str = RESULTS_DIR, out: str = OUT) -> str:
    recs = _latest(load_results(results_dir))
    ceilings = [r for r in recs if r.get("kind") == "ceiling"]
    ceil_val = max((_pd(r) for r in ceilings if _pd(r) is not None), default=None)

    scores = [_pd(r) for r in recs if _pd(r) is not None]
    scale_max = 0.9
    if scores:
        import math
        scale_max = min(1.0, math.ceil(max(scores) * 20) / 20 + 0.02)

    # nhóm theo tập đánh giá (split, subset); trong nhóm gắn cờ eval_set khác đa số
    groups = {}
    for r in recs:
        if r.get("kind") == "ceiling":
            continue
        groups.setdefault((r.get("split") or "?", r.get("subset")), []).append(r)
    for rows in groups.values():
        ids = [x.get("eval_set_id") for x in rows if x.get("eval_set_id")]
        maj = Counter(ids).most_common(1)[0][0] if ids else None
        for x in rows:
            x["_flag"] = bool(x.get("eval_set_id") and maj and x["eval_set_id"] != maj)

    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_methods = sum(1 for r in recs if r.get("kind") == "method")

    # ---------- dựng các thẻ biểu đồ theo nhóm ----------
    def refline(val, label, cls):
        pct = min(100, val / scale_max * 100)
        return (f'<div class="ref {cls}" style="left:{pct:.2f}%">'
                f'<span class="rl">{label} {val:.3f}</span></div>')

    cards = []
    for (split, subset) in sorted(groups, key=lambda g: (-(g[1] or 0), g[0])):
        rows = groups[(split, subset)]
        methods = sorted([r for r in rows if r.get("kind") == "method" and _pd(r) is not None],
                         key=lambda r: -_pd(r))
        base = max([r for r in rows if r.get("kind") == "baseline" and _pd(r) is not None],
                   key=lambda r: r.get("timestamp", ""), default=None)
        if not methods:
            continue
        best_pd = _pd(methods[0])
        bars = []
        for r in methods:
            v = _pd(r)
            w = min(100, v / scale_max * 100)
            top = " top" if v == best_pd else ""
            flag = ' <span class="warn" title="Tập đánh giá khác đa số — không so trực tiếp">⚠</span>' if r.get("_flag") else ""
            cfg = f'topk={r.get("topk")}' if r.get("topk") is not None else ""
            bars.append(
                f'<div class="row"><div class="ml">{html.escape(r.get("method",""))}'
                f'<span class="cfg">{cfg}</span>{flag}</div>'
                f'<div class="track"><div class="fill{top}" style="width:{w:.2f}%">'
                f'<span class="val">{v:.3f}</span></div></div></div>')
        refs = ""
        if base is not None:
            refs += refline(_pd(base), "baseline", "base")
        if ceil_val is not None:
            refs += refline(ceil_val, "trần câu", "ceil")
        sub_lbl = f"{subset} susp" if subset else "toàn bộ"
        cards.append(
            f'<section class="card"><div class="chd"><h2>{html.escape(split)} · {sub_lbl}</h2>'
            f'<span class="eid">{len(methods)} phương pháp</span></div>'
            f'<div class="chart"><div class="rows">{"".join(bars)}</div>'
            f'<div class="reflines">{refs}</div></div>'
            f'<div class="axis"><span>0</span><span>{scale_max:.2f}</span></div></section>')

    # ---------- bảng đầy đủ ----------
    trows = []
    for r in sorted(recs, key=lambda r: (-(r.get("subset") or 0), -(_pd(r) or 0))):
        m = r.get("metrics", {})
        kind = r.get("kind", "method")
        badge = {"method": "", "baseline": ' <span class="k base">baseline</span>',
                 "ceiling": ' <span class="k ceil">trần</span>'}.get(kind, "")
        delta = ""
        if kind == "method":
            grp = groups.get((r.get("split") or "?", r.get("subset")), [])
            b = next((x for x in grp if x.get("kind") == "baseline" and _pd(x) is not None), None)
            if b and _pd(r) is not None:
                d = _pd(r) - _pd(b)
                delta = f'<span class="{"up" if d>=0 else "dn"}">{d:+.3f}</span>'
        flag = ' ⚠' if r.get("_flag") else ""
        params = ", ".join(f"{k}={v}" for k, v in (r.get("params") or {}).items())
        cells = [f'{html.escape(r.get("method",""))}{badge}{flag}',
                 html.escape(r.get("split") or "—"), str(r.get("subset") or "—"),
                 str(r.get("topk")) if r.get("topk") is not None else "—"]
        cells += [f'{m[k]:.3f}' if isinstance(m.get(k), (int, float)) else "—" for k in METRICS]
        cells += [delta or "—", html.escape(params) or "—",
                  html.escape((r.get("timestamp") or "")[:16].replace("T", " "))]
        cls = ' class="pd"' if True else ""
        tds = "".join(f"<td{cls if i==4 else ''}>{c}</td>" for i, c in enumerate(cells))
        trows.append(f"<tr>{tds}</tr>")

    empty = '<p class="empty">Chưa có kết quả nào trong evaluation/results/. Chạy run_pipeline.py để tạo.</p>'
    body = ("".join(cards) or empty) + (f"""
 <section class="card wide"><div class="chd"><h2>Tất cả kết quả</h2>
  <span class="eid">bản mới nhất mỗi cấu hình</span></div>
  <div class="twrap"><table><thead><tr>
   <th>Phương pháp</th><th>Split</th><th>Subset</th><th>topk</th>
   <th class="pd">PlagDet</th><th>P</th><th>R</th><th>F1</th><th>Gran</th>
   <th>Δ baseline</th><th>Tham số</th><th>Thời điểm</th>
  </tr></thead><tbody>{''.join(trows)}</tbody></table></div></section>""" if trows else "")

    page = _CSS.replace("__GEN__", gen).replace("__N__", str(n_methods)) \
               .replace("__CEIL__", f"{ceil_val:.3f}" if ceil_val else "—") \
               .replace("__BODY__", body)
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out


_CSS = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Leaderboard — Đánh giá phát hiện đạo văn</title>
<style>
:root{
 --paper:#f4f5f2;--surface:#fff;--surface-2:#eceee9;--ink:#14181c;--ink-2:#545b61;--ink-3:#7c848a;
 --line:#d7dbd4;--accent:#2b6fb0;--accent-soft:rgba(43,111,176,.10);--best:#c05e10;--best-soft:#ffe6cc;
 --good:#3b7548;--bad:#b23b2e;--r:12px;
 --font:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:ui-monospace,"Cascadia Code","JetBrains Mono","SF Mono",Menlo,Consolas,monospace;
}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --paper:#0f1317;--surface:#171d22;--surface-2:#1e252b;--ink:#e7eae6;--ink-2:#9aa3aa;--ink-3:#6f787f;
 --line:#29313a;--accent:#5a9fe0;--accent-soft:rgba(90,159,224,.14);--best:#e08a3c;--best-soft:#4a3418;
 --good:#6db079;--bad:#e0796b;
}}
:root[data-theme="dark"]{
 --paper:#0f1317;--surface:#171d22;--surface-2:#1e252b;--ink:#e7eae6;--ink-2:#9aa3aa;--ink-3:#6f787f;
 --line:#29313a;--accent:#5a9fe0;--accent-soft:rgba(90,159,224,.14);--best:#e08a3c;--best-soft:#4a3418;
 --good:#6db079;--bad:#e0796b;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--font);line-height:1.6;-webkit-font-smoothing:antialiased}
.top{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:14px;padding:14px clamp(16px,4vw,32px);
 background:color-mix(in srgb,var(--paper) 88%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.top .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}
.top h1{font-size:16px;margin:0;font-weight:640;letter-spacing:-.01em}
.top .meta{font-family:var(--mono);font-size:12px;color:var(--ink-3)}
.toggle{margin-left:auto;border:1px solid var(--line);background:var(--surface);color:var(--ink-2);
 border-radius:999px;padding:6px 12px;font:inherit;font-size:12px;cursor:pointer}
.toggle:hover{border-color:var(--accent);color:var(--accent)}
.wrap{max-width:1080px;margin:0 auto;padding:clamp(18px,3vw,32px);display:flex;flex-direction:column;gap:20px}
.card{border:1px solid var(--line);border-radius:var(--r);background:var(--surface);padding:18px 20px}
.chd{display:flex;align-items:baseline;gap:12px;margin-bottom:16px}
.chd h2{font-size:15px;margin:0;font-weight:640}
.chd .eid{font-family:var(--mono);font-size:11px;color:var(--ink-3)}
.chart{position:relative}
.rows{display:flex;flex-direction:column;gap:10px}
.row{display:grid;grid-template-columns:190px 1fr;align-items:center;gap:12px}
.ml{font-size:13px;font-weight:560;display:flex;flex-wrap:wrap;align-items:baseline;gap:6px}
.ml .cfg{font-family:var(--mono);font-size:10px;color:var(--ink-3);font-weight:400}
.warn{color:var(--best);cursor:help}
.track{position:relative;height:22px;background:var(--surface-2);border-radius:5px;overflow:hidden}
.fill{height:100%;background:var(--accent);border-radius:5px 4px 4px 5px;min-width:38px;
 display:flex;align-items:center;justify-content:flex-end;transition:width .5s ease}
.fill.top{background:var(--best)}
.fill .val{font-family:var(--mono);font-size:11px;color:#fff;font-weight:600;padding-right:7px;font-variant-numeric:tabular-nums}
.reflines{position:absolute;left:202px;right:0;top:0;bottom:0;pointer-events:none}
.ref{position:absolute;top:-2px;bottom:-2px;width:0;border-left:2px dashed var(--ink-3)}
.ref.base{border-color:var(--ink-2)} .ref.ceil{border-color:var(--good)}
.ref .rl{position:absolute;top:-16px;left:3px;font-family:var(--mono);font-size:9.5px;white-space:nowrap;color:var(--ink-3)}
.ref.ceil .rl{color:var(--good)}
.axis{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;color:var(--ink-3);
 margin-top:8px;padding-left:202px}
.twrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-3);font-weight:600}
th:first-child,td:first-child{text-align:left}
td.pd,th.pd{font-weight:700;color:var(--ink);font-variant-numeric:tabular-nums}
tbody tr:hover{background:var(--surface-2)}
.k{font-family:var(--mono);font-size:9.5px;padding:1px 6px;border-radius:999px;font-weight:600}
.k.base{background:var(--accent-soft);color:var(--accent)} .k.ceil{background:color-mix(in srgb,var(--good) 15%,transparent);color:var(--good)}
.up{color:var(--good);font-variant-numeric:tabular-nums} .dn{color:var(--bad);font-variant-numeric:tabular-nums}
.empty{color:var(--ink-3);font-size:14px;padding:30px;text-align:center}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-family:var(--mono);font-size:11px;color:var(--ink-2)}
.legend i{display:inline-block;width:22px;height:10px;border-radius:3px;margin-right:6px;vertical-align:middle}
.legend .b1{background:var(--accent)} .legend .b2{background:var(--best)}
.legend .b3{width:0;height:14px;border-left:2px dashed var(--ink-2);border-radius:0}
.legend .b4{width:0;height:14px;border-left:2px dashed var(--good);border-radius:0}
.foot{color:var(--ink-3);font-family:var(--mono);font-size:11px;text-align:center;padding:8px}
</style></head><body>
<div class="top">
 <div><div class="eyebrow">PAN 2025 · Đánh giá</div><h1>Leaderboard phương pháp phát hiện đạo văn</h1></div>
 <span class="meta">__N__ phương pháp · trần câu __CEIL__ · dựng __GEN__</span>
 <button class="toggle" onclick="var r=document.documentElement;r.setAttribute('data-theme',r.getAttribute('data-theme')==='dark'?'light':'dark')">◐ Giao diện</button>
</div>
<div class="wrap">
 <div class="card" style="padding:14px 20px"><div class="legend">
  <span><i class="b1"></i>phương pháp</span><span><i class="b2"></i>tốt nhất nhóm</span>
  <span><i class="b3"></i>baseline (cả tài liệu)</span><span><i class="b4"></i>trần câu (giới hạn granularity câu)</span>
  <span><span class="warn">⚠</span> tập đánh giá khác — không so trực tiếp</span>
 </div></div>
 __BODY__
 <div class="foot">PlagDet (PAN 2015) · thanh bắt đầu từ 0 · so sánh chỉ hợp lệ trong cùng (split · subset)</div>
</div>
</body></html>"""


if __name__ == "__main__":
    print("->", build())
