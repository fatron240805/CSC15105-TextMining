#!/usr/bin/env python3
"""Demo end-to-end (A4-4): 1 tài liệu nghi vấn -> phát hiện đạo văn + báo cáo.

Retrieval (TF-IDF top-k trong thư mục nguồn) -> Alignment (tf-isf) -> highlight.
Xuất: console, JSON, và HTML tô màu các đoạn đạo văn (kèm nguồn khi hover).

  python demo/run_demo.py --input susp.txt --sources <thư_mục_nguồn> --topk 5
"""
from __future__ import annotations
import argparse, glob, html, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.alignment.align_tfisf import align_pair


def read(p):
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def detect(susp_text, source_files, topk, th, th3):
    """Trả list case {susp_start, susp_len, source, sim} + retrieval ranking."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    src_texts = [read(p) for p in source_files]
    src_ids = [os.path.basename(p) for p in source_files]
    vec = TfidfVectorizer(max_features=100000, sublinear_tf=True, stop_words="english")
    src_m = vec.fit_transform([t[:20000] for t in src_texts])
    susp_m = vec.transform([susp_text[:20000]])
    sims = cosine_similarity(susp_m, src_m)[0]
    k = min(topk, len(src_ids))
    order = np.argsort(-sims)[:k]
    ranking = [(src_ids[j], float(sims[j])) for j in order]

    cases = []
    for j in order:
        for off, ln, ss, sl in align_pair(susp_text, src_texts[j], th, th, th3, 4):
            cases.append({"susp_start": off, "susp_len": ln, "source": src_ids[j],
                          "src_start": ss, "src_len": sl, "src_text": src_texts[j][ss:ss + sl]})
    # dedupe span chồng lấn (giữ cái xuất hiện trước theo offset)
    cases.sort(key=lambda c: c["susp_start"])
    kept, occ = [], []
    for c in cases:
        s, e = c["susp_start"], c["susp_start"] + c["susp_len"]
        if any(not (e <= a or s >= b) for a, b in occ):
            continue
        kept.append(c); occ.append((s, e))
    return kept, ranking


def render_html(susp_text, cases, ranking, title):
    """UI local chất lượng artifact (self-contained, mở bằng browser)."""
    cases = sorted(cases, key=lambda c: c["susp_start"])
    ratio = sum(c["susp_len"] for c in cases) / max(1, len(susp_text))

    # văn bản có highlight, mỗi case 1 anchor
    parts, cur = [], 0
    for i, c in enumerate(cases):
        s, e = c["susp_start"], c["susp_start"] + c["susp_len"]
        parts.append(html.escape(susp_text[cur:s]))
        parts.append(
            f'<mark class="pl" id="c{i}" data-src="{html.escape(c["source"])}" '
            f'title="Nguồn: {html.escape(c["source"])} · ký tự {s}–{e}">'
            f'{html.escape(susp_text[s:e])}</mark>')
        cur = e
    parts.append(html.escape(susp_text[cur:]))
    doc_html = "".join(parts)

    # danh sách case (click để nhảy tới)
    case_items = "".join(
        f'<li><button class="case" data-goto="c{i}">'
        f'<span class="cn">#{i + 1}</span>'
        f'<span class="ct">{html.escape((susp_text[c["susp_start"]:c["susp_start"]+70]).strip()[:70])}…</span>'
        f'<span class="cm"><code>{html.escape(c["source"])}</code> · '
        f'{c["susp_start"]}–{c["susp_start"]+c["susp_len"]}</span></button></li>'
        for i, c in enumerate(cases)) or '<li class="empty">Không phát hiện đoạn đạo văn.</li>'

    rank_items = "".join(
        f'<li><code>{html.escape(sid)}</code>'
        f'<span class="bar"><span style="width:{min(100, sc*100):.0f}%"></span></span>'
        f'<b>{sc:.3f}</b></li>' for sid, sc in ranking)

    pct = round(ratio * 100)
    sev = "high" if ratio >= 0.5 else "med" if ratio >= 0.15 else "low"

    # dữ liệu cho modal đối chiếu (nhúng JSON an toàn cho <script>)
    cases_json = json.dumps([{
        "start": c["susp_start"], "len": c["susp_len"],
        "susp": susp_text[c["susp_start"]:c["susp_start"] + c["susp_len"]],
        "source": c["source"], "src_start": c.get("src_start", 0),
        "src_len": c.get("src_len", 0), "src_text": c.get("src_text", ""),
    } for c in cases], ensure_ascii=False).replace("</", "<\\/")

    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Đạo văn — {html.escape(title)}</title>
<style>
:root{{
 --paper:#f4f5f2;--surface:#fff;--surface-2:#eceee9;--ink:#14181c;--ink-2:#545b61;--ink-3:#7c848a;
 --line:#d7dbd4;--accent:#2b6fb0;--accent-soft:rgba(43,111,176,.10);--pl:#c05e10;--pl-soft:#ffe6cc;
 --good:#3b7548;--r:12px;
 --font:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:ui-monospace,"Cascadia Code","JetBrains Mono","SF Mono",Menlo,Consolas,monospace;
}}
:root[data-theme="dark"],:root:not([data-theme]){{}}
@media(prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
 --paper:#0f1317;--surface:#171d22;--surface-2:#1e252b;--ink:#e7eae6;--ink-2:#9aa3aa;--ink-3:#6f787f;
 --line:#29313a;--accent:#5a9fe0;--accent-soft:rgba(90,159,224,.14);--pl:#e08a3c;--pl-soft:#4a3418;--good:#6db079;
}}}}
:root[data-theme="dark"]{{
 --paper:#0f1317;--surface:#171d22;--surface-2:#1e252b;--ink:#e7eae6;--ink-2:#9aa3aa;--ink-3:#6f787f;
 --line:#29313a;--accent:#5a9fe0;--accent-soft:rgba(90,159,224,.14);--pl:#e08a3c;--pl-soft:#4a3418;--good:#6db079;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--font);line-height:1.6;-webkit-font-smoothing:antialiased}}
.top{{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:14px;padding:14px clamp(16px,4vw,32px);
 background:color-mix(in srgb,var(--paper) 88%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}}
.top .eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}}
.top h1{{font-size:16px;margin:0;font-weight:640;letter-spacing:-.01em}}
.top .doc{{font-family:var(--mono);font-size:12px;color:var(--ink-3)}}
.toggle{{margin-left:auto;border:1px solid var(--line);background:var(--surface);color:var(--ink-2);
 border-radius:999px;padding:6px 12px;font:inherit;font-size:12px;cursor:pointer}}
.toggle:hover{{border-color:var(--accent);color:var(--accent)}}
.wrap{{max-width:1180px;margin:0 auto;padding:clamp(18px,3vw,32px)}}
.hero{{display:flex;align-items:center;gap:22px;border:1px solid var(--line);border-radius:var(--r);
 background:var(--surface);padding:20px 24px;margin-bottom:20px}}
.gauge{{--v:{pct};position:relative;width:96px;height:96px;border-radius:50%;flex:none;
 background:conic-gradient(var(--pl) calc(var(--v)*1%),var(--surface-2) 0)}}
.gauge::after{{content:"";position:absolute;inset:11px;border-radius:50%;background:var(--surface)}}
.gauge b{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:1;
 font-size:24px;font-weight:680;font-variant-numeric:tabular-nums;color:var(--pl)}}
.hero .k{{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}}
.hero .big{{font-size:26px;font-weight:660}}
.hero .sub{{color:var(--ink-2);font-size:14px}}
.sev{{display:inline-block;font-family:var(--mono);font-size:11px;padding:2px 9px;border-radius:999px;letter-spacing:.04em}}
.sev.high{{background:var(--pl-soft);color:var(--pl)}} .sev.med{{background:var(--accent-soft);color:var(--accent)}}
.sev.low{{background:color-mix(in srgb,var(--good) 15%,transparent);color:var(--good)}}
.grid{{display:grid;grid-template-columns:1fr 320px;gap:20px;align-items:start}}
@media(max-width:820px){{.grid{{grid-template-columns:1fr}}}}
.panel{{border:1px solid var(--line);border-radius:var(--r);background:var(--surface)}}
.panel h2{{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);
 margin:0;padding:14px 18px;border-bottom:1px solid var(--line)}}
.doc{{padding:20px 22px;white-space:pre-wrap;word-wrap:break-word;font:14px/1.75 var(--mono);max-height:72vh;overflow:auto}}
mark.pl{{background:var(--pl-soft);color:inherit;border-bottom:2px solid var(--pl);border-radius:2px;padding:.5px 1px;cursor:help;scroll-margin-top:70px}}
mark.pl.flash{{animation:fl 1.1s ease}}
@keyframes fl{{0%,100%{{background:var(--pl-soft)}}30%{{background:var(--pl);color:#fff}}}}
.aside{{position:sticky;top:70px;display:flex;flex-direction:column;gap:16px}}
ul{{list-style:none;margin:0;padding:0}}
.cases li{{border-bottom:1px solid var(--line)}} .cases li:last-child{{border:0}}
.case{{width:100%;text-align:left;background:none;border:0;color:inherit;font:inherit;cursor:pointer;
 padding:11px 16px;display:grid;grid-template-columns:auto 1fr;gap:2px 8px}}
.case:hover{{background:var(--surface-2)}}
.cn{{grid-row:1/3;font-family:var(--mono);font-size:12px;color:var(--pl);font-weight:640;align-self:center}}
.ct{{font-size:13px;color:var(--ink)}} .cm{{font-family:var(--mono);font-size:11px;color:var(--ink-3)}}
.cm code{{color:var(--ink-2)}}
.rank li{{display:flex;align-items:center;gap:8px;padding:9px 16px;font-family:var(--mono);font-size:12px;color:var(--ink-2)}}
.rank .bar{{flex:1;height:5px;background:var(--surface-2);border-radius:3px;overflow:hidden}}
.rank .bar span{{display:block;height:100%;background:var(--accent)}}
.rank b{{color:var(--ink);font-variant-numeric:tabular-nums}}
.empty{{padding:16px;color:var(--ink-3);font-size:13px}}
.foot{{color:var(--ink-3);font-family:var(--mono);font-size:11px;margin-top:22px}}
.modal{{position:fixed;inset:0;z-index:20;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.45);padding:18px}}
.modal.open{{display:flex}}
.mcard{{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);max-width:900px;width:100%;max-height:86vh;display:flex;flex-direction:column;overflow:hidden}}
.mtop{{display:flex;align-items:center;gap:10px;padding:14px 18px;border-bottom:1px solid var(--line)}}
.mtop h3{{margin:0;font-size:15px;font-weight:640}}
.mtop .x{{margin-left:auto;border:0;background:none;color:var(--ink-2);font-size:22px;line-height:1;cursor:pointer;padding:0 4px}}
.mtop .x:hover{{color:var(--pl)}}
.mcols{{display:grid;grid-template-columns:1fr 1fr;gap:0;overflow:auto}}
@media(max-width:640px){{.mcols{{grid-template-columns:1fr}}}}
.mcol{{padding:16px 18px;min-width:0}} .mcol+.mcol{{border-left:1px solid var(--line)}}
@media(max-width:640px){{.mcol+.mcol{{border-left:0;border-top:1px solid var(--line)}}}}
.mcol .lab{{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px}}
.mcol.susp .lab{{color:var(--pl)}} .mcol.src .lab{{color:var(--accent)}}
.mcol .meta{{font-family:var(--mono);font-size:11px;color:var(--ink-3);margin-bottom:10px}}
.mcol .passage{{font:13px/1.7 var(--mono);white-space:pre-wrap;word-wrap:break-word}}
.mcol.susp .passage{{background:var(--pl-soft);border-radius:6px;padding:10px 12px}}
</style></head><body>
<div class="top">
 <div><div class="eyebrow">PAN 2025 · Phát hiện đạo văn</div><h1>Báo cáo</h1></div>
 <span class="doc">{html.escape(title)}</span>
 <button class="toggle" onclick="tt()">◐ Giao diện</button>
</div>
<div class="wrap">
 <div class="hero">
  <div class="gauge"><b>{pct}</b></div>
  <div>
   <div class="k">Điểm đạo văn</div>
   <div class="big">{ratio:.2f} <span class="sev {sev}">{ {'high':'CAO','med':'TRUNG BÌNH','low':'THẤP'}[sev] }</span></div>
   <div class="sub">{len(cases)} đoạn đạo văn · {len(ranking)} nguồn được truy hồi · {len(susp_text):,} ký tự</div>
  </div>
 </div>
 <div class="grid">
  <div class="panel"><h2>Văn bản — đoạn tô cam là đạo văn (hover xem nguồn)</h2>
   <div class="doc">{doc_html}</div></div>
  <div class="aside">
   <div class="panel cases"><h2>Đoạn phát hiện</h2><ul>{case_items}</ul></div>
   <div class="panel rank"><h2>Nguồn khả nghi (retrieval)</h2><ul>{rank_items}</ul></div>
  </div>
 </div>
 <div class="foot">Retrieval TF-IDF → Alignment tf-isf (seed-and-extend) → PlagDet · hệ thống chạy local</div>
</div>
<div class="modal" id="modal" onclick="if(event.target===this)closeM()"></div>
<script>
 var CASES={cases_json};
 function esc(s){{return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}
 function tt(){{var r=document.documentElement;var d=r.getAttribute('data-theme')==='dark';
  r.setAttribute('data-theme',d?'light':'dark');}}
 function openM(i){{var c=CASES[i];if(!c)return;var mo=document.getElementById('modal');
  mo.innerHTML='<div class="mcard"><div class="mtop"><h3>Đoạn #'+(i+1)+' — đối chiếu nghi vấn ↔ nguồn</h3>'
   +'<button class="x" title="Đóng (Esc)" onclick="closeM()">×</button></div><div class="mcols">'
   +'<div class="mcol susp"><div class="lab">Đoạn nghi vấn</div><div class="meta">ký tự '+c.start+'–'+(c.start+c.len)+'</div><div class="passage">'+esc(c.susp)+'</div></div>'
   +'<div class="mcol src"><div class="lab">Nguồn — đạo từ đây</div><div class="meta"><code>'+esc(c.source)+'</code> · ký tự '+c.src_start+'–'+(c.src_start+c.src_len)+'</div><div class="passage">'+esc(c.src_text||'(không có văn bản nguồn)')+'</div></div>'
   +'</div></div>';mo.classList.add('open');}}
 function closeM(){{document.getElementById('modal').classList.remove('open');}}
 function jump(i){{var m=document.getElementById('c'+i);if(m){{m.scrollIntoView({{behavior:'smooth',block:'start'}});
  m.classList.remove('flash');void m.offsetWidth;m.classList.add('flash');}}openM(i);}}
 document.querySelectorAll('.case').forEach(function(b){{b.onclick=function(){{jump(parseInt(b.dataset.goto.slice(1)));}};}});
 document.querySelectorAll('.doc mark.pl').forEach(function(m,i){{m.style.cursor='pointer';m.onclick=function(){{openM(i);}};}});
 document.addEventListener('keydown',function(e){{if(e.key==='Escape')closeM();}});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="tài liệu nghi vấn (.txt)")
    ap.add_argument("--sources", required=True, help="thư mục chứa nguồn (.txt)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--th", type=float, default=0.30)
    ap.add_argument("--th3", type=float, default=0.50)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    susp_text = read(args.input)
    src_files = sorted(glob.glob(os.path.join(args.sources, "*.txt")))
    if not src_files:
        ap.error(f"không có .txt trong {args.sources}")

    cases, ranking = detect(susp_text, src_files, args.topk, args.th, args.th3)
    ratio = sum(c["susp_len"] for c in cases) / max(1, len(susp_text))
    name = os.path.basename(args.input)

    print(f"\n📄 {name} — Điểm đạo văn: {ratio:.2f} — {len(cases)} đoạn")
    for c in sorted(cases, key=lambda c: c["susp_start"]):
        s = c["susp_start"]
        print(f"  ├─ [ký tự {s}–{s+c['susp_len']}] ← {c['source']}: "
              f"{susp_text[s:s+60].strip()[:60]!r}...")

    os.makedirs(args.out, exist_ok=True)
    json.dump({"input": name, "plagiarism_ratio": round(ratio, 4), "cases": cases,
               "retrieval": ranking}, open(os.path.join(args.out, "demo_report.json"), "w",
                                           encoding="utf-8"), ensure_ascii=False, indent=2)
    html_path = os.path.join(args.out, "demo_report.html")
    open(html_path, "w", encoding="utf-8").write(render_html(susp_text, cases, ranking, name))
    print(f"\n-> {html_path}\n-> {os.path.join(args.out, 'demo_report.json')}")


if __name__ == "__main__":
    main()
