#!/usr/bin/env python3
"""Demo app nhỏ — web UI phát hiện đạo văn (stdlib, KHÔNG cần Flask).

Backend: nạp kho nguồn + fit TF-IDF MỘT LẦN lúc khởi động; mỗi request chạy
Retrieval (TF-IDF top-k) -> Alignment (tf-isf seed-and-extend) -> trả JSON.
Frontend: dán/upload tài liệu -> bấm Kiểm tra -> highlight ngay (artifact-style).

  python demo/app.py --sources "C:/github/PAN2025/00_spot_check/00_spot_check/src"
  Mở http://localhost:8000
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.alignment.align_tfisf import align_pair


def _read(p):
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


class Detector:
    """Nạp kho nguồn + fit TF-IDF một lần; detect() dùng lại."""

    def __init__(self, sources_dir, topk=3, th=0.30, th3=0.50):
        from sklearn.feature_extraction.text import TfidfVectorizer
        files = sorted(glob.glob(os.path.join(sources_dir, "*.txt")))
        if not files:
            raise SystemExit(f"Không có .txt trong {sources_dir}")
        self.ids = [os.path.basename(p) for p in files]
        self.texts = [_read(p) for p in files]
        self.vec = TfidfVectorizer(max_features=100000, sublinear_tf=True, stop_words="english")
        self.src_m = self.vec.fit_transform([t[:20000] for t in self.texts])
        self.topk, self.th, self.th3 = topk, th, th3
        print(f"[detector] nạp {len(self.ids)} nguồn, TF-IDF sẵn sàng.", flush=True)

    def detect(self, text):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        q = self.vec.transform([text[:20000]])
        sims = cosine_similarity(q, self.src_m)[0]
        order = list(np.argsort(-sims)[:max(self.topk, 3)])       # hiện 3 nguồn top, align topk
        ranking = [{"source": self.ids[j], "score": round(float(sims[j]), 4)} for j in order]

        cases = []
        for j in order[:self.topk]:
            for off, ln, ss, sl in align_pair(text, self.texts[j], self.th, self.th, self.th3, 4):
                cases.append({"start": off, "len": ln, "source": self.ids[j],
                              "src_start": ss, "src_len": sl,
                              "src_text": self.texts[j][ss:ss + sl]})
        cases.sort(key=lambda c: c["start"])                      # dedupe chồng lấn
        kept, occ = [], []
        for c in cases:
            s, e = c["start"], c["start"] + c["len"]
            if any(not (e <= a or s >= b) for a, b in occ):
                continue
            kept.append(c); occ.append((s, e))
        ratio = sum(c["len"] for c in kept) / max(1, len(text))
        return {"ratio": round(ratio, 4), "cases": kept, "ranking": ranking, "length": len(text)}


PAGE = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kiểm tra đạo văn — PAN 2025</title>
<style>
:root{--paper:#f4f5f2;--surface:#fff;--surface-2:#eceee9;--ink:#14181c;--ink-2:#545b61;--ink-3:#7c848a;
 --line:#d7dbd4;--accent:#2b6fb0;--accent-soft:rgba(43,111,176,.10);--pl:#c05e10;--pl-soft:#ffe6cc;--good:#3b7548;--r:12px;
 --font:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:ui-monospace,"Cascadia Code","JetBrains Mono","SF Mono",Menlo,Consolas,monospace}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#0f1317;--surface:#171d22;--surface-2:#1e252b;
 --ink:#e7eae6;--ink-2:#9aa3aa;--ink-3:#6f787f;--line:#29313a;--accent:#5a9fe0;--accent-soft:rgba(90,159,224,.14);
 --pl:#e08a3c;--pl-soft:#4a3418;--good:#6db079}}
:root[data-theme="dark"]{--paper:#0f1317;--surface:#171d22;--surface-2:#1e252b;--ink:#e7eae6;--ink-2:#9aa3aa;--ink-3:#6f787f;
 --line:#29313a;--accent:#5a9fe0;--accent-soft:rgba(90,159,224,.14);--pl:#e08a3c;--pl-soft:#4a3418;--good:#6db079}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--font);line-height:1.6}
.top{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:14px;padding:14px clamp(16px,4vw,32px);
 background:color-mix(in srgb,var(--paper) 88%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.top .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}
.top h1{font-size:16px;margin:0;font-weight:640;letter-spacing:-.01em}
.toggle{margin-left:auto;border:1px solid var(--line);background:var(--surface);color:var(--ink-2);border-radius:999px;
 padding:6px 12px;font:inherit;font-size:12px;cursor:pointer}.toggle:hover{border-color:var(--accent);color:var(--accent)}
.wrap{max-width:1180px;margin:0 auto;padding:clamp(18px,3vw,32px)}
.panel{border:1px solid var(--line);border-radius:var(--r);background:var(--surface)}
.panel h2{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);
 margin:0;padding:14px 18px;border-bottom:1px solid var(--line)}
.input{padding:18px}
textarea{width:100%;min-height:180px;resize:vertical;border:1px solid var(--line);border-radius:8px;background:var(--paper);
 color:var(--ink);font:13.5px/1.6 var(--mono);padding:12px}
.row{display:flex;gap:12px;align-items:center;margin-top:12px;flex-wrap:wrap}
.btn{border:0;background:var(--accent);color:#fff;border-radius:8px;padding:10px 20px;font:inherit;font-weight:620;cursor:pointer}
.btn:hover{filter:brightness(1.06)}.btn:disabled{opacity:.5;cursor:default}
.file{font-size:13px;color:var(--ink-2)}.hint{font-size:12px;color:var(--ink-3);margin-left:auto;font-family:var(--mono)}
#results{margin-top:20px;display:none}
.hero{display:flex;align-items:center;gap:22px;border:1px solid var(--line);border-radius:var(--r);background:var(--surface);
 padding:20px 24px;margin-bottom:20px}
.gauge{position:relative;width:96px;height:96px;border-radius:50%;flex:none}
.gauge::after{content:"";position:absolute;inset:11px;border-radius:50%;background:var(--surface)}
.gauge b{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:1;font-size:24px;
 font-weight:680;font-variant-numeric:tabular-nums;color:var(--pl)}
.hero .k{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
.hero .big{font-size:26px;font-weight:660}.hero .sub{color:var(--ink-2);font-size:14px}
.sev{display:inline-block;font-family:var(--mono);font-size:11px;padding:2px 9px;border-radius:999px}
.sev.high{background:var(--pl-soft);color:var(--pl)}.sev.med{background:var(--accent-soft);color:var(--accent)}
.sev.low{background:color-mix(in srgb,var(--good) 15%,transparent);color:var(--good)}
.grid{display:grid;grid-template-columns:1fr 320px;gap:20px;align-items:start}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.doc{padding:20px 22px;white-space:pre-wrap;word-wrap:break-word;font:14px/1.75 var(--mono);max-height:66vh;overflow:auto}
mark.pl{background:var(--pl-soft);color:inherit;border-bottom:2px solid var(--pl);border-radius:2px;padding:.5px 1px;cursor:help;scroll-margin-top:70px}
mark.pl.flash{animation:fl 1.1s ease}@keyframes fl{0%,100%{background:var(--pl-soft)}30%{background:var(--pl);color:#fff}}
.aside{position:sticky;top:70px;display:flex;flex-direction:column;gap:16px}
ul{list-style:none;margin:0;padding:0}.cases li{border-bottom:1px solid var(--line)}.cases li:last-child{border:0}
.case{width:100%;text-align:left;background:none;border:0;color:inherit;font:inherit;cursor:pointer;padding:11px 16px;
 display:grid;grid-template-columns:auto 1fr;gap:2px 8px}.case:hover{background:var(--surface-2)}
.cn{grid-row:1/3;font-family:var(--mono);font-size:12px;color:var(--pl);font-weight:640;align-self:center}
.ct{font-size:13px}.cm{font-family:var(--mono);font-size:11px;color:var(--ink-3)}
.rank li{display:flex;align-items:center;gap:8px;padding:9px 16px;font-family:var(--mono);font-size:12px;color:var(--ink-2)}
.rank .bar{flex:1;height:5px;background:var(--surface-2);border-radius:3px;overflow:hidden}
.rank .bar span{display:block;height:100%;background:var(--accent)}.rank b{color:var(--ink)}
.empty{padding:16px;color:var(--ink-3);font-size:13px}.spin{color:var(--ink-3);font-size:13px;padding:8px 0}
.foot{color:var(--ink-3);font-family:var(--mono);font-size:11px;margin-top:22px}
.modal{position:fixed;inset:0;z-index:20;display:none;align-items:center;justify-content:center;
 background:rgba(0,0,0,.45);padding:18px}
.modal.open{display:flex}
.mcard{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);max-width:900px;width:100%;
 max-height:86vh;display:flex;flex-direction:column;overflow:hidden}
.mtop{display:flex;align-items:center;gap:10px;padding:14px 18px;border-bottom:1px solid var(--line)}
.mtop h3{margin:0;font-size:15px;font-weight:640}.mtop .x{margin-left:auto;border:0;background:none;color:var(--ink-2);
 font-size:22px;line-height:1;cursor:pointer;padding:0 4px}.mtop .x:hover{color:var(--pl)}
.mcols{display:grid;grid-template-columns:1fr 1fr;gap:0;overflow:auto}
@media(max-width:640px){.mcols{grid-template-columns:1fr}}
.mcol{padding:16px 18px;min-width:0}.mcol+.mcol{border-left:1px solid var(--line)}
@media(max-width:640px){.mcol+.mcol{border-left:0;border-top:1px solid var(--line)}}
.mcol .lab{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px}
.mcol.susp .lab{color:var(--pl)}.mcol.src .lab{color:var(--accent)}
.mcol .meta{font-family:var(--mono);font-size:11px;color:var(--ink-3);margin-bottom:10px}
.mcol .passage{font:13px/1.7 var(--mono);white-space:pre-wrap;word-wrap:break-word}
.mcol.susp .passage{background:var(--pl-soft);border-radius:6px;padding:10px 12px}
.mfoot{display:flex;align-items:center;gap:12px;padding:12px 18px;border-top:1px solid var(--line)}
.rwbtn{border:1px solid var(--good);background:none;color:var(--good);border-radius:8px;padding:7px 14px;
 font:inherit;font-size:13px;font-weight:560;cursor:pointer}
.rwbtn:hover{background:color-mix(in srgb,var(--good) 12%,transparent)}
.rwbtn:disabled{opacity:.55;cursor:default}
.rwnote{font-family:var(--mono);font-size:11px;color:var(--ink-3)}
.mrw{padding:0 18px 16px}
.mcol.rw{padding:0}.mcol.rw .lab{color:var(--good);margin-bottom:8px}
.mcol.rw .passage{background:color-mix(in srgb,var(--good) 9%,transparent);border-radius:6px;padding:10px 12px}
.rwbtn.cl{border-color:var(--accent);color:var(--accent)}
.rwbtn.cl:hover{background:color-mix(in srgb,var(--accent) 12%,transparent)}
.rwbtn.vf{border-color:var(--ink-2);color:var(--ink-2)}
.rwbtn.vf:hover{background:var(--surface-2)}
.mfoot{flex-wrap:wrap}
.mcol.cl .lab{color:var(--accent)} .mcol.vf .lab{color:var(--ink-2)}
.rwbtn.ex{background:var(--good);border-color:var(--good);color:#fff;font-weight:620}
.rwbtn.ex:hover{filter:brightness(1.07);background:var(--good)}
.mcol.ex .lab{color:var(--good)}
.exrow{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:2px}
.rwp{background:color-mix(in srgb,var(--good) 9%,transparent);border-radius:6px;padding:10px 12px}
.tag{display:inline-block;font-family:var(--mono);font-size:13px;font-weight:600;color:var(--accent);
 background:var(--accent-soft);border:1px solid color-mix(in srgb,var(--accent) 40%,var(--line));
 border-radius:999px;padding:4px 13px;margin-bottom:8px}
.verdict{font-size:14px;font-weight:640;margin-bottom:6px}
.verdict.yes{color:var(--good)} .verdict.no{color:var(--pl)}
.rtext{font-size:13.5px;color:var(--ink-2);line-height:1.6}
</style></head><body>
<div class="top"><div><div class="eyebrow">PAN 2025 · Text Mining</div><h1>Kiểm tra đạo văn</h1></div>
 <button class="toggle" onclick="tt()">◐ Giao diện</button></div>
<div class="wrap">
 <div class="panel"><h2>Tài liệu nghi vấn</h2><div class="input">
  <textarea id="txt" placeholder="Dán nội dung tài liệu cần kiểm tra vào đây…"></textarea>
  <div class="row">
   <button class="btn" id="go" onclick="detect()">Kiểm tra đạo văn</button>
   <label class="file">hoặc <input type="file" accept=".txt" id="file"></label>
   <span class="hint" id="hint">kho nguồn: __NSRC__ tài liệu</span>
  </div></div></div>
 <div id="results"></div>
 <div class="foot">Retrieval TF-IDF → Alignment tf-isf (seed-and-extend, PAN 2014) → PlagDet · chạy local</div>
</div>
<div class="modal" id="modal" onclick="if(event.target===this)closeM()"></div>
<script>
function tt(){var r=document.documentElement;r.setAttribute('data-theme',r.getAttribute('data-theme')==='dark'?'light':'dark')}
var esc=function(s){return s.replace(/[&<>]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;'}[c]})}
document.getElementById('file').onchange=function(e){var f=e.target.files[0];if(!f)return;
 var r=new FileReader();r.onload=function(){document.getElementById('txt').value=r.result};r.readAsText(f)}
async function detect(){
 var text=document.getElementById('txt').value;if(!text.trim())return;
 var go=document.getElementById('go');go.disabled=true;go.textContent='Đang phân tích…';
 var R=document.getElementById('results');R.style.display='block';R.innerHTML='<div class="spin">Đang chạy retrieval + alignment…</div>';
 try{
  var res=await fetch('/detect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text})});
  var d=await res.json();render(text,d);
 }catch(e){R.innerHTML='<div class="panel input">Lỗi: '+esc(String(e))+'</div>'}
 go.disabled=false;go.textContent='Kiểm tra đạo văn';
}
function render(text,d){
 var pct=Math.round(d.ratio*100);
 var sev=d.ratio>=.5?['high','CAO']:d.ratio>=.15?['med','TRUNG BÌNH']:['low','THẤP'];
 var cs=d.cases.slice().sort(function(a,b){return a.start-b.start});
 // văn bản highlight
 var h='',cur=0;
 cs.forEach(function(c,i){var s=c.start,e=c.start+c.len;
  h+=esc(text.slice(cur,s));
  h+='<mark class="pl" id="c'+i+'" title="Nguồn: '+esc(c.source)+' · ký tự '+s+'–'+e+'">'+esc(text.slice(s,e))+'</mark>';cur=e});
 h+=esc(text.slice(cur));
 // danh sách case
 var ci=cs.map(function(c,i){return '<li><button class="case" data-goto="c'+i+'">'
  +'<span class="cn">#'+(i+1)+'</span>'
  +'<span class="ct">'+esc(text.slice(c.start,c.start+70).trim())+'…</span>'
  +'<span class="cm"><code>'+esc(c.source)+'</code> · '+c.start+'–'+(c.start+c.len)+'</span></button></li>'}).join('')
  ||'<li class="empty">Không phát hiện đoạn đạo văn.</li>';
 var rk=d.ranking.map(function(r){return '<li><code>'+esc(r.source)+'</code>'
  +'<span class="bar"><span style="width:'+Math.min(100,r.score*100).toFixed(0)+'%"></span></span><b>'+r.score.toFixed(3)+'</b></li>'}).join('');
 document.getElementById('results').innerHTML=
  '<div class="hero"><div class="gauge" style="background:conic-gradient(var(--pl) '+pct+'%,var(--surface-2) 0)"><b>'+pct+'</b></div>'
  +'<div><div class="k">Điểm đạo văn</div><div class="big">'+d.ratio.toFixed(2)+' <span class="sev '+sev[0]+'">'+sev[1]+'</span></div>'
  +'<div class="sub">'+cs.length+' đoạn · '+d.ranking.length+' nguồn truy hồi · '+d.length.toLocaleString()+' ký tự</div></div></div>'
  +'<div class="grid"><div class="panel"><h2>Văn bản — đoạn cam là đạo văn (hover xem nguồn)</h2><div class="doc">'+h+'</div></div>'
  +'<div class="aside"><div class="panel cases"><h2>Đoạn phát hiện</h2><ul>'+ci+'</ul></div>'
  +'<div class="panel rank"><h2>Nguồn khả nghi</h2><ul>'+rk+'</ul></div></div></div>';
 window._C=cs;window._T=text;
 function jump(i){var m=document.getElementById('c'+i);if(m){m.scrollIntoView({behavior:'smooth',block:'start'});
  m.classList.remove('flash');void m.offsetWidth;m.classList.add('flash')}openM(i)}
 document.querySelectorAll('.case').forEach(function(b){b.onclick=function(){jump(parseInt(b.dataset.goto.slice(1)))}});
 document.querySelectorAll('.doc mark.pl').forEach(function(m,i){m.style.cursor='pointer';m.onclick=function(){openM(i)}});
}
function openM(i){var c=window._C[i],t=window._T;if(!c)return;
 var sp=t.slice(c.start,c.start+c.len);
 window._RW={susp:sp,src:c.src_text||''};
 var mo=document.getElementById('modal');
 mo.innerHTML='<div class="mcard"><div class="mtop"><h3>Đoạn #'+(i+1)+' — đối chiếu & viết lại</h3>'
  +'<button class="x" title="Đóng (Esc)" onclick="closeM()">×</button></div><div class="mcols">'
  +'<div class="mcol susp"><div class="lab">Đoạn nghi vấn</div><div class="meta">ký tự '+c.start+'–'+(c.start+c.len)+'</div><div class="passage">'+esc(sp)+'</div></div>'
  +'<div class="mcol src"><div class="lab">Nguồn — đạo từ đây</div><div class="meta"><code>'+esc(c.source)+'</code> · ký tự '+c.src_start+'–'+(c.src_start+c.src_len)+'</div><div class="passage">'+esc(c.src_text||'(không có văn bản nguồn)')+'</div></div>'
  +'</div>'
  +'<div class="mfoot">'
   +'<button class="rwbtn ex" id="exbtn" onclick="explainCase()">◆ Luận giải (RAG)</button>'
   +'<button class="rwbtn cl" id="clbtn" onclick="classifyCase()">◈ Phân loại kỹ thuật</button>'
   +'<button class="rwbtn vf" id="vfbtn" onclick="verifyCase()">✓ Xác minh (AI)</button>'
   +'<button class="rwbtn" id="rwbtn" onclick="rewriteCase()">✎ Viết lại</button>'
   +'<span class="rwnote" id="rwnote">__LLM__ · Luận giải = sinh grounded (kỹ thuật · giải thích · mức độ · viết lại) từ nguồn truy hồi.</span></div>'
  +'<div class="mrw" id="mrw"></div></div>';
 mo.classList.add('open');
}
function rewriteCase(){var d=window._RW;if(!d)return;
 var btn=document.getElementById('rwbtn'),note=document.getElementById('rwnote'),out=document.getElementById('mrw');
 btn.disabled=true;note.textContent='Đang gọi Gemini…';out.innerHTML='';
 fetch('/rewrite',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)})
  .then(function(r){return r.json()}).then(function(j){
   btn.disabled=false;
   if(j.error){note.textContent='Lỗi: '+j.error;return}
   note.textContent=j.changes||'';
   out.innerHTML='<div class="mcol rw"><div class="lab">Bản viết lại ('+esc(j.model||'AI')+')</div><div class="passage">'+esc(j.rewritten)+'</div></div>';
  }).catch(function(e){btn.disabled=false;note.textContent='Lỗi: '+e});
}
function classifyCase(){var d=window._RW;if(!d)return;
 var btn=document.getElementById('clbtn'),note=document.getElementById('rwnote'),out=document.getElementById('mrw');
 btn.disabled=true;note.textContent='Đang phân loại kỹ thuật…';out.innerHTML='';
 fetch('/classify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)})
  .then(function(r){return r.json()}).then(function(j){btn.disabled=false;
   if(j.error){note.textContent='Lỗi: '+j.error;return}
   note.textContent=j.confidence!=null?'độ tin cậy '+Math.round(j.confidence*100)+'%':'';
   out.innerHTML='<div class="mcol cl"><div class="lab">Kỹ thuật đạo văn</div>'
    +'<div class="tag">'+esc(j.technique_vi||j.technique||'—')+'</div>'
    +'<div class="rtext">'+esc(j.explanation||'')+'</div></div>';
  }).catch(function(e){btn.disabled=false;note.textContent='Lỗi: '+e});
}
function verifyCase(){var d=window._RW;if(!d)return;
 var btn=document.getElementById('vfbtn'),note=document.getElementById('rwnote'),out=document.getElementById('mrw');
 btn.disabled=true;note.textContent='Đang xác minh…';out.innerHTML='';
 fetch('/verify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)})
  .then(function(r){return r.json()}).then(function(j){btn.disabled=false;
   if(j.error){note.textContent='Lỗi: '+j.error;return}
   note.textContent=j.confidence!=null?'độ tin cậy '+Math.round(j.confidence*100)+'%':'';
   var ok=j.is_plagiarism;
   out.innerHTML='<div class="mcol vf"><div class="lab">Xác minh (khử báo giả)</div>'
    +'<div class="verdict '+(ok?'yes':'no')+'">'+(ok?'✓ Đúng là đạo văn':'✗ Chỉ trùng chủ đề — báo giả')+'</div>'
    +'<div class="rtext">'+esc(j.reason||'')+'</div></div>';
  }).catch(function(e){btn.disabled=false;note.textContent='Lỗi: '+e});
}
function explainCase(){var d=window._RW;if(!d)return;
 var btn=document.getElementById('exbtn'),note=document.getElementById('rwnote'),out=document.getElementById('mrw');
 btn.disabled=true;note.textContent='Đang sinh luận giải grounded từ nguồn truy hồi…';out.innerHTML='';
 fetch('/explain',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)})
  .then(function(r){return r.json()}).then(function(j){btn.disabled=false;
   if(j.error){note.textContent='Lỗi: '+j.error;return}
   note.textContent=j.confidence!=null?'độ tin cậy '+Math.round(j.confidence*100)+'%':'';
   var sv=(j.severity||'').toLowerCase(),svc=sv==='cao'?'high':sv==='thấp'?'low':'med';
   out.innerHTML='<div class="mcol ex"><div class="lab">Luận giải đạo văn — sinh grounded (RAG)</div>'
    +'<div class="exrow"><span class="tag">'+esc(j.technique_vi||j.technique||'—')+'</span>'
    +'<span class="sev '+svc+'">mức độ: '+esc(j.severity||'—')+'</span></div>'
    +'<div class="rtext" style="margin:8px 0 12px">'+esc(j.explanation||'')+'</div>'
    +'<div class="lab" style="color:var(--good)">Đề xuất viết lại khử đạo</div>'
    +'<div class="passage rwp">'+esc(j.suggested_rewrite||'(không có)')+'</div></div>';
  }).catch(function(e){btn.disabled=false;note.textContent='Lỗi: '+e});
}
function closeM(){document.getElementById('modal').classList.remove('open')}
document.addEventListener('keydown',function(e){if(e.key==='Escape')closeM()});
</script></body></html>"""


def make_handler(detector, page):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path not in ("/detect", "/rewrite", "/classify", "/verify", "/explain"):
                self._send(404, b"not found", "text/plain"); return
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                su, sr = body.get("susp", ""), body.get("src", "")
                if self.path == "/detect":
                    out = detector.detect(body.get("text", ""))
                elif self.path == "/rewrite":           # bước G — viết lại khử đạo văn
                    from generation.rewrite import rewrite_passage
                    out = rewrite_passage(su, sr)
                elif self.path == "/classify":          # #1 — phân loại kỹ thuật + giải thích
                    from generation.classify import classify_passage
                    out = classify_passage(su, sr)
                elif self.path == "/explain":           # bước G (RAG) — luận giải grounded
                    from generation.explain import explain_passage
                    out = explain_passage(su, sr)
                else:                                   # /verify — #3 — xác minh khử FP
                    from generation.verify import verify_pair
                    out = verify_pair(su, sr)
                self._send(200, json.dumps(out).encode("utf-8"), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode("utf-8"), "application/json")

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", default=r"C:/github/PAN2025/00_spot_check/00_spot_check/src",
                    help="thư mục kho nguồn (.txt)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--topk", type=int, default=3)   # align top-3 nguồn: recall cao hơn cho văn bản tùy ý
    ap.add_argument("--provider", default="fpt", choices=["fpt", "gemini"],
                    help="LLM cho vai trò classify/verify/rewrite (mặc định fpt = GLM)")
    ap.add_argument("--llm-model", default="GLM-5.2",
                    help="tên model LLM (fpt: GLM-5.2 tốt nhất; gemini: bỏ qua)")
    args = ap.parse_args()

    # Chốt provider/model cho tầng generation TRƯỚC khi phục vụ — verify/classify đọc
    # env LLM_PROVIDER+LLM_MODEL trong generate_json; rewrite đọc cùng env ở nhánh FPT.
    os.environ["LLM_PROVIDER"] = args.provider
    if args.provider == "fpt":
        os.environ["LLM_MODEL"] = args.llm_model
        llm_label = f"{args.llm_model} (FPT)"
        if not os.environ.get("LLM_API_KEY"):
            print("[cảnh báo] provider=fpt nhưng thiếu LLM_API_KEY trong .env — LLM sẽ lỗi.", flush=True)
    else:
        llm_label = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    print(f"[llm] provider={args.provider} · model={llm_label}", flush=True)

    det = Detector(args.sources, topk=args.topk)
    page = PAGE.replace("__NSRC__", str(len(det.ids))).replace("__LLM__", llm_label)
    srv = HTTPServer(("127.0.0.1", args.port), make_handler(det, page))
    print(f"\n  ► Mở http://localhost:{args.port}  (kho nguồn: {len(det.ids)} tài liệu)\n"
          f"    Ctrl+C để dừng.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\ndừng.")


if __name__ == "__main__":
    main()
