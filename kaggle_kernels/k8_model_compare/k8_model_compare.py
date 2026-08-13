#!/usr/bin/env python3
"""K8 — So sánh 6 model FPT cho verifier #3 (khử FP) trên val, chạy trên Kaggle.

Key FPT đọc từ Kaggle Secrets: LLM_API_KEY, LLM_API_BASE (KHÔNG nhúng trong script).
Bật Internet + gắn 2 secret trước khi chạy. Xuất /kaggle/working/model_compare.json.
"""
import subprocess, sys, importlib, os
def _ensure(p, imp=None):
    try: importlib.import_module(imp or p)
    except ImportError: subprocess.run([sys.executable,"-m","pip","install","-q",p])
_ensure("openai")

def _secret(name, default=None):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ.get(name, default)

LLM_API_KEY = _secret("LLM_API_KEY")
LLM_API_BASE = _secret("LLM_API_BASE", "https://mkp-api.fptcloud.com")
assert LLM_API_KEY, "Thiếu secret LLM_API_KEY — gắn trong Add-ons > Secrets."

N_SPANS = 1000
INNER = 2                # span đồng thời / model
MAX_CONCURRENT = 12      # trần call đồng thời toàn cục
MODELS = ["Llama-3.3-70B-Instruct","DeepSeek-V4-Flash","Qwen3.6-27B",
          "gpt-oss-20b","gpt-oss-120b","GLM-5.2"]

# ===== splitter lexical (giữ offset) cho tf-isf =====
from dataclasses import dataclass as _dc
@_dc(frozen=True)
class Sentence:
    start: int; end: int
    @property
    def length(self): return self.end - self.start
_END, _WS = ".!?", " \t\r\n"
def _lex_split(text):
    spans=[]; start=0; i=0; n=len(text)
    while i<n:
        if text[i] in _END:
            j=i+1
            while j<n and text[j] in _END: j+=1
            k=j
            while k<n and text[k] in _WS: k+=1
            spans.append(Sentence(start,k)); start=k; i=k
        else: i+=1
    if start<n: spans.append(Sentence(start,n))
    return spans

# ===== align_tfisf =====
#!/usr/bin/env python3
"""Aligner v2 — seed-and-extend theo Sánchez-Pérez (PAN 2014, thắng giải, PlagDet 0.878).

Khác v1 (neural, thất bại): dùng **tf-isf lexical** + seed = (cos≥th1 AND Dice≥th2),
extension **bilateral** (đoạn susp↔src cùng liền mạch — ràng buộc diagonal), filter
min-length + overlap. Chạy CPU nhanh, không cần GPU/model.

Ref: Sánchez-Pérez et al., "A Winning Approach to Text Alignment...", CLEF 2014.
"""

import math
import re
from collections import Counter

import numpy as np



_TOK = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list:
    return _TOK.findall(s.lower())


def _sent_units(text: str, min_words: int = 3):
    """Cắt câu (giữ offset) rồi gộp câu <min_words từ với câu sau (như paper)."""
    raw = _lex_split(text)                    # [Sentence(start,end)]
    units = []                                     # [(start, end, tokens)]
    i = 0
    while i < len(raw):
        s, e = raw[i].start, raw[i].end
        toks = _tokens(text[s:e])
        while len(toks) < min_words and i + 1 < len(raw):   # gộp câu ngắn
            i += 1
            e = raw[i].end
            toks = _tokens(text[s:e])
        units.append((s, e, toks))
        i += 1
    return units


def _tfisf_matrices(susp_units, src_units):
    """Trả (cos, dice) ma trận (n_susp × n_src). isf tính trên MỌI câu của cả 2 doc."""
    all_toks = [u[2] for u in susp_units] + [u[2] for u in src_units]
    N = len(all_toks)
    df = Counter()
    for toks in all_toks:
        for t in set(toks):
            df[t] += 1
    vocab = {t: k for k, t in enumerate(df)}
    isf = np.array([math.log(N / df[t]) for t in vocab], dtype="float32")

    def build(units):
        W = np.zeros((len(units), len(vocab)), dtype="float32")   # tf-isf
        B = np.zeros((len(units), len(vocab)), dtype="float32")   # nhị phân (cho Dice)
        for r, (_, _, toks) in enumerate(units):
            for t, c in Counter(toks).items():
                j = vocab[t]
                W[r, j] = c * isf[j]
                B[r, j] = 1.0
        return W, B

    Sw, Sb = build(susp_units)
    Rw, Rb = build(src_units)
    Sn = Sw / (np.linalg.norm(Sw, axis=1, keepdims=True) + 1e-9)
    Rn = Rw / (np.linalg.norm(Rw, axis=1, keepdims=True) + 1e-9)
    cos = Sn @ Rn.T
    inter = Sb @ Rb.T                                  # số term chung
    dice = 2 * inter / (Sb.sum(1, keepdims=True) + Rb.sum(1).reshape(1, -1) + 1e-9)
    return cos, dice, (Sw, Rw)


def _cluster(seeds, max_gap):
    """Bilateral clustering: gộp seed (i,j) thành case khi liền mạch ở CẢ i lẫn j.
    2 tầng: cụm theo i (gap<=max_gap), trong mỗi cụm i lại cụm theo j."""
    if not seeds:
        return []
    seeds = sorted(seeds)                               # theo i rồi j
    cases = []
    # tầng 1: run theo i
    i_runs, cur = [], [seeds[0]]
    for s in seeds[1:]:
        if s[0] - cur[-1][0] <= max_gap + 1:
            cur.append(s)
        else:
            i_runs.append(cur); cur = [s]
    i_runs.append(cur)
    # tầng 2: trong mỗi i-run, run theo j
    for run in i_runs:
        by_j = sorted(run, key=lambda x: x[1])
        jr, cj = [], [by_j[0]]
        for s in by_j[1:]:
            if s[1] - cj[-1][1] <= max_gap + 1:
                cj.append(s)
            else:
                jr.append(cj); cj = [s]
        jr.append(cj)
        for sub in jr:
            iis = [x[0] for x in sub]; jjs = [x[1] for x in sub]
            cases.append((min(iis), max(iis), min(jjs), max(jjs)))
    return cases


def align_pair(susp_text: str, src_text: str,
               th1: float = 0.33, th2: float = 0.33, th3: float = 0.34,
               max_gap: int = 4, min_plag_chars: int = 150) -> list:
    """Trả list (susp_offset, susp_length) các đoạn đạo văn dự đoán."""
    su = _sent_units(susp_text)
    ru = _sent_units(src_text)
    if not su or not ru:
        return []
    cos, dice, (Sw, Rw) = _tfisf_matrices(su, ru)
    seed = (cos >= th1) & (dice >= th2)
    seeds = list(zip(*np.where(seed)))
    cases = _cluster(seeds, max_gap)

    out = []
    for i0, i1, j0, j1 in cases:
        # similarity đoạn = cos(tổng vector câu susp, tổng vector câu src)  [Eq. extension]
        fs = Sw[i0:i1 + 1].sum(0); fr = Rw[j0:j1 + 1].sum(0)
        sim = float(fs @ fr / ((np.linalg.norm(fs) + 1e-9) * (np.linalg.norm(fr) + 1e-9)))
        if sim < th3:
            continue
        start, end = su[i0][0], su[i1][1]              # span susp (ký tự)
        ss, se = ru[j0][0], ru[j1][1]                  # span NGUỒN (ký tự) — để hiện đối chiếu
        if end - start >= min_plag_chars:
            out.append((start, end - start, ss, se - ss, sim))

    # filter overlap: sắp theo susp start, giữ non-overlap ưu tiên sim cao
    out.sort(key=lambda x: (-x[4]))                    # sim giảm dần
    kept, occupied = [], []
    for s, l, ss, sl, sim in out:
        e = s + l
        if any(not (e <= os or s >= oe) for os, oe in occupied):   # chồng lấn susp
            continue
        kept.append((s, l, ss, sl)); occupied.append((s, e))
    kept.sort()
    return kept                                        # [(susp_start, susp_len, src_start, src_len)]

# ===== PlagDet =====
import math as _math
from collections import defaultdict as _dd
class Span:
    __slots__=("doc","start","length")
    def __init__(s,doc,start,length): s.doc,s.start,s.length=doc,start,length
    @property
    def end(s): return s.start+s.length
def _ov(a0,a1,b0,b1): return max(0,min(a1,b1)-max(a0,b0))
def _uw(t,others):
    subs=[]
    for o in others:
        a=max(t.start,o.start); b=min(t.end,o.end)
        if b>a: subs.append((a,b))
    if not subs: return 0
    subs.sort(); tot=0; cs,ce=subs[0]
    for a,b in subs[1:]:
        if a<=ce: ce=max(ce,b)
        else: tot+=ce-cs; cs,ce=a,b
    return tot+ce-cs
def plagdet_score(truth,pred):
    R=[s for s in truth if s.length>0]; S=[s for s in pred if s.length>0]
    Rb,Sb=_dd(list),_dd(list)
    for s in R: Rb[s.doc].append(s)
    for s in S: Sb[s.doc].append(s)
    rec=1.0 if not R else sum(_uw(r,Sb.get(r.doc,[]))/r.length for r in R)/len(R)
    pre=1.0 if not S else sum(_uw(s,Rb.get(s.doc,[]))/s.length for s in S)/len(S)
    f1=0.0 if (pre+rec)==0 else 2*pre*rec/(pre+rec)
    cnt=[]
    for r in R:
        c=sum(1 for s in Sb.get(r.doc,[]) if _ov(r.start,r.end,s.start,s.end)>0)
        if c>0: cnt.append(c)
    g=1.0 if not cnt else sum(cnt)/len(cnt)
    return dict(precision=pre,recall=rec,f1=f1,granularity=g,
                plagdet=(f1/_math.log2(1+g) if g>=1 else f1))

# ===== FPT verifier (#3) =====
import re, json as _json
from openai import OpenAI
_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
_VSYS=("You are a strict plagiarism verifier. Given a SUSPICIOUS passage and a SOURCE passage "
"the detector matched it to, decide whether the suspicious passage is GENUINELY derived from that "
"source (copied, paraphrased, or reworded from it) — as opposed to merely sharing the same topic or "
"common domain phrasing by coincidence. Reused specific wording, structure, or a distinctive chain of "
"claims = plagiarism. Only generic overlap two independent authors would both write = NOT plagiarism. "
'Return strict JSON: {"is_plagiarism": true|false, "confidence": <0..1>, "reason":"..."}.')
def _strip(s):
    s=s.strip()
    if s.startswith("```"):
        s=s.split("\n",1)[-1] if "\n" in s else s
        s=s.rsplit("```",1)[0]
    return s.strip()
def verify(model, susp, src, retries=4):
    msgs=[{"role":"system","content":_VSYS},
          {"role":"user","content":f'SOURCE:\n\"\"\"\n{src}\n\"\"\"\n\nSUSPICIOUS:\n\"\"\"\n{susp}\n\"\"\"\nReturn ONLY JSON.'}]
    import time
    last=None
    for a in range(retries):
        try:
            r=_client.chat.completions.create(model=model,messages=msgs,temperature=0,max_tokens=4096)
            m=r.choices[0].message
            raw=_strip(m.content or getattr(m,"reasoning_content",None) or "")
            try: d=_json.loads(raw)
            except Exception:
                mm=re.search(r"\{.*\}",raw,re.S); d=_json.loads(mm.group(0)) if mm else {}
            v=d.get("is_plagiarism")
            return (bool(v) if isinstance(v,bool) else str(v).strip().lower() in ("true","1","yes")), d.get("confidence"), False
        except Exception as e:
            last=e; time.sleep(5*(a+1))
    return True, None, True   # lỗi -> giữ

# ===== DRIVER =====
import glob, csv, time, json, threading, statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

def _find(*names, root="/kaggle/input"):
    for n in names:
        h=glob.glob(os.path.join(root,"**",n),recursive=True)
        if h: return sorted(h,key=len)[0]
    return None

susp_dir=_find("susp"); src_dir=_find("src"); labels=_find("val_spans.csv")
print("susp:",susp_dir,"src:",src_dir,"labels:",labels,flush=True)

TH,TH3=0.30,0.50
def rd(p): return open(p,encoding="utf-8",errors="ignore").read()
def overlaps(o,l,golds):
    e=o+l
    return any(not (e<=go or o>=go+gl) for go,gl in golds)

gold_by,src_of={},{}
with open(labels,encoding="utf-8",newline="") as f:
    for r in csv.DictReader(f):
        if r["feature"]=="plagiarism" and r["source_reference"]:
            gold_by.setdefault(r["suspicious_reference"],[]).append((int(r["this_offset"]),int(r["this_length"])))
            src_of.setdefault(r["suspicious_reference"],r["source_reference"])

SPANS,gold_spans=[],[]
for su,golds in gold_by.items():
    if len(SPANS)>=N_SPANS: break
    sp=os.path.join(susp_dir,su); rp=os.path.join(src_dir,src_of[su])
    if not (os.path.exists(sp) and os.path.exists(rp)): continue
    st,rt=rd(sp),rd(rp)
    pred=align_pair(st,rt,TH,TH,TH3,4)
    if not pred: continue
    for go,gl in golds: gold_spans.append(Span(su,go,gl))
    for s,l,ss,sl in pred:
        if len(SPANS)>=N_SPANS: break
        SPANS.append({"doc":su,"s":s,"l":l,"susp":st[s:s+l],"src":rt[ss:ss+sl],"tp":overlaps(s,l,golds)})

n_tp=sum(x["tp"] for x in SPANS); n_fp=len(SPANS)-n_tp
pdb=plagdet_score(gold_spans,[Span(x["doc"],x["s"],x["l"]) for x in SPANS])
print(f"{len(SPANS)} span ({n_tp} TP, {n_fp} FP) · PlagDet nền {pdb['plagdet']:.3f}",flush=True)

gate=threading.Semaphore(MAX_CONCURRENT); done={}
def vone(model,x):
    with gate:
        t=time.time()
        keep,conf,fail=verify(model,x["susp"],x["src"])
        return keep,conf,time.time()-t,fail
def run_model(model):
    t0=time.time()
    with ThreadPoolExecutor(max_workers=INNER) as ex:
        out=list(ex.map(lambda x: vone(model,x), SPANS))
    keeps=[o[0] for o in out]; confs=[o[1] for o in out if isinstance(o[1],(int,float))]
    lats=[o[2] for o in out]; fails=sum(o[3] for o in out)
    keep_tp=sum(1 for x,k in zip(SPANS,keeps) if x["tp"] and k)
    keep_fp=sum(1 for x,k in zip(SPANS,keeps) if not x["tp"] and k)
    a=plagdet_score(gold_spans,[Span(x["doc"],x["s"],x["l"]) for x,k in zip(SPANS,keeps) if k])
    rec={"model":model,"plagdet_before":round(pdb["plagdet"],3),"plagdet_after":round(a["plagdet"],3),
         "delta":round(a["plagdet"]-pdb["plagdet"],3),
         "fp_reduction":round((n_fp-keep_fp)/n_fp,3) if n_fp else None,
         "tp_retention":round(keep_tp/n_tp,3) if n_tp else None,
         "prec_after":round(a["precision"],3),
         "avg_conf":round(statistics.mean(confs),2) if confs else None,
         "avg_lat_s":round(statistics.mean(lats),1),"errors":fails}
    done[model]=rec
    print(f"  ✓ {model:24} {pdb['plagdet']:.3f}->{a['plagdet']:.3f} (Δ{rec['delta']:+.3f}) "
          f"fp_red {rec['fp_reduction']} tp_ret {rec['tp_retention']} [{time.time()-t0:.0f}s] "
          f"({len(done)}/{len(MODELS)})",flush=True)
    return rec

t0=time.time()
with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
    res=list(ex.map(run_model,MODELS))
res.sort(key=lambda r:-r["plagdet_after"])
summary={"n_spans":len(SPANS),"n_TP":n_tp,"n_FP":n_fp,"plagdet_before":round(pdb["plagdet"],3),
         "results":res,"runtime_sec":round(time.time()-t0,1),"timestamp":datetime.now().isoformat(timespec="seconds")}
json.dump(summary,open("/kaggle/working/model_compare.json","w"),ensure_ascii=False,indent=2)
print(f"\n===== {len(MODELS)} MODEL · {len(SPANS)} span ({n_tp} TP/{n_fp} FP) · nền {pdb['plagdet']:.3f} · {summary['runtime_sec']:.0f}s =====")
print(f"{'model':24}{'PlagDet':>9}{'Δ':>8}{'fp_red':>8}{'tp_ret':>8}{'prec':>7}{'lat':>7}")
for r in res:
    print(f"{r['model']:24}{r['plagdet_after']:>9.3f}{r['delta']:>+8.3f}{str(r['fp_reduction']):>8}{str(r['tp_retention']):>8}{r['prec_after']:>7.3f}{r['avg_lat_s']:>7}")
print("-> /kaggle/working/model_compare.json")
