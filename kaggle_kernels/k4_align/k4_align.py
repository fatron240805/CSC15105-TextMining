#!/usr/bin/env python3
"""K4 — Text alignment (seed-and-extend) + tune threshold. Phase 2 (A2-1..A2-5).

Chạy trên Kaggle T4. Với mỗi susp (subset để tune), căn chỉnh với NGUỒN GOLD của nó
(tách khỏi retrieval): cắt câu -> embed -> câu susp khớp câu src (cosine>thr) = seed ->
gộp seed liên tiếp thành span. Sweep threshold+gap trong 1 lần chạy, chấm PlagDet (PAN 2015).

Self-contained (kernel standalone). Đọc pan25-corpus-val + pan25-labels.
So với: trần boundary ~0.86, baseline cả-doc ~0.67 (đo trên spot).
"""
from __future__ import annotations
import csv, glob, math, os, sys
from collections import defaultdict
import numpy as np

MODEL = os.environ.get("MODEL_NAME", "BAAI/bge-small-en-v1.5")
SUBSET = int(os.environ.get("SUBSET", "400"))       # số susp để tune
LABELS_ROOT = os.environ.get("LABELS_ROOT", "/kaggle/input")
CORPUS_ROOT = os.environ.get("CORPUS_ROOT", "/kaggle/input")
SPLIT = os.environ.get("SPLIT", "val")


def log(*a): print(*a, flush=True)

# ---- resolve mount (kaggle /input/datasets/<user>/<slug>/...) ----
def rfile(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else name

def rdocs(role):
    hits = glob.glob(f"/kaggle/input/**/docs/{role}/*.txt", recursive=True)
    return os.path.dirname(hits[0]) if hits else ""

# ---- sentence splitter (giữ offset, tile toàn text) ----
_END, _WS = ".!?", " \t\r\n"
def split_sentences(text):
    spans, start, i, n = [], 0, 0, len(text)
    while i < n:
        if text[i] in _END:
            j = i + 1
            while j < n and text[j] in _END: j += 1
            k = j
            while k < n and text[k] in _WS: k += 1
            spans.append((start, k)); start = k; i = k
        else:
            i += 1
    if start < n: spans.append((start, n))
    return spans

# ---- seed-and-extend ----
def best_sims(susp_text, src_texts, model):
    ss = split_sentences(susp_text)
    stx = [susp_text[a:b].strip() or " " for a, b in ss]
    rtx = []
    for st in src_texts:
        for a, b in split_sentences(st): rtx.append(st[a:b].strip() or " ")
    if not rtx: return ss, np.zeros(len(ss), "float32")
    S = np.asarray(model.encode(stx, batch_size=256, normalize_embeddings=True, show_progress_bar=False))
    R = np.asarray(model.encode(rtx, batch_size=256, normalize_embeddings=True, show_progress_bar=False))
    return ss, (S @ R.T).max(axis=1)

def spans_from_best(ss, best, thr, gap=1, min_chars=50):
    idx = [i for i in range(len(ss)) if best[i] > thr]
    if not idx: return []
    groups = [[idx[0]]]
    for i in idx[1:]:
        if i - groups[-1][-1] <= gap + 1: groups[-1].append(i)
        else: groups.append([i])
    out = []
    for g in groups:
        s, e = ss[g[0]][0], ss[g[-1]][1]
        if e - s >= min_chars: out.append((s, e - s))
    return out

# ---- PlagDet (PAN 2015, mức ký tự) ----
def _union_in(t0, t1, others):
    subs = [(max(t0, a), min(t1, b)) for a, b in others if min(t1, b) > max(t0, a)]
    if not subs: return 0
    subs.sort(); tot, cs, ce = 0, *subs[0]
    for s, e in subs[1:]:
        if s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    return tot + ce - cs

def plagdet(truth, pred):   # truth/pred: dict doc -> [(start,length)]
    R = [(d, s, s + l) for d, sp in truth.items() for s, l in sp if l > 0]
    S = [(d, s, s + l) for d, sp in pred.items() for s, l in sp if l > 0]
    Rby, Sby = defaultdict(list), defaultdict(list)
    for d, a, b in R: Rby[d].append((a, b))
    for d, a, b in S: Sby[d].append((a, b))
    rec = 1.0 if not R else sum(_union_in(a, b, Sby.get(d, [])) / (b - a) for d, a, b in R) / len(R)
    prec = 1.0 if not S else sum(_union_in(a, b, Rby.get(d, [])) / (b - a) for d, a, b in S) / len(S)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    cnts = []
    for d, a, b in R:
        c = sum(1 for x, y in Sby.get(d, []) if min(b, y) > max(a, x))
        if c: cnts.append(c)
    gran = 1.0 if not cnts else sum(cnts) / len(cnts)
    return prec, rec, f1, gran, f1 / math.log2(1 + gran)


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={dev} model={MODEL} subset={SUBSET}")
    model = SentenceTransformer(MODEL, device=dev)

    # gold spans + nguồn theo susp
    spans_csv = rfile(f"{SPLIT}_spans.csv")
    gold_by, srcs_by = defaultdict(list), defaultdict(set)
    for r in csv.DictReader(open(spans_csv, encoding="utf-8", newline="")):
        if r["feature"] != "plagiarism": continue
        gold_by[r["suspicious_reference"]].append((int(r["this_offset"]), int(r["this_length"])))
        if r.get("source_reference"): srcs_by[r["suspicious_reference"]].add(r["source_reference"])

    susp_dir, src_dir = rdocs("susp"), rdocs("src")
    log(f"gold {len(gold_by)} susp | susp_dir={susp_dir} | src_dir={src_dir}")

    subset = list(gold_by)[:SUBSET]
    cache, truth, wholedoc = {}, {}, {}
    for k, susp in enumerate(subset, 1):
        stext = open(os.path.join(susp_dir, susp), encoding="utf-8", newline="").read()
        srcs = [open(os.path.join(src_dir, s), encoding="utf-8", newline="").read()
                for s in srcs_by[susp] if os.path.exists(os.path.join(src_dir, s))]
        cache[susp] = (stext, best_sims(stext, srcs, model))
        truth[susp] = gold_by[susp]
        wholedoc[susp] = [(0, len(stext))]
        if k % 100 == 0: log(f"  embed {k}/{len(subset)}")

    ng = sum(len(v) for v in truth.values())
    log(f"\n{len(subset)} susp, {ng} gold spans")
    log(f"baseline cả-doc: PlagDet={plagdet(truth, wholedoc)[4]:.3f}\n")
    log(f"{'thr':>5} {'gap':>4} | {'P':>6} {'R':>6} {'F1':>6} {'gran':>6} {'PlagDet':>8}")
    log("-" * 48)
    best = (0, None)
    for gap in (0, 1, 2):
        for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            pred = {su: spans_from_best(c[1][0], c[1][1], thr, gap) for su, c in cache.items()}
            p, r, f, g, pd = plagdet(truth, pred)
            log(f"{thr:>5.2f} {gap:>4} | {p:>6.3f} {r:>6.3f} {f:>6.3f} {g:>6.3f} {pd:>8.3f}")
            if pd > best[0]: best = (pd, (thr, gap))
        log("-" * 48)
    log(f"\nTỐT NHẤT: PlagDet={best[0]:.3f} @ thr={best[1][0]} gap={best[1][1]}")
    with open("/kaggle/working/align_tune.csv", "w", encoding="utf-8", newline="") as fo:
        fo.write(f"best_plagdet,{best[0]:.4f}\nthreshold,{best[1][0]}\ngap,{best[1][1]}\nsubset,{len(subset)}\n")


if __name__ == "__main__":
    sys.exit(main())
