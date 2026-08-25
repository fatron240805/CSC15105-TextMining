#!/usr/bin/env python3
"""Aligner v2 — seed-and-extend theo Sánchez-Pérez (PAN 2014, thắng giải, PlagDet 0.878).

Khác v1 (neural, thất bại): dùng **tf-isf lexical** + seed = (cos≥th1 AND Dice≥th2),
extension **bilateral** (đoạn susp↔src cùng liền mạch — ràng buộc diagonal), filter
min-length + overlap. Chạy CPU nhanh, không cần GPU/model.

Ref: Sánchez-Pérez et al., "A Winning Approach to Text Alignment...", CLEF 2014.
"""
from __future__ import annotations
import math
import re
from collections import Counter

import numpy as np

from scripts.alignment.align_core import split_sentences   # offset-preserving, tile toàn doc

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list:
    return _TOK.findall(s.lower())


def _sent_units(text: str, min_words: int = 3):
    """Cắt câu (giữ offset) rồi gộp câu <min_words từ với câu sau (như paper)."""
    raw = split_sentences(text)                    # [Sentence(start,end)]
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
               max_gap: int = 4, min_plag_chars: int = 150, *,
               return_sim: bool = False) -> list:
    """Trả list (susp_offset, susp_length, src_offset, src_length) các đoạn đạo văn dự đoán.

    return_sim=True: thêm phần tử thứ 5 = sim đoạn (điểm alignment đã dùng để lọc th3).
    Dùng để phân biệt case alignment tự tin (sim >> th3) khỏi case biên (sim ~ th3) —
    caller downstream (vd. verifier LLM) chỉ nên xét lại case biên, tránh làm lại việc
    alignment đã quyết định xong."""
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
        kept.append((s, l, ss, sl, sim) if return_sim else (s, l, ss, sl))
        occupied.append((s, e))
    kept.sort()
    return kept                                        # [(susp_start, susp_len, src_start, src_len[, sim])]
