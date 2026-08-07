#!/usr/bin/env python3
"""Text alignment — seed-and-extend (A2-1). Nền cho Phase 2.

Hiện có: sentence splitter GIỮ OFFSET ký tự (tile toàn tài liệu — không hở, không
chồng), + kiểm chứng tái tạo. Đây là điều kiện tiên quyết vì PlagDet chấm mức ký tự:
lệch offset câu = lệch điểm mà không báo lỗi.

Seed-and-extend (sẽ thêm): mỗi câu susp khớp câu src (cosine > ngưỡng) = seed;
gộp các câu susp đạo văn liên tiếp (gap nhỏ) thành span (offset, length).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


@dataclass(frozen=True)
class Sentence:
    start: int      # offset ký tự bắt đầu (bao gồm khoảng trắng đầu nếu có)
    end: int        # offset kết thúc (sau dấu câu + whitespace theo sau)

    @property
    def length(self) -> int:
        return self.end - self.start


_END = ".!?"
_WS = " \t\r\n"


def split_sentences(text: str) -> list:
    """Cắt câu, GIỮ offset, TILE toàn text: spans[i].end == spans[i+1].start,
    spans[0].start == 0, spans[-1].end == len(text). Whitespace theo sau thuộc câu."""
    spans = []
    start = 0
    i, n = 0, len(text)
    while i < n:
        if text[i] in _END:
            j = i + 1
            while j < n and text[j] in _END:      # gộp chuỗi dấu ".!?"
                j += 1
            k = j
            while k < n and text[k] in _WS:        # nuốt whitespace theo sau
                k += 1
            spans.append(Sentence(start, k))
            start = k
            i = k
        else:
            i += 1
    if start < n:
        spans.append(Sentence(start, n))
    return spans


def verify_tiling(text: str, sents: list) -> None:
    """Assert splitter tile chính xác toàn text (offset không lệch)."""
    assert sents, "không có câu nào"
    assert sents[0].start == 0, f"start != 0: {sents[0].start}"
    assert sents[-1].end == len(text), f"end {sents[-1].end} != len {len(text)}"
    for a, b in zip(sents, sents[1:]):
        assert a.end == b.start, f"hở/chồng tại {a.end} vs {b.start}"
    # nối lại phải bằng text gốc
    rebuilt = "".join(text[s.start:s.end] for s in sents)
    assert rebuilt == text, "tái tạo KHÔNG khớp text gốc"


def snap_span_to_sentences(start: int, length: int, sents: list) -> tuple:
    """Mở span [start, start+length) ra ranh giới câu chứa nó → (snap_start, snap_len).
    Dùng để đo trần boundary: gold span snap vào câu tối đa đạt PlagDet bao nhiêu."""
    end = start + length
    lo, hi = None, None
    for s in sents:
        if s.end > start and s.start < end:        # câu giao với span
            lo = s.start if lo is None else min(lo, s.start)
            hi = s.end if hi is None else max(hi, s.end)
    if lo is None:                                  # span rỗng/ngoài text
        return start, length
    return lo, hi - lo


def sentence_best_sims(susp_text, src_texts, embed_fn):
    """Trả (susp_sentences, best_sim[]) — best_sim[i] = cosine cao nhất giữa câu susp i
    và bất kỳ câu nào của các src. PHẦN ĐẮT (embed) — chạy 1 lần/cặp, rồi sweep threshold rẻ."""
    import numpy as np
    ss = split_sentences(susp_text)
    susp_txt = [susp_text[s.start:s.end].strip() or " " for s in ss]

    src_txt = []
    for st in src_texts:
        for s in split_sentences(st):
            src_txt.append(st[s.start:s.end].strip() or " ")
    if not src_txt:
        return ss, np.zeros(len(ss), dtype="float32")

    S = np.asarray(embed_fn(susp_txt))     # (n_susp, d) đã chuẩn hoá
    R = np.asarray(embed_fn(src_txt))      # (n_src, d)
    best = (S @ R.T).max(axis=1)           # cosine cao nhất mỗi câu susp
    return ss, best


def spans_from_best(ss, best, threshold, merge_gap_sents=1, min_chars=50):
    """Seed-and-extend RẺ: câu susp có best>threshold = seed; gộp seed cách nhau
    <= merge_gap_sents câu → span (offset, length); bỏ span < min_chars."""
    idx = [i for i in range(len(ss)) if best[i] > threshold]
    if not idx:
        return []
    groups = [[idx[0]]]
    for i in idx[1:]:
        if i - groups[-1][-1] <= merge_gap_sents + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    spans = []
    for g in groups:
        start, end = ss[g[0]].start, ss[g[-1]].end
        if end - start >= min_chars:
            spans.append((start, end - start))
    return spans


if __name__ == "__main__":
    # self-test nhanh
    t = "Hello world. This is a test!  Another one?\nLast line without end"
    s = split_sentences(t)
    verify_tiling(t, s)
    print(f"OK — {len(s)} câu, tile khớp. spans:", [(x.start, x.end) for x in s])
