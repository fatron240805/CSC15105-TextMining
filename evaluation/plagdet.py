#!/usr/bin/env python3
"""PAN 2015 plagiarism-detection scoring — character-level PlagDet.

Bao trùm A0-5 (skeleton metric), A3-1 (Precision/Recall/F1 mức ký tự), và
A3-2 (Granularity + PlagDet). Cài đặt theo Potthast et al. (2010), là thước đo
chính thức của PAN 2015 mà đề bài yêu cầu.

Định nghĩa (mức ký tự, trên tài liệu nghi vấn):
  Mỗi "case" / "detection" là một đoạn = khoảng [offset, offset+length) trong
  một tài liệu susp. Chấm điểm ở cấp *bộ sưu tập*: gộp mọi đoạn của mọi tài liệu,
  nhưng phần giao chỉ tính giữa các đoạn *cùng một* tài liệu (khoá theo doc_id).

  R = tập đoạn đạo văn thật, S = tập đoạn hệ thống phát hiện.

  recall(S,R)    = (1/|R|) Σ_{r∈R} |⋃_{s∈S}(s∩r)| / |r|
  precision(S,R) = (1/|S|) Σ_{s∈S} |⋃_{r∈R}(r∩s)| / |s|
  F1             = 2·P·R / (P+R)
  granularity    = (1/|R_det|) Σ_{r∈R_det} |{s∈S : s∩r ≠ ∅}|      (R_det: r có ≥1 detection)
  PlagDet        = F1 / log2(1 + granularity)

Quy ước biên:
  |S| = 0  → precision = 1.0 (không có dương tính giả), recall = 0 → F1 = 0 → PlagDet = 0
  |R| = 0  → recall = 1.0 (rỗng đúng vô điều kiện)
  R_det rỗng → granularity = 1.0

Đoạn (Span) chỉ được coi là "plagiarism" khi chấm PlagDet — các span `altered`
(LLM tự sinh, không nguồn) KHÔNG nằm trong R của bài toán phát hiện đạo-văn-từ-nguồn.
Dùng cờ ``include_altered`` nếu muốn đưa cả altered vào (mặc định: chỉ plagiarism).

CLI:
    # Chấm dự đoán (pred CSV) so với ground-truth spans (truth CSV, từ parse_labels.py):
    python evaluation/plagdet.py --truth outputs/validation_spans.csv \
                                 --pred  outputs/val_predictions.csv

CSV schema (cả truth lẫn pred), tối thiểu các cột:
    suspicious_reference , this_offset , this_length , feature
`feature` chỉ cần ở truth (lọc plagiarism vs altered). Ở pred có thể bỏ.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


# --------------------------------------------------------------------------- #
# Mô hình đoạn
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Span:
    doc: str          # định danh tài liệu susp (khoá giao nhau)
    start: int        # offset ký tự bắt đầu
    length: int       # độ dài (ký tự)

    @property
    def end(self) -> int:
        return self.start + self.length


def _overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Số ký tự giao nhau của hai khoảng nửa mở."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _union_len_within(target: Span, others: list) -> int:
    """|⋃ (target ∩ o)| — độ dài phần của `target` bị phủ bởi bất kỳ span nào trong `others`.

    Vì mọi giao đều là khoảng con của `target`, ta gộp các khoảng con rồi tính tổng.
    """
    subs = []
    for o in others:
        s = max(target.start, o.start)
        e = min(target.end, o.end)
        if e > s:
            subs.append((s, e))
    if not subs:
        return 0
    subs.sort()
    total = 0
    cur_s, cur_e = subs[0]
    for s, e in subs[1:]:
        if s <= cur_e:                 # chồng lấn → mở rộng
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


# --------------------------------------------------------------------------- #
# Chấm điểm
# --------------------------------------------------------------------------- #
@dataclass
class PlagDetResult:
    precision: float
    recall: float
    f1: float
    granularity: float
    plagdet: float
    n_truth: int
    n_pred: int

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "granularity": round(self.granularity, 6),
            "plagdet": round(self.plagdet, 6),
            "n_truth": self.n_truth,
            "n_pred": self.n_pred,
        }


def plagdet_score(truth: Iterable, pred: Iterable) -> PlagDetResult:
    """Tính PlagDet mức bộ sưu tập từ hai tập :class:`Span`."""
    R = [s for s in truth if s.length > 0]
    S = [s for s in pred if s.length > 0]

    # Nhóm theo tài liệu để chỉ so span cùng doc
    R_by_doc = defaultdict(list)
    S_by_doc = defaultdict(list)
    for s in R:
        R_by_doc[s.doc].append(s)
    for s in S:
        S_by_doc[s.doc].append(s)

    # Recall: mỗi case thật r, tỉ lệ ký tự bị phủ bởi các detection cùng doc
    if not R:
        recall = 1.0
    else:
        acc = 0.0
        for r in R:
            covered = _union_len_within(r, S_by_doc.get(r.doc, []))
            acc += covered / r.length
        recall = acc / len(R)

    # Precision: mỗi detection s, tỉ lệ ký tự trùng với case thật cùng doc
    if not S:
        precision = 1.0
    else:
        acc = 0.0
        for s in S:
            covered = _union_len_within(s, R_by_doc.get(s.doc, []))
            acc += covered / s.length
        precision = acc / len(S)

    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    # Granularity: trung bình số detection phủ mỗi case thật (chỉ tính case có detection)
    detected_counts = []
    for r in R:
        cnt = sum(1 for s in S_by_doc.get(r.doc, [])
                  if _overlap_len(r.start, r.end, s.start, s.end) > 0)
        if cnt > 0:
            detected_counts.append(cnt)
    granularity = 1.0 if not detected_counts else sum(detected_counts) / len(detected_counts)

    plagdet = f1 / math.log2(1 + granularity) if granularity >= 1 else f1

    return PlagDetResult(precision, recall, f1, granularity, plagdet, len(R), len(S))


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_spans(csv_path: str, only_plagiarism: bool = True) -> list:
    """Đọc spans từ CSV (định dạng của parse_labels.py hoặc file dự đoán)."""
    spans = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature = row.get("feature", "plagiarism")
            if only_plagiarism and feature and feature != "plagiarism":
                continue
            try:
                doc = row["suspicious_reference"]
                start = int(row["this_offset"])
                length = int(row["this_length"])
            except (KeyError, ValueError):
                continue
            spans.append(Span(doc, start, length))
    return spans


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, help="CSV spans ground-truth")
    ap.add_argument("--pred", required=True, help="CSV spans dự đoán của hệ thống")
    ap.add_argument("--include-altered", action="store_true",
                    help="Đưa cả span altered vào R (mặc định: chỉ plagiarism)")
    args = ap.parse_args(argv)

    truth = load_spans(args.truth, only_plagiarism=not args.include_altered)
    pred = load_spans(args.pred, only_plagiarism=False)
    res = plagdet_score(truth, pred)

    print(f"Truth spans: {res.n_truth} | Pred spans: {res.n_pred}")
    print(f"  Precision   : {res.precision:.4f}")
    print(f"  Recall      : {res.recall:.4f}")
    print(f"  F1          : {res.f1:.4f}")
    print(f"  Granularity : {res.granularity:.4f}")
    print(f"  PlagDet     : {res.plagdet:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
