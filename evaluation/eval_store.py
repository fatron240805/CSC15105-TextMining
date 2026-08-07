#!/usr/bin/env python3
"""Kho kết quả đánh giá — mỗi lần chạy 1 file JSON trong evaluation/results/.

Mục tiêu: SO SÁNH được nhiều phương pháp về sau. Chốt schema ở đây để mọi runner
ghi cùng một định dạng; leaderboard đọc lại toàn bộ folder.

  eval_set_id: sha1 của danh sách susp đã chấm -> 2 kết quả chỉ so sánh trực tiếp
               được khi cùng eval_set_id (cùng split + cùng subset). Khác id -> gắn cờ.
"""
from __future__ import annotations
import glob
import hashlib
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def eval_set_id(susp_names) -> str:
    """Định danh tập đánh giá = sha1(danh sách susp đã sắp xếp)[:12]."""
    h = hashlib.sha1("\n".join(sorted(susp_names)).encode("utf-8")).hexdigest()
    return h[:12]


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-").lower()


def save_result(method: str, metrics: dict, *, kind: str = "method",
                split: str = "", subset=None, topk=None, params: dict = None,
                eval_set: str = "", runtime_sec=None, notes: str = "",
                results_dir: str = RESULTS_DIR) -> str:
    """Ghi 1 kết quả (JSON có timestamp). kind: method | baseline | ceiling."""
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now()
    rec = {
        "method": method,
        "kind": kind,
        "split": split,
        "subset": subset,
        "topk": topk,
        "eval_set_id": eval_set or None,
        "params": params or {},
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        "runtime_sec": round(runtime_sec, 1) if isinstance(runtime_sec, (int, float)) else None,
        "timestamp": ts.isoformat(timespec="seconds"),
        "notes": notes,
    }
    fname = f"{ts.strftime('%Y%m%d-%H%M%S')}__{_slug(method)}__{split or 'na'}.json"
    path = os.path.join(results_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return path


def load_results(results_dir: str = RESULTS_DIR) -> list:
    """Đọc mọi *.json (chỉ *.json — tránh desktop.ini của OneDrive)."""
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            rec["_file"] = os.path.basename(p)
            out.append(rec)
        except (json.JSONDecodeError, OSError):
            continue
    return out
