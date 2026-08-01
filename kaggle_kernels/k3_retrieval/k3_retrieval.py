#!/usr/bin/env python3
"""K3 — Truy hồi nguồn: FAISS index + Recall@k. Task A1-2, A1-3.

Đọc embeddings (từ K2 -> dataset pan25-embeddings), nhãn gold (pan25-labels),
và text thô (pan25-corpus, cho baseline TF-IDF). So sánh:
  (a) Embedding-based retrieval  — FAISS IndexFlatIP (exact, cosine)
  (b) TF-IDF baseline            — cosine trên vector TF-IDF

Gold: mỗi tài liệu susp -> tập source_reference thật (từ *_spans.csv). Đo:
  Recall@k (k∈{1,5,10,15}), MRR. Ghi retrieval_eval.csv.

FAISS IndexFlatIP là exact — với ~60k doc là đủ, KHÔNG cần IVF/HNSW (tránh mất recall).

Cấu hình qua env (mặc định = Kaggle). Smoke-test local:
  SPLIT=spot EMB_ROOT=outputs/smoke LABELS_ROOT=kaggle_datasets/pan25-labels \
  CORPUS_ROOT=kaggle_datasets/pan25-corpus OUT=outputs/smoke/retrieval_eval.csv \
  python k3_retrieval.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:                                    # console Windows cp1252 -> ép UTF-8 (vô hại trên Kaggle)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SPLIT        = os.environ.get("SPLIT", "val")    # Kaggle: sửa default (CLI không truyền env)
EMB_ROOT     = os.environ.get("EMB_ROOT", "/kaggle/input/pan25-embeddings")
LABELS_ROOT  = os.environ.get("LABELS_ROOT", "/kaggle/input/pan25-labels")
CORPUS_ROOT  = os.environ.get("CORPUS_ROOT", "/kaggle/input/pan25-corpus")
OUT          = os.environ.get("OUT", "/kaggle/working/retrieval_eval.csv")
KS           = [1, 5, 10, 15]
TFIDF_MAX_CHARS = 20000
TFIDF_MAX_FEATURES = 100000


def log(*a):
    print(*a, flush=True)


def resolve_file(default_path: str) -> str:
    """Tìm file — bền với mount /kaggle/input/<slug> hoặc /kaggle/input/datasets/<user>/<slug>."""
    if os.path.exists(default_path):
        return default_path
    hits = glob.glob(f"/kaggle/input/**/{os.path.basename(default_path)}", recursive=True)
    return hits[0] if hits else default_path


def resolve_docs(role: str) -> str:
    default = os.path.join(CORPUS_ROOT, "docs", role)
    if glob.glob(os.path.join(default, "*.txt")):
        return default
    hits = glob.glob(f"/kaggle/input/**/docs/{role}/*.txt", recursive=True)
    return os.path.dirname(hits[0]) if hits else default


# --------------------------------------------------------------------------- #
def load_gold(split: str) -> dict:
    """susp filename (.txt) -> set(source .txt) từ *_spans.csv (chỉ plagiarism có nguồn)."""
    path = resolve_file(os.path.join(LABELS_ROOT, f"{split}_spans.csv"))
    gold = defaultdict(set)
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("feature") != "plagiarism":
                continue
            src = row.get("source_reference") or ""
            if src:
                gold[row["suspicious_reference"]].add(src)
    return gold


def recall_at_k(ranked: dict, gold: dict, ks) -> dict:
    """ranked: susp -> list src theo thứ hạng. Trả Recall@k + MRR (macro theo susp)."""
    res = {f"recall@{k}": 0.0 for k in ks}
    mrr, n = 0.0, 0
    for susp, gold_srcs in gold.items():
        if susp not in ranked or not gold_srcs:
            continue
        n += 1
        r = ranked[susp]
        for k in ks:
            res[f"recall@{k}"] += len(gold_srcs & set(r[:k])) / len(gold_srcs)
        for rank, s in enumerate(r, 1):
            if s in gold_srcs:
                mrr += 1.0 / rank
                break
    if n:
        for k in ks:
            res[f"recall@{k}"] /= n
        mrr /= n
    res["mrr"] = mrr
    res["n_susp"] = n
    return res


# --------------------------------------------------------------------------- #
def embedding_retrieval(split: str) -> dict:
    # Exact top-k bằng numpy matmul (vector đã chuẩn hoá -> inner-product = cosine).
    # Tương đương faiss.IndexFlatIP nhưng không cần faiss (image Kaggle thiếu). Theo batch
    # để chặn RAM: batch 1024 susp × 60k src = ~245MB, không dựng ma trận 60k×60k.
    src_emb = np.load(resolve_file(os.path.join(EMB_ROOT, f"{split}_src_emb.npy"))).astype("float32")
    src_ids = json.load(open(resolve_file(os.path.join(EMB_ROOT, f"{split}_src_ids.json"))))
    susp_emb = np.load(resolve_file(os.path.join(EMB_ROOT, f"{split}_susp_emb.npy"))).astype("float32")
    susp_ids = json.load(open(resolve_file(os.path.join(EMB_ROOT, f"{split}_susp_ids.json"))))
    log(f"[emb] src {src_emb.shape} susp {susp_emb.shape}")

    k = min(max(KS), src_emb.shape[0])            # guard: spot set chỉ 50 doc
    src_t = src_emb.T
    ranked, B = {}, 1024
    for start in range(0, susp_emb.shape[0], B):
        sims = susp_emb[start:start + B] @ src_t
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for r, gi in enumerate(part):
            order = gi[np.argsort(-sims[r, gi])]
            ranked[susp_ids[start + r]] = [src_ids[j] for j in order]
    return ranked


def tfidf_retrieval(split: str) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def load_dir(role):
        d = resolve_docs(role)                          # tự dò mount
        files = sorted(glob.glob(os.path.join(d, "*.txt")))
        texts, ids = [], []
        for fp in files:
            with open(fp, encoding="utf-8", newline="") as f:
                texts.append(f.read()[:TFIDF_MAX_CHARS])
            ids.append(os.path.basename(fp))
        return texts, ids

    src_texts, src_ids = load_dir("src")
    susp_texts, susp_ids = load_dir("susp")
    log(f"[tfidf] src {len(src_ids)} susp {len(susp_ids)}")

    vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, sublinear_tf=True,
                          stop_words="english")
    src_m = vec.fit_transform(src_texts)
    susp_m = vec.transform(susp_texts)

    ranked, B = {}, 512
    k = min(max(KS), src_m.shape[0])              # guard topk > n_src (spot set)
    for start in range(0, susp_m.shape[0], B):    # theo batch để chặn RAM
        sims = cosine_similarity(susp_m[start:start + B], src_m)
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]   # kth=k-1 nhanh hơn range(topk)
        for r, gi in enumerate(part):
            order = gi[np.argsort(-sims[r, gi])]
            ranked[susp_ids[start + r]] = [src_ids[j] for j in order]
    return ranked


# --------------------------------------------------------------------------- #
def main() -> int:
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    gold = load_gold(SPLIT)
    log(f"gold: {len(gold)} susp có nguồn")

    rows = []
    for name, fn in [("embedding", embedding_retrieval), ("tfidf", tfidf_retrieval)]:
        try:
            ranked = fn(SPLIT)
            metrics = recall_at_k(ranked, gold, KS)
            dropped = len(gold) - metrics["n_susp"]
            if dropped:
                log(f"[{name}] CẢNH BÁO: {dropped} susp gold không có trong ranked (bỏ khỏi mẫu số)")
            log(f"[{name}] {metrics}")
            for k, v in metrics.items():
                rows.append({"method": name, "metric": k, "value": round(v, 6)})
        except Exception as e:
            log(f"[{name}] LỖI: {e}")

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "metric", "value"])
        w.writeheader()
        w.writerows(rows)
    log(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
