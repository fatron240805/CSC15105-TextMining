#!/usr/bin/env python3
"""K2 — Embed toàn bộ corpus (src + susp) trên Kaggle GPU. Task A1-1.

Chạy như 1 Kaggle kernel (enable_gpu + enable_internet). Đọc corpus read-only từ
dataset `pan25-corpus`, ghi embeddings ra /kaggle/working (persist, <=20GB).

THIẾT KẾ ĐỂ SỐNG SÓT TIMEOUT 12H:
  - Xử lý theo shard SHARD_SIZE doc; mỗi shard ghi ngay emb_shard_XXXX.npy + ids.
  - Khi rerun, shard đã có file thì BỎ QUA -> resume không mất công.
  - Cuối cùng gộp shard thành {split}_{role}_emb.npy + {split}_{role}_ids.json.

Embedding mức TÀI LIỆU cho retrieval (K3): mean-pool embedding của N chunk đầu.
(Passage-level cho alignment do K4 làm riêng — không embed toàn bộ ở đây để tiết kiệm quota.)

Cấu hình qua BIẾN MÔI TRƯỜNG (mặc định = Kaggle). Cho phép smoke-test local:
  SPLIT=spot MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 \
  CORPUS_ROOT=kaggle_datasets/pan25-corpus OUT_ROOT=outputs/smoke \
  SHARD_SIZE=25 LIMIT_DOCS=50 python k2_embed.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

try:                                    # console Windows cp1252 -> ép UTF-8 (vô hại trên Kaggle)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------------------- #
# CONFIG qua env (mặc định = môi trường Kaggle)
# --------------------------------------------------------------------------- #
MODEL_NAME   = os.environ.get("MODEL_NAME", "BAAI/bge-small-en-v1.5")  # 130MB, tải ổn định; PAN tiếng Anh. (BGE-M3 2GB hay bị HF rate-limit)
SPLIT        = os.environ.get("SPLIT", "val")                  # "spot"|"val"|"train" (Kaggle: sửa default vì CLI không truyền env)
CORPUS_ROOT  = os.environ.get("CORPUS_ROOT", "/kaggle/input/pan25-corpus")
OUT_ROOT     = os.environ.get("OUT_ROOT", "/kaggle/working")
ROLES        = tuple(os.environ.get("ROLES", "src,susp").split(","))
SHARD_SIZE   = int(os.environ.get("SHARD_SIZE", "2000"))      # doc / shard
CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", "400"))     # ~token / chunk
MAX_CHUNKS   = int(os.environ.get("MAX_CHUNKS", "8"))         # chunk đầu / doc
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "64"))
LIMIT_DOCS   = int(os.environ.get("LIMIT_DOCS", "0"))         # 0 = tất cả; >0 = smoke
MAX_CHARS    = CHUNK_TOKENS * MAX_CHUNKS * 5                  # cắt trước khi tokenize (tối ưu ~30%)

os.makedirs(OUT_ROOT, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def read_text(path: str) -> str:
    # Chỉ đọc phần đầu cần thiết — doc PAN dài ~17k token, ta chỉ dùng ~3.2k đầu.
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read(MAX_CHARS)


def chunk_text(text: str, tok, max_chunks: int, chunk_tokens: int) -> list:
    """Cắt text thành tối đa max_chunks đoạn theo KÝ TỰ (~chunk_tokens*4 char/đoạn).

    KHÔNG tokenize ở đây: tok.encode+decode toàn bộ doc (sentencepiece) rất chậm trên CPU
    (~45 phút cho val, ~6 giờ cho train). Model.encode tự tokenize 1 lần trên đường tối ưu.
    Ranh giới ký tự xấp xỉ ranh giới token — thừa đủ cho embedding doc mean-pool.
    """
    approx = chunk_tokens * 4                      # ~4 char/token
    end = min(len(text), approx * max_chunks)
    chunks = [text[i:i + approx] for i in range(0, end, approx)]
    return chunks or [""]


def resolve_docs(role: str) -> str:
    """Tìm dir docs/<role> chứa .txt. Bền với 2 kiểu mount Kaggle
    (/kaggle/input/<slug> hoặc /kaggle/input/datasets/<user>/<slug>) và local."""
    default = os.path.join(CORPUS_ROOT, "docs", role)
    if glob.glob(os.path.join(default, "*.txt")):
        return default
    hits = glob.glob(f"/kaggle/input/**/docs/{role}/*.txt", recursive=True)
    return os.path.dirname(hits[0]) if hits else default


def embed_split_role(model, tok, split: str, role: str) -> None:
    docs_dir = resolve_docs(role)
    files = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    if LIMIT_DOCS:
        files = files[:LIMIT_DOCS]
    log(f"[{split}/{role}] {len(files)} doc @ {docs_dir}")
    if not files:
        log(f"  !! không thấy .txt nào cho role={role} dưới /kaggle/input — kiểm tra dataset")
        return

    n_shards = (len(files) + SHARD_SIZE - 1) // SHARD_SIZE
    for si in range(n_shards):
        shard_emb = os.path.join(OUT_ROOT, f"{split}_{role}_emb_shard_{si:04d}.npy")
        shard_ids = os.path.join(OUT_ROOT, f"{split}_{role}_ids_shard_{si:04d}.json")
        if os.path.exists(shard_emb) and os.path.exists(shard_ids):
            log(f"  shard {si}/{n_shards} đã có -> skip (resume)")
            continue

        batch_files = files[si * SHARD_SIZE:(si + 1) * SHARD_SIZE]
        # Gom chunk của MỌI doc trong shard vào 1 lần encode (GPU batch thực sự);
        # encode per-doc chỉ batch ~8 chunk -> chậm gấp bội, cháy quota.
        all_chunks, counts, doc_ids = [], [], []
        for fp in batch_files:
            chunks = chunk_text(read_text(fp), tok, MAX_CHUNKS, CHUNK_TOKENS)
            all_chunks.extend(chunks)
            counts.append(len(chunks))
            doc_ids.append(os.path.basename(fp))
        emb = np.asarray(model.encode(all_chunks, batch_size=BATCH_SIZE,
                                      normalize_embeddings=True, show_progress_bar=False))
        # tách lại theo doc + mean-pool
        doc_vecs, pos = [], 0
        for c in counts:
            doc_vecs.append(emb[pos:pos + c].mean(axis=0))
            pos += c
        arr = np.vstack(doc_vecs).astype("float32")
        arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)   # chuẩn hoá lại sau mean-pool
        np.save(shard_emb, arr)
        with open(shard_ids, "w") as f:
            json.dump(doc_ids, f)
        log(f"  shard {si}/{n_shards} -> {arr.shape}")

    # gộp shard
    embs, ids = [], []
    for si in range(n_shards):
        embs.append(np.load(os.path.join(OUT_ROOT, f"{split}_{role}_emb_shard_{si:04d}.npy")))
        ids += json.load(open(os.path.join(OUT_ROOT, f"{split}_{role}_ids_shard_{si:04d}.json")))
    full = np.vstack(embs)
    np.save(os.path.join(OUT_ROOT, f"{split}_{role}_emb.npy"), full)
    with open(os.path.join(OUT_ROOT, f"{split}_{role}_ids.json"), "w") as f:
        json.dump(ids, f)
    log(f"[{split}/{role}] gộp xong -> {full.shape}")


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    log(f"torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
    device = "cuda" if (torch.cuda.is_available() and not os.environ.get("FORCE_CPU")) else "cpu"
    if device == "cuda":
        try:
            log(f"GPU={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")
        except Exception as e:
            log(f"GPU info err: {e}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    if device == "cuda":                     # sanity encode -> nếu 'no kernel image' thì fallback CPU
        try:
            model.encode(["warmup"], show_progress_bar=False)
        except Exception as e:
            log(f"GPU encode FAIL ({type(e).__name__}: {str(e)[:90]}) -> fallback CPU")
            device = "cpu"
            model = SentenceTransformer(MODEL_NAME, device="cpu")

    log(f"Model={MODEL_NAME} split={SPLIT} device={device} shard={SHARD_SIZE} limit={LIMIT_DOCS}")
    for role in ROLES:
        embed_split_role(model, tok, SPLIT, role)
    log("DONE. Nhớ promote OUT_ROOT -> dataset pan25-embeddings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
