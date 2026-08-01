# Kaggle Runbook — chạy pipeline trên free GPU qua CLI

Chuỗi lệnh chính xác. Chạy sau khi có `~/.kaggle/kaggle.json`.
Xem kiến trúc & lý do trong `IMPLEMENTATION_PLAN.md`.

## 0. Xác nhận auth (BLOCKING)
```bash
kaggle datasets list --mine              # ra danh sách/none = OK; 401 = token sai chỗ
```

## ⚠️ Gotcha đã XÁC MINH trên Kaggle (đọc kỹ — mất nhiều giờ mới ra)

1. **Token kiểu mới cho WRITE**: `datasets/kernels create/push` cần token `KGAT_...` ở
   `~/.kaggle/access_token` (không chỉ `kaggle.json` username+key — cái đó chỉ đủ read).
2. **Push kernel PHẢI có `PYTHONUTF8=1`** (comment tiếng Việt UTF-8, CLI Windows đọc cp1252 → crash).
   Pull output cũng vậy: `PYTHONUTF8=1 kaggle kernels output ...`.
3. **Dataset có subfolder → `--dir-mode zip`**; Windows còn cần tạo sẵn
   `mkdir -p "$TEMP/.kaggle/uploads/kaggle_datasets"` (bug sidecar CLI).
4. **Slug kernel = TITLE, không phải id** → để `id` khớp slug-của-title (vd title "K2 embed corpus"
   → slug `k2-embed-corpus`), nếu không `status/output` sẽ 403.
5. **Mount path Kaggle mới**: `/kaggle/input/datasets/<user>/<slug>/...` (không phải `/kaggle/input/<slug>`).
   Kernel đã **tự dò path** nên không cần lo, nhưng đừng hard-code.
6. **Zip giải nén PHẲNG**: upload `spot.zip` (chứa `docs/...`) → dataset có `docs/` ở root
   (mất cấp `spot/`). ⇒ **mỗi split = 1 dataset riêng** (pan25-corpus-val, -train), không nhét chung.
7. **P100 KHÔNG chạy được** torch 2.10 (`cc=6.0` bị bỏ hỗ trợ → "no kernel image").
   **BẮT BUỘC** `"machine_shape": "NvidiaTeslaT4"` trong kernel-metadata.json cho kernel GPU
   (T4 = cc 7.5, OK). Giá trị hợp lệ: `NvidiaTeslaT4` | `NvidiaTeslaP100` (đúng hoa/thường).
8. **faiss KHÔNG có** trong image Kaggle → K3 dùng numpy matmul top-k (exact, tương đương IndexFlatIP).

## 1. Điền Kaggle username vào metadata
Thay `TODO-USERNAME` bằng username thật trong:
`kaggle_datasets/*/dataset-metadata.json`, `kaggle_kernels/*/kernel-metadata.json`,
và `kaggle.username` trong `orchestration/config.yaml`.
```bash
# ví dụ (Git Bash): grep -rl TODO-USERNAME kaggle_datasets kaggle_kernels | \
#   xargs sed -i 's/TODO-USERNAME/hieutran123/g'
```

## 2. Đóng gói dữ liệu (local, một lần)
```bash
python scripts/kaggle_stage.py                    # labels (~480MB)
python scripts/kaggle_stage.py --corpus --split val   # + corpus val (8k doc, nhẹ — prototype)
# python scripts/kaggle_stage.py --corpus           # + corpus train (2.4GB, khi cần full)
```

## 3. Tạo 2 dataset (corpus + labels)
```bash
kaggle datasets create -p kaggle_datasets/pan25-labels
kaggle datasets create -p kaggle_datasets/pan25-corpus
# cập nhật về sau: kaggle datasets version -p <dir> -m "add train split"
```

## 4. K2 — embed (GPU). Prototype trên val trước!
```bash
kaggle kernels push -p kaggle_kernels/k2_embed
kaggle kernels status <user>/k2-embed             # đợi 'complete'
kaggle kernels output <user>/k2-embed -p ./outputs/k2
```
- Mặc định `SPLIT="val"` (7.9k doc, ~vài phút). Để chạy **train** (60k): sửa dòng
  `SPLIT = os.environ.get("SPLIT", "val")` trong `k2_embed.py` thành default `"train"`
  rồi push lại (Kaggle CLI không truyền env var được).
- **Đo trước:** lần đầu để `SHARD_SIZE` nhỏ, xem log thời gian/shard rồi ước lượng full.
- Timeout 12h? Chỉ cần push lại — shard đã xong tự skip (resume).

## 5. Promote embeddings thành dataset (để K3/K4 đọc, không tính lại)
```bash
cp ./outputs/k2/*_emb.npy ./outputs/k2/*_ids.json kaggle_datasets/pan25-embeddings/
kaggle datasets create -p kaggle_datasets/pan25-embeddings          # lần đầu
# kaggle datasets version -p kaggle_datasets/pan25-embeddings -m "val embeds"  # lần sau
```

## 6. K3 — retrieval eval (CPU). Recall@k: embedding vs TF-IDF
```bash
kaggle kernels push -p kaggle_kernels/k3_retrieval
kaggle kernels status <user>/k3-retrieval
kaggle kernels output <user>/k3-retrieval -p ./outputs/
cat ./outputs/retrieval_eval.csv                  # -> deliverable A1-3
```

## 7. Tiếp theo (chưa viết kernel)
- **K4 alignment** (Phase 2): đọc pan25-corpus + pan25-embeddings + top-k của K3 → seed-and-extend → span dự đoán CSV.
- **Scoring** (Phase 3): chạy `python evaluation/plagdet.py --truth outputs/val_spans.csv --pred outputs/val_pred_spans.csv` — **local, không cần Kaggle**.

## Quota (~30 GPU-h/tuần toàn team)
- Embedding = chi phí **một lần** → cache thành dataset, đừng chạy lại.
- Prototype trên **val (8k)** trước khi đụng **train (60k)**.
- SLM judge (Tier 2) = chi phí **lặp** → luôn sample vài trăm case.
