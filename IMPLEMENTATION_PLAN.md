# Kế hoạch triển khai end-to-end — chạy trên Kaggle free GPU (qua Kaggle CLI)

> Companion của `plagiarism_detection_project_design.md`. File này biến thiết kế thành
> các bước chạy được, ánh xạ vào 26 task trong Task Tracker, với ràng buộc **tài nguyên
> miễn phí của Kaggle** (P100/T4 16GB, ~30 GPU-giờ/tuần, kernel tối đa 12 giờ).

---

## 0. Điều kiện tiên quyết (BLOCKING — mọi bước Kaggle chết nếu thiếu)

Hiện trạng đã kiểm tra:

| Thành phần | Trạng thái |
|---|---|
| Kaggle CLI | ✅ `2.2.3` đã cài |
| `~/.kaggle/kaggle.json` | ❌ **CHƯA CÓ** — phải tạo trước |
| Python local | 3.11, có `torch`+`pandas`; thiếu `faiss`/`sentence-transformers`/`transformers`/`sklearn` (không sao — phần nặng chạy trên Kaggle) |
| Outputs Phase 0 | ✅ đã parse: `outputs/{spot,train,validation}_{labels.jsonl,spans.csv}` |

**Việc đầu tiên nhóm phải làm (1 người, 5 phút):**

1. Kaggle → Settings → **Create New API Token** → tải `kaggle.json`.
2. Đặt vào `C:\Users\<user>\.kaggle\kaggle.json` (Windows) hoặc `~/.kaggle/kaggle.json`.
3. Xác nhận auth — gõ trong Claude Code prompt: `! kaggle datasets list --max-size 1`
   (nếu ra danh sách dataset là OK; nếu 401 là token sai chỗ/hết hạn).

Sau khi xác nhận, mọi lệnh `kaggle` bên dưới mới chạy được.

---

## 1. Đối chiếu Thiết kế ↔ Dữ liệu thật (đọc trước khi code)

Design doc mô tả một số thứ **không tồn tại** trong nhãn PAN 2025 thật (đã parse XML để xác minh).
Không được kế thừa deliverable không có ground truth. Bảng quyết định:

| Design giả định | Dữ liệu PAN 2025 thật | Quyết định |
|---|---|---|
| 78.038 cặp (S,P) | train **62.160** + val **7.976** = 70.136 truth XML (+ spot 50) | Dùng số thật; 78k là con số của tổ chức (gồm test set không public) |
| Nhãn `technique` (verbatim_copy / shake_and_paraphrase / …) | Chỉ có `obfuscation ∈ {simple, medium, hard}` (per-span) + `severity ∈ {low, medium, high}` (per-doc) | **Reframe**: "technique classification" (A2-4) → **obfuscation-level classification** trên nhãn thật. Sinh technique-labeled data (§3 design) là **track phụ optional** (Tier 3) |
| Track "explanation eval" so `technique` dự đoán vs thật (§6) | Không có technique truth | Bỏ khỏi core; nếu làm Tier 3 thì mới có truth để chấm |
| Classifier "paraphrase vs altered" (A2-4) | `plagiarism` (có source_reference) vs `altered` (không có source) — **phân biệt được từ chính nhãn** | Giữ nguyên; đây là nhị phân có ground truth thật |
| Rewrite đoạn văn (§5) | Ngoài phạm vi bài toán PAN chấm điểm | Tier 2 — chỉ làm sau khi core có điểm |
| SPECTER cho candidate + PAN2015 plagdet | ✅ Khớp | Giữ nguyên |

**Số liệu thật để dùng xuyên suốt** (đã đo, không đoán):

- Train: 62.160 pairs · susp 60.759 · src 60.592 · **2.730.100 spans** (1.877.750 plagiarism / 852.350 altered)
- Validation: 7.976 pairs · susp 7.950 · src 7.949 · **348.629 spans** (238.242 / 110.387)
- Span = offset **ký tự** (không phải byte) vào text UTF-8; file dùng LF. Đọc `newline=''`.
- Nguồn của mỗi span lấy từ `source_reference` trong XML (không suy từ tên file).

---

## 2. Kiến trúc: Local orchestrator + Kaggle GPU worker

```
  MÁY LOCAL (Windows)                          KAGGLE (free P100/T4)
  ─ orchestrator ─                             ─ worker ─
  scripts/kaggle/*.py  ──push kernel──►  đọc dataset (read-only)
  kaggle kernels push                    chạy embed / FAISS / align / judge
  kaggle kernels status ◄──poll──        ghi /kaggle/working (≤20GB, chỉ dir này persist)
  kaggle kernels output ◄──pull──        output → promote thành Kaggle dataset (versioned)
```

- **Local = orchestrator**: viết script kernel, `push`, `poll`, `pull` kết quả. Không train nặng ở local.
- **Kaggle = worker**: đọc dữ liệu, chạy GPU, ghi output. Mỗi kernel phải **idempotent + resumable**.

### Ba Kaggle Dataset (đây là phần cốt lõi, không phải "cache tạm")

| Dataset | Nội dung | Ai tạo | Kích thước |
|---|---|---|---|
| `pan25-corpus` | susp/, src/, pairs của train+val (raw .txt) | 1 lần, upload từ `C:\github\PAN2025` | ~2.4 GB |
| `pan25-labels` | **`*_spans.csv`** (KHÔNG upload `*_labels.jsonl` 638MB trừ khi cần nesting) | từ `outputs/` | ~480 MB |
| `pan25-embeddings` | embeddings + FAISS index (output của K2/K3), versioned | kernel promote | ~0.5–1 GB |

> Kernel sau đọc `pan25-embeddings` như **input read-only** → không tính lại embedding →
> tiết kiệm quota. Mỗi lần recompute = bump version dataset.

### Ràng buộc kernel phải ghim (sẽ cắn giữa chừng nếu quên)

- `enable_internet: true` để tải model HuggingFace (hoặc attach model như 1 dataset nếu offline).
- `enable_gpu: true`; chọn accelerator P100 hoặc T4×2.
- Chỉ `/kaggle/working` persist (≤20GB). Ghi **partial shard** (vd `emb_shard_007.npy`) và
  **skip shard đã xong** khi resume — job embed 60k doc mà timeout 12h không resume = mất trắng.
- Quota ~30 GPU-giờ/tuần toàn team → xem mục 8 (budget).

### Lệnh Kaggle CLI cốt lõi (orchestrator dùng lặp lại)

```bash
# tạo/ cập nhật dataset corpus (chạy trong thư mục chứa dataset-metadata.json)
kaggle datasets create -p ./kaggle_datasets/pan25-corpus --dir-mode zip
kaggle datasets version -p ./kaggle_datasets/pan25-corpus -m "add val split" --dir-mode zip

# push 1 kernel (thư mục chứa kernel-metadata.json + script .py/.ipynb)
kaggle kernels push -p ./kaggle_kernels/k2_embed

# theo dõi + lấy kết quả
kaggle kernels status <user>/k2-embed-sources
kaggle kernels output <user>/k2-embed-sources -p ./outputs/k2/
```

`kernel-metadata.json` mẫu (K2):
```json
{
  "id": "<user>/k2-embed-sources",
  "title": "K2 embed sources",
  "code_file": "k2_embed.py",
  "language": "python",
  "kernel_type": "script",
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["<user>/pan25-corpus"],
  "is_private": true
}
```

---

## 3. Chia scope theo TIER (core no-LLM trước — có điểm PlagDet trước khi tốn quota cho 7B)

| Tier | Nội dung | Cần LLM? | Quota | Task tracker |
|---|---|---|---|---|
| **Tier 1 (core)** | Retrieval → Alignment → PlagDet. **Đây là toàn bộ bài PAN được chấm điểm, có ground truth thật.** | Không | Bounded, ~6–10 GPU-h một lần | A0-4,5 · A1-* · A2-* · A3-1,2,3 · A4-* |
| **Tier 2** | Rewrite (§5) + SLM-as-judge (§6) + highlight report (A3-4) | Có (SLM mở trên Kaggle GPU) | Cao, sample-based | A2-4 mở rộng · A3-4 · phần eval §6 |
| **Tier 3 (optional)** | Sinh technique-labeled data (§3) qua Claude API | Có (API ngoài) | Ngoài Kaggle | — (mở rộng) |

→ Làm **xong Tier 1 và có một con số PlagDet end-to-end** trước khi đụng Tier 2.

---

## 4. Kế hoạch theo Phase (ánh xạ 26 task → kernel Kaggle → người)

### Phase 0 — Setup & Data Understanding
| Task | Việc | Chạy ở đâu | Trạng thái |
|---|---|---|---|
| A0-1 (Khoa) | repo skeleton + `orchestration/config.yaml` | local | ⚠️ chưa push (chưa thấy trong repo) |
| A0-2 (Hiếu) | dataset + integrity check | local | ✅ xong (`check_integrity.py`) |
| A0-3 (Hiếu) | parse XML → JSONL/CSV | local | ✅ xong (`parse_labels.py`) |
| A0-4 (Phúc) | EDA: phân phối severity/obfuscation, độ dài span, plagiarism vs altered | **local hoặc kernel K1** (đọc `*_spans.csv`, nhẹ, CPU) | ⬜ |
| A0-5 (Khôi) | skeleton `evaluation/metrics.py` (P/R/F1/Granularity/PlagDet theo công thức PAN 2015) | local | ⬜ |

### Phase 1 — Source Retrieval (chủ yếu trên Kaggle GPU)
| Task | Kernel | Việc |
|---|---|---|
| A1-1 (Hưng) | **K2 (GPU)** | Embed toàn bộ `src/` (60.592) + `susp/` (60.759). Doc-level cho retrieval (SPECTER2/BGE-M3). **Chunk dài, ghi shard, resumable.** Output → `pan25-embeddings` |
| A1-2 (Hưng) | **K3 (CPU/GPU)** | FAISS `IndexFlatIP` (60k×768 ≈ 180MB, **exact — KHÔNG cần IVF/HNSW**). Truy vấn top-k src cho mỗi susp |
| A1-3 (Khoa) | K3 | Recall@k: **TF-IDF baseline** vs **embedding**, gold = `source_reference` từ labels. Output `outputs/retrieval_eval.csv` |
| A1-4 (Hưng) | K2/K3 | Tối ưu batching/streaming. Lưu ý: corpus thật ~60k/split, "100k" trong tracker là mục tiêu, `IndexFlatIP` đủ |

> **Rủi ro truncation (ghim để không phải khám phá lại ở scale):** K2 embed doc = mean-pool
> ~3.200 token đầu (MAX_CHUNKS×CHUNK_TOKENS). Đây là tín hiệu *chủ đề* tốt cho retrieval
> (doc nguồn cùng topic), nhưng **under-retrieve khi đoạn đạo văn nằm cuối doc dài** (EDA:
> span p90 = 1.427 ký tự, ~47 span/cặp rải khắp doc). **Thước đo:** Recall@15 của K3 trên val
> chính là con số cho biết truncation có tốn recall không. Nếu có → **fallback: index mức
> chunk** (index mọi chunk, retrieve doc theo max-chunk score) thay vì tăng số chunk trong mean.

### Phase 2 — Text Alignment (core)
| Task | Kernel | Việc |
|---|---|---|
| A2-1 (Hiếu) | **K4** | Seed-and-extend / DP alignment trên từng cặp (susp, src candidate). Sinh candidate span khớp |
| A2-2 (Khôi) | K4 | `alignment_score` = SPECTER + TF-IDF + section-title (Eq.1) |
| A2-3 (Khôi) | K4 | Trích (offset, length) span đạo văn + gộp span chồng lấn |
| A2-4 (Hưng) | K4 | Classifier **plagiarism (có source) vs altered (không source)** — nhị phân, ground truth thật. (Bản mở rộng technique = Tier 3) |
| A2-5 (Hưng) | K4 | Threshold theo `obfuscation`/`severity` (simple/medium/hard) để tăng recall |

### Phase 3 — Scoring (CPU, chạy local được)
| Task | Việc |
|---|---|
| A3-1 (Khôi) | Precision/Recall/F1 **mức ký tự** (công thức 2–4) |
| A3-2 (Khôi) | Granularity + PlagDet (công thức 5–6), test lại bằng ví dụ đề bài |
| A3-3 (Khôi) | Chuẩn hoá điểm tổng thể 0–1 mỗi doc |
| A3-4 (Phúc) | **[Tier 2]** highlight span trên P + link tới nguồn (HTML report) |

### Phase 4 — Integration & Optimization
| Task | Việc |
|---|---|
| A4-1 (Hiếu) | Ghép end-to-end (Retrieval→Alignment→Scoring), chạy trên **validation** (7.976 — nhỏ, chạy nhanh) |
| A4-2 (Hiếu) | Streaming line-by-line cho tập lớn (train 62k) tránh tràn RAM |
| A4-3 (Phúc) | Edge case: doc rỗng, không tìm thấy nguồn, span quá ngắn |
| A4-4 (Phúc) | Demo/CLI xuất báo cáo PlagDet trên vài mẫu |

### Phase 5 — Report
A5-1 Method (Khoa) · A5-2 Experiment/Results (Phúc) · A5-3 review cuối (Khoa).

---

## 5. Cấu trúc repo mục tiêu

```
CSC15105-TextMining/
├── data/raw/                      # git-ignored, chỉ README tracked
├── outputs/                       # git-ignored (spans.csv, eval, embeddings pull về)
├── scripts/
│   ├── parse_labels.py            # ✅
│   ├── check_integrity.py         # ✅
│   ├── eda.py                     # A0-4
│   ├── retrieval/                 # A1-* (embed, index, search, eval)
│   ├── alignment/                 # A2-*
│   └── visualize/                 # A3-4
├── evaluation/                    # A0-5, A3-* (metrics.py, plagdet.py, prf_score.py)
├── orchestration/
│   ├── config.yaml                # A0-1 (pipeline config chung)
│   └── run_pipeline.py            # A4-1
├── kaggle_datasets/               # dataset-metadata.json cho 3 dataset
│   ├── pan25-corpus/
│   ├── pan25-labels/
│   └── pan25-embeddings/
├── kaggle_kernels/                # kernel-metadata.json + script mỗi kernel
│   ├── k1_eda/ k2_embed/ k3_retrieval/ k4_align/ ...
├── demo/                          # A4-4
├── paper/                         # A5-*
└── IMPLEMENTATION_PLAN.md         # file này
```

---

## 6. Ngân sách GPU quota (~30 giờ/tuần toàn team — đo trước, đừng đoán)

| Job | Ước lượng | Ghi chú |
|---|---|---|
| K2 embed doc-level (121k doc) | ~1–2 GPU-h | **Đo trên 500 doc trước**, nhân lên. Cache → không lặp |
| K2 embed passage-level (cho alignment, chỉ trên candidate pairs) | ~3–5 GPU-h | Không embed toàn bộ cross-product; chỉ top-k pairs |
| K3 FAISS + retrieval eval | <1 GPU-h | Flat index nhanh |
| Tier 2 — 5 SLM judge (sample) | Đắt nhất | **Sample vài trăm case**, không chấm toàn bộ. Budget riêng |

Nguyên tắc: embedding là chi phí **một lần bounded** (cache lại); SLM judge là chi phí **lặp** →
luôn sample. Nếu cháy quota, dùng validation (7.976) thay train (62.160) để prototype.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Kernel timeout 12h mất embedding | Ghi shard `/kaggle/working`, skip shard đã xong (resume) |
| Hết quota tuần | Cache embeddings thành dataset; prototype trên validation trước |
| CRLF/offset lệch | Đã chuẩn hoá: đọc `newline=''`, offset ký tự (đã verify Phase 0) |
| Kéo `train_labels.jsonl` 638MB lên Kaggle vô ích | Upload `*_spans.csv` (phẳng, pandas đọc thẳng) |
| A2-4 chấm "technique" không có truth | Reframe sang obfuscation-level (bảng mục 1) |
| FAISS IVF/HNSW mất recall phải bào chữa trong paper | Dùng `IndexFlatIP` exact (60k đủ nhỏ) |

---

## 8. Việc cần làm ngay (thứ tự)

1. **[BLOCKING] 1 người**: tạo `kaggle.json`, xác nhận `! kaggle datasets list --max-size 1`.
2. **Hiếu/Khoa**: dựng `orchestration/config.yaml` (A0-1) + skeleton thư mục repo (mục 5).
3. **Hiếu**: viết `kaggle_datasets/pan25-corpus` + `pan25-labels` metadata, `kaggle datasets create` (upload 1 lần).
4. **Phúc**: A0-4 EDA đọc `outputs/*_spans.csv` (local, không tốn quota).
5. **Khôi**: A0-5 skeleton `evaluation/metrics.py` (PlagDet công thức PAN 2015) — làm sớm để Phase 3 sẵn sàng.
6. **Hưng**: K2 embed — **chạy thử 500 doc đo thời gian trước**, rồi mới full + cache thành `pan25-embeddings`.
7. Sau khi có embeddings: K3 (A1-2/3), rồi K4 (Phase 2), rồi Phase 3 → **có số PlagDet end-to-end (Tier 1 xong)**.
8. Chỉ sau đó mới đụng Tier 2 (rewrite + SLM judge).
```
```
