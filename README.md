# CSC15105 — Phát hiện đạo văn (PAN 2025)

Hệ RAG phát hiện đạo văn trên corpus khoa học arXiv của PAN 2025:

**Retrieval (TF-IDF) → Alignment (tf-isf seed-and-extend) → Scoring (PlagDet) → Verifier LLM → Generation grounded (luận giải).**

Kết quả chính (val): Retrieval R@1 **0.975** · Detection PlagDet **0.713** · Verifier GLM-5.2 **+0.044** PlagDet.
Báo cáo kiến trúc: `evaluation/architecture_report.html` · slide: `architecture_slides.pdf`.

---

## 1. Cài đặt

```bash
pip install numpy pandas scikit-learn openai google-genai
# (tuỳ chọn) aligner neural E5 của Khôi: pip install torch sentence-transformers
```

### Lấy data đã xử lý (từ Drive — nhanh nhất)

Toàn bộ data đã-xử-lý (labels/spans, embeddings, kết quả eval, báo cáo) được đóng gói sẵn:
`pan25_processed_data.zip` (~140MB nén / ~1,25GB) — **link nhóm**:
https://drive.google.com/open?id=1k0zI7lvhG84fwpLTsYT5McvCoJu0Jqg2

```bash
# 1) Tải & giải nén, rồi copy phần đã xử lý vào đúng chỗ repo:
#    (gói có cấu trúc pan25_processed/outputs/... , pan25_processed/evaluation/...)
cp -r pan25_processed/outputs/*       outputs/
cp -r pan25_processed/evaluation/results     evaluation/
cp -r pan25_processed/evaluation/generation  evaluation/
```
Sau bước này có `outputs/validation_spans.csv` (gold để chấm), embeddings, và các JSON kết quả.
`outputs/` bị `.gitignore` nên **không có sẵn khi clone** — phải lấy từ gói này (hoặc tự sinh lại, xem dưới).

> **Gói Drive KHÔNG chứa corpus thô** (các file `.txt` susp/src, ~2,4GB).
> - **Chỉ dựng lại báo cáo** (`build_leaderboard.py`, `build_system_report.py`) → *đủ với gói Drive*, không cần corpus.
> - **Chạy eval mới** (`run_pipeline.py`, `eval_verifier.py`, …) → còn cần **corpus PAN val** (`susp/` + `src/`):
>   tải từ Kaggle dataset `pan25-corpus-val` (hoặc bản corpus nhóm đang dùng), đặt vào một thư mục rồi
>   **sửa hằng `VAL`** đầu các script (`orchestration/run_pipeline.py`, `evaluation/*.py`) trỏ đúng đường dẫn máy bạn.

**Hoặc tự sinh lại gold** từ truth XML (nếu đã có corpus PAN, không cần Drive):

```bash
python scripts/parse_labels.py \
  --truth-dir "<PAN>/pan25-...-validation/02_validation/02_validation_truth" \
  --docs-dir  "<PAN>/pan25-...-validation/02_validation/02_validation" \
  --out-jsonl outputs/validation_labels.jsonl \
  --out-csv   outputs/validation_spans.csv
```

**Khoá LLM** (chỉ cần cho Verifier / Generation / Classify). Tạo `.env` ở gốc repo (đã gitignore):

```dotenv
# Google Gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-flash-latest
# FPT AI Marketplace (OpenAI-compatible) — dùng cho GLM-5.2 v.v.
LLM_API_KEY=...
LLM_API_BASE=https://mkp-api.fptcloud.com
LLM_MODEL=GLM-5.2
```
Chuyển provider bằng env `LLM_PROVIDER=gemini` (mặc định) hoặc `fpt`.

---

## 2. Inference (chạy phát hiện đạo văn)

### 2a. Pipeline end-to-end (batch, không cần GPU)
Mỗi tài liệu nghi vấn → TF-IDF truy hồi top-k nguồn → align tf-isf → gộp span → chấm PlagDet:

```bash
python orchestration/run_pipeline.py --split val --subset 200 --topk 1
```
| Cờ | Ý nghĩa | Mặc định |
|---|---|---|
| `--split` | `val` \| `spot` | `val` |
| `--subset` | số tài liệu nghi vấn | `200` |
| `--topk` | số nguồn để align (top-1 tốt nhất; top-k>1 hại precision) | `1` |
| `--th` / `--th3` | ngưỡng seed / mở rộng của aligner | `0.30` / `0.50` |
| `--aligner` | `tfisf` (lexical) \| `khoi` (E5 neural, cần torch+GPU) | `tfisf` |

Kết quả in ra màn hình + ghi `outputs/pipeline_eval.csv`, lưu vào kho `evaluation/results/`
và dựng lại `evaluation/leaderboard.html`.

### 2b. Demo web tương tác (dán/upload tài liệu → highlight + luận giải)
```bash
python demo/app.py \
  --sources "C:/github/PAN2025/pan25-generated-plagiarism-detection-validation/02_validation/02_validation/src" \
  --port 8010 --provider fpt --llm-model GLM-5.2
```
Mở `http://localhost:8010`. Nạp kho nguồn + TF-IDF một lần lúc khởi động; mỗi request chạy
Retrieval → Alignment và trả span. Bấm một span để mở modal có 4 nút LLM: **Luận giải (RAG)**,
Phân loại kỹ thuật, Xác minh, Viết lại (dùng model theo `--provider/--llm-model`).
Mẫu test: `demo/test_samples.txt`. (Bỏ `--provider/--llm-model` để mặc định GLM-5.2 qua FPT.)

---

## 3. Evaluation

Mọi script eval đọc gold từ `outputs/validation_spans.csv`; các script gọi LLM cần `.env`.
**Teammate**: lấy gói Drive (mục 1) trước. Chưa có corpus thô vẫn **dựng lại được báo cáo** bằng
`build_leaderboard.py` / `build_system_report.py` (đọc JSON/CSV, không cần văn bản gốc);
muốn **chạy eval mới** thì cần corpus PAN val + sửa `VAL` (mục 1).

| Tầng | Lệnh | Ghi ra |
|---|---|---|
| **Detection (PlagDet)** | `python orchestration/run_pipeline.py --split val --subset 1000 --topk 1` | `outputs/pipeline_eval.csv`, `evaluation/results/*.json` |
| **Verifier #3** (khử báo giả, nhãn gold khách quan) | `python evaluation/eval_verifier.py --max-spans 20 --sleep 5` | `evaluation/generation/verifier_eval_*.json` |
| **So sánh 6 LLM verifier** (song song, quy mô lớn) | `python evaluation/compare_models.py --n 1000 --inner 2 --max-concurrent 12` | `evaluation/generation/model_compare_*.json` |
| **Generation** (viết lại khử đạo — detector chạy lại) | `python evaluation/eval_generation.py --max-spans 15 --judge` | `evaluation/generation/gen_eval_*.json` |
| **Classify #1** (kỹ thuật, bộ synthetic — chỉ tham khảo) | `python evaluation/eval_classify.py --n 12 --sleep 5` | `evaluation/generation/*.json` |

Dựng lại các trang báo cáo tổng hợp:
```bash
python evaluation/build_leaderboard.py       # -> evaluation/leaderboard.html (so sánh phương pháp)
python evaluation/build_system_report.py     # -> evaluation/system_report.html (3 tầng)
```

**Retrieval (Recall@k, MRR)** đo ở quy mô đầy đủ trên **Kaggle** (GPU) — kernel `k3`, kết quả về
`outputs/retrieval_eval.csv`; các kernel eval quy mô lớn khác (k5–k8) xem `KAGGLE_RUNBOOK.md`.

> Lưu ý: các eval gọi LLM bị giới hạn **quota** (Gemini free-tier cạn quota-ngày → `429`;
> FPT ổn định hơn). Vì vậy `--max-spans` / `--n` nhỏ để chạy nhanh; chạy quy mô lớn nên đẩy Kaggle.

---

## 4. Cấu trúc thư mục

```
orchestration/run_pipeline.py     # inference + eval end-to-end
scripts/alignment/align_tfisf.py  # aligner seed-and-extend (lõi)
scripts/parse_labels.py           # truth XML -> outputs/*_labels.jsonl + *_spans.csv
generation/{verify,classify,rewrite,explain}.py  # vai trò LLM (đa provider)
evaluation/*.py                   # eval từng tầng + dựng báo cáo HTML
evaluation/{leaderboard,system_report,architecture_report}.html
demo/app.py                       # demo web (stdlib http.server)
outputs/                          # data đã xử lý: labels/spans, embeddings, eval csv
```

Tài liệu thêm: `plagiarism_detection_project_design.md`, `IMPLEMENTATION_PLAN.md`, `KAGGLE_RUNBOOK.md`.
