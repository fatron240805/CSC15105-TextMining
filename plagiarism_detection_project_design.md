# Thiết kế đồ án: Hệ thống phát hiện, giải thích và viết lại đạo văn

## 1. Tổng quan

Hệ thống nhận vào một tài liệu (`.txt`, `.docx`) chứa các đoạn văn nghi đạo văn, thực hiện:
1. Truy xuất (retrieval) các nguồn khả nghi.
2. Xác định tỉ lệ đạo văn, kỹ thuật đạo văn, highlight vị trí.
3. Sinh lại (rewrite) đoạn văn giữ nguyên nội dung nhưng loại bỏ các điểm đạo.
4. Đánh giá toàn bộ pipeline (retrieval + generation + explanation) bằng bộ SLM-as-judge, đối chiếu với ground truth.

Nền tảng dữ liệu: **PAN 2025 – Generative Plagiarism Detection task**. Task này dùng corpus **arXiv/ar5iv 2025** (structured HTML5), sinh đạo văn bằng 3 LLM (Llama, DeepSeek-R1, Mistral) qua việc paraphrase các đoạn $s \in S$ thành $s' \in P$ không trích dẫn nguồn, tạo ra **78.038 cặp tài liệu** $(S, P)$ với alignment $(s, s')$. Việc chọn candidate nguồn dùng embedding **SPECTER**, và đánh giá theo phương pháp luận **PAN 2015** (plagdet score = kết hợp precision/recall theo granularity).

---

## 2. Datasets đề xuất

| Dataset | Vai trò | Ghi chú |
|---|---|---|
| **PAN 2025 (Generative Plagiarism Detection)** | Nguồn chính — cặp $(S, P)$ đã có alignment $(s, s')$ | Xin quyền qua pan.webis.de / Zenodo, cần nêu mục đích sử dụng |
| **PAN-PC-11 / PAN-PC-13** | Bổ sung loại đạo văn cổ điển: near-copy, translation-based, đạo tư tưởng | Ground truth XML chi tiết, dùng đối chiếu robustness (giống cách PAN2025 test trên PAN2015) |
| **ETPC (Extended Typology Paraphrase Corpus)** | Gán nhãn *loại paraphrase* (lexical/syntactic/semantic-based) | Dùng để dạy LLM sinh "cách thức đạo văn" có kiểm soát |
| **PAWS / MRPC** | Cặp câu paraphrase vs non-paraphrase | Huấn luyện/đánh giá classifier phụ trợ, quy mô nhỏ |
| **S2ORC / arXiv full-text corpus** | Mở rộng nguồn ngoài ar5iv 2025 nếu cần domain khác | Miễn phí qua Semantic Scholar API |
| **ViWiki-Sum / Vietnamese Wikipedia dump** | Nếu mở rộng bài toán sang tiếng Việt | Chưa có shared task tương đương, cần tự sinh ground truth |

**Khuyến nghị**: dùng PAN2025 làm base, bổ sung PAN-PC cho đa dạng loại đạo văn, và tự generate thêm tập nhỏ (mục 3) để có ground truth chi tiết về *kỹ thuật đạo văn* — vì PAN2025 gốc chỉ có alignment $(s, s')$, chưa gán nhãn loại paraphrase.

---

## 3. Quy trình tạo tài liệu đạo văn (Data Generation)

### 3.1 Chọn nguồn và cặp $(S, P)$
- Lấy $S$ = văn bản gốc (từ PAN2025/arXiv hoặc nguồn tự thu thập).
- Với mỗi $S$, sample $k$ đoạn văn $\{s_1, ..., s_k\}$ làm điểm sẽ bị đạo.

### 3.2 Prompt LLM sinh đạo văn có kiểm soát

```python
PLAGIARISM_GEN_PROMPT = """
Bạn là một công cụ mô phỏng đạo văn phục vụ nghiên cứu học thuật.
Cho đoạn văn gốc sau, hãy tạo một đoạn văn đạo văn (s') sử dụng
MỘT trong các kỹ thuật sau: {technique_list}

Đoạn gốc (s):
\"\"\"{source_paragraph}\"\"\"

Yêu cầu output JSON đúng schema:
{{
  "plagiarized_text": "...",
  "technique": "verbatim_copy | shake_and_paraphrase |
                 synonym_substitution | sentence_reordering |
                 back_translation | idea_plagiarism | mosaic_patchwork",
  "technique_explanation": "Giải thích ngắn gọn cách kỹ thuật này biến đổi câu gốc",
  "modified_spans": [
    {{"original_span": "...", "modified_span": "..."}}
  ]
}}
Chỉ trả JSON, không thêm text khác.
"""
```

### 3.3 Gọi API theo dòng (line-by-line)

Xử lý từng dòng thay vì batch để kiểm soát bộ nhớ và quota API, dễ retry khi JSON invalid:

```python
import json, time
from anthropic import Anthropic

client = Anthropic()

def generate_plagiarized_paragraph(source_text: str, technique: str) -> dict:
    prompt = PLAGIARISM_GEN_PROMPT.format(
        technique_list=technique, source_paragraph=source_text
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None  # có thể retry 1 lần với instruction nhấn mạnh "chỉ JSON"

with open("source_paragraphs.jsonl") as fin, open("ground_truth.jsonl", "a") as fout:
    for line in fin:
        item = json.loads(line)
        result = generate_plagiarized_paragraph(item["text"], item["technique"])
        if result:
            result["source_id"] = item["id"]
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        time.sleep(0.5)  # tránh rate-limit
```

### 3.4 Cấu trúc ground truth

```json
{
  "doc_id": "P00042",
  "source_doc_id": "S00017",
  "alignment": {
    "source_span": {"start": 120, "end": 340, "text": "..."},
    "plagiarized_span": {"start": 45, "end": 260, "text": "..."}
  },
  "technique": "shake_and_paraphrase",
  "technique_explanation": "...",
  "plagiarism_ratio": 0.62
}
```

Ground truth này dùng ở bước Evaluation để chấm phần *giải thích* mà hệ thống sinh ra (so khớp `technique` dự đoán vs thật, và semantic similarity giữa `explanation` dự đoán vs thật).

### 3.5 Kiểm định chất lượng dữ liệu sinh
- Lọc theo similarity threshold: dùng embedding (SPECTER2 hoặc BGE-M3) tính $\cos(s, s')$, loại bỏ cặp quá giống (near-verbatim ngoài ý muốn) hoặc quá khác (không còn là đạo văn thật).
- Sample thủ công 5–10% để review chất lượng theo từng `technique`.

---

## 4. Kiến trúc pipeline hệ thống

```
Input tài liệu (txt, docx)
        │
        ▼
Parsing & chunking (tách đoạn, chuẩn hóa văn bản)
        │
        ▼
Retrieval nguồn (embedding + vector search)
        │
        ▼
Reranking & phát hiện (top-k, tính % đạo văn)
        │
        ▼
Sinh lại đoạn văn (LLM rewrite loại bỏ điểm đạo)
        │
        ▼
Đánh giá (SLM-as-judge so với ground truth)
```

**Chi tiết từng bước:**

1. **Input đa định dạng**: parser cho `.txt`/`.docx` (dùng `python-docx`, `docx2python`), chuẩn hóa encoding, xử lý bảng/footnote nếu có.
2. **Parsing & chunking**: tách đoạn theo câu/đoạn logic (không cắt cứng theo token) — giữ ranh giới đạo văn tự nhiên khớp với PAN alignment format.
3. **Retrieval nguồn**: embedding từng chunk (SPECTER2 cho scientific text, hoặc BGE/E5 cho general text), lưu vào vector DB (Supabase/pgvector hoặc FAISS), truy vấn top-N candidate nguồn.
4. **Reranking & phát hiện**: dùng cross-encoder (ví dụ `bge-reranker-v2-m3`) rerank top-N còn top-k, tính tỉ lệ đạo văn:

$$
\text{PlagRatio}(P) = \frac{\sum_{i} |s'_i|}{|P|}, \quad \text{với } s'_i \text{ là các span được gán nhãn đạo}
$$

Highlight các span có $\cos(s_i, s'_i) > \tau$ (threshold tune qua validation set).

5. **Sinh lại đoạn văn**: LLM rewrite với constraint "giữ nội dung, loại bỏ trùng lặp cấu trúc/câu chữ với nguồn nghi ngờ" — có thể dùng RAG ngược: đưa cả `plagiarized_span` và `source_span` vào context để LLM tránh lặp lại pattern.

6. **Đánh giá**: SLM-as-judge (4–5 model nhỏ chạy trên Kaggle/vast.ai) đánh giá 2 tầng:
   - **Retrieval**: Precision@k, Recall@k, MRR, nDCG so với `source_doc_id` ground truth.
   - **Generation/rewrite**: ROUGE-L, BERTScore so với bản gốc không đạo văn (nếu có), cộng LLM-as-judge chấm theo rubric (độ trung thực nội dung, mức giảm similarity với nguồn đạo).
   - **Explanation**: so khớp `technique` dự đoán (accuracy/F1) và semantic similarity của `technique_explanation`.

---

## 5. Câu hỏi mở cần nhóm quyết định

- Ngôn ngữ dữ liệu: chỉ tiếng Anh (theo PAN2025) hay mở rộng song ngữ Việt–Anh?
- Framework RAG cụ thể: LangChain, LlamaIndex, hay tự viết pipeline retrieval?
- Vector DB: Supabase/pgvector hay FAISS local cho giai đoạn thử nghiệm?
- Danh sách 4–5 SLM cụ thể dùng làm judge (ví dụ: Qwen2.5-7B, Llama-3.1-8B, Phi-3.5, Gemma-2-9B, Mistral-7B)?
