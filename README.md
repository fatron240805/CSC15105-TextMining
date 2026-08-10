# Text Mining Project - Plagiarism / Source Alignment

Dự án này thực hiện việc tìm và nối các đoạn văn bản tương đồng giữa một tài liệu nghi ngờ (suspicious document) và một tài liệu nguồn (source document) bằng phương pháp embedding ngữ nghĩa và nối đường đi (alignment path).

## Mục tiêu

- Tách văn bản thành các câu riêng lẻ.
- Biểu diễn từng câu bằng vector embedding ngữ nghĩa.
- Tính độ tương đồng cosine giữa câu nghi ngờ và câu nguồn.
- Chọn các cặp câu phù hợp làm candidate matches.
- Gộp và nối các cặp này thành các vùng alignment để tạo ra dự đoán cuối cùng.
- Xuất kết quả ra file text và XML.

## Cấu trúc thư mục

```text
CSC15105-TextMining/
├── main.py                      # Entry point, điều phối toàn bộ pipeline
├── requirements.txt            # Danh sách dependency cần cài đặt
├── dataset/
│   ├── validation-data/
│   │   └── validation/
│   │       ├── susp/            # Tài liệu suspicious
│   │       └── src/             # Tài liệu source
│   └── validation-groundtruth/ # Dữ liệu ground truth để đánh giá
├── src/
│   ├── preprocess.py           # Load tài liệu, tách câu, bỏ phần tham khảo
│   ├── embedding.py            # Load mô hình embedding và encode câu
│   ├── similarity.py          # Tính cosine similarity và chọn top-k match
│   ├── alignment.py            # Nhóm và nối các match thành alignment
│   └── utils.py                # Xuất kết quả ra file text/XML
├── outputs/                    # Thư mục kết quả sau khi chạy
├── docs/                       # Tài liệu mô tả logic và thuật toán
└── README.md                   # Tài liệu giới thiệu dự án
```

## Vai trò các file chính

### 1. main.py
- Là file chạy chính của dự án.
- Khởi tạo model embedding, load dữ liệu, chạy toàn bộ pipeline từ preprocessing đến xuất kết quả.

### 2. src/preprocess.py
- Đọc nội dung tài liệu từ file.
- Xác định phạm vi nội dung chính bằng cách bỏ qua abstract và references/bibliography nếu có.
- Tách văn bản thành các câu bằng spaCy.
- Loại bỏ các câu chỉ là citation/reference markers.

### 3. src/embedding.py
- Tải mô hình sentence-transformers, ví dụ: intfloat/e5-base-v2.
- Mã hóa các câu thành vector embedding.

### 4. src/similarity.py
- Tính ma trận độ tương đồng cosine giữa tất cả câu suspicious và câu source.
- Chọn các cặp câu có điểm số cao, lọc theo ngưỡng threshold và top-k.

### 5. src/alignment.py
- Nhóm các candidate matches theo vị trí câu suspicious.
- Gộp các match trên cùng một câu source trong vùng gần nhau.
- Xây dựng đường đi (path) giữa các nhóm source theo thứ tự tăng dần.
- Chọn path tốt nhất để tạo thành một alignment.

### 6. src/utils.py
- Ghi kết quả ra file text và XML để phục vụ đánh giá hoặc trình bày.

## Luồng chạy chương trình

1. Đọc file suspicious và file source.
2. Tách thành các câu.
3. Embedding từng câu bằng mô hình ngữ nghĩa.
4. Tính ma trận similarity.
5. Tìm các cặp câu tương đồng.
6. Tạo alignment từ các cặp câu này.
7. Chuyển các alignment thành các span ở cấp độ ký tự.
8. Ghi kết quả ra outputs/predictions.txt và outputs/predictions.xml.

## Cài đặt

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Chạy chương trình

```bash
python main.py
```

## Kết quả đầu ra

Sau khi chạy, dự án sẽ tạo:

- outputs/predictions.txt
- outputs/predictions.xml

## Ghi chú

- Các tham số quan trọng nằm ở đầu file main.py, bao gồm:
  - THRESHOLD: ngưỡng similarity
  - TOP_K: số candidate match cao nhất cho mỗi câu
  - MAX_SUSPICIOUS_GAP: khoảng cách cho phép giữa các câu suspicious trong cùng một group
  - MAX_SOURCE_GAP: khoảng cách cho phép giữa các câu source trong cùng một nhóm
  - MIN_PATH_LENGTH: độ dài tối thiểu của đường alignment
