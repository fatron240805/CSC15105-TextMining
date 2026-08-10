# Plagiarism Detection & Alignment

## Overview

Hệ thống phát hiện các đoạn văn có khả năng đạo văn bằng **semantic similarity** và **source-path alignment**.

```text
Document
   ↓
Sentence Splitting
   ↓
E5 Embedding
   ↓
Cosine Similarity
   ↓
Top-k Candidate Matches
   ↓
Suspicious Grouping
   ↓
Source Merging
   ↓
Source Path Construction
   ↓
Best Path Selection
   ↓
Character-level Spans
   ↓
TXT / XML Output
```

## 1. Candidate Matching

Sử dụng:

```text
intfloat/e5-base-v2
```

Suspicious sentences dùng prefix `query:`, source sentences dùng `passage:`.

Với mỗi suspicious sentence, giữ tối đa `TOP_K` source sentences có similarity:

```text
similarity >= THRESHOLD
```

Ví dụ:

```text
S8 -> T206  0.8608
S8 -> T207  0.8833
S8 -> T254  0.8628
```

---

## 2. Suspicious Grouping

Các suspicious sentences được group nếu khoảng cách giữa hai sentence liên tiếp không vượt quá:

```text
MAX_SUSPICIOUS_GAP = 2
```

Ví dụ:

```text
[0, 8, 17, 18, 36, 60, 61, 62, 64]
```

thành:

```text
[0]
[8]
[17, 18]
[36]
[60, 61, 62, 64]
```

---

## 3. Source Merging

Các source matches của cùng một suspicious sentence được merge nếu khoảng cách source không vượt quá:

```text
MAX_SOURCE_GAP = 5
```

Ví dụ:

```text
S8 -> T206
S8 -> T207
S8 -> T254
```

trở thành:

```text
S8 -> [T206, T207]
S8 -> [T254]
```

Score của source group là trung bình similarity của các matches bên trong group.

---

## 4. Source Path Alignment

Với một suspicious group có nhiều sentences, hệ thống tìm các **source paths**.

Một path hợp lệ khi source index:

1. Tăng dần.
2. Khoảng cách giữa các source groups không vượt quá `MAX_SOURCE_GAP`.

Ví dụ:

```text
S60 -> T290
S61 -> T291
S62 -> T292
S63 -> T293
S64 -> T295
```

tạo thành một path hợp lệ.

Nếu có nhiều source paths:

```text
Path A:
290 → 291 → 292 → 293 → 295

Path B:
390 → 391 → 392 → 393 → 395
```

thì ưu tiên:

1. Path dài hơn.
2. Nếu cùng độ dài → average similarity cao hơn.

`MIN_PATH_LENGTH = 2` quy định path multi-sentence phải có ít nhất 2 suspicious sentences.

---

## 5. Single Sentence

Nếu group chỉ có một suspicious sentence, hệ thống không xây path mà:

1. Merge source matches.
2. Tính average similarity.
3. Chọn source group có score cao nhất.

Ví dụ:

```text
S8 -> T206  0.8608
S8 -> T207  0.8833
S8 -> T254  0.8628
```

sẽ chọn:

```text
S8 -> [T206, T207]
```

vì:

```text
(0.8608 + 0.8833) / 2 = 0.87205
```

cao hơn `0.8628`.

---

## 6. Character-level Span

Sau khi chọn alignment, sentence indices được chuyển thành character offsets.

Ví dụ:

```text
Suspicious:
start  = 1639
end    = 1984
length = 345

Source:
start  = 30764
end    = 31053
length = 289
```

Prediction có dạng:

```python
{
    "suspicious_start": 1639,
    "suspicious_length": 345,
    "source_start": 30764,
    "source_length": 289
}
```

---

## 7. Output

Hệ thống xuất hai file:

### TXT

Dùng để debug và kiểm tra trực tiếp:

```text
PREDICTION 1

SUSPICIOUS
Adversarial data poisoning ...

SOURCE
Regression models are widely used ...
```

### XML

Dùng làm prediction output:

```xml
<document reference="suspicious-document010237.txt">
  <feature
      name="plagiarism"
      type="prediction"
      this_offset="1639"
      this_length="345"
      source_reference="source-document010237.txt"
      source_offset="30764"
      source_length="289"/>
</document>
```

## Parameters

```python
THRESHOLD = 0.86
TOP_K = 5
MAX_SUSPICIOUS_GAP = 2
MAX_SOURCE_GAP = 5
MIN_PATH_LENGTH = 2
```

Tóm lại, **E5 similarity chỉ dùng để tạo candidate matches; alignment dựa thêm vào thứ tự của suspicious sentences và source sentences để tìm source path hợp lý.**
