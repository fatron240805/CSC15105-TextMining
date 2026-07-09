# Raw data — PAN 2025 Generated Plagiarism Detection

Task **A0-2** (Phase 0). This folder is where the raw PAN 2025 dataset lives
locally. The data itself (~2.4 GB train + ~340 MB validation) is **git-ignored**
— only this README is tracked. Every teammate downloads the data once and points
the scripts at it.

## 1. How to obtain

Download the two archives from the project Google Drive (PAN 2025
"generated-plagiarism-detection"):

| Archive | Size | Contents |
|---|---|---|
| `pan25-generated-plagiarism-detection-train.zip` | ~2.4 GB | train split |
| `pan25-generated-plagiarism-detection-validation.zip` | ~337 MB | validation split |

A tiny `00_spot_check` set (50 pairs) ships alongside for quick sanity checks.

Extract them so this folder looks like the layout below. On the reference
machine the data currently lives at `C:\github\PAN2025`; you may either extract
here under `data/raw/` or point the scripts at any absolute path via
`--truth-dir` / `--docs-dir`.

## 2. Directory layout (canonical paths)

Each split has **two** sibling folders: the documents and their ground truth.

```
data/raw/
├── 00_spot_check/
│   ├── 00_spot_check/            # documents  (pairs, src/, susp/)
│   └── 00_spot_check_truth/      # 50 annotation XMLs
├── pan25-generated-plagiarism-detection-train/
│   └── 01_train/
│       ├── 01_train/             # documents
│       │   ├── pairs             # susp src filename pairs (62,160 lines)
│       │   ├── susp/             # 60,759 suspicious-document*.txt
│       │   └── src/              # 60,592 source-document*.txt
│       └── 01_train_truth/       # 62,160 annotation XMLs + metadata.json (~153 MB)
└── pan25-generated-plagiarism-detection-validation/
    └── 02_validation/
        ├── 02_validation/        # documents (susp: 7,950 / src: 7,949)
        └── 02_validation_truth/  # 7,976 annotation XMLs + metadata.json (~100 MB)
```

> **Note:** the standalone top-level `01_train/` and `02_validation/` folders
> (containing only `*_truth`) are **incomplete duplicates** — do not use them as
> the data root. The complete documents live under the
> `pan25-generated-plagiarism-detection-*` folders shown above.

### File roles

| Path | What it is |
|---|---|
| `susp/suspicious-documentNNNNNN.txt` | Suspicious document (may contain plagiarism). Plain UTF-8 text. |
| `src/source-documentNNNNNN.txt` | Source document a suspicious doc may have plagiarised from. |
| `pairs` | Whitespace-separated `<susp.txt> <src.txt>` candidate pairs, one per line. |
| `*_truth/suspicious-…-source-….xml` | Ground-truth annotations for one pair (see §3). |
| `*_truth/metadata.json` | arXiv-level metadata (date, archive, categories, authors, doc_id) keyed by filename. Large; optional. |

## 3. Ground-truth XML schema

One XML per suspicious/source pair, e.g.
`suspicious-document020468-source-document020468.xml`:

```xml
<document reference="suspicious-document020468.txt">
  <feature name="about" title="..." authors="A and B and C"
           similarity="0.9910" severity="medium"
           prompt_tokens="6833" output_tokens="4424"/>
  <feature name="md5Hash" value="71b6f195..."/>
  <feature name="plagiarism" type="llm_prompted" llm="DeepSeek-R1"
           this_language="en" this_offset="117" this_length="1465"
           source_reference="source-document020468.txt"
           source_offset="82" source_length="1677" obfuscation="simple"/>
  <feature name="altered" type="llm_prompted" llm="DeepSeek-R1"
           this_language="en" this_offset="17974" this_length="301"/>
</document>
```

| Feature | Meaning | Key attributes |
|---|---|---|
| `about` | Document-level metadata | `title`, `authors`, `similarity` (0–1), `severity` (`low`/`medium`/`high`) |
| `md5Hash` | Integrity hash of the suspicious doc | `value` |
| **`plagiarism`** | A passage plagiarised **from a source** | `this_offset`/`this_length` (in susp), `source_reference`, `source_offset`/`source_length`, `obfuscation` (`simple`/`medium`/`hard`) |
| **`altered`** | LLM-generated passage with **no source** | `this_offset`/`this_length` only |

### Offset semantics (verified — important)

- `*_offset` / `*_length` are **character** positions into the UTF-8-decoded
  `.txt` file, **not** byte offsets.
- Train/validation `.txt` files use **LF** newlines. Still read with
  `open(..., encoding="utf-8", newline="")` so a stray CRLF file cannot shift
  every subsequent offset (universal-newline mode would silently collapse
  `\r\n` → `\n`).
- The suspicious document comes from `<document reference=...>`; each source
  comes from that feature's `source_reference`. **Do not infer the source from
  the filename** — susp-id and source-id can differ (e.g.
  `suspicious-document020491-source-document052407.xml`).

## 4. Parsing the labels (A0-3)

Use `scripts/parse_labels.py` to turn the XML truth into structured JSONL
(per-pair) + CSV (per-span, for EDA/A0-4):

```bash
# Sanity check on the 50-pair spot-check set (prints extracted spans):
python scripts/parse_labels.py \
    --truth-dir "data/raw/00_spot_check/00_spot_check_truth" \
    --docs-dir  "data/raw/00_spot_check/00_spot_check" \
    --out-jsonl outputs/spot_labels.jsonl \
    --out-csv   outputs/spot_spans.csv \
    --verify 3

# Full train split:
python scripts/parse_labels.py \
    --truth-dir "data/raw/pan25-generated-plagiarism-detection-train/01_train/01_train_truth" \
    --out-jsonl outputs/train_labels.jsonl \
    --out-csv   outputs/train_spans.csv
```

## 5. Dataset facts at a glance

| Split | susp docs | src docs | pairs | Truth XMLs | Annotated spans |
|---|---:|---:|---:|---:|---:|
| spot-check | 50 | 50 | 50 | 50 | 1,639 |
| train | 60,759 | 60,592 | 62,160 | 62,160 | 2,730,100 |
| validation | 7,950 | 7,949 | 7,976 | 348,629 | 348,629 |

- Plagiarised passages are LLM-generated paraphrases (DeepSeek-R1, Llama-3, …) at
  three obfuscation levels: `simple`, `medium`, `hard`.
- Document `severity` (`low`/`medium`/`high`) is the per-document plagiarism
  intensity referenced by the EDA task (A0-4).
- Span breakdown: train 1,877,750 plagiarism / 852,350 altered;
  validation 238,242 plagiarism / 110,387 altered (~2.2 : 1 in both splits).

## 6. Integrity verification (A0-2)

Run `scripts/check_integrity.py` to confirm a split is self-consistent. It fails
(non-zero exit) on any dangling reference, orphan file, or pairs/XML count
mismatch, so it can gate the pipeline:

```bash
python scripts/check_integrity.py \
    --docs-dir  "data/raw/pan25-generated-plagiarism-detection-validation/02_validation/02_validation" \
    --truth-dir "data/raw/pan25-generated-plagiarism-detection-validation/02_validation/02_validation_truth"
```

**Verified result — all three splits PASS:**

| Split | Missing files | Orphan files | pairs == XMLs | susp reused | src reused |
|---|:--:|:--:|:--:|--:|--:|
| spot-check | 0 | 0 | ✅ | 0 | 0 |
| train | 0 | 0 | ✅ (62,160) | 1,401 | 1,568 |
| validation | 0 | 0 | ✅ (7,976) | 26 | 27 |

Why `pairs` > unique file counts: a suspicious doc can plagiarise **several**
sources and a source can be cited by **several** suspicious docs, so the same
`.txt` appears in multiple pairs. Those reuse counts (not any missing data)
fully account for the difference. No `.txt` referenced by `pairs` is absent, and
no `.txt` on disk is left out of every pair.
