#!/usr/bin/env python3
"""K5 — Aligner của Khôi (E5-base-v2 + source-path) chạy trên Kaggle GPU, chấm PlagDet.

Tự chứa: nhúng nguyên văn code alignment của Khôi + PlagDet + driver. So trực tiếp
với tf-isf@150 nhờ nhúng đúng 150 susp của eval set (eval_set_id=f3c526f7d87f).
Xuất /kaggle/working/result_khoi.json (+ baseline) để tải về evaluation/results/.
"""
import subprocess, sys, importlib

def _ensure(pip_name, import_name=None):
    try:
        importlib.import_module(import_name or pip_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name])

_ensure("sentence-transformers", "sentence_transformers")
try:
    import spacy; spacy.load("en_core_web_sm")
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "spacy"])
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])


# ===== preprocess.py (Khôi) =====
from pathlib import Path
import re
import spacy

# Load English sentence segmentation model
nlp = spacy.load("en_core_web_sm")

def load_document(path):
    '''
    Load document from given path.
    '''
    path = Path(path)

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        return f.read()


def find_content_range(text):
    '''
    Find the main content range of an academic document.

    The returned offsets refer to the original document.
    '''

    # Determine the abstract part
    abstract_pattern = re.compile(
        r"(?im)^\s*abstract\s*$"
    )
    abstract_match = abstract_pattern.search(text)
    if abstract_match:
        content_start = abstract_match.end()
        while (
            content_start < len(text)
            and text[content_start].isspace()
        ):
            content_start += 1
    else:
        content_start = 0

    # Determine the reference part
    references_pattern = re.compile(
        r"(?im)^\s*"
        r"(references|bibliography|works\s+cited)"
        r"\s*$"
    )
    references_match = references_pattern.search(
        text,
        pos=content_start,
    )
    if references_match:
        content_end = references_match.start()
    else:
        content_end = len(text)

    return content_start, content_end

def is_reference_marker(text):
    '''
    Check whether a sentence is only a citation/reference marker.
    '''

    text = text.strip()
    normalized = re.sub(r"\s*[.,;:!?]+\s*$", "", text).strip()

    pattern = re.compile(
        r"^(\[\s*\d+(\s*[-,]\s*\d+)*\s*\]|\(\s*\d+(\s*[-,]\s*\d+)*\s*\))$"
    )

    return bool(pattern.match(normalized))


def split_sentences(text):
    '''
    Split document into sentences using spaCy.

    Sentence offsets are always measured against
    the original document.
    '''

    content_start, content_end = find_content_range(text)
    content = text[content_start:content_end]

    doc = nlp(content)
    sentences = []

    for sent in doc.sents:
        raw_text = sent.text
        sentence = raw_text.strip()

        if not sentence:
            continue

        if is_reference_marker(sentence):
            continue

        left_trim = (len(raw_text) - len(raw_text.lstrip()))
        right_trim = (len(raw_text) - len(raw_text.rstrip()))
        offset_start = (content_start + sent.start_char + left_trim)
        offset_end = (content_start + sent.end_char - right_trim)

        sentences.append({
            "text": text[offset_start:offset_end],
            "offset_start": offset_start,
            "offset_end": offset_end,
            "length": offset_end - offset_start,
        })

    return sentences
# ===== embedding.py (Khôi) =====
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = "intfloat/e5-base-v2"

def encode_sentences(
    sentences: Sequence[dict],
    model: SentenceTransformer,
    prefix: str = "passage:",
) -> np.ndarray:
    '''
    Encode sentences into dense semantic embeddings.

    Args:
        sentences: Output from preprocess.split_sentences().
        model: SentenceTransformer model.
        prefix: Use "query" for suspisous and "passage" for source
    '''
    texts = [
        f"{prefix} {sentence['text']}"
        for sentence in sentences
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(embeddings, dtype=np.float32)

def load_embedding_model(model_name: str = MODEL_NAME, device: str | None = None) -> SentenceTransformer:
    '''
    Load the sentence embedding model.
    '''
    return SentenceTransformer(
        model_name,
        device=device,
    )
# ===== similarity.py (Khôi) =====
import numpy as np

def cosine_similarity(
    suspicious_embeddings: np.ndarray,
    source_embeddings: np.ndarray,
) -> np.ndarray:
    '''
    Compute cosine similarity between all suspicious and source sentences.
    Embeddings are expected to be normalized.
    '''
    return suspicious_embeddings @ source_embeddings.T

def get_top_k_matches(
    similarity_matrix: np.ndarray,
    threshold: float,
    top_k: int = 5,
) -> list[dict]:
    '''
    Get top-k source sentence matches for each suspicious sentence with given threshold
    '''
    matches = []

    num_suspicious = similarity_matrix.shape[0]

    for susp_idx in range(num_suspicious):
        scores = similarity_matrix[susp_idx]

        top_indices = np.argsort(scores)[-top_k:][::-1]

        for source_idx in top_indices:
            score = float(scores[source_idx])

            if score > threshold:
                matches.append({
                    "suspicious_idx": susp_idx,
                    "source_idx": int(source_idx),
                    "score": score,
                })

    return matches
# ===== alignment.py (Khôi) =====
def group_suspicious_matches(matches, max_suspicious_gap: int = 2):
    '''
    Group matches by suspicious sentence position.

    Suspicious sentences belong to the same group when
    the distance between consecutive suspicious indices
    is <= max_suspicious_gap.

    Example:
        S0, S8, S17, S18, S36, S60, S61, S62

    becomes:
        [S0]
        [S8]
        [S17, S18]
        [S36]
        [S60, S61, S62]
    '''

    if not matches:
        return []

    matches_by_suspicious = {}

    for match in matches:
        suspicious_idx = match["suspicious_idx"]

        if suspicious_idx not in matches_by_suspicious:
            matches_by_suspicious[suspicious_idx] = []

        matches_by_suspicious[suspicious_idx].append(match)

    suspicious_indices = sorted(matches_by_suspicious.keys())
    groups = []
    current_group = [suspicious_indices[0]]

    for suspicious_idx in suspicious_indices[1:]:
        previous_idx = current_group[-1]

        if (suspicious_idx - previous_idx <= max_suspicious_gap):
            current_group.append(suspicious_idx)
        else:
            groups.append({
                "suspicious_indices": current_group,
                "matches": [
                    match
                    for idx in current_group
                    for match in matches_by_suspicious[idx]
                ],
            })
            current_group = [suspicious_idx]

    groups.append({
        "suspicious_indices": current_group,
        "matches": [
            match
            for idx in current_group
            for match in matches_by_suspicious[idx]
        ],
    })

    return groups


def merge_source_matches(matches, max_source_gap: int = 5):
    '''
    Merge source matches that are close to each other
    for the same suspicious sentence.

    Example:
        S8 -> T206
        S8 -> T207
        S8 -> T254
    becomes:
        S8 -> [T206, T207]
        S8 -> [T254]
    '''

    if not matches:
        return []

    matches_by_suspicious = {}

    for match in matches:
        suspicious_idx = (match["suspicious_idx"])

        if suspicious_idx not in matches_by_suspicious:
            matches_by_suspicious[suspicious_idx] = []

        matches_by_suspicious[suspicious_idx].append(match)

    merged = []

    for suspicious_idx in sorted(matches_by_suspicious.keys()):
        suspicious_matches = sorted(
            matches_by_suspicious[
                suspicious_idx
            ],
            key=lambda match: match[
                "source_idx"
            ],
        )

        current = [suspicious_matches[0]]

        for match in suspicious_matches[1:]:
            previous = current[-1]
            source_gap = (match["source_idx"] - previous["source_idx"])

            if source_gap <= max_source_gap:
                current.append(match)

            else:
                merged.append({
                    "suspicious_idx": suspicious_idx,
                    "source_start": current[0][
                        "source_idx"
                    ],
                    "source_end": current[-1][
                        "source_idx"
                    ],
                    "matches": current,
                    "score": sum(
                        item["score"]
                        for item in current
                    ) / len(current),
                })

                current = [match]

        merged.append({
            "suspicious_idx": suspicious_idx,
            "source_start": current[0][
                "source_idx"
            ],
            "source_end": current[-1][
                "source_idx"
            ],
            "matches": current,
            "score": sum(
                item["score"]
                for item in current
            ) / len(current),
        })

    return merged


def build_source_paths(
    suspicious_group,
    max_source_gap: int = 5,
):
    '''
    Build source paths inside one suspicious group.

    A path connects source groups from consecutive
    suspicious sentences when their source positions
    are increasing and close enough.

    Example:
        S60 -> 290
        S61 -> 291
        S62 -> 292

    creates:
        S60 -> T290
        S61 -> T291
        S62 -> T292
    '''

    suspicious_indices = (suspicious_group["suspicious_indices"])
    matches = suspicious_group["matches"]

    source_groups = merge_source_matches(
        matches,
        max_source_gap=max_source_gap,
    )

    groups_by_suspicious = {}
    for source_group in source_groups:
        suspicious_idx = (source_group["suspicious_idx"])

        if (
            suspicious_idx
            not in groups_by_suspicious
        ):
            groups_by_suspicious[
                suspicious_idx
            ] = []

        groups_by_suspicious[
            suspicious_idx
        ].append(source_group)

    paths = []
    first_suspicious = suspicious_indices[0]

    for source_group in groups_by_suspicious.get(first_suspicious, []):
        paths.append({
            "suspicious_indices": [
                first_suspicious
            ],
            "source_groups": [
                source_group
            ],
            "score": source_group["score"],
        })

    for suspicious_idx in suspicious_indices[1:]:

        candidates = groups_by_suspicious.get(
            suspicious_idx,
            [],
        )

        new_paths = []

        for path in paths:

            previous_group = path[
                "source_groups"
            ][-1]

            previous_source_end = (
                previous_group["source_end"]
            )

            extended = False

            for source_group in candidates:

                current_source_start = (
                    source_group["source_start"]
                )

                source_gap = (
                    current_source_start
                    - previous_source_end
                )

                # Source position must move forward.
                if current_source_start <= (
                    previous_source_end
                ):
                    continue

                # Source groups must be close.
                if source_gap > max_source_gap:
                    continue

                new_path = {
                    "suspicious_indices": (
                        path[
                            "suspicious_indices"
                        ]
                        + [suspicious_idx]
                    ),
                    "source_groups": (
                        path[
                            "source_groups"
                        ]
                        + [source_group]
                    ),
                }

                scores = [
                    group["score"]
                    for group in new_path[
                        "source_groups"
                    ]
                ]

                new_path["score"] = (
                    sum(scores) / len(scores)
                )

                new_paths.append(
                    new_path
                )

                extended = True

            # If this path cannot continue,
            # keep the old path as it is.
            if not extended:
                new_paths.append(path)

        paths = new_paths

    return paths


def select_best_paths(paths, min_path_length: int = 2):
    '''
    Select the best source path(s).

    Longer paths are preferred first.
    Among paths with the same length,
    higher average similarity is preferred.

    Only the best path is returned for now.
    '''

    if not paths:
        return []

    valid_paths = [
        path
        for path in paths
        if len(
            path["suspicious_indices"]
        ) >= min_path_length
    ]

    if not valid_paths:
        return []

    valid_paths = sorted(
        valid_paths,
        key=lambda path: (
            len(
                path[
                    "suspicious_indices"
                ]
            ),
            path["score"],
        ),
        reverse=True,
    )

    best_path = valid_paths[0]

    return [best_path]


def align_matches(
    matches,
    max_suspicious_gap: int = 2,
    max_source_gap: int = 5,
    min_path_length: int = 2,
):
    '''
    Perform source-path based alignment.

    Pipeline:
        candidate matches
            ->
        suspicious grouping
            ->
        source grouping
            ->
        source path construction
            ->
        best path selection

    Returns:
        List of alignment paths.
    '''

    if not matches:
        return []

    suspicious_groups = group_suspicious_matches(
        matches,
        max_suspicious_gap=max_suspicious_gap,
    )

    alignments = []

    for suspicious_group in suspicious_groups:

        paths = build_source_paths(
            suspicious_group,
            max_source_gap=max_source_gap,
        )

        best_paths = select_best_paths(
            paths,
            min_path_length=min_path_length,
        )

        if len(
            suspicious_group[
                "suspicious_indices"
            ]
        ) == 1:

            source_groups = (
                merge_source_matches(
                    suspicious_group["matches"],
                    max_source_gap=max_source_gap,
                )
            )

            if source_groups:

                best_source_group = max(
                    source_groups,
                    key=lambda group: group[
                        "score"
                    ],
                )

                alignments.append({
                    "suspicious_indices": (
                        suspicious_group[
                            "suspicious_indices"
                        ]
                    ),
                    "source_groups": [
                        best_source_group
                    ],
                    "score": (
                        best_source_group["score"]
                    ),
                })

            continue

        alignments.extend(best_paths)

    return alignments


def alignments_to_spans(
    alignments,
    suspicious_sentences,
    source_sentences,
):
    '''
    Convert alignment paths into character-level spans.

    Sentence indices are converted to character offsets
    using the sentence metadata.
    '''

    spans = []

    for alignment in alignments:

        suspicious_indices = alignment[
            "suspicious_indices"
        ]

        source_groups = alignment[
            "source_groups"
        ]

        if not suspicious_indices:
            continue

        if not source_groups:
            continue

        suspicious_start = (
            suspicious_sentences[
                suspicious_indices[0]
            ]["offset_start"]
        )

        suspicious_end = (
            suspicious_sentences[
                suspicious_indices[-1]
            ]["offset_end"]
        )


        source_start_idx = (
            source_groups[0]["source_start"]
        )

        source_end_idx = (
            source_groups[-1]["source_end"]
        )

        source_start = (
            source_sentences[
                source_start_idx
            ]["offset_start"]
        )

        source_end = (
            source_sentences[
                source_end_idx
            ]["offset_end"]
        )

        all_matches = []

        for source_group in source_groups:
            all_matches.extend(
                source_group["matches"]
            )

        spans.append({
            "suspicious_start": suspicious_start,
            "suspicious_length": (
                suspicious_end
                - suspicious_start
            ),

            "source_start": source_start,
            "source_length": (
                source_end
                - source_start
            ),

            "score": alignment["score"],

            "suspicious_indices": (
                suspicious_indices
            ),

            "source_groups": source_groups,

            "matches": all_matches,
        })

    return spans
# ================= PlagDet (PAN 2015, mức ký tự) =================
import math
from collections import defaultdict

class Span:
    __slots__ = ("doc", "start", "length")
    def __init__(self, doc, start, length):
        self.doc, self.start, self.length = doc, start, length
    @property
    def end(self):
        return self.start + self.length

def _overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))

def _union_within(t, others):
    subs = []
    for o in others:
        s = max(t.start, o.start); e = min(t.end, o.end)
        if e > s:
            subs.append((s, e))
    if not subs:
        return 0
    subs.sort(); tot = 0; cs, ce = subs[0]
    for s, e in subs[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs; cs, ce = s, e
    return tot + ce - cs

def plagdet_score(truth, pred):
    R = [s for s in truth if s.length > 0]
    S = [s for s in pred if s.length > 0]
    Rb, Sb = defaultdict(list), defaultdict(list)
    for s in R: Rb[s.doc].append(s)
    for s in S: Sb[s.doc].append(s)
    if not R:
        recall = 1.0
    else:
        recall = sum(_union_within(r, Sb.get(r.doc, [])) / r.length for r in R) / len(R)
    if not S:
        precision = 1.0
    else:
        precision = sum(_union_within(s, Rb.get(s.doc, [])) / s.length for s in S) / len(S)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    counts = []
    for r in R:
        c = sum(1 for s in Sb.get(r.doc, []) if _overlap(r.start, r.end, s.start, s.end) > 0)
        if c > 0: counts.append(c)
    gran = 1.0 if not counts else sum(counts) / len(counts)
    pd = f1 / math.log2(1 + gran) if gran >= 1 else f1
    return dict(precision=precision, recall=recall, f1=f1, granularity=gran, plagdet=pd,
                n_truth=len(R), n_pred=len(S))

EVAL_SUSP = ["suspicious-document010237.txt", "suspicious-document010241.txt", "suspicious-document010242.txt", "suspicious-document010246.txt", "suspicious-document010248.txt", "suspicious-document010249.txt", "suspicious-document010250.txt", "suspicious-document010253.txt", "suspicious-document010254.txt", "suspicious-document010255.txt", "suspicious-document010256.txt", "suspicious-document010258.txt", "suspicious-document010259.txt", "suspicious-document010261.txt", "suspicious-document010263.txt", "suspicious-document010267.txt", "suspicious-document010270.txt", "suspicious-document010273.txt", "suspicious-document010274.txt", "suspicious-document010275.txt", "suspicious-document010284.txt", "suspicious-document010285.txt", "suspicious-document010287.txt", "suspicious-document010288.txt", "suspicious-document010291.txt", "suspicious-document010294.txt", "suspicious-document010299.txt", "suspicious-document010301.txt", "suspicious-document010303.txt", "suspicious-document010304.txt", "suspicious-document010306.txt", "suspicious-document010307.txt", "suspicious-document010308.txt", "suspicious-document010309.txt", "suspicious-document010310.txt", "suspicious-document010311.txt", "suspicious-document010312.txt", "suspicious-document010314.txt", "suspicious-document010315.txt", "suspicious-document010316.txt", "suspicious-document010321.txt", "suspicious-document010325.txt", "suspicious-document010326.txt", "suspicious-document010327.txt", "suspicious-document010331.txt", "suspicious-document010332.txt", "suspicious-document010333.txt", "suspicious-document010334.txt", "suspicious-document010336.txt", "suspicious-document010339.txt", "suspicious-document010341.txt", "suspicious-document010342.txt", "suspicious-document010343.txt", "suspicious-document010344.txt", "suspicious-document010348.txt", "suspicious-document010350.txt", "suspicious-document010356.txt", "suspicious-document010357.txt", "suspicious-document010358.txt", "suspicious-document010361.txt", "suspicious-document010364.txt", "suspicious-document010365.txt", "suspicious-document010366.txt", "suspicious-document010367.txt", "suspicious-document010368.txt", "suspicious-document010379.txt", "suspicious-document010380.txt", "suspicious-document010383.txt", "suspicious-document010384.txt", "suspicious-document010388.txt", "suspicious-document010389.txt", "suspicious-document010390.txt", "suspicious-document010392.txt", "suspicious-document010393.txt", "suspicious-document010394.txt", "suspicious-document010395.txt", "suspicious-document010396.txt", "suspicious-document010398.txt", "suspicious-document010399.txt", "suspicious-document010401.txt", "suspicious-document010403.txt", "suspicious-document010404.txt", "suspicious-document010405.txt", "suspicious-document010408.txt", "suspicious-document010409.txt", "suspicious-document010411.txt", "suspicious-document010413.txt", "suspicious-document010416.txt", "suspicious-document010417.txt", "suspicious-document010418.txt", "suspicious-document010420.txt", "suspicious-document010421.txt", "suspicious-document010423.txt", "suspicious-document010425.txt", "suspicious-document010428.txt", "suspicious-document010429.txt", "suspicious-document010430.txt", "suspicious-document010433.txt", "suspicious-document010435.txt", "suspicious-document010436.txt", "suspicious-document010437.txt", "suspicious-document010441.txt", "suspicious-document010443.txt", "suspicious-document010444.txt", "suspicious-document010446.txt", "suspicious-document010447.txt", "suspicious-document010448.txt", "suspicious-document010449.txt", "suspicious-document010450.txt", "suspicious-document010452.txt", "suspicious-document010453.txt", "suspicious-document010454.txt", "suspicious-document010457.txt", "suspicious-document010461.txt", "suspicious-document010464.txt", "suspicious-document010466.txt", "suspicious-document010467.txt", "suspicious-document010468.txt", "suspicious-document010469.txt", "suspicious-document010470.txt", "suspicious-document010472.txt", "suspicious-document010473.txt", "suspicious-document010474.txt", "suspicious-document010476.txt", "suspicious-document010481.txt", "suspicious-document010486.txt", "suspicious-document010493.txt", "suspicious-document010494.txt", "suspicious-document010495.txt", "suspicious-document010498.txt", "suspicious-document010500.txt", "suspicious-document010502.txt", "suspicious-document010505.txt", "suspicious-document010506.txt", "suspicious-document010509.txt", "suspicious-document010511.txt", "suspicious-document010512.txt", "suspicious-document010515.txt", "suspicious-document010516.txt", "suspicious-document010517.txt", "suspicious-document010518.txt", "suspicious-document010519.txt", "suspicious-document010521.txt", "suspicious-document010524.txt", "suspicious-document010525.txt", "suspicious-document010526.txt", "suspicious-document010530.txt", "suspicious-document010531.txt", "suspicious-document010532.txt", "suspicious-document010533.txt"]

# ================= DRIVER: retrieval TF-IDF top-1 -> align (Khôi) -> PlagDet =================
import os, glob, json, time, hashlib
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import torch

def _find(*names, root="/kaggle/input"):
    for n in names:
        hits = glob.glob(os.path.join(root, "**", n), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]
    return None

def eval_set_id(names):
    return hashlib.sha1("\n".join(sorted(names)).encode("utf-8")).hexdigest()[:12]

def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    esid = eval_set_id(EVAL_SUSP)
    print(f"device={dev} | eval_set_id={esid} | n_susp={len(EVAL_SUSP)}", flush=True)

    susp_dir = _find("susp")            # docs/susp
    src_dir = _find("src")              # docs/src
    labels = _find("val_spans.csv")
    print("susp_dir:", susp_dir, "| src_dir:", src_dir, "| labels:", labels, flush=True)

    # gold (chỉ cho eval susp)
    import csv
    want = set(EVAL_SUSP)
    gold_by = {}
    with open(labels, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["feature"] == "plagiarism" and r["suspicious_reference"] in want:
                gold_by.setdefault(r["suspicious_reference"], []).append(
                    (int(r["this_offset"]), int(r["this_length"])))

    # retrieval: TF-IDF toàn bộ src, top-1
    src_files = sorted(glob.glob(os.path.join(src_dir, "*.txt")))
    src_ids = [os.path.basename(p) for p in src_files]
    print(f"[retrieval] {len(src_files)} nguồn, fit TF-IDF...", flush=True)
    src_full = [open(p, encoding="utf-8", errors="ignore").read() for p in src_files]
    vec = TfidfVectorizer(max_features=100000, sublinear_tf=True, stop_words="english")
    src_m = vec.fit_transform([t[:20000] for t in src_full])
    susp_texts = [open(os.path.join(susp_dir, su), encoding="utf-8", errors="ignore").read()
                  for su in EVAL_SUSP]
    susp_m = vec.transform([t[:20000] for t in susp_texts])
    sims = sk_cosine(susp_m, src_m)
    top1 = np.argmax(sims, axis=1)
    print(f"[retrieval] xong ({time.time()-t0:.0f}s). Tải E5-base-v2...", flush=True)

    model = load_embedding_model("intfloat/e5-base-v2", device=dev)

    gold, pred = [], []
    for i, (su, stext) in enumerate(zip(EVAL_SUSP, susp_texts), 1):
        for off, ln in gold_by.get(su, []):
            gold.append(Span(su, off, ln))
        rtext = src_full[top1[i - 1]]
        ss, rs = split_sentences(stext), split_sentences(rtext)
        if ss and rs:
            se = encode_sentences(ss, model, prefix="query:")
            re_ = encode_sentences(rs, model, prefix="passage:")
            matches = get_top_k_matches(cosine_similarity(se, re_), threshold=0.86, top_k=5)
            for p in alignments_to_spans(align_matches(matches, 2, 5, 2), ss, rs):
                pred.append(Span(su, p["suspicious_start"], p["suspicious_length"]))
        if i % 25 == 0 or i == len(EVAL_SUSP):
            print(f"  align {i}/{len(EVAL_SUSP)} ({time.time()-t0:.0f}s)", flush=True)

    whole = [Span(su, 0, len(t)) for su, t in zip(EVAL_SUSP, susp_texts)]
    r = plagdet_score(gold, pred)
    b = plagdet_score(gold, whole)
    runtime = time.time() - t0
    print(f"\n=== KHOI E5+path (val, {len(EVAL_SUSP)} susp, {len(gold)} gold) ===")
    print(f"baseline cả-doc: PlagDet={b['plagdet']:.3f}")
    print(f"khoi           : P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
          f"gran={r['granularity']:.3f} PlagDet={r['plagdet']:.3f}  ({runtime:.0f}s)")

    ts = datetime.now().isoformat(timespec="seconds")
    def rec(method, metrics, kind, topk, params, notes):
        return {"method": method, "kind": kind, "split": "val", "subset": len(EVAL_SUSP),
                "topk": topk, "eval_set_id": esid, "params": params,
                "metrics": {k: round(v, 4) for k, v in metrics.items()
                            if k in ("plagdet", "precision", "recall", "f1", "granularity")},
                "runtime_sec": round(runtime, 1), "timestamp": ts, "notes": notes}
    json.dump(rec("tfidf+e5-path (Khôi)", r, "method", 1,
                  {"model": "e5-base-v2", "threshold": 0.86, "top_k": 5},
                  "Kaggle T4 GPU; E5 threshold=0.86; source-path align"),
              open("/kaggle/working/result_khoi.json", "w"), ensure_ascii=False, indent=2)
    json.dump(rec("baseline (whole-doc)", b, "baseline", None, {}, "gán cả tài liệu là đạo văn"),
              open("/kaggle/working/result_baseline.json", "w"), ensure_ascii=False, indent=2)
    print("-> /kaggle/working/result_khoi.json + result_baseline.json")

if __name__ == "__main__":
    main()
