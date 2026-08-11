#!/usr/bin/env python3
"""K7 — Quét ngưỡng cosine E5 (Khôi) để cứu recall. Encode 1 lần/susp, sweep rẻ.

Cùng subset 1000 + eval_set_id với k6 -> chồng lên thẻ val.1000. Mỗi ngưỡng = 1 kết quả.
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
# ===== alignment.py (Khôi, + beam cap) =====
_BEAM = 64
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
        if len(paths) > _BEAM:
            paths = sorted(paths, key=lambda p: p.get('score', 0.0), reverse=True)[:_BEAM]

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
# ================= PlagDet =================
import math as _math
from collections import defaultdict as _dd

class Span:
    __slots__ = ("doc", "start", "length")
    def __init__(self, doc, start, length):
        self.doc, self.start, self.length = doc, start, length
    @property
    def end(self):
        return self.start + self.length

def _ov(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))

def _uw(t, others):
    subs = []
    for o in others:
        s = max(t.start, o.start); e = min(t.end, o.end)
        if e > s: subs.append((s, e))
    if not subs: return 0
    subs.sort(); tot = 0; cs, ce = subs[0]
    for s, e in subs[1:]:
        if s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    return tot + ce - cs

def plagdet_score(truth, pred):
    R = [s for s in truth if s.length > 0]; S = [s for s in pred if s.length > 0]
    Rb, Sb = _dd(list), _dd(list)
    for s in R: Rb[s.doc].append(s)
    for s in S: Sb[s.doc].append(s)
    recall = 1.0 if not R else sum(_uw(r, Sb.get(r.doc, [])) / r.length for r in R) / len(R)
    precision = 1.0 if not S else sum(_uw(s, Rb.get(s.doc, [])) / s.length for s in S) / len(S)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    counts = []
    for r in R:
        c = sum(1 for s in Sb.get(r.doc, []) if _ov(r.start, r.end, s.start, s.end) > 0)
        if c > 0: counts.append(c)
    gran = 1.0 if not counts else sum(counts) / len(counts)
    pd = f1 / _math.log2(1 + gran) if gran >= 1 else f1
    return dict(precision=precision, recall=recall, f1=f1, granularity=gran, plagdet=pd)

# ================= DRIVER: sweep ngưỡng E5 =================
import os, glob, json, time, hashlib, csv
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import torch

SUBSET = 1000
THRS = [0.78, 0.80, 0.82, 0.84, 0.86]

def _find(*names, root="/kaggle/input"):
    for n in names:
        hits = glob.glob(os.path.join(root, "**", n), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]
    return None

def esid(names):
    return hashlib.sha1("\n".join(sorted(names)).encode("utf-8")).hexdigest()[:12]

def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    susp_dir = _find("susp"); src_dir = _find("src"); labels = _find("val_spans.csv")
    print(f"device={dev} | susp={susp_dir} src={src_dir} labels={labels}", flush=True)

    gold_by = {}
    with open(labels, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["feature"] == "plagiarism":
                gold_by.setdefault(r["suspicious_reference"], []).append(
                    (int(r["this_offset"]), int(r["this_length"])))
    SUB = sorted(gold_by)[:SUBSET]
    eid = esid(SUB)
    print(f"SUB={len(SUB)} eval_set_id={eid} | THRS={THRS}", flush=True)

    src_files = sorted(glob.glob(os.path.join(src_dir, "*.txt")))
    print(f"[retrieval] {len(src_files)} nguồn, fit TF-IDF top-1...", flush=True)
    src_full = [open(p, encoding="utf-8", errors="ignore").read() for p in src_files]
    vec = TfidfVectorizer(max_features=100000, sublinear_tf=True, stop_words="english")
    src_m = vec.fit_transform([t[:20000] for t in src_full])
    susp_txt = {su: open(os.path.join(susp_dir, su), encoding="utf-8", errors="ignore").read() for su in SUB}
    susp_m = vec.transform([susp_txt[su][:20000] for su in SUB])
    sims = sk_cosine(susp_m, src_m)
    top1 = {su: int(np.argmax(sims[i])) for i, su in enumerate(SUB)}
    print(f"[retrieval] xong ({time.time()-t0:.0f}s). Tải E5, encode 1 lần + sweep...", flush=True)

    model = load_embedding_model("intfloat/e5-base-v2", device=dev)
    pred = {t: {} for t in THRS}
    for i, su in enumerate(SUB, 1):
        j = top1[su]
        ss, rs = split_sentences(susp_txt[su]), split_sentences(src_full[j])
        if ss and rs:
            se = encode_sentences(ss, model, prefix="query:")
            re_ = encode_sentences(rs, model, prefix="passage:")
            sim = cosine_similarity(se, re_)               # encode 1 lần, tái dùng cho mọi ngưỡng
            for t in THRS:
                m = get_top_k_matches(sim, threshold=t, top_k=5)
                spans = [(p["suspicious_start"], p["suspicious_length"])
                         for p in alignments_to_spans(align_matches(m, 2, 5, 2), ss, rs)
                         if p["suspicious_length"] > 0]
                pred[t][su] = spans
        else:
            for t in THRS:
                pred[t][su] = []
        if i % 100 == 0 or i == len(SUB):
            print(f"  {i}/{len(SUB)} ({time.time()-t0:.0f}s)", flush=True)

    gold = [Span(su, o, l) for su in SUB for (o, l) in gold_by[su]]
    ts = datetime.now().isoformat(timespec="seconds")
    runtime = time.time() - t0
    results = []
    for t in THRS:
        S = [Span(su, o, l) for su in SUB for (o, l) in pred[t][su]]
        r = plagdet_score(gold, S)
        results.append({"method": f"tfidf+e5-path th{t} (Khôi)", "kind": "method", "split": "val",
                        "subset": len(SUB), "topk": 1, "eval_set_id": eid,
                        "params": {"model": "e5-base-v2", "threshold": t, "top_k": 5},
                        "metrics": {k: round(v, 4) for k, v in r.items()},
                        "runtime_sec": round(runtime, 1), "timestamp": ts,
                        "notes": f"E5 sweep threshold={t}, source-path"})
        print(f"  th={t}: PlagDet={r['plagdet']:.3f} P={r['precision']:.3f} R={r['recall']:.3f}", flush=True)
    json.dump(results, open("/kaggle/working/results_sweep.json", "w"), ensure_ascii=False, indent=2)
    print(f"-> /kaggle/working/results_sweep.json ({runtime:.0f}s)")

if __name__ == "__main__":
    main()
