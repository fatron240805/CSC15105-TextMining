"""Build one immutable detector manifest shared by every rewrite model.

Input contract cho baseline rewrite đa-model (kaggle NVIDIA free-tier, xem
generation/experiments/nvidia_free.yaml): mỗi case chỉ đến từ align_pair (alignment
thuần) trên cặp (susp, nguồn GOLD). CỐ Ý không đi qua generation.verify / generation.classify
— hai vai trò đó là chú giải/phân xử case biên (xem generation/verify.py), không phải nơi
quyết định span nào được đưa vào manifest. Đừng nối verify.py vào đây làm gate; nếu cần lọc
case biên cho một benchmark khác, làm ở một bước rõ ràng riêng, đừng ẩn trong builder này.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from evaluation.generation_store import canonical_hash, resolve_repo_path, utc_now
from scripts.alignment.align_tfisf import align_pair


def _read(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def _dedupe_cases(cases: list[dict]) -> list[dict]:
    """Keep deterministic, non-overlapping suspicious spans across gold sources."""
    cases.sort(key=lambda c: (c["start"], -c["length"], c["source"]))
    kept = []
    occupied = []
    for case in cases:
        start, end = case["start"], case["start"] + case["length"]
        if any(not (end <= old_start or start >= old_end)
               for old_start, old_end in occupied):
            continue
        kept.append(case)
        occupied.append((start, end))
    return kept


def build_manifest(config: dict, *, force: bool = False) -> tuple[Path, dict]:
    exp = config["experiment"]
    data = config["data"]
    detector = config.get("detector", {})
    output = resolve_repo_path(exp["manifest"])
    if output.exists() and not force:
        raise FileExistsError(f"Manifest already exists: {output}. Use --force to rebuild.")

    spans_csv = resolve_repo_path(data["spans_csv"])
    validation_root = resolve_repo_path(data["validation_root"])
    susp_dir, src_dir = validation_root / "susp", validation_root / "src"
    for required in (spans_csv, susp_dir, src_dir):
        if not required.exists():
            raise FileNotFoundError(f"Required generation input not found: {required}")

    sources_by_doc = defaultdict(set)
    with open(spans_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("feature") == "plagiarism" and row.get("source_reference"):
                sources_by_doc[row["suspicious_reference"]].add(row["source_reference"])

    th = float(detector.get("th", 0.30))
    th3 = float(detector.get("th3", 0.50))
    max_gap = int(detector.get("max_gap", 4))
    min_chars = int(detector.get("min_plag_chars", 150))
    max_spans = int(exp.get("max_spans", 15))
    documents = []
    selected = 0
    partial_documents = 0

    for doc_name in sorted(sources_by_doc):
        if selected >= max_spans:
            break
        susp_path = susp_dir / doc_name
        if not susp_path.exists():
            continue
        suspicious_text = _read(susp_path)
        source_texts = {}
        detected = []
        for source_name in sorted(sources_by_doc[doc_name]):
            source_path = src_dir / source_name
            if not source_path.exists():
                continue
            source_text = _read(source_path)
            source_texts[source_name] = source_text
            spans = align_pair(suspicious_text, source_text, th, th, th3,
                               max_gap, min_chars)
            for start, length, source_start, source_length in spans:
                detected.append({
                    "start": start,
                    "length": length,
                    "source": source_name,
                    "source_start": source_start,
                    "source_length": source_length,
                })
        detected = _dedupe_cases(detected)
        if not detected:
            continue
        remaining = max_spans - selected
        chosen = detected[:remaining]
        if len(chosen) < len(detected):
            partial_documents += 1
        for case in chosen:
            source_text = source_texts[case["source"]]
            case["sample_id"] = canonical_hash({
                "doc": doc_name,
                "start": case["start"],
                "length": case["length"],
                "source": case["source"],
                "source_start": case["source_start"],
                "source_length": case["source_length"],
                "suspicious_text": suspicious_text[
                    case["start"]:case["start"] + case["length"]],
                "source_text": source_text[
                    case["source_start"]:case["source_start"] + case["source_length"]],
            })
        selected += len(chosen)
        documents.append({
            "type": "document",
            "doc": doc_name,
            "suspicious_text": suspicious_text,
            "sources": {name: source_texts[name] for name in sorted({c["source"] for c in chosen})},
            "cases": chosen,
            "partial_document": len(chosen) < len(detected),
        })

    if not documents:
        raise RuntimeError("Detector produced no cases; check paths and thresholds.")
    manifest_hash = canonical_hash(documents)
    metadata = {
        "type": "manifest",
        "version": 1,
        "created_at": utc_now(),
        "manifest_hash": manifest_hash,
        "n_documents": len(documents),
        "n_spans": selected,
        "partial_documents": partial_documents,
        "detector": {"th": th, "th3": th3, "max_gap": max_gap,
                     "min_plag_chars": min_chars},
        "selection": "sorted docs/sources; non-overlap; exact max_spans",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    return output, metadata


def load_manifest(path: str | Path) -> tuple[dict, list[dict]]:
    source = resolve_repo_path(path)
    with open(source, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records or records[0].get("type") != "manifest":
        raise ValueError(f"Invalid generation manifest: {source}")
    metadata, documents = records[0], records[1:]
    actual_hash = canonical_hash(documents)
    if actual_hash != metadata.get("manifest_hash"):
        raise ValueError("Manifest hash mismatch; file may have been modified.")
    return metadata, documents


def iter_samples(documents: list[dict]):
    for document in documents:
        for case in document["cases"]:
            yield document, case
