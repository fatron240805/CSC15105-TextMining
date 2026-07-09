#!/usr/bin/env python3
"""Parse PAN 2025 plagiarism-detection ground-truth XML into structured data.

Task A0-3 (Phase 0). Reads the ``*_truth`` annotation files of the PAN 2025
"Generated Plagiarism Detection" dataset and emits two structured views:

* **JSONL** (``--out-jsonl``): one record per suspicious/source *pair* — the rich,
  nested form. Keeps document-level metadata (title, authors, similarity,
  severity) plus the full list of annotated spans.
* **CSV** (``--out-csv``): one row per *span* — the flat form for pandas / EDA
  (task A0-4). Every ``plagiarism`` and ``altered`` feature becomes a row.

Ground-truth XML schema (one file per pair, e.g.
``suspicious-document020468-source-document020468.xml``)::

    <document reference="suspicious-document020468.txt">
      <feature name="about"   title="..." authors="A and B" similarity="0.99"
                              severity="medium" prompt_tokens=".." output_tokens=".."/>
      <feature name="md5Hash" value="..."/>
      <feature name="plagiarism" type="llm_prompted" llm="DeepSeek-R1"
               this_offset="117" this_length="1465"
               source_reference="source-document020468.txt"
               source_offset="82" source_length="1677" obfuscation="simple"/>
      <feature name="altered"    type="llm_prompted" llm="DeepSeek-R1"
               this_offset="17974" this_length="301"/>
      ...
    </document>

Key facts verified against the data (do NOT assume otherwise):

* Offsets/lengths are **character** positions into the UTF-8-decoded ``.txt``
  file (not byte offsets). The train/validation ``.txt`` files use LF newlines,
  but we still read with ``newline=''`` so a stray CRLF file cannot shift spans.
* ``plagiarism`` features carry both ``this_*`` (in the suspicious doc) and
  ``source_*`` (in the source doc). ``altered`` features are LLM-generated text
  with **no source**, so they carry only ``this_*``.
* The source document is taken from each feature's ``source_reference`` attribute
  and the suspicious document from ``<document reference=...>`` — NOT inferred
  from the filename, because susp-id and source-id can differ (e.g.
  ``suspicious-document020491-source-document052407.xml``).

Usage
-----
    # Sanity-check on the tiny spot-check set, printing a few extracted spans:
    python scripts/parse_labels.py \
        --truth-dir "C:/github/PAN2025/00_spot_check/00_spot_check_truth" \
        --docs-dir  "C:/github/PAN2025/00_spot_check/00_spot_check" \
        --out-jsonl outputs/spot_labels.jsonl \
        --out-csv   outputs/spot_spans.csv \
        --verify 3

    # Full train split (no span extraction, faster):
    python scripts/parse_labels.py \
        --truth-dir ".../pan25-generated-plagiarism-detection-train/01_train/01_train_truth" \
        --out-jsonl outputs/train_labels.jsonl \
        --out-csv   outputs/train_spans.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Span:
    """A single annotated passage (one ``plagiarism`` or ``altered`` feature)."""

    feature: str                 # "plagiarism" | "altered"
    this_offset: int             # char offset into the suspicious document
    this_length: int
    source_reference: Optional[str] = None   # source .txt filename (plagiarism only)
    source_offset: Optional[int] = None
    source_length: Optional[int] = None
    obfuscation: Optional[str] = None        # "simple" | "medium" | "hard" (plagiarism only)
    llm: Optional[str] = None
    this_language: Optional[str] = None


@dataclass
class PairRecord:
    """All ground-truth annotations for one suspicious/source pair."""

    xml_file: str
    suspicious_reference: str                 # from <document reference=...>
    source_references: list = field(default_factory=list)  # distinct sources cited
    # Document-level metadata from the <feature name="about"> element:
    title: Optional[str] = None
    authors: Optional[str] = None
    similarity: Optional[float] = None
    severity: Optional[str] = None            # doc-level Low/Medium/High proxy
    md5hash: Optional[str] = None
    n_plagiarism: int = 0
    n_altered: int = 0
    spans: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _to_int(v: Optional[str]) -> Optional[int]:
    return int(v) if v is not None and v != "" else None


def _to_float(v: Optional[str]) -> Optional[float]:
    try:
        return float(v) if v is not None and v != "" else None
    except ValueError:
        return None


def parse_truth_xml(xml_path: Path) -> PairRecord:
    """Parse one ground-truth XML file into a :class:`PairRecord`."""
    root = ET.parse(xml_path).getroot()
    rec = PairRecord(
        xml_file=xml_path.name,
        suspicious_reference=root.get("reference", ""),
    )
    sources: set = set()

    for feat in root.findall("feature"):
        name = feat.get("name")

        if name == "about":
            rec.title = feat.get("title")
            rec.authors = feat.get("authors")
            rec.similarity = _to_float(feat.get("similarity"))
            rec.severity = feat.get("severity")

        elif name == "md5Hash":
            rec.md5hash = feat.get("value")

        elif name in ("plagiarism", "altered"):
            span = Span(
                feature=name,
                this_offset=_to_int(feat.get("this_offset")) or 0,
                this_length=_to_int(feat.get("this_length")) or 0,
                this_language=feat.get("this_language"),
                llm=feat.get("llm"),
            )
            if name == "plagiarism":
                span.source_reference = feat.get("source_reference")
                span.source_offset = _to_int(feat.get("source_offset"))
                span.source_length = _to_int(feat.get("source_length"))
                span.obfuscation = feat.get("obfuscation")
                if span.source_reference:
                    sources.add(span.source_reference)
                rec.n_plagiarism += 1
            else:
                rec.n_altered += 1
            rec.spans.append(span)
        # Any other feature name is metadata we don't model; ignore quietly.

    rec.source_references = sorted(sources)
    return rec


def iter_truth_files(truth_dir: Path) -> Iterator[Path]:
    """Yield ground-truth XML files (skips metadata.json, desktop.ini, ...)."""
    for p in sorted(truth_dir.glob("*.xml")):
        yield p


# --------------------------------------------------------------------------- #
# Span extraction / verification
# --------------------------------------------------------------------------- #
def read_text(path: Path) -> str:
    """Read a document as text with newlines preserved (char-offset faithful).

    ``newline=''`` disables universal-newline translation so a CRLF file is not
    silently collapsed, which would shift every subsequent character offset.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def extract_span(text: str, offset: int, length: int) -> str:
    return text[offset:offset + length]


def verify_spans(records: list, docs_dir: Path, n: int) -> None:
    """Extract and print up to ``n`` spans so offsets can be eyeballed."""
    susp_dir = docs_dir / "susp"
    src_dir = docs_dir / "src"
    shown = 0
    print(f"\n--- span verification (up to {n}) -------------------------------")
    for rec in records:
        if shown >= n:
            break
        susp_path = susp_dir / rec.suspicious_reference
        if not susp_path.exists():
            continue
        susp_text = read_text(susp_path)
        for span in rec.spans:
            if shown >= n:
                break
            snippet = extract_span(susp_text, span.this_offset, span.this_length)
            print(f"\n[{rec.xml_file}] {span.feature} "
                  f"this_offset={span.this_offset} len={span.this_length}")
            print(f"  SUSP: {snippet[:160]!r}...")
            if span.feature == "plagiarism" and span.source_reference:
                src_path = src_dir / span.source_reference
                if src_path.exists():
                    src_text = read_text(src_path)
                    src_snippet = extract_span(
                        src_text, span.source_offset or 0, span.source_length or 0)
                    print(f"  SRC : {src_snippet[:160]!r}...")
            shown += 1
    print("-----------------------------------------------------------------\n")


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_jsonl(records: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            d = asdict(rec)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


CSV_FIELDS = [
    "xml_file", "suspicious_reference", "source_reference",
    "feature", "obfuscation", "severity", "similarity", "llm",
    "this_offset", "this_length", "source_offset", "source_length",
]


def write_csv(records: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for rec in records:
            for span in rec.spans:
                w.writerow({
                    "xml_file": rec.xml_file,
                    "suspicious_reference": rec.suspicious_reference,
                    "source_reference": span.source_reference or "",
                    "feature": span.feature,
                    "obfuscation": span.obfuscation or "",
                    "severity": rec.severity or "",
                    "similarity": rec.similarity if rec.similarity is not None else "",
                    "llm": span.llm or "",
                    "this_offset": span.this_offset,
                    "this_length": span.this_length,
                    "source_offset": span.source_offset if span.source_offset is not None else "",
                    "source_length": span.source_length if span.source_length is not None else "",
                })


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth-dir", required=True, type=Path,
                    help="Directory of ground-truth *.xml files (a *_truth folder).")
    ap.add_argument("--docs-dir", type=Path, default=None,
                    help="Directory containing susp/ and src/ (needed only for --verify).")
    ap.add_argument("--out-jsonl", type=Path, default=None,
                    help="Write one JSON record per pair to this path.")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Write one CSV row per span to this path.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Parse at most N XML files (for quick tests).")
    ap.add_argument("--verify", type=int, default=0,
                    help="Extract & print N spans to sanity-check offsets "
                         "(requires --docs-dir).")
    args = ap.parse_args(argv)

    if not args.truth_dir.is_dir():
        ap.error(f"--truth-dir not found: {args.truth_dir}")
    if args.verify and args.docs_dir is None:
        ap.error("--verify requires --docs-dir")

    records: list = []
    n_spans = n_plag = n_alt = 0
    for i, xml_path in enumerate(iter_truth_files(args.truth_dir)):
        if args.limit is not None and i >= args.limit:
            break
        try:
            rec = parse_truth_xml(xml_path)
        except ET.ParseError as e:
            print(f"WARN: skipping malformed XML {xml_path.name}: {e}",
                  file=sys.stderr)
            continue
        records.append(rec)
        n_spans += len(rec.spans)
        n_plag += rec.n_plagiarism
        n_alt += rec.n_altered
        if (i + 1) % 5000 == 0:
            print(f"  parsed {i + 1} files...", file=sys.stderr)

    print(f"Parsed {len(records)} pairs | {n_spans} spans "
          f"({n_plag} plagiarism, {n_alt} altered)")

    if args.verify:
        verify_spans(records, args.docs_dir, args.verify)
    if args.out_jsonl:
        write_jsonl(records, args.out_jsonl)
        print(f"Wrote JSONL -> {args.out_jsonl}")
    if args.out_csv:
        write_csv(records, args.out_csv)
        print(f"Wrote CSV   -> {args.out_csv}")
    if not (args.out_jsonl or args.out_csv or args.verify):
        print("(No output requested. Use --out-jsonl / --out-csv / --verify.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
