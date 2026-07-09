#!/usr/bin/env python3
"""Check integrity of a PAN 2025 plagiarism-detection split.

Task A0-2 (Phase 0). Verifies that a split's ``susp/``, ``src/`` and ``pairs``
are internally consistent and that the ground truth lines up with the documents:

* every susp/src file named in ``pairs`` actually exists on disk (no dangling
  references);
* no document on disk is left out of every pair (no orphans);
* the number of ground-truth XMLs equals the number of ``pairs`` lines
  (one annotation per pair);
* reports how many susp/src docs are *reused* across multiple pairs, which is the
  benign reason ``len(pairs)`` exceeds the unique-file counts (one suspicious doc
  can plagiarise several sources, and one source can be cited by several susps).

Exit code is non-zero if any hard integrity violation is found (missing files,
orphans, or a pairs/XML count mismatch), so it can gate a pipeline.

Usage
-----
    python scripts/check_integrity.py \
        --docs-dir  ".../02_validation/02_validation" \
        --truth-dir ".../02_validation/02_validation_truth"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def load_pairs(pairs_fp: Path) -> list:
    pairs = []
    with open(pairs_fp, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def check_split(docs_dir: Path, truth_dir: Path) -> bool:
    """Return True if the split passes all hard integrity checks."""
    susp_dir, src_dir = docs_dir / "susp", docs_dir / "src"
    pairs_fp = docs_dir / "pairs"

    for p in (susp_dir, src_dir, pairs_fp, truth_dir):
        if not p.exists():
            print(f"  ERROR: expected path missing: {p}", file=sys.stderr)
            return False

    susp_files = {f.name for f in susp_dir.glob("*.txt")}
    src_files = {f.name for f in src_dir.glob("*.txt")}
    pairs = load_pairs(pairs_fp)
    xmls = list(truth_dir.glob("*.xml"))

    uniq_susp = {s for s, _ in pairs}
    uniq_src = {r for _, r in pairs}
    missing_susp = {s for s, _ in pairs if s not in susp_files}
    missing_src = {r for _, r in pairs if r not in src_files}
    orphan_susp = susp_files - uniq_susp
    orphan_src = src_files - uniq_src

    print(f"  susp/ on disk : {len(susp_files):>7}   src/ on disk : {len(src_files):>7}")
    print(f"  pairs lines   : {len(pairs):>7}   truth XMLs   : {len(xmls):>7}")
    print(f"  unique susp   : {len(uniq_susp):>7}   unique src   : {len(uniq_src):>7}")
    print(f"  susp reused   : {len(pairs) - len(uniq_susp):>7}   src reused   : {len(pairs) - len(uniq_src):>7}")
    print(f"  missing susp  : {len(missing_susp):>7}   missing src  : {len(missing_src):>7}")
    print(f"  orphan susp   : {len(orphan_susp):>7}   orphan src   : {len(orphan_src):>7}")

    ok = True
    if missing_susp:
        ok = False
        print(f"  FAIL: {len(missing_susp)} susp referenced by pairs but absent on disk, "
              f"e.g. {sorted(missing_susp)[:3]}", file=sys.stderr)
    if missing_src:
        ok = False
        print(f"  FAIL: {len(missing_src)} src referenced by pairs but absent on disk, "
              f"e.g. {sorted(missing_src)[:3]}", file=sys.stderr)
    if orphan_susp:
        ok = False
        print(f"  FAIL: {len(orphan_susp)} susp on disk not referenced by any pair", file=sys.stderr)
    if orphan_src:
        ok = False
        print(f"  FAIL: {len(orphan_src)} src on disk not referenced by any pair", file=sys.stderr)
    if len(xmls) != len(pairs):
        ok = False
        print(f"  FAIL: truth XML count ({len(xmls)}) != pairs count ({len(pairs)})", file=sys.stderr)

    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-dir", required=True, type=Path,
                    help="Split documents dir containing susp/, src/, pairs.")
    ap.add_argument("--truth-dir", required=True, type=Path,
                    help="Split ground-truth dir containing the *.xml files.")
    args = ap.parse_args(argv)

    print(f"Integrity check: {args.docs_dir}")
    ok = check_split(args.docs_dir, args.truth_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
