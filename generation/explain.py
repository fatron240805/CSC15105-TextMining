#!/usr/bin/env python3
"""Bước Generation (chữ G của RAG) — SINH bản "luận giải đạo văn" grounded.

Đây là mắt xích khiến hệ đúng nghĩa RAG: LLM đọc (đoạn nghi vấn + đoạn NGUỒN do
tầng Retrieval trả về) rồi SINH một bản luận giải bám hoàn toàn vào bằng chứng
truy hồi — nêu kỹ thuật đạo, giải thích, mức độ, và đề xuất viết lại khử đạo.

"Query" của người dùng = tài liệu nghi vấn (text/file); prompt gửi LLM do hệ tự
dựng: template cố định + đoạn nguồn truy hồi (augmented) + span. Người dùng KHÔNG
tự viết prompt.

  from generation.explain import explain_passage
  explain_passage(susp, src) -> {"technique","technique_vi","explanation",
                                 "severity","suggested_rewrite","confidence"}
"""
from __future__ import annotations
from generation._gemini import generate_json
from generation.classify import TECHNIQUES

_SYSTEM = (
    "You are a plagiarism analyst producing a grounded explanation report. "
    "You are given a SUSPICIOUS passage and the SOURCE passage that a retrieval "
    "step matched it to. Ground EVERYTHING you write in these two passages only — "
    "do not invent facts not present in them. Produce, in one call:\n"
    "1) technique: EXACTLY ONE key from {verbatim_copy, synonym_substitution, "
    "sentence_reordering, back_translation, mosaic_patchwork, idea_plagiarism};\n"
    "2) explanation: 1-2 câu tiếng Việt, chỉ ra bằng chứng cụ thể (cụm từ/cấu trúc "
    "dùng lại) vì sao đây là đạo văn theo kỹ thuật đó;\n"
    "3) severity: one of {cao, trung bình, thấp} theo mức trùng lặp;\n"
    "4) suggested_rewrite: viết lại đoạn nghi vấn bằng tiếng Anh học thuật, GIỮ nội "
    "dung nhưng KHÔNG lặp câu chữ/cấu trúc của nguồn;\n"
    'Return strict JSON: {"technique":"<key>","explanation":"<vi>",'
    '"severity":"<cao|trung bình|thấp>","suggested_rewrite":"<en>","confidence":<0..1>}.'
)

_SEV = {"cao", "trung bình", "thấp"}


def explain_passage(susp: str, src: str, *, model: str = None) -> dict:
    """Sinh bản luận giải grounded cho một span (nghi vấn ↔ nguồn truy hồi)."""
    prompt = (f"SOURCE passage (retrieved):\n\"\"\"\n{src.strip()}\n\"\"\"\n\n"
              f"SUSPICIOUS passage:\n\"\"\"\n{susp.strip()}\n\"\"\"\n")
    d = generate_json(prompt, system=_SYSTEM, model=model, temperature=0.3)

    tech = str(d.get("technique", "")).strip()
    if tech not in TECHNIQUES:                       # chuẩn hoá nếu model trả lệch
        tech = tech.lower().replace(" ", "_")
        tech = tech if tech in TECHNIQUES else "idea_plagiarism"

    sev = str(d.get("severity", "")).strip().lower()
    if sev not in _SEV:
        sev = "trung bình"

    return {
        "technique": tech,
        "technique_vi": TECHNIQUES[tech],
        "explanation": str(d.get("explanation", "")).strip(),
        "severity": sev,
        "suggested_rewrite": str(d.get("suggested_rewrite", "")).strip(),
        "confidence": d.get("confidence"),
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    o = explain_passage(
        "The velocity fluctuations dominate the small-scale structure seen in "
        "spectral-line data cubes, crucial for interpreting observations.",
        "Small-scale structure observed in spectral-line data cubes is dominated by "
        "velocity fluctuations; this is crucial for the interpretation of observational data.")
    for k, v in o.items():
        print(f"{k}: {v}")
