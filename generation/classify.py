#!/usr/bin/env python3
"""Vai trò #1 — Giải thích + phân loại kỹ thuật đạo văn (reverse RAG).

Cho cặp (đoạn nghi vấn, đoạn nguồn), LLM chọn MỘT kỹ thuật đạo văn + giải thích vì sao.
Biến detector thành hệ CÓ THỂ DIỄN GIẢI thay vì chỉ tô cam.

  from generation.classify import classify_passage
  classify_passage(susp, src) -> {"technique","technique_vi","explanation","confidence"}
"""
from __future__ import annotations
from generation._gemini import generate_json

# 6 kỹ thuật theo design doc (mục 3.2)
TECHNIQUES = {
    "verbatim_copy": "Sao chép nguyên văn",
    "synonym_substitution": "Thay từ đồng nghĩa",
    "sentence_reordering": "Đảo trật tự câu/mệnh đề",
    "back_translation": "Dịch đi–dịch lại (back-translation)",
    "mosaic_patchwork": "Chắp vá nhiều nguồn (mosaic)",
    "idea_plagiarism": "Đạo ý (giữ ý, viết lại hoàn toàn)",
}

_SYSTEM = (
    "You classify how a suspicious passage was plagiarized from a source passage. "
    "Choose EXACTLY ONE technique from this fixed set (use the English key):\n"
    "- verbatim_copy: copied word-for-word or near-identical.\n"
    "- synonym_substitution: same structure, words swapped for synonyms.\n"
    "- sentence_reordering: same content, clauses/sentences reordered.\n"
    "- back_translation: reworded as if translated to another language and back.\n"
    "- mosaic_patchwork: stitched from multiple fragments / interleaved phrases.\n"
    "- idea_plagiarism: same ideas but almost fully rewritten wording.\n"
    "Judge by how the SUSPICIOUS text relates to the SOURCE. "
    'Return strict JSON: {"technique":"<key>","explanation":"<1-2 câu tiếng Việt vì sao>",'
    '"confidence":<0..1>}.'
)


def classify_passage(susp: str, src: str, *, model: str = None) -> dict:
    prompt = (f"SOURCE passage:\n\"\"\"\n{src.strip()}\n\"\"\"\n\n"
              f"SUSPICIOUS passage:\n\"\"\"\n{susp.strip()}\n\"\"\"\n")
    d = generate_json(prompt, system=_SYSTEM, model=model, temperature=0.2)
    tech = str(d.get("technique", "")).strip()
    if tech not in TECHNIQUES:                 # chuẩn hoá nếu model trả lệch
        tech = tech.lower().replace(" ", "_")
        tech = tech if tech in TECHNIQUES else "idea_plagiarism"
    return {"technique": tech, "technique_vi": TECHNIQUES[tech],
            "explanation": str(d.get("explanation", "")).strip(),
            "confidence": d.get("confidence")}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    o = classify_passage(
        "Linear regression is a core supervised learning method used in many fields.",
        "Linear regression models are a fundamental class of supervised learning, with applications in many fields.")
    print(o)
