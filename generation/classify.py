#!/usr/bin/env python3
"""Vai trò #1 — phân loại kỹ thuật đạo văn (reverse RAG). KHÔNG chồng lấn alignment:
alignment chỉ quyết định CÓ đạo văn + Ở ĐÂU (offset/length); classify chỉ diễn giải
BẰNG CÁCH NÀO cho một span alignment đã chốt xong — không tự quyết định lại span nào
được tính. Lưu ý: PAN không có ground-truth `technique` (chỉ có `obfuscation`
simple/medium/hard — xem IMPLEMENTATION_PLAN.md mục A2-4/A2-5), nên nhãn ở đây chỉ
mang tính diễn giải/tham khảo, KHÔNG dùng để gate điểm số.

Sibling rẻ hơn của generation.explain (cùng 1 lần gọi LLM cho technique+explanation,
không kèm severity/suggested_rewrite) — dùng khi chỉ cần phân loại nhanh, không cần
báo cáo đầy đủ.

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


def normalize_technique(tech: str) -> str:
    """Chuẩn hoá key kỹ thuật nếu model trả lệch (khoảng trắng, hoa/thường, key lạ).
    Dùng chung bởi classify.py và explain.py để tránh 2 bản sao logic lệch nhau."""
    tech = (tech or "").strip()
    if tech in TECHNIQUES:
        return tech
    tech = tech.lower().replace(" ", "_")
    return tech if tech in TECHNIQUES else "idea_plagiarism"


def classify_passage(susp: str, src: str, *, model: str = None) -> dict:
    prompt = (f"SOURCE passage:\n\"\"\"\n{src.strip()}\n\"\"\"\n\n"
              f"SUSPICIOUS passage:\n\"\"\"\n{susp.strip()}\n\"\"\"\n")
    d = generate_json(prompt, system=_SYSTEM, model=model, temperature=0.2)
    tech = normalize_technique(str(d.get("technique", "")))
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
