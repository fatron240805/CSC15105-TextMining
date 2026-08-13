#!/usr/bin/env python3
"""Vai trò #3 — LLM-verifier: xác minh một span đạo văn để KHỬ dương-tính-giả.

Aligner (tf-isf) có thể gắn cờ nhầm khi hai đoạn chỉ TRÙNG CHỦ ĐỀ chứ không thật sự
sao chép. Verifier đọc cặp (nghi vấn, nguồn) và quyết định GIỮ / BỎ, kèm độ tin cậy.
Áp sau alignment -> tăng precision của PlagDet.

  from generation.verify import verify_pair
  verify_pair(susp, src) -> {"is_plagiarism": bool, "confidence": 0..1, "reason": str}
"""
from __future__ import annotations
from generation._gemini import generate_json

_SYSTEM = (
    "You are a strict plagiarism verifier. Given a SUSPICIOUS passage and a SOURCE passage "
    "the detector matched it to, decide whether the suspicious passage is GENUINELY derived "
    "from that source (copied, paraphrased, or reworded from it) — as opposed to merely "
    "sharing the same topic or common domain phrasing by coincidence. "
    "Reused specific wording, structure, or a distinctive chain of claims = plagiarism. "
    "Only generic overlap two independent authors would both write = NOT plagiarism. "
    'Return strict JSON: {"is_plagiarism": true|false, "confidence": <0..1>, '
    '"reason": "<1 câu tiếng Việt>"}.'
)


def verify_pair(susp: str, src: str, *, model: str = None) -> dict:
    prompt = (f"SOURCE passage:\n\"\"\"\n{src.strip()}\n\"\"\"\n\n"
              f"SUSPICIOUS passage:\n\"\"\"\n{susp.strip()}\n\"\"\"\n")
    d = generate_json(prompt, system=_SYSTEM, model=model, temperature=0.0)
    val = d.get("is_plagiarism")
    is_pl = bool(val) if isinstance(val, bool) else str(val).strip().lower() in ("true", "1", "yes", "có")
    conf = d.get("confidence")
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = None
    return {"is_plagiarism": is_pl, "confidence": conf, "reason": str(d.get("reason", "")).strip()}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("khớp thật:", verify_pair(
        "The velocity fluctuations dominate the small-scale structure in spectral-line data cubes.",
        "Small-scale structure observed in spectral-line data cubes is dominated by velocity fluctuations."))
    print("trùng chủ đề:", verify_pair(
        "Machine learning has many applications in science and engineering today.",
        "Deep neural networks are widely used across many engineering and scientific domains."))
