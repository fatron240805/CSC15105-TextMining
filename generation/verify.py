#!/usr/bin/env python3
"""Vai trò #3 — LLM-verifier: xác minh CASE BIÊN để KHỬ dương-tính-giả.

Ranh giới với Alignment (rà soát sau góp ý thầy — 2026-08-24): tầng Alignment (th/th1/
th2/th3, A2-4/A2-5) đã SỞ HỮU quyết định "span này có tính không" — quyết định đó có thể
tune bằng ngưỡng, tái lập được, và đã được kiểm định trên gold. Verifier KHÔNG được phép
phán lại toàn bộ quyết định đó (sẽ chỉ là làm lại việc alignment đã làm, tốn API mà không
rõ thêm giá trị gì). Verifier chỉ nên được gọi cho case BIÊN — span có sim alignment nằm
sát th3 (in_edge_band) — nơi ngưỡng tĩnh tự nó không đủ tin cậy để tự quyết một mình.
Case sim >> th3 (alignment tự tin) thì KHÔNG gọi verifier, cứ tin alignment.

  from generation.verify import verify_pair, in_edge_band
  in_edge_band(sim, th3) -> bool                 # có đáng gọi verifier không
  verify_pair(susp, src) -> {"is_plagiarism": bool, "confidence": 0..1, "reason": str}
"""
from __future__ import annotations
from generation._gemini import generate_json

EDGE_BAND = 0.05   # rộng dải biên quanh th3; TODO hiệu chỉnh bằng sweep (xem
                   # evaluation/tune_tfisf_threshold.py) khi có corpus — giá trị này
                   # là placeholder hợp lý, chưa được đo thực nghiệm.


def in_edge_band(sim: float, th3: float, band: float = EDGE_BAND) -> bool:
    """True nếu sim nằm trong dải biên [th3, th3+band) — nơi alignment tự nó không
    chắc chắn và verifier đáng để gọi. sim đã qua lọc th3 nên luôn >= th3;
    sim >= th3+band nghĩa là alignment đã tự tin, không cần verifier."""
    return th3 <= sim < th3 + band

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
