#!/usr/bin/env python3
"""Generation (bước G của RAG) — viết lại đoạn nghi vấn để KHỬ đạo văn.

"Reverse RAG": đưa CẢ đoạn nghi vấn lẫn đoạn nguồn (retrieval trả về) vào context,
yêu cầu LLM viết lại đoạn nghi vấn sao cho GIỮ nội dung nhưng KHÔNG lặp câu chữ /
cấu trúc của nguồn. Dùng Google Gemini.

Key đọc từ env GEMINI_API_KEY, hoặc file ~/.gemini/api_key (ngoài repo — không commit).
Không hardcode key ở đây.

  from generation.rewrite import rewrite_passage
  out = rewrite_passage(susp_text, src_text)   # {"rewritten": ..., "changes": ...}
"""
from __future__ import annotations
import json
import os
from pathlib import Path

def _load_dotenv():
    """Nạp .env (repo root hoặc thư mục hiện tại) vào os.environ — không ghi đè biến đã có."""
    for base in (Path(__file__).resolve().parent.parent, Path.cwd()):
        f = base / ".env"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k and k not in os.environ:
                os.environ[k] = v.strip().strip('"').strip("'")


_load_dotenv()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

_SYSTEM = (
    "You are an academic writing assistant that removes plagiarism. You are given a "
    "SUSPICIOUS passage (suspected of being plagiarized) and the SOURCE passage it was "
    "likely copied or paraphrased from. Rewrite ONLY the suspicious passage so that it: "
    "(1) preserves the same meaning, facts, and technical content; "
    "(2) does NOT reuse the source's wording, phrasing, or sentence structure — express "
    "the ideas in genuinely original language; "
    "(3) reads as fluent, academic English; "
    "(4) keeps roughly the same length. "
    "Do not add commentary, citations, or content not present in the suspicious passage. "
    'Return strict JSON: {"rewritten": "<the rewritten passage>", '
    '"changes": "<one short sentence, in Vietnamese, on what you changed>"}.'
)


def _load_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()
    f = Path.home() / ".gemini" / "api_key"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "Không tìm thấy Gemini API key. Đặt biến môi trường GEMINI_API_KEY "
        "hoặc lưu vào ~/.gemini/api_key (ngoài repo)."
    )


_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=_load_key())
    return _client


def rewrite_passage(suspicious: str, source: str, *, model: str = None,
                    temperature: float = 0.7, provider: str = None) -> dict:
    """Viết lại đoạn nghi vấn để khử đạo văn. Trả {"rewritten", "changes"}.

    provider: 'gemini' (mặc định — client riêng) hoặc 'fpt' (đi qua generate_json chung,
    dùng GLM-5.2 / model FPT). Nếu None → đọc env LLM_PROVIDER.
    """
    import time
    prompt = (
        f"SOURCE passage (do NOT reuse its wording):\n\"\"\"\n{source.strip()}\n\"\"\"\n\n"
        f"SUSPICIOUS passage (rewrite THIS):\n\"\"\"\n{suspicious.strip()}\n\"\"\"\n"
    )
    prov = (provider or os.environ.get("LLM_PROVIDER", "gemini")).lower()
    if prov == "fpt":                              # nhánh FPT — dùng client OpenAI-compat chung
        from generation._gemini import generate_json
        mdl = model or os.environ.get("LLM_MODEL")
        data = generate_json(prompt, system=_SYSTEM, model=mdl,
                             temperature=temperature, provider="fpt")
        rewritten = str(data.get("rewritten", "")).strip()
        changes = str(data.get("changes", "")).strip()
        if not rewritten:
            raise RuntimeError(f"FPT ({mdl}) không trả về nội dung viết lại.")
        return {"rewritten": rewritten, "changes": changes, "model": mdl}
    from google.genai import types
    client = _get_client()
    cfg = types.GenerateContentConfig(system_instruction=_SYSTEM, temperature=temperature,
                                      response_mime_type="application/json")
    last = None
    for attempt in range(5):                       # retry lỗi tạm thời (429 rate-limit/503/hiccup)
        try:
            resp = client.models.generate_content(model=model or MODEL, contents=prompt, config=cfg)
            break
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))          # backoff 5,10,15,20s — chờ quota/phút hồi
    else:
        raise RuntimeError(f"Gemini lỗi sau 5 lần thử: {last}")
    raw = (resp.text or "").strip()
    try:
        data = json.loads(raw)
        rewritten = str(data.get("rewritten", "")).strip()
        changes = str(data.get("changes", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        rewritten, changes = raw, ""     # model trả text thô -> dùng luôn
    if not rewritten:
        raise RuntimeError("Gemini không trả về nội dung viết lại.")
    return {"rewritten": rewritten, "changes": changes, "model": model or MODEL}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    susp = ("The velocity fluctuations dominate the small-scale structure that is seen "
            "in the spectral-line data cubes, which is crucial for interpreting observations.")
    src = ("Small-scale structure observed in spectral-line data cubes is dominated by "
           "velocity fluctuations; this is crucial for the interpretation of observational data.")
    out = rewrite_passage(susp, src)
    print("MODEL:", out["model"])
    print("REWRITTEN:", out["rewritten"])
    print("CHANGES :", out["changes"])
