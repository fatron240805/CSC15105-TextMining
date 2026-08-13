#!/usr/bin/env python3
"""Client Gemini dùng chung cho các vai trò generation (rewrite / classify / verify).

Key: env GEMINI_API_KEY → .env (repo root) → ~/.gemini/api_key. Không hardcode.
generate_json(): gọi Gemini trả JSON, retry backoff cho 429 free-tier.
"""
from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path


def _load_dotenv():
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
MODEL_DEFAULT = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")


def _load_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()
    f = Path.home() / ".gemini" / "api_key"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    raise RuntimeError("Không tìm thấy Gemini API key (env GEMINI_API_KEY / .env / ~/.gemini/api_key).")


_client = None


def get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=_load_key())
    return _client


def generate_json(prompt: str, *, system: str = None, model: str = None,
                  temperature: float = 0.4, retries: int = 5, provider: str = None) -> dict:
    """Gọi LLM, ép JSON. provider: 'gemini' (mặc định) hoặc 'fpt' (OpenAI-compat).
    Trả dict đã parse (hoặc {'_raw': ...} nếu không phải JSON)."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "gemini")).lower()
    if provider == "fpt":
        return _fpt_json(prompt, system, model, temperature, retries)
    from google.genai import types
    client = get_client()
    cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                      response_mime_type="application/json")
    last = None
    for attempt in range(retries):
        try:
            r = client.models.generate_content(model=model or MODEL_DEFAULT, contents=prompt, config=cfg)
            raw = (r.text or "").strip()
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {"_raw": raw}
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))       # backoff 5,10,15,20,25s cho 429/phút
    raise RuntimeError(f"Gemini lỗi sau {retries} lần thử: {last}")


# ---- Provider FPT (AI Marketplace, tương thích OpenAI) ----
_fpt = None


def _fpt_client():
    global _fpt
    if _fpt is None:
        from openai import OpenAI
        key = os.environ.get("LLM_API_KEY")
        base = os.environ.get("LLM_API_BASE", "https://mkp-api.fptcloud.com")
        if not key:
            raise RuntimeError("Thiếu LLM_API_KEY (FPT) trong .env.")
        _fpt = OpenAI(api_key=key, base_url=base)
    return _fpt


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    return s.strip()


def _fpt_json(prompt, system, model, temperature, retries) -> dict:
    client = _fpt_client()
    mdl = model or os.environ.get("LLM_MODEL")
    if not mdl:
        raise RuntimeError("Thiếu LLM_MODEL (tên model FPT) — đặt trong .env hoặc truyền model=...")
    msgs = ([{"role": "system", "content": system}] if system else [])
    msgs.append({"role": "user", "content": prompt + "\n\nReturn ONLY valid JSON, no markdown."})
    last = None
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(model=mdl, messages=msgs, temperature=temperature,
                                               max_tokens=4096)   # reasoning model tiêu token trước
            msg = r.choices[0].message
            content = msg.content or getattr(msg, "reasoning_content", None) or ""
            raw = _strip_fences(content)
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                m = re.search(r"\{.*\}", raw, re.S)        # bắt khối {...} lẫn trong text/reasoning
                if m:
                    try:
                        return json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass
                return {"_raw": raw[:600]}
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"FPT ({mdl}) lỗi sau {retries} lần thử: {last}")
