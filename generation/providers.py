"""Provider adapters for plagiarism-removal rewriting.

The rewriting prompt is shared across providers so model experiments differ only
in model/provider parameters. NVIDIA hosted NIM is exposed through its
OpenAI-compatible endpoint; Gemini keeps using the native Google SDK.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any


SYSTEM_PROMPT = (
    "You are an academic writing assistant that removes plagiarism. You are given a "
    "SUSPICIOUS passage and the SOURCE passage it was likely copied or paraphrased "
    "from. Rewrite ONLY the suspicious passage so that it: "
    "(1) preserves the same meaning, facts, and technical content; "
    "(2) does NOT reuse the source's wording, phrasing, or sentence structure; "
    "(3) reads as fluent, academic English; "
    "(4) keeps roughly the same length. "
    "Do not add commentary, citations, or facts not present in the suspicious passage. "
    'Return strict JSON: {"rewritten":"...","changes":"one short Vietnamese sentence"}.'
)


def build_user_prompt(suspicious: str, source: str) -> str:
    return (
        f'SOURCE passage (do NOT reuse its wording):\n"""\n{source.strip()}\n"""\n\n'
        f'SUSPICIOUS passage (rewrite THIS):\n"""\n{suspicious.strip()}\n"""\n'
    )


def parse_rewrite_payload(raw: str) -> tuple[str, str]:
    """Parse strict JSON, tolerating markdown fences and surrounding prose."""
    raw = (raw or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        rewritten = str(data.get("rewritten", "")).strip()
        changes = str(data.get("changes", "")).strip()
        if rewritten:
            return rewritten, changes
    if not cleaned:
        raise RuntimeError("Model returned an empty rewrite.")
    return cleaned, ""


@dataclass
class RewriteResult:
    rewritten: str
    changes: str
    provider: str
    model: str
    usage: dict
    raw_finish_reason: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class RewriteProvider:
    name = "base"

    def rewrite(self, suspicious: str, source: str, *, model: str,
                temperature: float, top_p: float, max_tokens: int) -> RewriteResult:
        raise NotImplementedError

    def list_models(self) -> list[str]:
        return []


class GeminiRewriteProvider(RewriteProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None):
        from google import genai
        key = (api_key or os.environ.get("GEMINI_API_KEY") or
               os.environ.get("GOOGLE_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("Missing GEMINI_API_KEY.")
        self.client = genai.Client(api_key=key)

    def rewrite(self, suspicious: str, source: str, *, model: str,
                temperature: float, top_p: float, max_tokens: int) -> RewriteResult:
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
        response = self.client.models.generate_content(
            model=model,
            contents=build_user_prompt(suspicious, source),
            config=config,
        )
        rewritten, changes = parse_rewrite_payload(response.text or "")
        usage_meta = getattr(response, "usage_metadata", None)
        usage = {}
        if usage_meta is not None:
            usage = {
                "input_tokens": getattr(usage_meta, "prompt_token_count", None),
                "output_tokens": getattr(usage_meta, "candidates_token_count", None),
                "total_tokens": getattr(usage_meta, "total_token_count", None),
            }
        return RewriteResult(rewritten, changes, self.name, model, usage)

    def list_models(self) -> list[str]:
        return [m.name for m in self.client.models.list() if getattr(m, "name", None)]


class NvidiaRewriteProvider(RewriteProvider):
    name = "nvidia"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        from openai import OpenAI
        key = (api_key or os.environ.get("NVIDIA_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("Missing NVIDIA_API_KEY.")
        self.base_url = (base_url or os.environ.get("NVIDIA_BASE_URL") or
                         "https://integrate.api.nvidia.com/v1").rstrip("/")
        # Retry is owned by the experiment runner so every network attempt is
        # visible, paced, and accounted for in the experiment record.
        self.client = OpenAI(api_key=key, base_url=self.base_url,
                             max_retries=0, timeout=120.0)

    def rewrite(self, suspicious: str, source: str, *, model: str,
                temperature: float, top_p: float, max_tokens: int) -> RewriteResult:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(suspicious, source)},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
        )
        choice = response.choices[0]
        content: Any = choice.message.content
        if isinstance(content, list):
            content = "".join(str(getattr(x, "text", x)) for x in content)
        rewritten, changes = parse_rewrite_payload(str(content or ""))
        usage_obj = getattr(response, "usage", None)
        usage = {}
        if usage_obj is not None:
            usage = {
                "input_tokens": getattr(usage_obj, "prompt_tokens", None),
                "output_tokens": getattr(usage_obj, "completion_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }
        return RewriteResult(
            rewritten, changes, self.name, model, usage,
            raw_finish_reason=str(getattr(choice, "finish_reason", "") or "") or None,
        )

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list().data)


def get_provider(name: str, *, api_key_env: str | None = None,
                 base_url: str | None = None) -> RewriteProvider:
    normalized = name.strip().lower()
    explicit_key = os.environ.get(api_key_env, "") if api_key_env else None
    if normalized == "gemini":
        return GeminiRewriteProvider(api_key=explicit_key)
    if normalized in {"nvidia", "nim"}:
        return NvidiaRewriteProvider(api_key=explicit_key, base_url=base_url)
    raise ValueError(f"Unknown rewrite provider: {name}")
