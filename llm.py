"""FPT AI Factory client wrapper (OpenAI SDK, custom base_url).

Per FPT docs: base_url="https://mkp-api.fptcloud.com" — the SDK appends
/chat/completions automatically, reaching the correct endpoint.

No response envelope unwrapping is needed; the API returns standard
OpenAI-compatible ChatCompletion objects when called via the SDK.

This module is tool-agnostic: it knows nothing about the tools defined in
tools.py. It receives tool schemas from the caller and returns a normalized
LLMResponse.
"""

import json
import logging
import os
from dataclasses import dataclass, field

from openai import OpenAI

logger = logging.getLogger(__name__)

_BASE_URL = "https://mkp-api.fptcloud.com"


@dataclass
class ToolCall:
    """A single tool call requested by the LLM.

    Attributes:
        id: The tool_call_id, required when sending the result back.
        name: The function name the model wants to invoke.
        arguments: Parsed keyword arguments (already deserialized from JSON).
    """

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Normalized response from a single LLM chat call.

    Attributes:
        content: Plain-text reply from the model, or None if it only made tool calls.
        tool_calls: List of tool calls requested by the model (may be empty).
    """

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient:
    """OpenAI SDK client pointed at FPT AI Factory.

    Args:
        model: Model name to use (default: Qwen3-32B).
        temperature: Sampling temperature (default: 1).
        max_tokens: Maximum tokens to generate (default: 1024).
        top_p: Nucleus sampling probability (default: 1).
        top_k: Top-k sampling cutoff; passed as extra_body (default: 40).
        presence_penalty: Presence penalty (default: 0).
        frequency_penalty: Frequency penalty (default: 0).
    """

    def __init__(
        self,
        model: str = "Qwen3-32B",
        temperature: float = 1,
        max_tokens: int = 1024,
        top_p: float = 1,
        top_k: int = 40,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ) -> None:
        """Initialize the OpenAI SDK client with FPT's base URL.

        Reads FPT_AI_API_KEY from the environment (caller is responsible for
        loading .env before instantiation).

        Args:
            model: FPT model identifier, e.g. "Qwen3-32B".
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling cutoff (FPT-specific, sent via extra_body).
            presence_penalty: Presence penalty coefficient.
            frequency_penalty: Frequency penalty coefficient.
        """
        api_key = os.environ.get("FPT_AI_API_KEY", "")
        self._client = OpenAI(api_key=api_key, base_url=_BASE_URL)
        self._model = model
        self._params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        # top_k is not a standard OpenAI parameter; pass it via extra_body
        self._extra_body: dict = {"top_k": top_k}

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Send a chat request and return a normalized response.

        Args:
            messages: The full conversation history in OpenAI message format.
            tools: List of tool schemas in OpenAI function-calling format.

        Returns:
            LLMResponse with either content or tool_calls (or both) populated.
        """
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "extra_body": self._extra_body,
            **self._params,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
        )
