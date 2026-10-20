"""Deterministic mock upstream LLM provider for zero-network testing and replay."""

import hashlib
import math
import time
from typing import Protocol

from prompt_memo.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    Usage,
)
from prompt_memo.normalizer import extract_prompt_text


class UpstreamProvider(Protocol):
    """Protocol defining the completion interface for LLM providers."""

    def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generates a chat completion response for the given request."""
        ...


def estimate_token_count(text: str) -> int:
    """Estimates token count using a deterministic standard heuristic (~4 characters per token).

    Guarantees at least 1 token for non-empty text, 0 for empty strings.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


class MockUpstreamProvider:
    """Deterministic offline mock upstream LLM provider.

    Generates reproducible completions and accurate usage counts without external network calls.
    Completions are deterministically derived from the prompt contents and target model.
    """

    def __init__(self, fixed_timestamp: int | None = None) -> None:
        self.fixed_timestamp = fixed_timestamp

    def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generates an OpenAI-compatible ChatCompletionResponse deterministically."""
        prompt_text = extract_prompt_text(request.messages)
        prompt_tokens = estimate_token_count(prompt_text)

        # Compute deterministic seed hash from prompt + model + temperature
        signature = f"{request.model}:{request.temperature}:{prompt_text}"
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        short_id = digest[:12]

        # Generate deterministic response content
        last_user_message = ""
        for msg in reversed(request.messages):
            if msg.role.lower() == "user" and msg.content:
                last_user_message = msg.content.strip()
                break

        reply_prefix = f"[Mock:{request.model}]"
        if last_user_message:
            content = f"{reply_prefix} Echoing query: '{last_user_message}' (seed: {short_id})"
        else:
            content = f"{reply_prefix} Processed empty query (seed: {short_id})"

        completion_tokens = estimate_token_count(content)
        total_tokens = prompt_tokens + completion_tokens

        created_time = self.fixed_timestamp if self.fixed_timestamp is not None else int(time.time())

        choices = [
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ]

        return ChatCompletionResponse(
            id=f"chatcmpl-mock-{short_id}",
            object="chat.completion",
            created=created_time,
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            cached=False,
        )
