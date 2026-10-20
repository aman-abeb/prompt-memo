"""Utilities for canonical text normalization and deterministic prompt hashing."""

import hashlib
import re
import unicodedata
from typing import Any, Sequence, Union

from prompt_memo.models import Message


def normalize_text(text: str) -> str:
    """Normalizes text by applying unicode NFKC normalization and collapsing whitespace.

    - Normalizes unicode characters.
    - Standardizes line breaks (\r\n -> \n).
    - Trims leading/trailing whitespace on each line.
    - Collapses multiple horizontal spaces or tabs into a single space.
    - Collapses three or more consecutive line breaks into two line breaks.
    """
    if not text:
        return ""

    # Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)

    # Standardize line endings
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse horizontal whitespace per line and strip each line
    lines = [
        re.sub(r"[\t \f\v]+", " ", line).strip()
        for line in normalized.split("\n")
    ]

    # Join lines and collapse redundant blank lines
    collapsed = "\n".join(lines)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)

    return collapsed.strip()


def canonicalize_messages(messages: Sequence[Union[Message, dict[str, Any]]]) -> list[dict[str, str]]:
    """Converts a sequence of messages (Message instances or dicts) to canonical form.

    Each canonical message dictionary contains:
    - 'role': lowercased and trimmed role string
    - 'content': normalized content string
    """
    canonical: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, Message):
            role = msg.role
            content = msg.content or ""
        elif isinstance(msg, dict):
            role = str(msg.get("role", ""))
            content = str(msg.get("content", "") or "")
        else:
            raise TypeError(f"Unsupported message type: {type(msg)}")

        canonical.append({
            "role": role.strip().lower(),
            "content": normalize_text(content),
        })
    return canonical


def extract_prompt_text(messages: Sequence[Union[Message, dict[str, Any]]]) -> str:
    """Extracts concatenated normalized prompt text from messages.

    Used for similarity shingling and token estimation.
    """
    canonical = canonicalize_messages(messages)
    return "\n".join(item["content"] for item in canonical if item["content"])


def build_canonical_representation(
    messages: Sequence[Union[Message, dict[str, Any]]],
    model: str,
    temperature: float | None = None,
) -> str:
    """Builds an unambiguous canonical string representation for hashing.

    Includes model ID and sequenced canonical messages. Optionally binds
    sampling temperature if provided and non-default.
    """
    canonical = canonicalize_messages(messages)
    parts = [f"model:{model.strip().lower()}"]
    if temperature is not None:
        parts.append(f"temp:{temperature:.4f}")

    for msg in canonical:
        parts.append(f"role:{msg['role']}:{msg['content']}")

    return "\n".join(parts)


def hash_prompt(
    messages: Sequence[Union[Message, dict[str, Any]]],
    model: str,
    temperature: float | None = None,
) -> str:
    """Computes a deterministic SHA-256 hex digest for the canonical prompt representation."""
    representation = build_canonical_representation(messages, model=model, temperature=temperature)
    return hashlib.sha256(representation.encode("utf-8")).hexdigest()
