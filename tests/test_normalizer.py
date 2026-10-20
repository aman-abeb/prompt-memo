"""Unit tests for prompt normalization, canonicalization, and hashing."""

import pytest
from prompt_memo.models import ChatCompletionRequest, ChatCompletionResponse, Message
from prompt_memo.normalizer import (
    build_canonical_representation,
    canonicalize_messages,
    extract_prompt_text,
    hash_prompt,
    normalize_text,
)


def test_normalize_text_whitespace_and_newlines() -> None:
    raw = "   Hello    world!   \r\n  Line 2   with   extra spaces.  \n\n\n\nLine 3.  "
    expected = "Hello world!\nLine 2 with extra spaces.\n\nLine 3."
    assert normalize_text(raw) == expected


def test_normalize_text_unicode_nfkc() -> None:
    # Full-width characters and ligature
    raw = "Ｈｅｌｌｏ ﬀ world"
    normalized = normalize_text(raw)
    assert normalized == "Hello ff world"


def test_canonicalize_messages_from_dicts_and_models() -> None:
    from_dict = [{"role": "USER", "content": "  Test message  "}]
    from_model = [Message(role="user", content="Test message")]

    canon_dict = canonicalize_messages(from_dict)
    canon_model = canonicalize_messages(from_model)

    assert canon_dict == canon_model
    assert canon_dict[0] == {"role": "user", "content": "Test message"}


def test_hash_prompt_deterministic() -> None:
    messages = [
        Message(role="system", content="You are an expert mathematician."),
        Message(role="user", content="Calculate 2 + 2."),
    ]
    hash_1 = hash_prompt(messages, model="gpt-4o")
    hash_2 = hash_prompt(messages, model="gpt-4o")
    assert hash_1 == hash_2
    assert len(hash_1) == 64


def test_hash_prompt_whitespace_invariance() -> None:
    msg1 = [
        Message(role="system", content="You are an assistant."),
        Message(role="user", content="What is 1 + 1?"),
    ]
    msg2 = [
        Message(role="SYSTEM", content="   You are   an assistant.   "),
        Message(role="USER", content="   What is   1 + 1?   "),
    ]
    assert hash_prompt(msg1, model="gpt-4o") == hash_prompt(msg2, model="gpt-4o")


def test_hash_prompt_model_sensitivity() -> None:
    messages = [Message(role="user", content="Hello")]
    hash_gpt4 = hash_prompt(messages, model="gpt-4o")
    hash_claude = hash_prompt(messages, model="claude-3-opus")
    assert hash_gpt4 != hash_claude


def test_hash_prompt_temperature_sensitivity() -> None:
    messages = [Message(role="user", content="Hello")]
    hash_default = hash_prompt(messages, model="gpt-4o")
    hash_zero_temp = hash_prompt(messages, model="gpt-4o", temperature=0.0)
    hash_high_temp = hash_prompt(messages, model="gpt-4o", temperature=0.7)

    assert hash_default != hash_zero_temp
    assert hash_zero_temp != hash_high_temp


def test_extract_prompt_text() -> None:
    messages = [
        Message(role="system", content="Instruction text"),
        Message(role="user", content="Question text"),
    ]
    extracted = extract_prompt_text(messages)
    assert extracted == "Instruction text\nQuestion text"


def test_empty_messages_handling() -> None:
    assert normalize_text("") == ""
    assert normalize_text("    \n\n  ") == ""
    canon = canonicalize_messages([Message(role="user", content="")])
    assert canon == [{"role": "user", "content": ""}]
    assert extract_prompt_text([Message(role="user", content="")]) == ""


def test_chat_completion_request_response_models() -> None:
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="Hello world")],
        temperature=0.5,
    )
    assert req.model == "gpt-4o"
    assert req.messages[0].content == "Hello world"

    # Test JSON serialization / deserialization roundtrip
    dumped = req.model_dump_json()
    loaded = ChatCompletionRequest.model_validate_json(dumped)
    assert loaded.model == req.model
    assert loaded.messages[0].content == req.messages[0].content

    # Response model
    resp = ChatCompletionResponse(
        id="chatcmpl-test-123",
        created=1700000000,
        model="gpt-4o",
        choices=[],
        cached=True,
    )
    assert resp.cached is True
    assert resp.id == "chatcmpl-test-123"
