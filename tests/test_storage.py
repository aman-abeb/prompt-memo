"""Unit tests for SQLite persistence, hit metrics, and deterministic mock upstream."""

import concurrent.futures
import tempfile
import pytest

from prompt_memo.models import ChatCompletionRequest, Message
from prompt_memo.normalizer import build_canonical_representation, hash_prompt
from prompt_memo.storage import SQLiteCache
from prompt_memo.upstream import MockUpstreamProvider, estimate_token_count


@pytest.fixture
def cache() -> SQLiteCache:
    """Provides an in-memory SQLiteCache instance for isolated testing."""
    return SQLiteCache(db_path=":memory:")


@pytest.fixture
def mock_upstream() -> MockUpstreamProvider:
    """Provides a deterministic mock upstream provider with a fixed timestamp."""
    return MockUpstreamProvider(fixed_timestamp=1700000000)


def test_cache_miss_returns_none(cache: SQLiteCache) -> None:
    res = cache.get("nonexistent_hash_0000")
    assert res is None


def test_cache_set_and_get(cache: SQLiteCache, mock_upstream: MockUpstreamProvider) -> None:
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="What is the capital of France?")],
    )
    resp = mock_upstream.generate(req)
    assert resp.cached is False

    h = hash_prompt(req.messages, model=req.model)
    canonical = build_canonical_representation(req.messages, model=req.model)
    cache.set(prompt_hash=h, response=resp, request=req, canonical_prompt=canonical)

    # Retrieve from cache
    cached_resp = cache.get(h)
    assert cached_resp is not None
    assert cached_resp.cached is True
    assert cached_resp.id == resp.id
    assert cached_resp.model == req.model
    assert cached_resp.choices[0].message.content == resp.choices[0].message.content
    assert cached_resp.usage is not None
    assert resp.usage is not None
    assert cached_resp.usage.total_tokens == resp.usage.total_tokens


def test_hit_counter_and_access_tracking(cache: SQLiteCache, mock_upstream: MockUpstreamProvider) -> None:
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="Explain quantum computing.")],
    )
    resp = mock_upstream.generate(req)
    h = hash_prompt(req.messages, model=req.model)

    cache.set(prompt_hash=h, response=resp, request=req)

    # Initially 0 hits in record
    record_before = cache.get_record(h)
    assert record_before is not None
    assert record_before.hit_count == 0

    # First hit
    cached1 = cache.get(h)
    assert cached1 is not None

    record_after_1 = cache.get_record(h)
    assert record_after_1 is not None
    assert record_after_1.hit_count == 1
    assert record_after_1.last_accessed_at >= record_before.created_at

    # Second hit
    cached2 = cache.get(h)
    assert cached2 is not None

    record_after_2 = cache.get_record(h)
    assert record_after_2 is not None
    assert record_after_2.hit_count == 2


def test_metrics_aggregation(cache: SQLiteCache, mock_upstream: MockUpstreamProvider) -> None:
    req1 = ChatCompletionRequest(model="gpt-4o", messages=[Message(role="user", content="Query 1")])
    req2 = ChatCompletionRequest(model="gpt-4o", messages=[Message(role="user", content="Query 2")])

    resp1 = mock_upstream.generate(req1)
    resp2 = mock_upstream.generate(req2)

    h1 = hash_prompt(req1.messages, model=req1.model)
    h2 = hash_prompt(req2.messages, model=req2.model)

    cache.set(h1, resp1, req1)
    cache.set(h2, resp2, req2)

    # 2 hits on req1, 1 hit on req2
    cache.get(h1)
    cache.get(h1)
    cache.get(h2)

    metrics = cache.get_metrics()
    assert metrics["total_entries"] == 2
    assert metrics["total_hits"] == 3

    assert resp1.usage is not None and resp2.usage is not None
    expected_prompt_saved = (resp1.usage.prompt_tokens * 2) + (resp2.usage.prompt_tokens * 1)
    expected_completion_saved = (resp1.usage.completion_tokens * 2) + (resp2.usage.completion_tokens * 1)
    expected_total_saved = (resp1.usage.total_tokens * 2) + (resp2.usage.total_tokens * 1)

    assert metrics["saved_prompt_tokens"] == expected_prompt_saved
    assert metrics["saved_completion_tokens"] == expected_completion_saved
    assert metrics["saved_total_tokens"] == expected_total_saved


def test_delete_and_clear(cache: SQLiteCache, mock_upstream: MockUpstreamProvider) -> None:
    req = ChatCompletionRequest(model="gpt-4o", messages=[Message(role="user", content="Delete me")])
    resp = mock_upstream.generate(req)
    h = hash_prompt(req.messages, model=req.model)

    cache.set(h, resp, req)
    assert cache.count() == 1
    assert cache.delete(h) is True
    assert cache.delete(h) is False
    assert cache.count() == 0
    assert cache.get(h) is None

    # Test clear
    cache.set("h1", resp, req)
    cache.set("h2", resp, req)
    assert cache.count() == 2
    cleared = cache.clear()
    assert cleared == 2
    assert cache.count() == 0


def test_file_based_sqlite_concurrency() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        file_cache = SQLiteCache(db_path=tmp.name)
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Concurrent test prompt")],
        )
        upstream = MockUpstreamProvider(fixed_timestamp=1700000000)
        resp = upstream.generate(req)
        h = hash_prompt(req.messages, model=req.model)
        file_cache.set(h, resp, req)

        def worker_read() -> bool:
            res = file_cache.get(h)
            return res is not None and res.cached is True

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: worker_read(), range(20)))

        assert all(results)
        record = file_cache.get_record(h)
        assert record is not None
        assert record.hit_count == 20
        file_cache.close()


def test_mock_upstream_determinism() -> None:
    upstream = MockUpstreamProvider(fixed_timestamp=1234567890)
    req1 = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="Deterministic output check")],
    )
    req2 = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="Deterministic output check")],
    )
    resp1 = upstream.generate(req1)
    resp2 = upstream.generate(req2)

    assert resp1.id == resp2.id
    assert resp1.created == 1234567890
    assert resp1.choices[0].message.content == resp2.choices[0].message.content
    assert resp1.usage is not None and resp2.usage is not None
    assert resp1.usage.total_tokens == resp2.usage.total_tokens

    # Different input yields different ID and content
    req_diff = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="Completely different check")],
    )
    resp_diff = upstream.generate(req_diff)
    assert resp_diff.id != resp1.id


def test_mock_upstream_token_estimation() -> None:
    assert estimate_token_count("") == 0
    assert estimate_token_count("abc") == 1
    assert estimate_token_count("12345678") == 2
