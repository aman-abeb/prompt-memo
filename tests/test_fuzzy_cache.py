"""Unit and integration tests for MinHash fuzzy cache matching, TTL, and LRU eviction."""

import time
import pytest
from fastapi.testclient import TestClient

from prompt_memo.app import create_app
from prompt_memo.models import ChatCompletionRequest, Message
from prompt_memo.storage import SQLiteCache
from prompt_memo.upstream import MockUpstreamProvider


@pytest.fixture
def cache() -> SQLiteCache:
    """Provides an in-memory SQLiteCache instance."""
    return SQLiteCache(db_path=":memory:")


@pytest.fixture
def upstream() -> MockUpstreamProvider:
    """Provides a fixed deterministic mock upstream provider."""
    return MockUpstreamProvider(fixed_timestamp=1700000000)


def test_cache_stores_minhash_signature(cache: SQLiteCache, upstream: MockUpstreamProvider) -> None:
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="Calculate the Fibonacci sequence.")],
    )
    resp = upstream.generate(req)
    cache.set("fib_hash_001", resp, req)

    record = cache.get_record("fib_hash_001")
    assert record is not None
    assert record.minhash_signature is not None
    assert len(record.minhash_signature.split(",")) == 64


def test_fuzzy_match_hit_and_miss(cache: SQLiteCache, upstream: MockUpstreamProvider) -> None:
    req_base = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="What is the capital of France?")],
    )
    resp_base = upstream.generate(req_base)
    cache.set("france_hash_1", resp_base, req_base)

    # Near-duplicate query (omits question mark)
    req_fuzzy = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="What is the capital of France")],
    )

    # Exact get returns None
    assert cache.get("france_hash_2") is None

    # Fuzzy match succeeds
    fuzzy_match = cache.find_fuzzy_match(req_fuzzy, similarity_threshold=0.85)
    assert fuzzy_match is not None
    assert fuzzy_match.response.cached is True
    assert fuzzy_match.prompt_hash == "france_hash_1"
    assert fuzzy_match.similarity >= 0.85

    # Metrics reflect 1 fuzzy hit
    metrics = cache.get_metrics()
    assert metrics["total_hits"] == 1
    assert metrics["fuzzy_hits"] == 1

    # Dissimilar query does not match
    req_dissimilar = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="How do I bake sourdough bread from scratch?")],
    )
    assert cache.find_fuzzy_match(req_dissimilar, similarity_threshold=0.85) is None


def test_fuzzy_match_model_isolation(cache: SQLiteCache, upstream: MockUpstreamProvider) -> None:
    req_gpt = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="Summarize the history of Rome.")],
    )
    resp = upstream.generate(req_gpt)
    cache.set("rome_gpt4", resp, req_gpt)

    # Same prompt text, but targeted at claude-3-opus
    req_claude = ChatCompletionRequest(
        model="claude-3-opus",
        messages=[Message(role="user", content="Summarize the history of Rome.")],
    )

    # Must not match across model boundaries
    fuzzy_match = cache.find_fuzzy_match(req_claude, similarity_threshold=0.85)
    assert fuzzy_match is None


def test_ttl_expiration_policy(upstream: MockUpstreamProvider) -> None:
    # Cache with 0.05s default TTL
    short_ttl_cache = SQLiteCache(db_path=":memory:", default_ttl=0.05)
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="Short lived prompt")],
    )
    resp = upstream.generate(req)
    short_ttl_cache.set("ttl_prompt", resp, req)

    # Immediate lookup hits
    assert short_ttl_cache.get("ttl_prompt") is not None

    # Wait for TTL to expire
    time.sleep(0.08)

    # Expired entry returns None on get
    assert short_ttl_cache.get("ttl_prompt") is None

    # Expired entry is not returned on fuzzy lookup
    assert short_ttl_cache.find_fuzzy_match(req, similarity_threshold=0.85) is None

    # Maintenance evict
    evicted = short_ttl_cache.evict_expired()
    assert evicted >= 0


def test_lru_capacity_eviction_policy(upstream: MockUpstreamProvider) -> None:
    # Cache limited to 2 entries
    lru_cache = SQLiteCache(db_path=":memory:", max_entries=2)

    req1 = ChatCompletionRequest(model="gpt-4o", messages=[Message(role="user", content="Prompt 1")])
    req2 = ChatCompletionRequest(model="gpt-4o", messages=[Message(role="user", content="Prompt 2")])
    req3 = ChatCompletionRequest(model="gpt-4o", messages=[Message(role="user", content="Prompt 3")])

    resp1 = upstream.generate(req1)
    resp2 = upstream.generate(req2)
    resp3 = upstream.generate(req3)

    lru_cache.set("h1", resp1, req1)
    time.sleep(0.01)
    lru_cache.set("h2", resp2, req2)
    assert lru_cache.count() == 2

    # Access h1 so h2 becomes the least recently accessed
    time.sleep(0.01)
    assert lru_cache.get("h1") is not None

    # Insert 3rd entry: h2 should be evicted
    time.sleep(0.01)
    lru_cache.set("h3", resp3, req3)
    assert lru_cache.count() == 2

    assert lru_cache.get("h1") is not None
    assert lru_cache.get("h2") is None  # Evicted
    assert lru_cache.get("h3") is not None


def test_proxy_fuzzy_hit_headers() -> None:
    cache = SQLiteCache(db_path=":memory:")
    upstream = MockUpstreamProvider(fixed_timestamp=1700000000)
    app = create_app(cache=cache, upstream=upstream, similarity_threshold=0.85)
    client = TestClient(app)

    payload_base = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Explain the theory of general relativity."}],
    }
    # First request: MISS
    res1 = client.post("/v1/chat/completions", json=payload_base)
    assert res1.status_code == 200
    assert res1.headers.get("x-cache") == "MISS"
    original_hash = res1.headers.get("x-prompt-hash")

    # Second request with slight punctuation difference (near-duplicate)
    payload_fuzzy = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Explain the theory of general relativity!"}],
    }
    res2 = client.post("/v1/chat/completions", json=payload_fuzzy)
    assert res2.status_code == 200
    assert res2.headers.get("x-cache") == "HIT-FUZZY"
    assert res2.headers.get("x-prompt-hash") == original_hash

    sim_header = res2.headers.get("x-cache-similarity")
    assert sim_header is not None
    assert float(sim_header) >= 0.85

    data2 = res2.json()
    assert data2["cached"] is True


def test_proxy_fuzzy_matching_disabled_flag() -> None:
    cache = SQLiteCache(db_path=":memory:")
    upstream = MockUpstreamProvider(fixed_timestamp=1700000000)
    app = create_app(cache=cache, upstream=upstream, fuzzy_matching=False)
    client = TestClient(app)

    payload_base = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is asynchronous programming?"}],
    }
    payload_fuzzy = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is asynchronous programming!"}],
    }

    res1 = client.post("/v1/chat/completions", json=payload_base)
    assert res1.headers.get("x-cache") == "MISS"

    # Fuzzy matching is disabled -> MISS
    res2 = client.post("/v1/chat/completions", json=payload_fuzzy)
    assert res2.headers.get("x-cache") == "MISS"
