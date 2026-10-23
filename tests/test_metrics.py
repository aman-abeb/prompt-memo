"""Unit and integration tests for the /v1/metrics endpoint and pricing models."""

import pytest
from fastapi.testclient import TestClient

from prompt_memo.app import create_app
from prompt_memo.models import DEFAULT_MODEL_PRICING, ModelPricing, get_model_pricing
from prompt_memo.storage import SQLiteCache
from prompt_memo.upstream import MockUpstreamProvider


@pytest.fixture
def isolated_client() -> TestClient:
    """Provides a TestClient with an isolated in-memory SQLiteCache and fixed upstream."""
    cache = SQLiteCache(db_path=":memory:")
    upstream = MockUpstreamProvider(fixed_timestamp=1700000000)
    app = create_app(cache=cache, upstream=upstream, similarity_threshold=0.85)
    return TestClient(app)


def test_model_pricing_computation() -> None:
    pricing = ModelPricing(prompt_price_per_million=2.50, completion_price_per_million=10.00)
    # 1M prompt tokens ($2.50) + 1M completion tokens ($10.00) = $12.50
    assert pricing.compute_cost(1_000_000, 1_000_000) == pytest.approx(12.50)

    # 1000 prompt tokens + 2000 completion tokens
    # (1000/1M * 2.50) + (2000/1M * 10.00) = 0.0025 + 0.0200 = 0.0225
    assert pricing.compute_cost(1000, 2000) == pytest.approx(0.0225)


def test_get_model_pricing_matching_and_fallbacks() -> None:
    # Exact match
    p_gpt4o = get_model_pricing("gpt-4o")
    assert p_gpt4o.prompt_price_per_million == 2.50
    assert p_gpt4o.completion_price_per_million == 10.00

    # Variant / prefix match
    p_variant = get_model_pricing("gpt-4o-2024-08-06")
    assert p_variant.prompt_price_per_million == 2.50

    # Case insensitivity
    p_upper = get_model_pricing("GPT-4O-MINI")
    assert p_upper.prompt_price_per_million == 0.15

    # Unknown model falls back safely
    p_unknown = get_model_pricing("unknown-exotic-model-v99")
    assert p_unknown.prompt_price_per_million == 2.50
    assert p_unknown.completion_price_per_million == 10.00


def test_metrics_empty_cache_state(isolated_client: TestClient) -> None:
    response = isolated_client.get("/v1/metrics")
    assert response.status_code == 200
    data = response.json()

    assert data["total_entries"] == 0
    assert data["total_requests"] == 0
    assert data["total_hits"] == 0
    assert data["exact_hits"] == 0
    assert data["fuzzy_hits"] == 0
    assert data["total_misses"] == 0
    assert data["hit_rate"] == 0.0
    assert data["saved_prompt_tokens"] == 0
    assert data["saved_completion_tokens"] == 0
    assert data["saved_total_tokens"] == 0
    assert data["estimated_dollar_savings"] == 0.0
    assert data["by_model"] == {}
    assert "gpt-4o" in data["pricing"]


def test_metrics_cold_misses(isolated_client: TestClient) -> None:
    # Two distinct queries -> 2 misses
    req1 = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Query number one"}]}
    req2 = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Query number two"}]}

    res1 = isolated_client.post("/v1/chat/completions", json=req1)
    res2 = isolated_client.post("/v1/chat/completions", json=req2)
    assert res1.headers.get("x-cache") == "MISS"
    assert res2.headers.get("x-cache") == "MISS"

    metrics_resp = isolated_client.get("/v1/metrics")
    assert metrics_resp.status_code == 200
    data = metrics_resp.json()

    assert data["total_entries"] == 2
    assert data["total_requests"] == 2
    assert data["total_misses"] == 2
    assert data["total_hits"] == 0
    assert data["exact_hits"] == 0
    assert data["fuzzy_hits"] == 0
    assert data["hit_rate"] == 0.0
    assert data["saved_total_tokens"] == 0
    assert data["estimated_dollar_savings"] == 0.0

    # Model level stats
    assert "gpt-4o" in data["by_model"]
    m = data["by_model"]["gpt-4o"]
    assert m["entries"] == 2
    assert m["total_misses"] == 2
    assert m["total_hits"] == 0


def test_metrics_exact_hits_and_cost_savings(isolated_client: TestClient) -> None:
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is the speed of light?"}],
    }

    # 1. Miss
    r1 = isolated_client.post("/v1/chat/completions", json=payload)
    assert r1.headers.get("x-cache") == "MISS"
    usage1 = r1.json()["usage"]
    prompt_tokens = usage1["prompt_tokens"]
    comp_tokens = usage1["completion_tokens"]

    # 2. Exact Hit #1
    r2 = isolated_client.post("/v1/chat/completions", json=payload)
    assert r2.headers.get("x-cache") == "HIT"

    # 3. Exact Hit #2
    r3 = isolated_client.post("/v1/chat/completions", json=payload)
    assert r3.headers.get("x-cache") == "HIT"

    metrics_resp = isolated_client.get("/v1/metrics")
    data = metrics_resp.json()

    assert data["total_entries"] == 1
    assert data["total_requests"] == 3
    assert data["total_misses"] == 1
    assert data["total_hits"] == 2
    assert data["exact_hits"] == 2
    assert data["fuzzy_hits"] == 0
    assert data["hit_rate"] == pytest.approx(round(2 / 3, 4))

    expected_saved_prompt = prompt_tokens * 2
    expected_saved_comp = comp_tokens * 2
    expected_saved_total = (prompt_tokens + comp_tokens) * 2

    assert data["saved_prompt_tokens"] == expected_saved_prompt
    assert data["saved_completion_tokens"] == expected_saved_comp
    assert data["saved_total_tokens"] == expected_saved_total

    # Verify dollar calculation: gpt-4o rate is $2.50 / $10.00
    pricing = DEFAULT_MODEL_PRICING["gpt-4o"]
    expected_dollars = pricing.compute_cost(expected_saved_prompt, expected_saved_comp)
    assert data["estimated_dollar_savings"] == pytest.approx(round(expected_dollars, 6))

    m_stats = data["by_model"]["gpt-4o"]
    assert m_stats["total_hits"] == 2
    assert m_stats["exact_hits"] == 2
    assert m_stats["fuzzy_hits"] == 0
    assert m_stats["estimated_dollar_savings"] == pytest.approx(round(expected_dollars, 6))


def test_metrics_fuzzy_hits(isolated_client: TestClient) -> None:
    # 1. Miss
    r1 = isolated_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Explain machine learning in simple terms."}]},
    )
    assert r1.headers.get("x-cache") == "MISS"

    # 2. Fuzzy Hit (minor punctuation modification)
    r2 = isolated_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Explain machine learning in simple terms!"}]},
    )
    assert r2.headers.get("x-cache") == "HIT-FUZZY"

    metrics_resp = isolated_client.get("/v1/metrics")
    data = metrics_resp.json()

    assert data["total_requests"] == 2
    assert data["total_misses"] == 1
    assert data["total_hits"] == 1
    assert data["exact_hits"] == 0
    assert data["fuzzy_hits"] == 1
    assert data["hit_rate"] == 0.5
    assert data["saved_total_tokens"] > 0
    assert data["estimated_dollar_savings"] > 0.0

    m_stats = data["by_model"]["gpt-4o"]
    assert m_stats["exact_hits"] == 0
    assert m_stats["fuzzy_hits"] == 1
    assert m_stats["total_hits"] == 1


def test_metrics_mixed_workload_across_multiple_models(isolated_client: TestClient) -> None:
    # Model 1: gpt-4o
    # Request 1: Miss
    isolated_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Analyze climate change trends."}]},
    )
    # Request 2: Exact hit
    isolated_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Analyze climate change trends."}]},
    )
    # Request 3: Fuzzy hit
    isolated_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Analyze climate change trends!"}]},
    )

    # Model 2: gpt-4o-mini
    # Request 4: Miss
    isolated_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is recursion?"}]},
    )
    # Request 5: Exact hit
    isolated_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is recursion?"}]},
    )

    metrics_resp = isolated_client.get("/v1/metrics")
    data = metrics_resp.json()

    assert data["total_entries"] == 2
    assert data["total_requests"] == 5
    assert data["total_misses"] == 2
    assert data["total_hits"] == 3
    assert data["exact_hits"] == 2  # 1 from gpt-4o, 1 from gpt-4o-mini
    assert data["fuzzy_hits"] == 1  # 1 from gpt-4o
    assert data["hit_rate"] == pytest.approx(0.60)

    # Check per-model records
    assert "gpt-4o" in data["by_model"]
    assert "gpt-4o-mini" in data["by_model"]

    gpt4o_stats = data["by_model"]["gpt-4o"]
    assert gpt4o_stats["total_hits"] == 2
    assert gpt4o_stats["exact_hits"] == 1
    assert gpt4o_stats["fuzzy_hits"] == 1

    mini_stats = data["by_model"]["gpt-4o-mini"]
    assert mini_stats["total_hits"] == 1
    assert mini_stats["exact_hits"] == 1
    assert mini_stats["fuzzy_hits"] == 0

    # Total savings must equal sum of model savings
    sum_model_savings = gpt4o_stats["estimated_dollar_savings"] + mini_stats["estimated_dollar_savings"]
    assert data["estimated_dollar_savings"] == pytest.approx(sum_model_savings, abs=1e-5)


def test_custom_pricing_configuration() -> None:
    custom_pricing = {
        "custom-llm": ModelPricing(prompt_price_per_million=100.0, completion_price_per_million=200.0),
    }
    cache = SQLiteCache(db_path=":memory:")
    upstream = MockUpstreamProvider(fixed_timestamp=1700000000)
    app = create_app(cache=cache, upstream=upstream, pricing=custom_pricing)
    client = TestClient(app)

    payload = {
        "model": "custom-llm",
        "messages": [{"role": "user", "content": "Run custom analytics evaluation."}],
    }

    # 1 Miss
    r1 = client.post("/v1/chat/completions", json=payload)
    # 1 Hit
    r2 = client.post("/v1/chat/completions", json=payload)
    assert r2.headers.get("x-cache") == "HIT"

    usage = r1.json()["usage"]
    p_tok = usage["prompt_tokens"]
    c_tok = usage["completion_tokens"]

    expected_cost = custom_pricing["custom-llm"].compute_cost(p_tok, c_tok)

    metrics = client.get("/v1/metrics").json()
    assert metrics["estimated_dollar_savings"] == pytest.approx(round(expected_cost, 6))
    assert metrics["pricing"]["custom-llm"]["prompt_price_per_million"] == 100.0


def test_metrics_reset_after_cache_clear(isolated_client: TestClient) -> None:
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Clear verification prompt"}],
    }
    isolated_client.post("/v1/chat/completions", json=payload)
    isolated_client.post("/v1/chat/completions", json=payload)

    cache: SQLiteCache = isolated_client.app.state.cache
    stats_before = cache.get_stats()
    assert stats_before["total_entries"] == 1
    assert stats_before["total_hits"] == 1

    cache.clear()

    metrics_after = isolated_client.get("/v1/metrics").json()
    assert metrics_after["total_entries"] == 0
    assert metrics_after["total_hits"] == 0
    assert metrics_after["total_misses"] == 0
    assert metrics_after["total_requests"] == 0
    assert metrics_after["hit_rate"] == 0.0
    assert metrics_after["saved_total_tokens"] == 0
    assert metrics_after["estimated_dollar_savings"] == 0.0
