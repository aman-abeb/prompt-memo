"""Unit and integration tests for FastAPI proxy and /v1/chat/completions endpoint."""

import pytest
from fastapi.testclient import TestClient

from prompt_memo.app import create_app
from prompt_memo.storage import SQLiteCache
from prompt_memo.upstream import MockUpstreamProvider


@pytest.fixture
def isolated_client() -> TestClient:
    """Provides a TestClient with an isolated in-memory SQLiteCache and fixed upstream."""
    cache = SQLiteCache(db_path=":memory:")
    upstream = MockUpstreamProvider(fixed_timestamp=1700000000)
    app = create_app(cache=cache, upstream=upstream)
    return TestClient(app)


def test_health_check_endpoint(isolated_client: TestClient) -> None:
    response = isolated_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "prompt-memo", "version": "0.1.0"}

    root_response = isolated_client.get("/")
    assert root_response.status_code == 200
    assert root_response.json()["status"] == "ok"


def test_chat_completion_miss_and_subsequent_hit(isolated_client: TestClient) -> None:
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Japan?"},
        ],
    }

    # Request 1: Cache MISS
    res1 = isolated_client.post("/v1/chat/completions", json=payload)
    assert res1.status_code == 200
    assert res1.headers.get("x-cache") == "MISS"
    assert "x-prompt-hash" in res1.headers
    assert len(res1.headers["x-prompt-hash"]) == 64
    assert float(res1.headers["x-latency-ms"]) >= 0.0

    data1 = res1.json()
    assert data1["cached"] is False
    assert data1["model"] == "gpt-4o"
    assert len(data1["choices"]) == 1
    assert "capital of Japan" in data1["choices"][0]["message"]["content"]

    # Request 2: Identical prompt -> Cache HIT
    res2 = isolated_client.post("/v1/chat/completions", json=payload)
    assert res2.status_code == 200
    assert res2.headers.get("x-cache") == "HIT"
    assert res2.headers.get("x-prompt-hash") == res1.headers.get("x-prompt-hash")
    assert float(res2.headers["x-latency-ms"]) >= 0.0

    data2 = res2.json()
    assert data2["cached"] is True
    assert data2["choices"][0]["message"]["content"] == data1["choices"][0]["message"]["content"]
    assert data2["usage"] == data1["usage"]


def test_chat_completion_whitespace_invariance_hit(isolated_client: TestClient) -> None:
    base_payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello world!"},
        ],
    }
    equivalent_payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "USER", "content": "   Hello    world!   \n\n"},
        ],
    }

    res1 = isolated_client.post("/v1/chat/completions", json=base_payload)
    assert res1.status_code == 200
    assert res1.headers.get("x-cache") == "MISS"

    # Whitespace and casing normalized to the exact same hash
    res2 = isolated_client.post("/v1/chat/completions", json=equivalent_payload)
    assert res2.status_code == 200
    assert res2.headers.get("x-cache") == "HIT"
    assert res2.headers.get("x-prompt-hash") == res1.headers.get("x-prompt-hash")
    assert res2.json()["cached"] is True


def test_chat_completion_different_prompts_separate(isolated_client: TestClient) -> None:
    req_a = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Question A"}]}
    req_b = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Question B"}]}

    res_a = isolated_client.post("/v1/chat/completions", json=req_a)
    res_b = isolated_client.post("/v1/chat/completions", json=req_b)

    assert res_a.headers.get("x-cache") == "MISS"
    assert res_b.headers.get("x-cache") == "MISS"
    assert res_a.headers.get("x-prompt-hash") != res_b.headers.get("x-prompt-hash")


def test_chat_completion_temperature_sensitivity(isolated_client: TestClient) -> None:
    req_temp0 = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Temperature prompt"}],
        "temperature": 0.0,
    }
    req_temp1 = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Temperature prompt"}],
        "temperature": 0.7,
    }

    res0 = isolated_client.post("/v1/chat/completions", json=req_temp0)
    res1 = isolated_client.post("/v1/chat/completions", json=req_temp1)

    assert res0.headers.get("x-cache") == "MISS"
    assert res1.headers.get("x-cache") == "MISS"
    assert res0.headers.get("x-prompt-hash") != res1.headers.get("x-prompt-hash")


def test_streaming_not_supported_returns_400(isolated_client: TestClient) -> None:
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Stream this please"}],
        "stream": True,
    }
    response = isolated_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    assert "Streaming" in response.json()["detail"]


def test_invalid_request_returns_422(isolated_client: TestClient) -> None:
    # Missing 'messages' field
    response = isolated_client.post("/v1/chat/completions", json={"model": "gpt-4o"})
    assert response.status_code == 422

    # Missing 'model' field
    response_no_model = isolated_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response_no_model.status_code == 422


def test_proxy_updates_cache_metrics(isolated_client: TestClient) -> None:
    cache = isolated_client.app.state.cache
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Metrics verification prompt"}],
    }

    # Initial miss
    isolated_client.post("/v1/chat/completions", json=payload)
    metrics_after_miss = cache.get_metrics()
    assert metrics_after_miss["total_entries"] == 1
    assert metrics_after_miss["total_hits"] == 0

    # Two consecutive hits
    isolated_client.post("/v1/chat/completions", json=payload)
    isolated_client.post("/v1/chat/completions", json=payload)

    metrics_after_hits = cache.get_metrics()
    assert metrics_after_hits["total_entries"] == 1
    assert metrics_after_hits["total_hits"] == 2
    assert metrics_after_hits["saved_total_tokens"] > 0
