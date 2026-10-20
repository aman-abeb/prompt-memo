# PromptMemo

<!-- builder:badges:start -->
[![CI](https://github.com/aman-abeb/prompt-memo/actions/workflows/ci.yml/badge.svg)](https://github.com/aman-abeb/prompt-memo/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white) ![License: MIT](https://img.shields.io/badge/license-MIT-green)
<!-- builder:badges:end -->

Fast local caching proxy for LLM APIs with normalized MinHash fuzzy matching, token savings analytics, and zero-network mock replay.

## Features

- **OpenAI-Compatible Schemas**: Standard chat completion models (`ChatCompletionRequest`, `ChatCompletionResponse`, `Message`, and `Usage`).
- **FastAPI ASGI Proxy**: Transparent `/v1/chat/completions` proxy routing with `X-Cache`, `X-Prompt-Hash`, and `X-Latency-Ms` headers.
- **Canonical Prompt Normalization**: Unicode NFKC normalization, whitespace collapse, and structured key serialization ensuring identical semantic prompts map to the same cache key.
- **Deterministic SHA-256 Hashing**: Model-aware, parameter-configurable hashing for exact-match retrieval.
- **SQLite Storage Engine**: Thread-safe SQLite persistence layer with connection pooling, hit counting, and token savings analytics.
- **Deterministic Mock Upstream**: Offline LLM provider generating reproducible OpenAI-compatible responses without external network calls.
- **Zero-Network Ready**: Fully offline test harness with deterministic behavior and no external network calls required.

## Quickstart

### Installation

```bash
pip install -r requirements.txt
```

### Running the Demo

Execute the entrypoint directly to observe canonical normalization, prompt hashing, and SQLite caching with mock replay:

```bash
python main.py
```

### Running the Proxy Server

Start the ASGI proxy server locally:

```bash
python main.py --serve
# or
uvicorn main:app --port 8000
```

## Example

Running `python main.py` produces the following output:

```text
PromptMemo CLI Demo
====================
Canonical representation:
  model:gpt-4o
  role:system:You are a helpful assistant.
  role:user:Hello world!
  How are you doing?

Exact SHA-256 Hash: e346513ddc4caaf7b86e0fc25102a7b8e50b73c467aae162b7ae4d039f99aa9f
Invariant Match: SUCCESS

Cache lookup #1 (before set): MISS
Upstream generated reply: '[Mock:gpt-4o] Echoing query: 'Hello world!
How are you doing?' (seed: e346513ddc4c)'
Cache lookup #2 (whitespace-equivalent): HIT

Metrics Summary:
  Cached Entries: 1
  Total Hits:     1
  Tokens Saved:   30
```

## How it works

1. **Message Sanitization**: Messages undergo unicode normalization (`unicodedata.normalize("NFKC", ...)`), uniform line break translation, and whitespace collapse while preserving essential line breaks.
2. **Canonical Serialization**: Prompts are converted to a sorted, unambiguous key format combining the target model and sequenced messages (`role:content`).
3. **Hash Generation**: The serialized representation is hashed using SHA-256 to produce an exact lookup key.
4. **SQLite Cache Storage**: Exact hashes index persisted completions along with token usage and access timestamps in SQLite.
5. **Deterministic Mock Fallback**: When upstream network access is disabled, `MockUpstreamProvider` simulates LLM responses deterministically.

## Running tests

Run the test suite offline with pytest:

```bash
pytest
```

<!-- builder:roadmap:start -->
## Status

**In active development:** 0 of 3 milestones done.

## Roadmap

- [ ] **Core proxy routing & exact-match SQLite cache**: Build the ASGI application with /v1/chat/completions endpoint, deterministic mock upstream provider, exact normalized prompt hashing, SQLite persistence layer, and unit tests using TestClient.
- [ ] **Fuzzy MinHash shingling & LRU eviction**: Implement k-shingle MinHash indexing for fuzzy prompt similarity matching against a configurable similarity threshold, add TTL/LRU cache eviction policies, and benchmark similarity retrieval accuracy.
- [ ] **Streaming SSE replay, metrics endpoint & CLI demo**: Add Server-Sent Events (SSE) streaming replay for cached chunk responses, real upstream HTTP forwarding fallback via env var, /v1/metrics analytics endpoint for token/cost accounting, and an end-to-end integration test suite.
<!-- builder:roadmap:end -->
