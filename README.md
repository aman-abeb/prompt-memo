# PromptMemo

<!-- builder:badges:start -->
[![CI](https://github.com/aman-abeb/prompt-memo/actions/workflows/ci.yml/badge.svg)](https://github.com/aman-abeb/prompt-memo/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white) ![License: MIT](https://img.shields.io/badge/license-MIT-green)
<!-- builder:badges:end -->


Fast local caching proxy for LLM APIs with normalized MinHash fuzzy matching, token savings analytics, and zero-network mock replay.

## Features

- **OpenAI-Compatible Schemas**: Standard chat completion models (`ChatCompletionRequest`, `ChatCompletionResponse`, `Message`, and `Usage`).
- **Canonical Prompt Normalization**: Unicode NFKC normalization, whitespace collapse, and structured key serialization ensuring identical semantic prompts map to the same cache key.
- **Deterministic SHA-256 Hashing**: Model-aware, parameter-configurable hashing for exact-match retrieval.
- **Zero-Network Ready**: Fully offline test harness with deterministic behavior and no external network calls required.

## Quickstart

### Installation

```bash
pip install -r requirements.txt
```

### Running the Demo

Execute the entrypoint directly to observe canonical normalization and prompt hashing:

```bash
python main.py
```

## Example

Running `python main.py` produces the following output:

```text
PromptMemo CLI Demo
===================
Raw prompt input:
  Role: user
  Content: '  Hello   world!  \n\n How are you doing?   '

Canonical representation:
  model:gpt-4o
  role:user:Hello world!
  How are you doing?

Exact SHA-256 Hash:
  e346513ddc4caaf7b86e0fc25102a7b8e50b73c467aae162b7ae4d039f99aa9f

Whitespace-invariant match verification: SUCCESS (hashes match!)
```

## How it works

1. **Message Sanitization**: Messages undergo unicode normalization (`unicodedata.normalize("NFKC", ...)`), uniform line break translation, and whitespace collapse while preserving essential line breaks.
2. **Canonical Serialization**: Prompts are converted to a sorted, unambiguous key format combining the target model and sequenced messages (`role:content`).
3. **Hash Generation**: The serialized representation is hashed using SHA-256 to produce an exact lookup key for downstream storage layers (e.g. SQLite).

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
