"""FastAPI ASGI proxy application for OpenAI-compatible chat completions."""

import time
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status

from prompt_memo.models import ChatCompletionRequest, ChatCompletionResponse
from prompt_memo.normalizer import build_canonical_representation, hash_prompt
from prompt_memo.storage import SQLiteCache
from prompt_memo.upstream import MockUpstreamProvider, UpstreamProvider


def get_cache(request: Request) -> SQLiteCache:
    """Dependency provider for the cache instance stored on app state."""
    return request.app.state.cache


def get_upstream(request: Request) -> UpstreamProvider:
    """Dependency provider for the upstream provider instance stored on app state."""
    return request.app.state.upstream


def create_app(
    cache: Optional[SQLiteCache] = None,
    upstream: Optional[UpstreamProvider] = None,
) -> FastAPI:
    """Factory function configuring and returning the FastAPI proxy application."""
    app = FastAPI(
        title="PromptMemo",
        description="Fast local caching proxy for LLM APIs with exact-match SQLite storage.",
        version="0.1.0",
    )

    app.state.cache = cache if cache is not None else SQLiteCache(":memory:")
    app.state.upstream = upstream if upstream is not None else MockUpstreamProvider()

    @app.get("/", tags=["Health"])
    @app.get("/health", tags=["Health"])
    async def health_check() -> dict[str, str]:
        """Service health check endpoint."""
        return {"status": "ok", "service": "prompt-memo", "version": "0.1.0"}

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        summary="Create chat completion",
        tags=["Chat Completions"],
    )
    async def create_chat_completion(
        payload: ChatCompletionRequest,
        response: Response,
        cache_dep: Annotated[SQLiteCache, Depends(get_cache)],
        upstream_dep: Annotated[UpstreamProvider, Depends(get_upstream)],
    ) -> ChatCompletionResponse:
        """Transparently intercepts chat completion requests.

        Computes canonical prompt normalization and SHA-256 hash:
        - Cache HIT: returns cached response with X-Cache: HIT.
        - Cache MISS: generates response via upstream provider, persists in cache,
          and returns response with X-Cache: MISS.
        """
        if payload.stream:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Streaming completions are currently not supported (scheduled for milestone 3).",
            )

        start_time = time.perf_counter()
        prompt_hash = hash_prompt(
            messages=payload.messages,
            model=payload.model,
            temperature=payload.temperature,
        )

        cached_response = cache_dep.get(prompt_hash)
        if cached_response is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Prompt-Hash"] = prompt_hash
            response.headers["X-Latency-Ms"] = f"{elapsed_ms:.2f}"
            return cached_response

        # Cache miss: query upstream provider
        completion_response = upstream_dep.generate(payload)

        # Store in SQLite cache
        canonical_prompt = build_canonical_representation(
            messages=payload.messages,
            model=payload.model,
            temperature=payload.temperature,
        )
        cache_dep.set(
            prompt_hash=prompt_hash,
            response=completion_response,
            request=payload,
            canonical_prompt=canonical_prompt,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Prompt-Hash"] = prompt_hash
        response.headers["X-Latency-Ms"] = f"{elapsed_ms:.2f}"

        return completion_response

    return app


# Default ASGI application instance for uvicorn
app = create_app()
