"""PromptMemo runnable entrypoint and quickstart demonstration."""

import sys
from prompt_memo.app import create_app
from prompt_memo.models import ChatCompletionRequest, Message
from prompt_memo.normalizer import build_canonical_representation, hash_prompt
from prompt_memo.storage import SQLiteCache
from prompt_memo.upstream import MockUpstreamProvider

# ASGI application instance for uvicorn (e.g. uvicorn main:app)
app = create_app()


def run_demo() -> None:
    """Runs a quick local verification of normalization, mock upstream, and SQLite caching."""
    model = "gpt-4o"
    raw_messages = [
        Message(role="system", content="  You are a helpful   assistant.  "),
        Message(role="user", content="  Hello   world!  \n\n How are you doing?   "),
    ]
    equivalent_raw_messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="USER", content="Hello world!\nHow are you doing?"),
    ]

    hash1 = hash_prompt(raw_messages, model=model)
    hash2 = hash_prompt(equivalent_raw_messages, model=model)
    canonical_str = build_canonical_representation(raw_messages, model=model)

    print("PromptMemo CLI Demo")
    print("=" * 20)
    print("Canonical representation:")
    for line in canonical_str.splitlines():
        print(f"  {line}")
    print()
    print(f"Exact SHA-256 Hash: {hash1}")
    print(f"Invariant Match: {'SUCCESS' if hash1 == hash2 else 'FAILED'}")
    print()

    # Demonstrate SQLite persistence with deterministic mock LLM upstream
    cache = SQLiteCache(db_path=":memory:")
    upstream = MockUpstreamProvider(fixed_timestamp=1700000000)

    req = ChatCompletionRequest(model=model, messages=raw_messages)

    # First call: cache miss, query upstream
    cached_resp = cache.get(hash1)
    print(f"Cache lookup #1 (before set): {'HIT' if cached_resp else 'MISS'}")

    upstream_resp = upstream.generate(req)
    cache.set(prompt_hash=hash1, response=upstream_resp, request=req, canonical_prompt=canonical_str)
    print(f"Upstream generated reply: '{upstream_resp.choices[0].message.content}'")

    # Second call: cache hit
    cached_resp = cache.get(hash2)
    print(f"Cache lookup #2 (whitespace-equivalent): {'HIT' if cached_resp and cached_resp.cached else 'MISS'}")

    metrics = cache.get_metrics()
    print("\nMetrics Summary:")
    print(f"  Cached Entries: {metrics['total_entries']}")
    print(f"  Total Hits:     {metrics['total_hits']}")
    print(f"  Tokens Saved:   {metrics['saved_total_tokens']}")


def main() -> None:
    """Main CLI entrypoint. Starts server if --serve flag passed, else runs demo."""
    if "--serve" in sys.argv or "-s" in sys.argv:
        import uvicorn
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
    else:
        run_demo()


if __name__ == "__main__":
    main()
    sys.exit(0)
