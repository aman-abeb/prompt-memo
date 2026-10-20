"""PromptMemo: Local caching proxy for LLM APIs."""

from prompt_memo.app import create_app
from prompt_memo.minhash import (
    MinHasher,
    compute_shingles,
    estimate_similarity,
    exact_jaccard,
)
from prompt_memo.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    Usage,
)
from prompt_memo.storage import SQLiteCache
from prompt_memo.upstream import MockUpstreamProvider

__version__ = "0.1.0"

__all__ = [
    "create_app",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChoice",
    "Message",
    "Usage",
    "SQLiteCache",
    "MockUpstreamProvider",
    "MinHasher",
    "compute_shingles",
    "exact_jaccard",
    "estimate_similarity",
]
