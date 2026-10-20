"""SQLite persistence layer for exact prompt hash caching and metrics tracking."""

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, Optional

from prompt_memo.models import ChatCompletionRequest, ChatCompletionResponse


@dataclass
class CacheRecord:
    """Metadata record for a stored completion in SQLite."""

    prompt_hash: str
    model: str
    canonical_prompt: str
    response_json: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    hit_count: int
    created_at: float
    last_accessed_at: float


class SQLiteCache:
    """Thread-safe SQLite storage engine for exact-match prompt completion caching."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Retrieves or creates a thread-local SQLite connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for high concurrency when using disk databases
            if self.db_path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            self._local.conn = conn
        return conn

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for thread-safe database operations."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def _init_db(self) -> None:
        """Initializes the SQLite schema if not already present."""
        with self._cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    canonical_prompt TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prompt_cache_last_accessed
                ON prompt_cache(last_accessed_at);
                """
            )

    def get(self, prompt_hash: str) -> Optional[ChatCompletionResponse]:
        """Retrieves cached ChatCompletionResponse for prompt_hash and increments hit counter.

        Returns None if not present in cache. Response is returned with cached=True.
        """
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                """
                UPDATE prompt_cache
                SET hit_count = hit_count + 1, last_accessed_at = ?
                WHERE prompt_hash = ?
                RETURNING response_json;
                """,
                (now, prompt_hash),
            )
            row = cur.fetchone()
            if row is None:
                return None

            response_data = json.loads(row["response_json"])
            response_data["cached"] = True
            return ChatCompletionResponse.model_validate(response_data)

    def get_record(self, prompt_hash: str) -> Optional[CacheRecord]:
        """Retrieves the raw CacheRecord metadata without incrementing hit count."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT prompt_hash, model, canonical_prompt, response_json,
                       prompt_tokens, completion_tokens, total_tokens,
                       hit_count, created_at, last_accessed_at
                FROM prompt_cache
                WHERE prompt_hash = ?;
                """,
                (prompt_hash,),
            )
            row = cur.fetchone()
            if row is None:
                return None

            return CacheRecord(
                prompt_hash=row["prompt_hash"],
                model=row["model"],
                canonical_prompt=row["canonical_prompt"],
                response_json=row["response_json"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                total_tokens=row["total_tokens"],
                hit_count=row["hit_count"],
                created_at=row["created_at"],
                last_accessed_at=row["last_accessed_at"],
            )

    def set(
        self,
        prompt_hash: str,
        response: ChatCompletionResponse,
        request: Optional[ChatCompletionRequest] = None,
        canonical_prompt: str = "",
    ) -> None:
        """Stores or updates a ChatCompletionResponse in SQLite."""
        now = time.time()
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        total_tokens = response.usage.total_tokens if response.usage else (prompt_tokens + completion_tokens)
        model_name = request.model if request else response.model

        # Ensure serialized payload has cached=False as base state
        response_dict = response.model_dump()
        response_dict["cached"] = False
        serialized = json.dumps(response_dict)

        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO prompt_cache (
                    prompt_hash, model, canonical_prompt, response_json,
                    prompt_tokens, completion_tokens, total_tokens,
                    hit_count, created_at, last_accessed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(prompt_hash) DO UPDATE SET
                    response_json = excluded.response_json,
                    prompt_tokens = excluded.prompt_tokens,
                    completion_tokens = excluded.completion_tokens,
                    total_tokens = excluded.total_tokens,
                    last_accessed_at = excluded.last_accessed_at;
                """,
                (
                    prompt_hash,
                    model_name,
                    canonical_prompt,
                    serialized,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    now,
                    now,
                ),
            )

    def delete(self, prompt_hash: str) -> bool:
        """Deletes a cached item by prompt_hash. Returns True if deleted."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM prompt_cache WHERE prompt_hash = ?;", (prompt_hash,))
            return cur.rowcount > 0

    def clear(self) -> int:
        """Clears all entries from the cache table. Returns deleted row count."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM prompt_cache;")
            return cur.rowcount

    def count(self) -> int:
        """Returns total number of cached entries."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM prompt_cache;")
            row = cur.fetchone()
            return int(row["total"]) if row else 0

    def get_metrics(self) -> dict[str, Any]:
        """Aggregates cache performance metrics including hits and tokens saved."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) as total_entries,
                    COALESCE(SUM(hit_count), 0) as total_hits,
                    COALESCE(SUM(prompt_tokens * hit_count), 0) as saved_prompt_tokens,
                    COALESCE(SUM(completion_tokens * hit_count), 0) as saved_completion_tokens,
                    COALESCE(SUM(total_tokens * hit_count), 0) as saved_total_tokens
                FROM prompt_cache;
                """
            )
            row = cur.fetchone()
            if not row:
                return {
                    "total_entries": 0,
                    "total_hits": 0,
                    "saved_prompt_tokens": 0,
                    "saved_completion_tokens": 0,
                    "saved_total_tokens": 0,
                }

            return {
                "total_entries": int(row["total_entries"]),
                "total_hits": int(row["total_hits"]),
                "saved_prompt_tokens": int(row["saved_prompt_tokens"]),
                "saved_completion_tokens": int(row["saved_completion_tokens"]),
                "saved_total_tokens": int(row["saved_total_tokens"]),
            }

    def close(self) -> None:
        """Closes the active connection for the current thread if open."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
