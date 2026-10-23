"""SQLite persistence layer with MinHash fuzzy matching, TTL, and LRU eviction policies."""

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, Optional, Sequence, Union

from prompt_memo.minhash import MinHasher, deserialize_signature, serialize_signature
from prompt_memo.models import ChatCompletionRequest, ChatCompletionResponse
from prompt_memo.normalizer import build_canonical_representation, extract_prompt_text


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
    fuzzy_hit_count: int = 0
    expires_at: Optional[float] = None
    minhash_signature: Optional[str] = None
    temperature: Optional[float] = None


@dataclass
class FuzzyMatchResult:
    """Result of a fuzzy cache lookup."""

    response: ChatCompletionResponse
    similarity: float
    prompt_hash: str
    canonical_prompt: str


class SQLiteCache:
    """Thread-safe SQLite storage engine for exact and fuzzy prompt completion caching."""

    def __init__(
        self,
        db_path: str = ":memory:",
        default_ttl: Optional[float] = None,
        max_entries: Optional[int] = None,
        minhasher: Optional[MinHasher] = None,
    ) -> None:
        self.db_path = db_path
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self.minhasher = minhasher if minhasher is not None else MinHasher()
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Retrieves or creates the SQLite connection."""
        if self._conn is None:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            if self.db_path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            self._conn = conn
        return self._conn

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
        """Initializes the SQLite schema and performs backward-compatible column upgrades."""
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
                    fuzzy_hit_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    expires_at REAL,
                    minhash_signature TEXT,
                    temperature REAL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_stats (
                    metric_name TEXT NOT NULL,
                    model TEXT NOT NULL DEFAULT '',
                    count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(metric_name, model)
                );
                """
            )
            # Schema migration verification
            cur.execute("PRAGMA table_info(prompt_cache);")
            columns = {row["name"] for row in cur.fetchall()}
            if "fuzzy_hit_count" not in columns:
                cur.execute("ALTER TABLE prompt_cache ADD COLUMN fuzzy_hit_count INTEGER NOT NULL DEFAULT 0;")
            if "expires_at" not in columns:
                cur.execute("ALTER TABLE prompt_cache ADD COLUMN expires_at REAL;")
            if "minhash_signature" not in columns:
                cur.execute("ALTER TABLE prompt_cache ADD COLUMN minhash_signature TEXT;")
            if "temperature" not in columns:
                cur.execute("ALTER TABLE prompt_cache ADD COLUMN temperature REAL;")

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prompt_cache_last_accessed
                ON prompt_cache(last_accessed_at);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prompt_cache_expires_at
                ON prompt_cache(expires_at);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prompt_cache_model
                ON prompt_cache(model);
                """
            )

    def record_miss(self, model: str = "") -> None:
        """Records a cache miss for the specified model in cache_stats."""
        clean_model = model.strip() if model else ""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO cache_stats (metric_name, model, count)
                VALUES ('miss', ?, 1)
                ON CONFLICT(metric_name, model) DO UPDATE SET count = count + 1;
                """,
                (clean_model,),
            )

    def get(self, prompt_hash: str) -> Optional[ChatCompletionResponse]:
        """Retrieves cached ChatCompletionResponse for prompt_hash and increments hit counter.

        Returns None if not present or expired. Response is returned with cached=True.
        """
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT response_json, expires_at
                FROM prompt_cache
                WHERE prompt_hash = ?;
                """,
                (prompt_hash,),
            )
            row = cur.fetchone()
            if row is None:
                return None

            # Enforce TTL expiration
            if row["expires_at"] is not None and row["expires_at"] <= now:
                cur.execute("DELETE FROM prompt_cache WHERE prompt_hash = ?;", (prompt_hash,))
                return None

            cur.execute(
                """
                UPDATE prompt_cache
                SET hit_count = hit_count + 1, last_accessed_at = ?
                WHERE prompt_hash = ?;
                """,
                (now, prompt_hash),
            )

            response_data = json.loads(row["response_json"])
            response_data["cached"] = True
            return ChatCompletionResponse.model_validate(response_data)

    def find_fuzzy_match(
        self,
        query: Union[str, ChatCompletionRequest],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        similarity_threshold: float = 0.85,
        now: Optional[float] = None,
    ) -> Optional[FuzzyMatchResult]:
        """Searches for the most similar cached prompt using MinHash Jaccard similarity.

        If the highest similarity score exceeds similarity_threshold, increments both
        hit_count and fuzzy_hit_count and returns the cached response with cached=True.
        Filters candidates by matching model and temperature.
        """
        current_time = time.time() if now is None else now

        if isinstance(query, ChatCompletionRequest):
            query_text = extract_prompt_text(query.messages)
            if not query_text:
                query_text = build_canonical_representation(query.messages, model=query.model, temperature=query.temperature)
            target_model = query.model if model is None else model
            target_temp = query.temperature if temperature is None else temperature
        else:
            query_text = str(query)
            target_model = model
            target_temp = temperature

        if not query_text or not query_text.strip():
            return None

        query_sig = self.minhasher.compute_signature(query_text)

        with self._cursor() as cur:
            conditions = ["(expires_at IS NULL OR expires_at > ?)"]
            params: list[Any] = [current_time]

            if target_model is not None:
                conditions.append("model = ?")
                params.append(target_model)

            if target_temp is not None:
                conditions.append("(temperature IS NULL OR ABS(temperature - ?) < 0.001)")
                params.append(target_temp)

            where_clause = " AND ".join(conditions)
            cur.execute(
                f"""
                SELECT prompt_hash, canonical_prompt, response_json,
                       minhash_signature, last_accessed_at
                FROM prompt_cache
                WHERE {where_clause};
                """,
                tuple(params),
            )
            rows = cur.fetchall()

            best_sim = -1.0
            best_row = None
            for row in rows:
                sig_raw = row["minhash_signature"]
                if not sig_raw:
                    continue
                cand_sig = deserialize_signature(sig_raw)
                sim = self.minhasher.estimate_similarity(query_sig, cand_sig)
                if sim >= similarity_threshold:
                    if sim > best_sim:
                        best_sim = sim
                        best_row = row
                    elif abs(sim - best_sim) < 1e-6 and best_row is not None:
                        if row["last_accessed_at"] > best_row["last_accessed_at"]:
                            best_sim = sim
                            best_row = row

            if best_row is None:
                return None

            cur.execute(
                """
                UPDATE prompt_cache
                SET hit_count = hit_count + 1,
                    fuzzy_hit_count = fuzzy_hit_count + 1,
                    last_accessed_at = ?
                WHERE prompt_hash = ?;
                """,
                (current_time, best_row["prompt_hash"]),
            )
            response_data = json.loads(best_row["response_json"])
            response_data["cached"] = True
            return FuzzyMatchResult(
                response=ChatCompletionResponse.model_validate(response_data),
                similarity=best_sim,
                prompt_hash=best_row["prompt_hash"],
                canonical_prompt=best_row["canonical_prompt"],
            )

    def get_record(self, prompt_hash: str) -> Optional[CacheRecord]:
        """Retrieves the raw CacheRecord metadata without incrementing hit count."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT prompt_hash, model, canonical_prompt, response_json,
                       prompt_tokens, completion_tokens, total_tokens,
                       hit_count, fuzzy_hit_count, created_at, last_accessed_at,
                       expires_at, minhash_signature, temperature
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
                fuzzy_hit_count=row["fuzzy_hit_count"],
                created_at=row["created_at"],
                last_accessed_at=row["last_accessed_at"],
                expires_at=row["expires_at"],
                minhash_signature=row["minhash_signature"],
                temperature=row["temperature"],
            )

    def set(
        self,
        prompt_hash: str,
        response: ChatCompletionResponse,
        request: Optional[ChatCompletionRequest] = None,
        canonical_prompt: str = "",
        minhash_signature: Optional[Union[str, Sequence[int]]] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """Stores or updates a ChatCompletionResponse with MinHash signature and TTL in SQLite."""
        now = time.time()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = (now + effective_ttl) if effective_ttl is not None else None

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        total_tokens = response.usage.total_tokens if response.usage else (prompt_tokens + completion_tokens)
        model_name = request.model if request else response.model

        # Compute MinHash signature if not provided
        if minhash_signature is None:
            text_for_minhash = extract_prompt_text(request.messages) if request else canonical_prompt
            if not text_for_minhash:
                text_for_minhash = canonical_prompt
            sig = self.minhasher.compute_signature(text_for_minhash)
            sig_str = serialize_signature(sig)
        elif isinstance(minhash_signature, (list, tuple)):
            sig_str = serialize_signature(minhash_signature)
        else:
            sig_str = str(minhash_signature)

        temp_val: Optional[float] = None
        if request is not None and request.temperature is not None:
            temp_val = float(request.temperature)
        elif canonical_prompt:
            m = re.search(r"(?:^|\n)temp:([0-9.]+)(?:$|\n)", canonical_prompt)
            if m:
                temp_val = float(m.group(1))

        response_dict = response.model_dump()
        response_dict["cached"] = False
        serialized = json.dumps(response_dict)

        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO prompt_cache (
                    prompt_hash, model, canonical_prompt, response_json,
                    prompt_tokens, completion_tokens, total_tokens,
                    hit_count, fuzzy_hit_count, created_at, last_accessed_at,
                    expires_at, minhash_signature, temperature
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?)
                ON CONFLICT(prompt_hash) DO UPDATE SET
                    response_json = excluded.response_json,
                    prompt_tokens = excluded.prompt_tokens,
                    completion_tokens = excluded.completion_tokens,
                    total_tokens = excluded.total_tokens,
                    last_accessed_at = excluded.last_accessed_at,
                    expires_at = excluded.expires_at,
                    minhash_signature = excluded.minhash_signature,
                    temperature = excluded.temperature;
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
                    expires_at,
                    sig_str,
                    temp_val,
                ),
            )

        # Evict expired and enforce LRU capacity if configured
        self.evict_expired(now)
        if self.max_entries is not None:
            self.evict_lru(self.max_entries)

    def evict_expired(self, now: Optional[float] = None) -> int:
        """Evicts all records whose TTL has elapsed. Returns count of deleted entries."""
        current_time = time.time() if now is None else now
        with self._cursor() as cur:
            cur.execute(
                """
                DELETE FROM prompt_cache
                WHERE expires_at IS NOT NULL AND expires_at <= ?;
                """,
                (current_time,),
            )
            return cur.rowcount

    def evict_lru(self, max_entries: Optional[int] = None) -> int:
        """Evicts least recently accessed entries until count <= max_entries. Returns count."""
        limit = self.max_entries if max_entries is None else max_entries
        if limit is None or limit <= 0:
            return 0

        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM prompt_cache;")
            row = cur.fetchone()
            cnt = int(row["cnt"]) if row else 0
            if cnt <= limit:
                return 0

            excess = cnt - limit
            cur.execute(
                """
                DELETE FROM prompt_cache
                WHERE prompt_hash IN (
                    SELECT prompt_hash FROM prompt_cache
                    ORDER BY last_accessed_at ASC, created_at ASC
                    LIMIT ?
                );
                """,
                (excess,),
            )
            return cur.rowcount

    def delete(self, prompt_hash: str) -> bool:
        """Deletes a cached item by prompt_hash. Returns True if deleted."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM prompt_cache WHERE prompt_hash = ?;", (prompt_hash,))
            return cur.rowcount > 0

    def clear(self) -> int:
        """Clears all entries from cache and resets stats counters. Returns deleted row count."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM cache_stats;")
            cur.execute("DELETE FROM prompt_cache;")
            return cur.rowcount

    def count(self) -> int:
        """Returns total number of cached entries."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM prompt_cache;")
            row = cur.fetchone()
            return int(row["total"]) if row else 0

    def get_stats(self) -> dict[str, Any]:
        """Aggregates cache performance metrics including exact vs fuzzy hit breakdowns."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as total_entries FROM prompt_cache;")
            row_cnt = cur.fetchone()
            total_entries = int(row_cnt["total_entries"]) if row_cnt else 0

            cur.execute(
                """
                SELECT
                    COALESCE(SUM(hit_count), 0) as total_hits,
                    COALESCE(SUM(fuzzy_hit_count), 0) as fuzzy_hits,
                    COALESCE(SUM(prompt_tokens * hit_count), 0) as saved_prompt_tokens,
                    COALESCE(SUM(completion_tokens * hit_count), 0) as saved_completion_tokens,
                    COALESCE(SUM(total_tokens * hit_count), 0) as saved_total_tokens
                FROM prompt_cache;
                """
            )
            agg = cur.fetchone()
            total_hits = int(agg["total_hits"]) if agg else 0
            fuzzy_hits = int(agg["fuzzy_hits"]) if agg else 0
            exact_hits = total_hits - fuzzy_hits
            saved_prompt_tokens = int(agg["saved_prompt_tokens"]) if agg else 0
            saved_completion_tokens = int(agg["saved_completion_tokens"]) if agg else 0
            saved_total_tokens = int(agg["saved_total_tokens"]) if agg else 0

            cur.execute(
                """
                SELECT COALESCE(SUM(count), 0) as total_misses
                FROM cache_stats
                WHERE metric_name = 'miss';
                """
            )
            miss_row = cur.fetchone()
            total_misses = int(miss_row["total_misses"]) if miss_row else 0

            total_requests = total_hits + total_misses
            hit_rate = (total_hits / total_requests) if total_requests > 0 else 0.0

            # Per-model breakdown
            cur.execute(
                """
                SELECT
                    model,
                    COUNT(*) as entries,
                    COALESCE(SUM(hit_count), 0) as hits,
                    COALESCE(SUM(fuzzy_hit_count), 0) as fuzzy_hits,
                    COALESCE(SUM(prompt_tokens * hit_count), 0) as saved_prompt_tokens,
                    COALESCE(SUM(completion_tokens * hit_count), 0) as saved_completion_tokens,
                    COALESCE(SUM(total_tokens * hit_count), 0) as saved_total_tokens
                FROM prompt_cache
                GROUP BY model;
                """
            )
            model_rows = cur.fetchall()

            cur.execute(
                """
                SELECT model, count as misses
                FROM cache_stats
                WHERE metric_name = 'miss' AND model != '';
                """
            )
            miss_by_model = {row["model"]: int(row["misses"]) for row in cur.fetchall()}

            by_model: dict[str, dict[str, Any]] = {}
            for r in model_rows:
                m_name = r["model"]
                m_hits = int(r["hits"])
                m_fuzzy = int(r["fuzzy_hits"])
                m_exact = m_hits - m_fuzzy
                m_prompt_tok = int(r["saved_prompt_tokens"])
                m_comp_tok = int(r["saved_completion_tokens"])
                m_total_tok = int(r["saved_total_tokens"])
                m_misses = miss_by_model.get(m_name, 0)
                by_model[m_name] = {
                    "entries": int(r["entries"]),
                    "total_hits": m_hits,
                    "exact_hits": m_exact,
                    "fuzzy_hits": m_fuzzy,
                    "total_misses": m_misses,
                    "saved_prompt_tokens": m_prompt_tok,
                    "saved_completion_tokens": m_comp_tok,
                    "saved_total_tokens": m_total_tok,
                }

            for m_name, m_misses in miss_by_model.items():
                if m_name not in by_model:
                    by_model[m_name] = {
                        "entries": 0,
                        "total_hits": 0,
                        "exact_hits": 0,
                        "fuzzy_hits": 0,
                        "total_misses": m_misses,
                        "saved_prompt_tokens": 0,
                        "saved_completion_tokens": 0,
                        "saved_total_tokens": 0,
                    }

            return {
                "total_entries": total_entries,
                "total_requests": total_requests,
                "total_hits": total_hits,
                "exact_hits": exact_hits,
                "fuzzy_hits": fuzzy_hits,
                "total_misses": total_misses,
                "hit_rate": hit_rate,
                "saved_prompt_tokens": saved_prompt_tokens,
                "saved_completion_tokens": saved_completion_tokens,
                "saved_total_tokens": saved_total_tokens,
                "by_model": by_model,
            }

    def get_metrics(self) -> dict[str, Any]:
        """Aggregates cache performance metrics including hits and tokens saved."""
        return self.get_stats()

    def close(self) -> None:
        """Closes the active connection if open."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
