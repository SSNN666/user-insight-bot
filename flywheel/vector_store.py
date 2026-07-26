"""Vector store with pluggable backends — persistent Milvus-lite or in-memory.

Backends:
- ``milvus`` — Milvus-lite (persistent, survives restarts).  Default.
- ``memory`` — in-memory cosine similarity via numpy (fast, zero-dependency).

Both implement the same interface: ``insert``, ``search``, ``delete_by_id``, ``count``.
"""

import hashlib
import os
from functools import lru_cache

import numpy as np

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)


def _dim() -> int:
    """Return the configured embedding dimension (lazy, so tests can override)."""
    return get_settings().EMBEDDING_DIM

# ── Embedding ─────────────────────────────────────────────────────


def _ollama_embed(text: str, dim: int | None = None) -> list[float] | None:
    """Get real semantic embedding from Ollama's nomic-embed-text."""
    if dim is None:
        dim = _dim()
    try:
        import httpx
        settings = get_settings()
        base = settings.OPENAI_BASE_URL.rstrip("/v1").rstrip("/")
        resp = httpx.post(
            f"{base}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=15.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            vec = data.get("embedding", [])
            if len(vec) == dim:
                return vec
            if len(vec) < dim:
                return vec + [0.0] * (dim - len(vec))
            return vec[:dim]
        return None
    except Exception:
        return None


def _bigram_embed(text: str, dim: int = _dim()) -> list[float]:
    """Character-bigram fallback embedding. Normalized L2 vector."""
    bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
    vec = np.zeros(dim, dtype=np.float32)
    for bg in bigrams:
        idx = hash(bg) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


def _embed_cache_key(text: str, dim: int) -> str:
    """Stable cache key for a (text, dim) pair."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{digest}:{dim}"


@lru_cache(maxsize=512)
def _cached_embed_full(key: str) -> list[float] | None:
    """Cached embedding computation.  key = _embed_cache_key(text, dim).

    Returns None when Ollama is unavailable (triggers bigram fallback).
    Because lru_cache caches None, we don't retry failed Ollama calls for
    the same text within the cache lifetime.
    """
    # key format: "sha256hex:dim"
    parts = key.rsplit(":", 1)
    dim = int(parts[1])
    # We don't have the original text, so this is a design trade-off:
    # the caller passes the key and the text separately.
    # See embed_text() below for the actual flow.
    raise NotImplementedError("use embed_text() instead")


def embed_text(text: str, dim: int = _dim()) -> list[float]:
    """Primary embedding function with LRU cache + automatic fallback.

    Caches Ollama embeddings by text hash.  Bigram embeddings are NOT
    cached (they're cheap to compute locally).
    """
    # Try Ollama with caching
    cache_key = _embed_cache_key(text, dim)
    result = _cached_ollama_embed(cache_key, text, dim)
    if result is not None:
        return result
    logger.debug("embedding_fallback_to_bigram", extra={"text_len": len(text)})
    return _bigram_embed(text, dim)


@lru_cache(maxsize=512)
def _cached_ollama_embed(key: str, text: str, dim: int) -> list[float] | None:
    """Cached wrapper around _ollama_embed.  key is _embed_cache_key(text, dim).

    Passes both key (for lru_cache) and text+dim (for the actual call).
    Returns None when Ollama returns no embedding — this is cached too
    (negative caching) so we don't hammer Ollama with repeated failures.
    """
    return _ollama_embed(text, dim)


def invalidate_embedding_cache() -> None:
    """Clear the embedding LRU cache (useful for tests)."""
    _cached_ollama_embed.cache_clear()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    dot = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ── Abstract interface (protocol) ─────────────────────────────────


class _VectorStoreBase:
    """Protocol that both backends satisfy.

    Subclasses use ``_dim()`` directly for the embedding dimension.
    """

    def insert(self, sample_id: int, question: str, reply: str) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        raise NotImplementedError

    def delete_by_id(self, sample_id: int) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════
# Backend 1: In-memory (original, kept as fallback)
# ══════════════════════════════════════════════════════════════════


class InMemoryVectorStore(_VectorStoreBase):
    """In-memory vector store with cosine similarity search.

    Fast and zero-dependency, but all vectors are lost on restart.
    """

    def __init__(self):
        self._store: list[dict] = []  # [{id, question, reply, embedding}]
        self._id_set: set[int] = set()
        logger.info("vector_store_init", extra={"backend": "in-memory", "dim": _dim()})

    def insert(self, sample_id: int, question: str, reply: str) -> None:
        if sample_id in self._id_set:
            self.delete_by_id(sample_id)

        embedding = embed_text(question, _dim())
        self._store.append({
            "id": sample_id,
            "question": question[:500],
            "reply": reply[:1000],
            "embedding": embedding,
        })
        self._id_set.add(sample_id)
        logger.debug("vector_inserted", extra={"id": sample_id, "total": len(self._store)})

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._store:
            return []

        query_vec = embed_text(query, _dim())
        scored = []
        for item in self._store:
            sim = cosine_similarity(query_vec, item["embedding"])
            scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": item["id"],
                "question": item["question"],
                "reply": item["reply"],
                "similarity": round(sim, 3),
            }
            for sim, item in scored[:top_k]
        ]

    def delete_by_id(self, sample_id: int) -> None:
        self._store = [s for s in self._store if s["id"] != sample_id]
        self._id_set.discard(sample_id)

    def count(self) -> int:
        return len(self._store)


# ══════════════════════════════════════════════════════════════════
# Backend 2: Milvus-lite (persistent, production-ready)
# ══════════════════════════════════════════════════════════════════

_MILVUS_COLLECTION_NAME = "flywheel_samples"
_AUTO_FLUSH_INTERVAL = 20  # flush to disk every N inserts (crash safety)


class MilvusVectorStore(_VectorStoreBase):
    """Persistent vector store backed by Milvus-lite.

    Data survives process restarts.  The collection is created once
    on first access and reused thereafter.

    Uses brute-force (FLAT) search — no index needed.  This avoids
    Windows file-locking issues with IVF_FLAT index creation and is
    fast enough for the expected scale (< 10k vectors).

    Graceful fallback: if Milvus cannot start (locked dir, corrupt WAL,
    import error), falls back to in-memory store automatically.
    """

    def __init__(self, data_dir: str | None = None):
        settings = get_settings()
        self._data_dir = data_dir or settings.FLYWHEEL_MILVUS_DATA_DIR
        self._milvus = None
        self._collection = None
        self._fallback: InMemoryVectorStore | None = None
        self._dirty_count: int = 0
        self._init_milvus()

    # ── Init ──────────────────────────────────────────────────

    def _init_milvus(self) -> None:
        """Connect to Milvus-lite and ensure the collection exists."""
        try:
            from milvus_lite import (
                MilvusLite, CollectionSchema, FieldSchema, DataType,
            )
        except ImportError:
            logger.warning("milvus_not_installed", extra={
                "hint": "pip install milvus-lite",
            })
            self._fallback = InMemoryVectorStore()
            return

        os.makedirs(self._data_dir, exist_ok=True)

        try:
            self._milvus = MilvusLite(self._data_dir)
        except Exception as exc:
            logger.warning("milvus_connection_failed", extra={
                "data_dir": self._data_dir, "error": str(exc)[:200],
            })
            self._fallback = InMemoryVectorStore()
            return

        # Create collection if it doesn't exist
        try:
            if not self._milvus.has_collection(_MILVUS_COLLECTION_NAME):
                self._create_collection()
        except Exception as exc:
            logger.warning("milvus_has_collection_failed", extra={
                "error": str(exc)[:200],
            })
            self._fallback = InMemoryVectorStore()
            return

        try:
            self._collection = self._milvus.get_collection(_MILVUS_COLLECTION_NAME)
            self._collection.load()
            logger.info("vector_store_init", extra={
                "backend": "milvus-lite",
                "data_dir": self._data_dir,
                "dim": _dim(),
                "count": self._collection.num_entities,
            })
        except Exception as exc:
            logger.warning("milvus_load_failed", extra={"error": str(exc)[:200]})
            self._fallback = InMemoryVectorStore()

    def _create_collection(self) -> None:
        """Define the collection schema and create it.

        No index is created — brute-force (FLAT) search is used.
        This is appropriate for the flywheel scale (< 10k vectors)
        and avoids Windows file-locking issues with index builds.
        """
        from milvus_lite import CollectionSchema, FieldSchema, DataType

        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="id", dtype=DataType.INT64,
                    is_primary=True, auto_id=False,
                ),
                FieldSchema(
                    name="question", dtype=DataType.VARCHAR,
                    max_length=512,
                ),
                FieldSchema(
                    name="reply", dtype=DataType.VARCHAR,
                    max_length=1024,
                ),
                FieldSchema(
                    name="embedding", dtype=DataType.FLOAT_VECTOR,
                    dim=_dim(),
                ),
            ],
            enable_dynamic_field=False,
        )
        self._milvus.create_collection(_MILVUS_COLLECTION_NAME, schema)
        self._collection = self._milvus.get_collection(_MILVUS_COLLECTION_NAME)
        self._collection.load()
        logger.info("milvus_collection_created", extra={
            "collection": _MILVUS_COLLECTION_NAME, "dim": _dim(),
        })

    # ── CRUD ──────────────────────────────────────────────────

    def insert(self, sample_id: int, question: str, reply: str) -> None:
        if self._fallback is not None:
            self._fallback.insert(sample_id, question, reply)
            return

        try:
            # Upsert: delete existing row with same PK, then insert new.
            # We use delete+insert (rather than upsert) to avoid a
            # Milvus-lite 3.1 bug where upsert silently drops vectors.
            self.delete_by_id(sample_id)

            embedding = embed_text(question, _dim())
            self._collection.insert([
                {
                    "id": sample_id,
                    "question": question[:500],
                    "reply": reply[:1000],
                    "embedding": embedding,
                },
            ])
            # NOTE: do NOT flush after every insert — on Windows,
            # milvus-lite's flush() does os.rename(tmp, target) which
            # fails with FileExistsError if the previous flush hasn't
            # released its file handle.  Data is searchable in-memory
            # immediately.  We auto-flush every N inserts for crash
            # safety, and do a final flush on close().
            self._dirty_count += 1
            if self._dirty_count % _AUTO_FLUSH_INTERVAL == 0:
                self._collection.flush()
                logger.debug("milvus_auto_flush", extra={
                    "dirty_count": self._dirty_count,
                })

            logger.debug("vector_inserted", extra={
                "id": sample_id, "backend": "milvus",
            })
        except Exception as exc:
            logger.error("milvus_insert_failed", extra={
                "id": sample_id, "error": str(exc)[:200],
            })

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._fallback is not None:
            return self._fallback.search(query, top_k)

        try:
            query_vec = embed_text(query, _dim())
            results = self._collection.search(
                query_vectors=[query_vec],
                top_k=top_k,
                metric_type="COSINE",
                output_fields=["id", "question", "reply"],
            )
            # results is list[list[dict]] — one inner list per query vector
            hits = results[0] if results else []

            parsed = []
            for hit in hits:
                # Milvus-lite nests non-vector fields under "entity"
                entity = hit.get("entity", {})
                parsed.append({
                    "id": hit.get("id"),
                    "question": entity.get("question", ""),
                    "reply": entity.get("reply", ""),
                    "similarity": round(float(hit.get("distance", 0.0)), 3),
                })
            return parsed
        except Exception as exc:
            logger.warning("milvus_search_failed", extra={
                "error": str(exc)[:200],
            })
            return []

    def delete_by_id(self, sample_id: int) -> None:
        if self._fallback is not None:
            self._fallback.delete_by_id(sample_id)
            return

        try:
            self._collection.delete(pks=[sample_id])
        except Exception:
            # Row may not exist — that's fine
            pass

    def count(self) -> int:
        if self._fallback is not None:
            return self._fallback.count()

        try:
            # milvus-lite 3.1: .count() can return 0 incorrectly;
            # num_entities is reliable.
            return self._collection.num_entities
        except Exception:
            return 0

    def close(self) -> None:
        """Flush pending writes and release Milvus resources."""
        if self._milvus is not None:
            try:
                if self._collection is not None:
                    # Persist any unflushed inserts before shutdown
                    self._collection.flush()
                    self._collection.release()
                self._milvus.close()
                logger.info("milvus_closed")
            except Exception as exc:
                logger.debug("milvus_close_error", extra={"error": str(exc)[:200]})


# ── Factory ───────────────────────────────────────────────────────

import threading as _threading

_vector_store: _VectorStoreBase | None = None
_vector_store_lock = _threading.Lock()


def get_vector_store() -> _VectorStoreBase:
    """Return the configured vector store backend.

    Reads ``FLYWHEEL_VECTOR_BACKEND`` from settings:
    - ``"milvus"`` → persistent Milvus-lite (default)
    - ``"memory"`` → in-memory cosine similarity

    Falls back to in-memory if Milvus fails to initialise.
    Thread-safe: uses double-checked locking to avoid race on the singleton.
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    with _vector_store_lock:
        if _vector_store is not None:
            return _vector_store

        settings = get_settings()
        backend = settings.FLYWHEEL_VECTOR_BACKEND

        if backend == "milvus":
            store = MilvusVectorStore()
        else:
            store = InMemoryVectorStore()

        _vector_store = store
        return store


def reset_vector_store() -> None:
    """Close and recreate the vector store (useful in tests or reset flows)."""
    global _vector_store
    with _vector_store_lock:
        if _vector_store is not None:
            if isinstance(_vector_store, MilvusVectorStore):
                _vector_store.close()
            _vector_store = None
