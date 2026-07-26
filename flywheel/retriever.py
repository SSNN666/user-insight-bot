"""Hybrid retrieval with LLM re-ranking for the flywheel sample library.

Combines:
1. Dense retrieval — semantic search via Milvus (nomic-embed-text)
2. Sparse retrieval — BM25 keyword search with jieba tokenization
3. RRF fusion — Reciprocal Rank Fusion to merge both result sets
4. LLM re-rank — cross-encoder style relevance scoring

Interview pitch: "Dense semantic search (768-dim Milvus) captures meaning;
sparse BM25 with jieba tokenization captures precise term matching with
IDF-weighted importance — stopwords like '用户' are automatically
down-weighted while rare signal words like '流失' get boosted. RRF fuses
both in rank space, then an LLM cross-encoder does final re-ranking."

All retrieval paths have graceful fallbacks — if Ollama or Milvus is down,
the system degrades to BM25-only retrieval automatically.
"""

from concurrent.futures import ThreadPoolExecutor

from flywheel.store import get_sample_store
from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)

# Module-level thread pool reused across retrieval calls (lazy-init).
_retrieval_pool: ThreadPoolExecutor | None = None


# ── BM25 index cache ────────────────────────────────────────────

_bm25_index: "BM25Okapi | None" = None
_bm25_candidates_hash: int = 0


def _build_bm25(candidates: list[dict]) -> "BM25Okapi":
    """Build a BM25 index from candidate questions.

    Tokenises each question with jieba and returns a BM25Okapi instance.
    Cached per candidate set — invalidated automatically when the pool
    changes (detected via hash of question texts).
    """
    global _bm25_index, _bm25_candidates_hash

    # Fast path: same candidates as last time → reuse index
    import hashlib, json
    h = hashlib.md5(
        json.dumps([s.get("question", "")[:100] for s in candidates], sort_keys=True).encode()
    ).hexdigest() if candidates else ""
    # Use a simpler hash for performance (first+last+len is enough)
    if candidates:
        h = hash(
            (len(candidates),
             candidates[0].get("question", "")[:80] if candidates else "",
             candidates[-1].get("question", "")[:80] if len(candidates) > 1 else "",
             sum(len(s.get("question", "")) for s in candidates[:5]),
            )
    )

    if _bm25_index is not None and h == _bm25_candidates_hash:
        return _bm25_index

    # Build fresh index
    import jieba
    from rank_bm25 import BM25Okapi

    corpus: list[list[str]] = []
    for s in candidates:
        tokens = list(jieba.cut(s.get("question", "")))
        corpus.append(tokens)

    _bm25_index = BM25Okapi(corpus)
    _bm25_candidates_hash = h
    logger.debug("bm25_index_built", extra={"docs": len(corpus)})
    return _bm25_index


def _sparse_retrieve(
    query: str, candidates: list[dict], top_k: int = 5,
) -> list[tuple[float, dict]]:
    """Sparse retrieval: BM25 with jieba tokenization.

    Builds a BM25 index over candidate questions and scores the query.
    IDF automatically down-weights common words (e.g. '用户') and boosts
    rare signal words (e.g. '流失', '召回').

    Falls back to character-bigram Jaccard if jieba / BM25 is unavailable.
    """
    if not candidates:
        return []

    try:
        import jieba
        bm25 = _build_bm25(candidates)
        tokenized_query = list(jieba.cut(query))
        scores = bm25.get_scores(tokenized_query)

        scored = [(float(scores[i]), candidates[i]) for i in range(len(candidates))]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k * 3]

    except Exception:
        # Graceful fallback to bigram Jaccard (zero-dependency)
        logger.debug("bm25_fallback_to_bigram", extra={"query_len": len(query)})
        scored = []
        for s in candidates:
            q_text = query
            s_text = s.get("question", "")

            def _bigrams(t):
                return {t[i:i+2] for i in range(len(t) - 1)}

            bq = _bigrams(q_text)
            bs = _bigrams(s_text)
            sim = len(bq & bs) / len(bq | bs) if (bq and bs) else 0.0
            scored.append((sim, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k * 3]


# ── Dense Retrieval (semantic) ──────────────────────────────────


def _dense_retrieve(query: str, top_k: int = 5) -> list[tuple[float, dict]]:
    """Dense (semantic) retrieval via Milvus vector search."""
    try:
        from flywheel.vector_store import get_vector_store
        vs = get_vector_store()
        results = vs.search(query, top_k=top_k * 2)
        # Convert Milvus results to (score, sample) tuples
        return [
            (r.get("similarity", 0.0), {
                "id": r.get("id"),
                "question": r.get("question", ""),
                "reply": r.get("reply", ""),
                "quality_score": 0.8,  # default for vector-stored samples
            })
            for r in results
        ]
    except Exception as e:
        logger.warning("dense_retrieval_failed", extra={"error": str(e)})
        return []


# ── RRF Fusion ──────────────────────────────────────────────────


def _stable_dedup_key(sample: dict) -> str:
    """Return a stable, process-independent dedup key for a sample.

    Prefers the DB ``id`` field; falls back to a hex digest of the question
    text (SHA-256 truncated).  Unlike Python's ``hash()``, this is consistent
    across process restarts and independent of PYTHONHASHSEED.
    """
    if "id" in sample and sample["id"] is not None:
        return f"id:{sample['id']}"
    import hashlib
    digest = hashlib.sha256(
        sample.get("question", "").encode("utf-8")
    ).hexdigest()[:16]
    return f"q:{digest}"


def _rrf_fusion(
    sparse_results: list[tuple[float, dict]],
    dense_results: list[tuple[float, dict]],
    k: int = 60,
) -> list[tuple[float, dict]]:
    """Reciprocal Rank Fusion: merge two ranked lists into one.

    RRF score = sum(1 / (k + rank_i)) for each result list.
    k=60 is the recommended default from the RRF paper.
    """
    scores: dict[str, float] = {}        # dedup_key → RRF score
    samples: dict[str, dict] = {}        # dedup_key → sample dict

    for rank, (_, s) in enumerate(sparse_results):
        key = _stable_dedup_key(s)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        samples[key] = s

    for rank, (_, s) in enumerate(dense_results):
        key = _stable_dedup_key(s)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        samples[key] = s

    fused = [(scores[key], samples[key]) for key in scores]
    fused.sort(key=lambda x: x[0], reverse=True)
    return fused


# ── LLM Re-ranking ──────────────────────────────────────────────


def _llm_rerank(query: str, candidates: list[tuple[float, dict]], top_k: int = 3) -> list[dict]:
    """Use LLM as a cross-encoder to re-rank top candidates.

    Sends a batch of (query, candidate_question) pairs to the LLM
    for relevance scoring on a 1-5 scale.
    """
    if len(candidates) <= top_k:
        return [s for _, s in candidates]

    try:
        from llm.client import get_llm_client
        llm = get_llm_client()

        # Build a batch ranking prompt
        pairs_text = ""
        for i, (_, s) in enumerate(candidates[:10]):  # max 10 for re-rank
            pairs_text += f"[{i+1}] {s.get('question', '')[:150]}\n"

        prompt = (
            f"评估以下候选问答与用户问题的相关性（1=无关, 5=高度相关）：\n\n"
            f"用户问题: {query}\n\n"
            f"候选问答:\n{pairs_text}\n"
            f"请按格式输出: [[序号, 评分], ...] 例如: [[3,5],[1,4],[5,2]]\n"
            f"只输出评分最高的前{top_k}个。"
        )

        response = llm.invoke(prompt)
        # Parse [[idx, score], ...] from response
        import re
        pattern = r'\[(\d+)\s*,\s*(\d+)\]'
        matches = re.findall(pattern, str(response))
        if matches:
            scored = [(int(idx) - 1, int(score)) for idx, score in matches]
            scored.sort(key=lambda x: x[1], reverse=True)
            result = []
            seen = set()
            for idx, score in scored[:top_k]:
                if 0 <= idx < len(candidates) and idx not in seen:
                    seen.add(idx)
                    s = candidates[idx][1].copy()
                    s["similarity"] = score / 5.0
                    result.append(s)
            if result:
                logger.debug("llm_rerank_done", extra={"input": len(candidates), "output": len(result)})
                return result
    except Exception as e:
        logger.debug("llm_rerank_fallback", extra={"error": str(e)})

    # Fallback: return top by fusion score
    return [s for _, s in candidates[:top_k]]


# ── Quality Filter ───────────────────────────────────────────────


def _quality_rerank(candidates: list[dict]) -> list[dict]:
    """Down-rank low-quality samples so genuine Q&A pairs float to the top.

    Applies penalties for:
      - Watcher alerts (auto-generated event descriptions, not real questions)
      - Template-only answers (no narrative, just markdown tables)
      - Vague / too-short questions

    Returns the list re-sorted by adjusted similarity.
    """
    for s in candidates:
        penalty = 0.0
        question = s.get("question", "")
        reply = s.get("reply", "")

        # Watcher alert: long question with detection markers
        if len(question) > 150 and any(
            m in question for m in ("检测到", "监测窗口", "请综合分析")
        ):
            penalty += 0.30

        # Template-only answer: all table formatting, no narrative
        lines = [l.strip() for l in reply.split("\n") if l.strip()]
        if lines:
            table_or_header = sum(
                1 for l in lines
                if l.startswith("|") or l.startswith("###") or l.startswith("-")
            )
            narrative = len(lines) - table_or_header
            if narrative == 0:
                penalty += 0.20
            elif narrative <= 2 and table_or_header >= 8:
                penalty += 0.10

        # Vague question
        if len(question.strip()) < 10:
            penalty += 0.15

        # Apply penalty to the similarity score
        s["similarity"] = max(0.0, s.get("similarity", 0.0) - penalty)

    # Re-sort by adjusted similarity
    candidates.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
    return candidates


# ── Main Retrieval API ──────────────────────────────────────────


def retrieve_relevant_samples(query: str, top_k: int | None = None) -> list[dict]:
    """Hybrid retrieval: sparse + dense fusion → LLM re-rank.

    Args:
        query: User's current question.
        top_k: Number of samples to return (default from settings).

    Returns:
        List of sample dicts with question, reply, quality_score, similarity.

    Falls back gracefully:
    - Milvus unavailable → sparse-only
    - Ollama unavailable → sparse-only (bigram fallback in vector_store)
    - LLM unavailable → skip re-ranking, return top RRF results
    """
    settings = get_settings()
    top_k = top_k or settings.FLYWHEEL_RETRIEVAL_TOP_K

    # Phase 1: Get candidate pool from sample store
    store = get_sample_store()
    positives = store.get_positives(limit=200)
    if not positives:
        logger.debug("retrieval_no_candidates")
        return []

    # Phase 2: Sparse + Dense retrieval concurrently.
    # Uses a module-level pool (lazy-init) to avoid per-call overhead.
    global _retrieval_pool
    if _retrieval_pool is None:
        _retrieval_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="retrieval",
        )
    fut_sparse = _retrieval_pool.submit(_sparse_retrieve, query, positives, top_k)
    fut_dense = _retrieval_pool.submit(_dense_retrieve, query, top_k)
    sparse = fut_sparse.result()
    dense = fut_dense.result()

    # Phase 3: RRF fusion (if dense results exist)
    if dense:
        fused = _rrf_fusion(sparse, dense)
        logger.debug("retrieval_fusion", extra={
            "sparse": len(sparse), "dense": len(dense), "fused": len(fused),
        })
    else:
        fused = sparse
        logger.debug("retrieval_sparse_only", extra={"results": len(fused)})

    if not fused:
        return []

    # Phase 4: LLM re-ranking
    result = _llm_rerank(query, fused, top_k)

    # Phase 4b: Quality filter — penalise watcher alerts and template answers
    # so genuine Q&A samples surface even with lower similarity scores.
    result = _quality_rerank(result)

    # Filter low-relevance results
    filtered = [s for s in result if s.get("similarity", 0) >= 0.05]

    logger.debug("retrieval_done", extra={
        "query_len": len(query),
        "candidates": len(positives),
        "returned": len(filtered),
    })
    return filtered
