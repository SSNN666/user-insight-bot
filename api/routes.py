"""REST API endpoints — chat, health, feedback, debug, stats."""

import json
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.agent import ask_agent
from errors.exceptions import APIError, ParameterError
from log.logger import get_logger

logger = get_logger(__name__, log_type="user_chat")
router = APIRouter()

# ── Pydantic models ────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=64)


class AskResponse(BaseModel):
    reply: str
    session_id: str
    elapsed_ms: float


class ProductCreateResponse(BaseModel):
    status: str = Field(examples=["已创建"])
    product_id: int = Field(examples=[1001])


class ProductDeleteResponse(BaseModel):
    status: str = Field(examples=["已下架"])
    product_id: int = Field(examples=[1001])


class OrderCreateResponse(BaseModel):
    status: str = Field(examples=["已录入"])
    order_id: int = Field(examples=[10001])


class ResetMockResponse(BaseModel):
    status: str = Field(examples=["ok"])
    users: int = Field(examples=[132])
    segments: int = Field(examples=[4])
    products: int = Field(examples=[50])


class HealthResponse(BaseModel):
    status: str
    version: str


class FeedbackRequest(BaseModel):
    question: str
    reply: str
    rating: str = Field(..., pattern="^(up|down)$")
    session_id: str = "default"


class ProductRequest(BaseModel):
    product_name: str
    category: str = "General"
    price: float = Field(..., gt=0)


class OrderRequest(BaseModel):
    user_id: int
    product_id: int
    quantity: int = Field(default=1, ge=1)
    total_amount: float = Field(..., gt=0)


class UserUpdate(BaseModel):
    city: str | None = None
    age: int | None = None


# ── Root ──────────────────────────────────────────────────────

@router.get("/")
async def root():
    from fastapi.responses import HTMLResponse
    return HTMLResponse("""
    <html><head><title>智能电商平台</title><meta charset="UTF-8">
    <style>body{font-family:sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;background:#f5f7fa}
    .box{text-align:center;background:white;padding:60px;border-radius:16px;box-shadow:0 4px 20px rgba(0,0,0,.08)}
    h1{margin:0 0 8px 0}a{display:inline-block;margin:8px 12px;padding:12px 28px;border-radius:8px;text-decoration:none;font-weight:bold}
    .shop{background:#409EFF;color:white}.admin{background:#67C23A;color:white}.docs{background:#E6A23C;color:white}</style></head>
    <body><div class="box"><h1>🛒 智能电商平台</h1><p>选择一个入口</p>
    <a class="shop" href="http://localhost:5173">🛒 电商商城</a>
    <a class="admin" href="http://localhost:7860">📊 管理后台</a>
    <a class="docs" href="/docs">📘 API 文档</a></div></body></html>""")


# ── Health ─────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health():
    from config.settings import get_settings
    return HealthResponse(status="ok", version=get_settings().APP_VERSION)


@router.get("/health/full")
async def health_full():
    from api.health import get_full_health
    return await get_full_health()


# ── Chat ───────────────────────────────────────────────────────


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        start = time.monotonic()
        reply = ask_agent(req.question, session_id=req.session_id)
        elapsed = (time.monotonic() - start) * 1000
        return AskResponse(
            reply=reply, session_id=req.session_id,
            elapsed_ms=round(elapsed, 2),
        )
    except ParameterError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except APIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error("unexpected_error", extra={"error": str(e), "question": req.question[:100]})
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """Stream agent response via Server-Sent Events."""
    from fastapi.responses import StreamingResponse
    from agent.agent import _ask_agent_stream

    async def event_stream():
        try:
            async for chunk in _ask_agent_stream(
                req.question, session_id=req.session_id,
            ):
                if chunk:
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except ParameterError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except APIError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            logger.error("stream_error", extra={"error": str(e)})
            yield f"data: {json.dumps({'error': '服务器内部错误'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Feedback ───────────────────────────────────────────────────


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    from api.feedback import save_feedback
    fid = save_feedback(req.question, req.reply, req.rating, req.session_id)
    return {"id": fid, "status": "ok"}


@router.get("/feedback/stats")
async def feedback_stats():
    from api.feedback import get_feedback_stats
    return get_feedback_stats()


@router.get("/feedback/history")
async def feedback_history(limit: int = 20):
    from api.feedback import get_feedback_history
    return {"entries": get_feedback_history(limit)}


# ── Flywheel Stats ─────────────────────────────────────────────


@router.get("/flywheel/stats")
async def flywheel_stats():
    """Return flywheel sample library statistics."""
    try:
        from flywheel.store import get_sample_store
        from flywheel.vector_store import get_vector_store
        store = get_sample_store()
        vs = get_vector_store()
        return {
            "samples": store.count_by_source(),
            "total_positives": len(store.get_positives(1000)),
            "total_negatives": len(store.get_negatives(1000)),
            "vectors_indexed": vs.count(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flywheel/trigger")
async def flywheel_trigger():
    """Manually trigger one flywheel collection + scoring + ingestion cycle."""
    try:
        from flywheel.scheduler import get_flywheel_scheduler
        ingested = await get_flywheel_scheduler().update_once()
        return {"status": "ok", "ingested": ingested}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Debug Console ──────────────────────────────────────────────


@router.get("/debug/products")
async def debug_list_products():
    try:
        from api.data_store import list_products as ds_products
        return ds_products()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/debug/products", response_model=ProductCreateResponse)
async def debug_add_product(req: ProductRequest):
    from api.data_store import add_product as ds_add_product
    product = ds_add_product(req.product_name, req.category, req.price)
    return {"status": "已创建", "product_id": product["product_id"]}


@router.delete("/debug/products/{product_id}", response_model=ProductDeleteResponse)
async def debug_delete_product(product_id: int):
    from api.data_store import delete_product as ds_delete_product
    deleted = ds_delete_product(product_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="商品不存在")
    return {"status": "已下架", "product_id": product_id}


@router.post("/debug/orders", response_model=OrderCreateResponse)
async def debug_add_order(req: OrderRequest):
    from api.data_store import add_order as ds_add_order, get_product
    # Auto-compute total_amount if not provided
    amount = req.total_amount
    if amount <= 0:
        p = get_product(req.product_id)
        amount = (p["price"] if p else 99) * req.quantity
    order = ds_add_order(req.user_id, req.product_id, req.quantity, amount)
    return {"status": "已录入", "order_id": order["order_id"]}


@router.get("/debug/users")
async def debug_list_users():
    try:
        from api.data_store import list_users as ds_users
        return ds_users()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/debug/users/{user_id}")
async def debug_update_user(user_id: int, req: UserUpdate):
    from api.data_store import update_user as ds_update_user
    result = ds_update_user(
        user_id,
        city=req.city,
        age=req.age,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"user_id": user_id, "updates": result}


@router.post("/debug/reset-mock", response_model=ResetMockResponse)
async def debug_reset_mock():
    from api.data_store import reset_all
    from skills.user_segment import invalidate_pipeline_cache, _load_and_process
    from flywheel.vector_store import reset_vector_store

    # Reset in-memory stores
    store_info = reset_all()

    # Reset pipeline cache
    invalidate_pipeline_cache()
    rfm, seg, rules = _load_and_process(force_refresh=True)

    # Reset vector store (Milvus or in-memory)
    reset_vector_store()

    return {
        "status": "ok",
        "users": len(rfm),
        "segments": int(seg['segment'].nunique()) if seg is not None else 0,
        "products": store_info["products"],
    }


@router.post("/debug/trigger-event")
async def debug_trigger_event():
    from watcher.engine import get_watcher_engine
    engine = get_watcher_engine()
    tasks = await engine.poll_once()
    return {
        "status": "ok",
        "events_detected": len(tasks),
        "task_ids": [t.id for t in tasks] if tasks else [],
    }


# ── Manual Annotation (Phase 7) ───────────────────────────────

class AnnotateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    reply: str = Field(..., min_length=1)
    rating: str = Field(..., pattern="^(positive|negative)$")


@router.post("/annotate")
async def annotate(req: AnnotateRequest):
    try:
        import sqlite3, os
        from config.settings import get_settings
        from datetime import datetime, timezone

        db = get_settings().FLYWHEEL_DB_PATH
        os.makedirs(os.path.dirname(db), exist_ok=True)
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS manual_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                reply TEXT NOT NULL,
                rating TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO manual_annotations (question, reply, rating, created_at) VALUES (?, ?, ?, ?)",
            (req.question[:2000], req.reply[:5000], req.rating, now),
        )
        conn.commit()
        conn.close()
        return {"id": cur.lastrowid, "status": "ok", "rating": req.rating}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save annotation: {e}")


# ── Agent Trace API ────────────────────────────────────────────


@router.get("/traces")
async def list_traces(limit: int = 20):
    """Get recent agent execution traces for the Trace panel."""
    try:
        from agent.tracer import get_recent_traces
        traces = get_recent_traces(limit)
        return {
            "traces": [t.to_dict() for t in traces],
            "total": len(traces),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces/{session_id}")
async def get_trace_detail(session_id: str):
    """Get the full trace for a specific session."""
    try:
        from agent.tracer import get_trace
        trace = get_trace(session_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/traces")
async def clear_traces():
    """Clear all agent traces."""
    try:
        from agent.tracer import clear_traces
        clear_traces()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Observability ─────────────────────────────────────────────


@router.get("/stats/observability")
async def observability_stats():
    try:
        from agent.observability import get_stats
        return get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Stats (for charts) ─────────────────────────────────────────


@router.get("/stats/rfm")
async def stats_rfm(force: bool = False):
    try:
        from skills.user_segment import _load_and_process, invalidate_pipeline_cache
        if force:
            invalidate_pipeline_cache()
        rfm, seg, _ = _load_and_process(force_refresh=force)
        stats = []
        for seg_id in sorted(seg['segment'].unique()):
            sdf = seg[seg['segment'] == seg_id]
            stats.append({
                "segment": int(seg_id),
                "user_count": int(len(sdf)),
                "avg_recency": round(float(sdf['recency'].mean()), 1),
                "avg_frequency": round(float(sdf['frequency'].mean()), 2),
                "avg_monetary": round(float(sdf['monetary'].mean()), 2),
            })
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/segment-ratio")
async def stats_segment_ratio(force: bool = False):
    try:
        from skills.user_segment import _load_and_process, invalidate_pipeline_cache
        if force:
            invalidate_pipeline_cache()
        _, seg, _ = _load_and_process(force_refresh=force)
        ratios = []
        total = len(seg)
        for seg_id in sorted(seg['segment'].unique()):
            count = int((seg['segment'] == seg_id).sum())
            ratios.append({"segment": int(seg_id), "count": count, "ratio": round(count / total * 100, 1)})
        return ratios
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/flow")
async def stats_flow(force: bool = False):
    try:
        from skills.user_segment import _load_and_process, invalidate_pipeline_cache
        if force:
            invalidate_pipeline_cache()
        _, seg, _ = _load_and_process(force_refresh=force)
        if 'flow_tag' not in seg.columns:
            return []
        flows = []
        for seg_id in sorted(seg['segment'].unique()):
            sdf = seg[seg['segment'] == seg_id]
            tags = sdf['flow_tag'].value_counts().to_dict()
            flows.append({"segment": int(seg_id), **{str(k): v for k, v in tags.items()}})
        return flows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
