"""REST API endpoints — chat, health, feedback, debug, stats."""

import asyncio
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agent.agent import ask_agent, _ask_agent_internal
from errors.exceptions import APIError, ParameterError
from log.logger import get_logger

logger = get_logger(__name__, log_type="user_chat")
router = APIRouter()

# ── 会话级执行串行化 ────────────────────────────────────────────
# graph invoke/astream 共享 InMemorySaver checkpoint,同一 thread_id 并发
# 会引发 checkpoint 版本冲突 → 同一 session 的图执行加锁串行化。
_ask_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ask")

_session_locks: dict[str, asyncio.Lock] = {}
_default_session_lock = asyncio.Lock()


def _get_session_lock(session_id: str) -> asyncio.Lock:
    if not session_id or session_id == "default":
        return _default_session_lock
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = _session_locks[session_id] = asyncio.Lock()
    return lock


def _get_censor():
    from common.content_moderation import get_censor
    return get_censor()

# ── Pydantic models ────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=64)
    # 场景强制:auto=意图分类 | analysis=强制分析模式(管理台) | shopping=强制购物模式
    mode: str = Field(default="auto", max_length=16)


class AskResponse(BaseModel):
    reply: str
    session_id: str
    elapsed_ms: float
    products: list[dict] | None = None   # 商品类 Skill 返回的商品(前端卡片/一键加购)
    charts: list[dict] | None = None     # 分析类 Skill 返回的图表数据(前端内联 SVG 渲染)


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


# ── 分群运营建议 ───────────────────────────────────────────────


@router.get("/api/suggestions")
async def api_suggestions():
    """分群级运营建议:LLM 生成 + Layer 1 数值核查(与图表同源数据)。"""
    try:
        from agent.suggestions import generate_segment_suggestions
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _ask_executor, generate_segment_suggestions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat ───────────────────────────────────────────────────────


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        # ── 输入防护:Prompt 注入检测 + 百度内容审核(均 fail-open 除注入硬拦截) ──
        from common.guardrails import detect_injection, WARN_NOTICE
        verdict = detect_injection(req.question)
        if verdict.blocked:
            raise HTTPException(status_code=403, detail="输入包含指令注入迹象，已拒绝处理")
        censor = _get_censor()
        input_check = censor.check_text(req.question, task="SHOP_QA_INPUT")
        if not input_check.passed:
            raise HTTPException(status_code=403, detail="输入内容未通过安全审核")

        start = time.monotonic()
        # 场景强制:管理台发 mode=analysis(屏蔽购物误路由);mode=shopping 强制购物人设
        system_override = None
        force_analysis = req.mode == "analysis"
        if req.mode == "shopping":
            from agent.prompts import PROMPT_SHOPPING
            system_override = PROMPT_SHOPPING

        # 同步 Agent 图跑在线程池,不阻塞事件循环(健康检查/商城 API 不受影响)
        loop = asyncio.get_event_loop()
        async with _get_session_lock(req.session_id):
            reply, _, products, charts = await loop.run_in_executor(
                _ask_executor, _ask_agent_internal, req.question, req.session_id,
                None, None, system_override, force_analysis,
            )

        # ── 输出防护:内容审核(fail-open)+ 注入提示附加 ──
        output_check = censor.check_text(reply, task="SHOP_QA_OUTPUT")
        if not output_check.passed:
            reply = "抱歉，本次回答未通过内容安全审核，已替换为系统提示。请换个问法再试。"
        if verdict.warned:
            reply += WARN_NOTICE

        elapsed = (time.monotonic() - start) * 1000
        return AskResponse(
            reply=reply, session_id=req.session_id,
            elapsed_ms=round(elapsed, 2),
            products=products or None,
            charts=charts or None,
        )
    except ParameterError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except APIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("unexpected_error", extra={"error": str(e), "question": req.question[:100]})
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """Stream agent execution via SSE.

    Event schema(每帧 `data: {"event": ..., ...}`):
      meta / node_start / node_end_detail / tool_call / tool_result /
      delta / fact_check / answer / done / error
    """
    from fastapi.responses import StreamingResponse
    from agent.agent import _ask_agent_stream

    # ── 输入防护:注入检测 + 内容审核(fail-open) ──
    from common.guardrails import detect_injection, WARN_NOTICE
    verdict = detect_injection(req.question)
    input_ok = _get_censor().check_text(req.question, task="SHOP_QA_INPUT").passed

    lock = _get_session_lock(req.session_id)

    async def event_stream():
        async with lock:
            try:
                if verdict.blocked or not input_ok:
                    yield f"data: {json.dumps({'event': 'error', 'message': '输入未通过安全校验，已拒绝处理'}, ensure_ascii=False)}\n\n"
                    return
                final_answer = ""
                async for event in _ask_agent_stream(
                    req.question, session_id=req.session_id,
                ):
                    if not event:
                        continue
                    if event["event"] == "answer":
                        final_answer = event["content"]
                        # 输出审核(fail-open)
                        if not _get_censor().check_text(final_answer, task="SHOP_QA_OUTPUT").passed:
                            final_answer = "抱歉，本次回答未通过内容安全审核，已替换为系统提示。请换个问法再试。"
                            event = dict(event, content=final_answer)
                        if verdict.warned:
                            final_answer += WARN_NOTICE
                            event = dict(event, content=final_answer)
                    elif event["event"] == "done":
                        event = dict(event, reply=final_answer or event.get("reply", ""))
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except ParameterError as e:
                yield f"data: {json.dumps({'event': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            except APIError as e:
                yield f"data: {json.dumps({'event': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error("stream_error", extra={"error": str(e)})
                yield f"data: {json.dumps({'event': 'error', 'message': '服务器内部错误'}, ensure_ascii=False)}\n\n"

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


# ── 商品图像解析 (VL 扩展, Phase: 加分项) ──────────────────


@router.post("/api/product-image/analyze")
async def analyze_product_image(
    file: UploadFile = File(...),
    user_id: int = Form(default=1),
):
    """上传商品图片 → qwen3-vl-plus 结构化标签 → 并入用户画像偏好。

    VL 只做图像内容理解(品类/外观/颜色/材质),不做文字提取;
    无多模态模型时返回 503 明确提示。
    """
    from config.settings import get_settings

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="仅支持图片文件")
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="图片过大(>8MB)")

    upload_dir = os.path.join(get_settings().CACHE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    ext = (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else "jpg"
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    image_path = os.path.join(upload_dir, f"{uuid.uuid4().hex[:12]}.{ext}")
    with open(image_path, "wb") as f:
        f.write(data)

    from skills import SkillRegistry
    skill = SkillRegistry.get("analyze_product_image")
    if skill is None:
        raise HTTPException(status_code=503, detail="商品图像解析 Skill 未注册")
    result = skill.execute(image_path=image_path, mime=file.content_type or "image/jpeg")

    if result.status.value == "missing":
        raise HTTPException(status_code=503, detail=result.summary)
    if result.status.value == "error":
        raise HTTPException(status_code=502, detail=result.error or "图像解析失败")

    # 标签并入用户画像偏好
    from api.data_store import add_user_preference
    profile = add_user_preference(user_id, result.data or {})

    return {
        "status": "ok",
        "user_id": user_id,
        "tags": result.data,
        "confidence": result.confidence,
        "profile_updated": profile is not None,
    }


@router.get("/api/recommendations")
async def get_recommendations(user_id: int = 1):
    """个性化推荐(画像驱动):分群+品类偏好+标签 → 商品打分。"""
    from skills import SkillRegistry
    skill = SkillRegistry.get("get_personal_recommendations")
    if skill is None:
        raise HTTPException(status_code=503, detail="推荐 Skill 未注册")
    result = skill.execute(user_id=user_id)
    if result.status.value == "error":
        raise HTTPException(status_code=502, detail=result.error or "推荐计算失败")
    return {
        "user_id": user_id,
        "status": result.status.value,
        "products": result.data or [],
        "summary": result.summary,
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


@router.get("/stats/segment-trend")
async def stats_segment_trend(limit: int = 20):
    """分群时间趋势:消费磁盘上的历史快照,返回每快照×每分群的时序行。

    Watcher 每轮 poll 都落一个快照,此端点把累积的历史变成可见的时间序列。
    """
    try:
        from pipeline.user_segmentation import load_snapshots
        snaps = load_snapshots()[-limit:]
        rows = []
        for s in snaps:
            ts = s.timestamp[:16].replace("T", " ")
            for seg in sorted(s.segment_stats.keys()):
                stats = s.segment_stats[seg]
                rows.append({
                    "timestamp": ts,
                    "segment": int(seg),
                    "user_count": stats.get("user_count", 0),
                    "avg_monetary": round(stats.get("avg_monetary", 0), 2),
                    "avg_frequency": round(stats.get("avg_frequency", 0), 2),
                })
        return rows
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
