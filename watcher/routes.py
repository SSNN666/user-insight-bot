"""Task management REST API endpoints — mounted under /tasks."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from watcher.task_manager import get_task_manager
from log.logger import get_logger

logger = get_logger(__name__, log_type="user_chat")
router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskListResponse(BaseModel):
    tasks: list[dict]
    total: int
    limit: int
    offset: int


@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    status: str | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    tm = get_task_manager()
    tasks = tm.list_tasks(status=status, limit=limit + 1, offset=offset)
    has_more = len(tasks) > limit
    result = tasks[:limit]
    return {
        "tasks": [t.to_dict() for t in result],
        "total": len(result),
        "limit": limit,
        "offset": offset,
    }


@router.get("/{task_id}")
async def get_task(task_id: int):
    tm = get_task_manager()
    task = tm.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task.to_dict()


@router.patch("/{task_id}/retry")
async def retry_task(task_id: int):
    """Re-queue a failed/timed-out task back to pending."""
    tm = get_task_manager()
    task = tm.retry_task(task_id)
    if not task:
        raise HTTPException(
            status_code=400,
            detail="任务不存在或状态不允许重试",
        )
    logger.info("task_retried", extra={"task_id": task_id})
    return task.to_dict()


@router.patch("/{task_id}/ignore")
async def ignore_task(task_id: int):
    """Mark a pending task as ignored."""
    tm = get_task_manager()
    ok = tm.ignore_task(task_id)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail="任务不存在或不在待处理状态",
        )
    logger.info("task_ignored", extra={"task_id": task_id})
    task = tm.get_task(task_id)
    return task.to_dict()
