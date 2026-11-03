"""
SSE Streaming Router — LangGraph-powered goal agent pipeline.

Emits three SSE event types:
  {event: "started",  run_id, raw_goal}
  {event: "progress", stage, stage_index, progress, confidence_score}
  {event: "token",    delta}              ← only when ?tokens=1 AND node in STREAM_TOKEN_NODES
  {event: "complete", run_id, confidence_score, degraded, roadmap, ...}
  {event: "error",    run_id, error}

Fix #2 applied: events are filtered by metadata.langgraph_node, not event["name"].
Fix Q1 applied: token events are gated behind ?tokens=1 query param.

Resume semantics (fix #3):
  Pass run_id as a query param to resume an interrupted run.
  The graph's ainvoke(None, config) call picks up from the last checkpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from auth import get_current_user
from models import User
from services.graph.builder import get_goal_graph
from services.graph.state import GoalState, initial_state
from services.checkpoint import get_checkpoint_store   # kept for /checkpoint/{run_id}

logger = logging.getLogger("edxiom.goal_stream")

router = APIRouter(prefix="/goals", tags=["Goal Agent Streaming"])

# Stage nodes whose on_chain_start events map to a progress frame.
# Ordered by pipeline position; used to compute progress fraction.
STAGE_NODES: list[str] = [
    "classify", "intake", "detect_entity", "research",
    "roadmap_draft", "verify", "repair", "youtube_enrich", "youtube_attach", "assembly",
]

# Only emit token-streaming events from conversational / user-facing nodes.
# JSON-generation nodes (roadmap_draft, verify, repair) would flood the stream
# with raw JSON fragments that the frontend can't use.
STREAM_TOKEN_NODES: set[str] = {"classify", "intake", "detect_entity"}


def _progress_fraction(node_name: str) -> float:
    try:
        idx = STAGE_NODES.index(node_name)
        return round((idx + 1) / len(STAGE_NODES), 2)
    except ValueError:
        return 0.0


async def _stream_pipeline(
    raw_goal: str,
    user_id: str,
    run_id: str,
    *,
    clarification_reply: Optional[str] = None,
    preferred_language: Optional[str] = None,
    target_months: Optional[int] = None,
    daily_hours: Optional[float] = None,
    playlist_url: Optional[str] = None,
    website_url: Optional[str] = None,
    reference_text: Optional[str] = None,
    stream_tokens: bool = False,
    resume: bool = False,
):
    """Async generator yielding Server-Sent Events from graph.astream_events()."""
    graph = get_goal_graph()
    config = {"configurable": {"thread_id": run_id}}

    yield f"data: {json.dumps({'event': 'started', 'run_id': run_id, 'raw_goal': raw_goal})}\n\n"

    try:
        if resume:
            # Fix #3: resume from last checkpoint — pass None as input.
            # The graph reads the last persisted state from the checkpointer.
            input_state = None
            logger.info("[%s] Resuming from checkpoint", run_id)
        else:
            input_state = initial_state(
                user_id=user_id,
                raw_goal=raw_goal,
                run_id=run_id,
                clarification_reply=clarification_reply,
                preferred_language=preferred_language,
                target_months=target_months,
                daily_hours=daily_hours,
                playlist_url=playlist_url,
                website_url=website_url,
                reference_text=reference_text,
            )
            # If user provided a clarification reply, merge it before handing to graph
            if clarification_reply:
                from services.graph.nodes import apply_clarification
                updates = apply_clarification(input_state, clarification_reply)
                input_state.update(updates)

        final_state: Optional[GoalState] = None

        # astream_events(version="v2") gives typed events with metadata
        async for event in graph.astream_events(input_state, config, version="v2"):
            kind: str = event["event"]

            # Fix #2: use metadata.langgraph_node — reliable regardless of nesting depth
            node_name: str = event.get("metadata", {}).get("langgraph_node", "")

            # ── Progress events (stage start) ──────────────────────────
            if kind == "on_chain_start" and node_name in STAGE_NODES:
                yield f"data: {json.dumps({'event': 'progress', 'stage': node_name, 'stage_index': STAGE_NODES.index(node_name), 'progress': _progress_fraction(node_name), 'confidence_score': 0.0})}\n\n"

            # ── Token events (LLM streaming) ───────────────────────────
            elif kind == "on_chat_model_stream" and stream_tokens and node_name in STREAM_TOKEN_NODES:
                chunk = event.get("data", {}).get("chunk")
                delta = getattr(chunk, "content", "") if chunk else ""
                if delta:
                    yield f"data: {json.dumps({'event': 'token', 'delta': delta, 'node': node_name})}\n\n"

            # ── Capture final state from assembly completion ────────────
            elif kind == "on_chain_end" and node_name == "assembly":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    final_state = output  # type: ignore[assignment]

            # ── Keepalive on graph updates (prevents proxy timeout) ─────
            elif kind == "on_chain_end" and node_name in STAGE_NODES:
                # Emit updated confidence_score if available in output
                output = event.get("data", {}).get("output", {})
                conf = output.get("confidence_score", 0.0) if isinstance(output, dict) else 0.0
                if conf:
                    yield f"data: {json.dumps({'event': 'progress_update', 'stage': node_name, 'confidence_score': conf})}\n\n"

        # ── Complete event ─────────────────────────────────────────────
        if final_state is None:
            # Graph exited early (clarification needed) — pull state from checkpointer
            store = get_checkpoint_store()
            checkpoint = await store.load(run_id)
            final_state = checkpoint.get("state", {}) if checkpoint else {}

        syllabus = final_state.get("verified_syllabus") or final_state.get("draft_syllabus") or []
        yield f"data: {json.dumps({'event': 'complete', 'run_id': run_id, 'confidence_score': final_state.get('confidence_score', 0.0), 'degraded': final_state.get('degraded', False), 'detected_entity': final_state.get('detected_entity'), 'needs_clarification': final_state.get('needs_clarification', False), 'clarification_question': final_state.get('clarification_question'), 'repair_attempts': final_state.get('repair_attempts', 0), 'roadmap': syllabus})}\n\n"

    except Exception as exc:
        logger.error("[%s] Stream pipeline failed: %s", run_id, exc, exc_info=True)
        yield f"data: {json.dumps({'event': 'error', 'run_id': run_id, 'error': str(exc)})}\n\n"


@router.get("/stream")
async def stream_goal_progress(
    goal: str = Query(..., description="Target learning goal or exam title"),
    clarification: Optional[str] = Query(None, description="User clarification reply for previous round"),
    language: Optional[str] = Query(None, description="Preferred language (English, Hindi, Hinglish)"),
    months: Optional[int] = Query(None, description="Target timeline in months"),
    hours: Optional[float] = Query(None, description="Daily study hours"),
    playlist: Optional[str] = Query(None, description="YouTube playlist URL"),
    website: Optional[str] = Query(None, description="Supplementary website URL"),
    run_id: Optional[str] = Query(None, description="Existing run_id to resume from checkpoint"),
    tokens: bool = Query(False, description="Set ?tokens=1 to enable token-level streaming (Q1 gate)"),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Server-Sent Events endpoint — streams the 8-stage AI goal agent in real-time.

    Pass an existing run_id with ?run_id=<id> to resume an interrupted run
    (only works with Redis-backed checkpointer, i.e. REDIS_URL must be set).
    """
    user_id = str(current_user.id) if current_user else "anonymous"
    is_resume = bool(run_id)
    effective_run_id = run_id or str(uuid.uuid4())

    return StreamingResponse(
        _stream_pipeline(
            raw_goal=goal,
            user_id=user_id,
            run_id=effective_run_id,
            clarification_reply=clarification,
            preferred_language=language,
            target_months=months,
            daily_hours=hours,
            playlist_url=playlist,
            website_url=website,
            stream_tokens=tokens,
            resume=is_resume,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/checkpoint/{run_id}")
async def get_goal_checkpoint(
    run_id: str,
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Retrieve checkpointed state for a specific run_id.
    Kept for backward compatibility — reads from the legacy CheckpointStore.
    LangGraph's own checkpointer state is accessible via the /stream?run_id= endpoint.
    """
    store = get_checkpoint_store()
    checkpoint = await store.load(run_id)
    if not checkpoint:
        return {"found": False, "run_id": run_id, "state": None}
    return {
        "found": True,
        "run_id": run_id,
        "stage": checkpoint.get("stage"),
        "state": checkpoint.get("state"),
    }
