"""
goal_agent.py — Public backward-compatible shim for the goal pipeline.

All orchestration logic has moved to services/graph/.
This module preserves the original run_goal_agent() signature so that
routers/goals.py, tests, and any other callers need zero changes.

Fix #3 (resume semantics):
  - Fresh run:  graph.ainvoke(initial_state, config)
  - Resume run: graph.ainvoke(None, config)  ← reads from checkpointer
    Pass resume=True only when run_id already has a checkpoint and the
    worker was interrupted mid-run.

The legacy CheckpointStore is kept alive for the /goals/checkpoint/{run_id}
endpoint. It will be retired in a future release once the LangGraph saver
is confirmed to serve that endpoint cleanly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from services.graph.state import GoalState, initial_state
from services.graph.builder import get_goal_graph
from services.graph.nodes import (
    _CREATOR_MAP,
    apply_clarification,
    assembly_node,
    build_clarification_question,
    classify_node,
    detect_entity_node,
    enrich_with_youtube_node,
    fetch_entity_context_node,
    intake_node,
    repair_syllabus_node,
    resolve_best_creator_node,
    synthesize_draft_node,
    verify_syllabus_node,
)
from services.checkpoint import get_checkpoint_store

__all__ = [
    "run_goal_agent",
    "GoalState",
    "initial_state",
    "_CREATOR_MAP",
    "classify_node",
    "intake_node",
    "detect_entity_node",
    "fetch_entity_context_node",
    "synthesize_draft_node",
    "verify_syllabus_node",
    "repair_syllabus_node",
    "resolve_best_creator_node",
    "enrich_with_youtube_node",
    "assembly_node",
    "apply_clarification",
    "build_clarification_question",
]

logger = logging.getLogger("edxiom.goal_agent")


async def run_goal_agent(
    user_id: str,
    raw_goal: str,
    clarification_reply: Optional[str] = None,
    preferred_language: Optional[str] = None,
    target_months: Optional[int] = None,
    daily_hours: Optional[float] = None,
    playlist_url: Optional[str] = None,
    website_url: Optional[str] = None,
    reference_text: Optional[str] = None,
    run_id: Optional[str] = None,
    # progress_queue is accepted for API compatibility but is no longer used.
    # The LangGraph streaming router (goal_stream.py) uses astream_events()
    # directly and does not need a side-channel queue.
    progress_queue: Optional[asyncio.Queue] = None,
    resume: bool = False,
) -> GoalState:
    """
    Execute the 8-stage goal pipeline and return the final GoalState.

    For streaming / SSE use, call the /goals/stream endpoint instead —
    it uses graph.astream_events() for real-time progress and token delivery.

    Args:
        user_id:              User identifier (UUID string or 'anonymous').
        raw_goal:             The raw learning goal string from the user.
        clarification_reply:  If the previous call returned needs_clarification=True,
                              pass the user's reply here to continue.
        preferred_language:   "Hindi" | "Hinglish" | "English"
        target_months:        Target timeline in months.
        daily_hours:          Daily available study hours.
        playlist_url:         Optional YouTube playlist URL.
        website_url:          Optional supplementary website URL.
        reference_text:       Optional pasted document / syllabus text.
        run_id:               Reuse an existing run_id (enables resume).
        progress_queue:       Deprecated — accepted for compat, ignored.
        resume:               If True, resume from the last checkpoint for run_id
                              using graph.ainvoke(None, config). Requires
                              Redis-backed checkpointer (REDIS_URL set).

    Returns:
        Final GoalState dict after assembly (or after clarification pause).
    """
    start_time = time.perf_counter()
    rid = run_id or str(uuid.uuid4())
    graph = get_goal_graph()
    config = {"configurable": {"thread_id": rid}}
    store = get_checkpoint_store()  # legacy store kept for checkpoint endpoint

    if resume:
        # Fix #3: resume from last checkpoint by passing None as input.
        logger.info("[%s] Resuming pipeline from checkpoint", rid)
        input_data = None
    else:
        state = initial_state(
            user_id=user_id,
            raw_goal=raw_goal,
            run_id=rid,
            clarification_reply=clarification_reply,
            preferred_language=preferred_language,
            target_months=target_months,
            daily_hours=daily_hours,
            playlist_url=playlist_url,
            website_url=website_url,
            reference_text=reference_text,
        )
        if clarification_reply:
            updates = apply_clarification(state, clarification_reply)
            state.update(updates)
        input_data = state

    try:
        if progress_queue is not None:
            stage_map = {
                "classify": ("classify", 0),
                "intake": ("intake", 1),
                "detect_entity": ("detect_entity", 2),
                "research": ("research", 3),
                "roadmap_draft": ("roadmap_draft", 4),
                "verify": ("verification", 5),
                "repair": ("verification", 5),
                "youtube_enrich": ("youtube_enrich", 6),
                "youtube_attach": ("youtube_enrich", 6),
                "assembly": ("assembly", 7),
            }
            emitted_stages = set()
            final_state = None
            async for event in graph.astream_events(input_data, config, version="v2"):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node")
                if kind == "on_chain_start" and node in stage_map and node not in emitted_stages:
                    emitted_stages.add(node)
                    stage_name, stage_idx = stage_map[node]
                    try:
                        progress_queue.put_nowait({
                            "run_id": rid,
                            "stage": stage_name,
                            "stage_index": stage_idx,
                            "progress": round((stage_idx + 1) / 8.0, 2),
                            "confidence_score": 0.0,
                        })
                    except Exception:
                        pass
                elif kind == "on_chain_end" and node == "assembly":
                    out = event.get("data", {}).get("output")
                    if isinstance(out, dict):
                        final_state = out
            if final_state is None:
                # E.g. paused at clarification
                checkpoint = await store.load(rid)
                final_state = checkpoint.get("state") if checkpoint else None
                if not final_state:
                    final_state = await graph.ainvoke(None, config)
        else:
            final_state: GoalState = await graph.ainvoke(input_data, config)
    except Exception as exc:
        logger.error("[%s] ❌ Graph invocation failed: %s", rid, exc, exc_info=True)
        # Return minimal error state so callers get something usable
        return GoalState(
            run_id=rid,
            user_id=user_id,
            raw_goal=raw_goal,
            error=str(exc),
            degraded=True,
            stage="error",
            stage_index=0,
            needs_clarification=False,
            clarification_question=None,
            missing_slots=[],
            conversation_turns=0,
            goal_type=None,
            clarified_goal=None,
            detected_entity=None,
            realtime_context=None,
            citations=[],
            draft_syllabus=None,
            verified_syllabus=None,
            verification_passed=False,
            repair_attempts=0,
            confidence_score=0.0,
            selected_creator=None,
            selected_creator_name=None,
            preferred_language=preferred_language,
            target_months=target_months,
            daily_hours=daily_hours,
            playlist_url=playlist_url,
            website_url=website_url,
            reference_text=reference_text,
            resource_warnings=[],
            execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            metrics={},
            messages=[],
        )

    total_ms = round((time.perf_counter() - start_time) * 1000, 2)
    final_state["execution_time_ms"] = total_ms

    # Persist final state in legacy store so /goals/checkpoint/{run_id} still works.
    await store.save(
        rid,
        "done" if not final_state.get("error") else "error",
        {k: v for k, v in final_state.items() if k != "messages"},  # strip non-JSON messages
    )

    logger.info(
        '{"event":"goal_agent_run","run_id":"%s","total_ms":%.1f,'
        '"confidence":%.2f,"degraded":%s,"entity":"%s","metrics":%s}',
        rid,
        total_ms,
        final_state.get("confidence_score", 0.0),
        str(final_state.get("degraded", False)).lower(),
        final_state.get("detected_entity") or "none",
        json.dumps(final_state.get("metrics", {})),
    )

    return final_state
