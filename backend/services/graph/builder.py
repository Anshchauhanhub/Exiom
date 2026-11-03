"""
graph/builder.py — CompiledStateGraph factory for the Exiom goal pipeline.

Fix #1 applied: get_goal_graph() returns a module-level singleton so the
graph is compiled **once at startup**, not per request.  Redis connection
pool creation happens once too.

Graph topology:
  classify → intake →(clarify|skip_research|continue)
           → detect_entity →(clarify|continue|skip_research)
           → research → roadmap_draft → verify
           →(pass|repair|fail)→ youtube_enrich → assembly → END

The verify→repair edge forms a bounded loop (capped by edges.MAX_REPAIR).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("edxiom.graph.builder")


def _make_checkpointer():
    """
    Return AsyncRedisSaver if REDIS_URL is set, otherwise MemorySaver.

    Note: MemorySaver is per-process only.  Multi-worker deployments MUST
    set REDIS_URL so checkpoints survive worker restarts (fix #3 semantics).
    """
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver
            checkpointer = AsyncRedisSaver.from_conn_string(redis_url)
            logger.info("[builder] Checkpointer: AsyncRedisSaver (%s…)", redis_url[:30])
            return checkpointer
        except ImportError:
            logger.warning("[builder] langgraph[redis] not installed — falling back to MemorySaver")
        except Exception as exc:
            logger.warning("[builder] AsyncRedisSaver init failed (%s) — falling back to MemorySaver", exc)

    from langgraph.checkpoint.memory import MemorySaver
    logger.info("[builder] Checkpointer: MemorySaver (single-process only)")
    return MemorySaver()


def build_goal_graph():
    """
    Compile and return the CompiledStateGraph.  Called once at startup.

    Import langgraph here (not at module top-level) so that the package
    is optional at import time — tests that mock the graph can patch
    before this is called.
    """
    from langgraph.graph import StateGraph, END

    from services.graph.state import GoalState
    from services.graph import nodes
    from services.graph.edges import route_after_intake, route_after_entity, route_after_verify

    graph = StateGraph(GoalState)

    # ── Nodes ─────────────────────────────────────────────────────────
    graph.add_node("classify",        nodes.classify_node)
    graph.add_node("intake",          nodes.intake_node)
    graph.add_node("detect_entity",   nodes.detect_entity_node)
    graph.add_node("research",        nodes.fetch_entity_context_node)
    graph.add_node("roadmap_draft",   nodes.synthesize_draft_node)
    graph.add_node("verify",          nodes.verify_syllabus_node)
    graph.add_node("repair",          nodes.repair_syllabus_node)
    graph.add_node("youtube_enrich",  nodes.resolve_best_creator_node)   # 6A
    graph.add_node("youtube_attach",  nodes.enrich_with_youtube_node)    # 6B
    graph.add_node("assembly",        nodes.assembly_node)

    # ── Entry ─────────────────────────────────────────────────────────
    graph.set_entry_point("classify")

    # ── Edges ─────────────────────────────────────────────────────────
    graph.add_edge("classify", "intake")

    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {"clarify": END, "continue": "detect_entity"},
    )

    graph.add_conditional_edges(
        "detect_entity",
        route_after_entity,
        {
            "clarify":         END,
            "skip_research":   "roadmap_draft",
            "continue":        "research",
        },
    )

    graph.add_edge("research",      "roadmap_draft")
    graph.add_edge("roadmap_draft", "verify")

    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "pass":   "youtube_enrich",
            "repair": "repair",
            "fail":   "youtube_enrich",   # degraded — skip extra repair, continue
        },
    )

    graph.add_edge("repair",         "verify")         # bounded loop ←→
    graph.add_edge("youtube_enrich", "youtube_attach")
    graph.add_edge("youtube_attach", "assembly")
    graph.add_edge("assembly",       END)

    checkpointer = _make_checkpointer()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("[builder] Goal graph compiled successfully.")
    return compiled


# ── Module-level singleton (Fix #1) ──────────────────────────────────

_graph = None


def get_goal_graph():
    """Return the compiled graph singleton. Thread-safe via GIL for CPython."""
    global _graph
    if _graph is None:
        _graph = build_goal_graph()
    return _graph
