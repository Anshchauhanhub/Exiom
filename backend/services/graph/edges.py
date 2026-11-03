"""
graph/edges.py — Routing functions for LangGraph conditional edges.

Each function receives the current GoalState and returns a string
that LangGraph resolves to the next node name via the edge map
passed to graph.add_conditional_edges().

Fix #4 applied: route_after_verify caps the repair loop with MAX_REPAIR.
"""

from __future__ import annotations

import logging
from langgraph.graph import END

from services.graph.state import GoalState

logger = logging.getLogger("edxiom.graph.edges")

# Maximum number of verify→repair cycles before degraded pass-through.
MAX_REPAIR: int = 2

# ── Stage 1 ──────────────────────────────────────────────────────────

def route_after_intake(state: GoalState) -> str:
    """
    After intake_node:
      - "clarify" → END (pipeline paused, caller returns clarification question)
      - "continue" → detect_entity
    """
    if state.get("needs_clarification"):
        logger.debug("[%s] intake → clarify (pause)", state.get("run_id"))
        return "clarify"
    return "continue"


# ── Stage 2 ──────────────────────────────────────────────────────────

def route_after_entity(state: GoalState) -> str:
    """
    After detect_entity_node:
      - "clarify" → END (ambiguous exam, ask user)
      - "skip_research" → roadmap_draft (skill goal, no entity, no external URLs)
      - "continue" → research
    """
    if state.get("needs_clarification"):
        logger.debug("[%s] entity → clarify (pause)", state.get("run_id"))
        return "clarify"

    # Skip expensive research for generic skill goals with no external resources
    has_entity = bool(state.get("detected_entity"))
    has_resources = bool(
        state.get("website_url")
        or state.get("playlist_url")
        or state.get("reference_text")
    )
    is_exam_or_career = state.get("goal_type") in ("exam", "career")

    if not has_entity and not has_resources and not is_exam_or_career:
        logger.debug("[%s] entity → skip_research", state.get("run_id"))
        return "skip_research"

    return "continue"


# ── Stage 5 ──────────────────────────────────────────────────────────

def route_after_verify(state: GoalState) -> str:
    """
    After verify_syllabus_node:
      - "pass"   → youtube_enrich  (verification passed)
      - "repair" → repair           (failed, repair budget remaining)
      - "fail"   → youtube_enrich  (repair budget exhausted — degraded path)

    Fix #4: always check repair_attempts BEFORE deciding to repair so that
    a stubbornly-failing roadmap never loops forever.
    """
    if state.get("verification_passed"):
        logger.debug("[%s] verify → pass", state.get("run_id"))
        return "pass"

    attempts = state.get("repair_attempts", 0)
    if attempts >= MAX_REPAIR:
        logger.warning(
            "[%s] verify → fail (repair cap %d/%d reached — degraded path)",
            state.get("run_id"), attempts, MAX_REPAIR,
        )
        return "fail"

    logger.info(
        "[%s] verify → repair (attempt %d/%d)",
        state.get("run_id"), attempts + 1, MAX_REPAIR,
    )
    return "repair"
