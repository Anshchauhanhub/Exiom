"""
graph/state.py — Canonical GoalState TypedDict for the LangGraph pipeline.

All node functions receive and return subsets of this dict.
LangGraph merges partial return values into the accumulated state.
The `messages` field uses the built-in `add_messages` reducer so that
nodes can append to it without reading the full history.
"""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

try:
    from langgraph.graph.message import add_messages
except ImportError:  # langgraph < 0.2 fallback
    def add_messages(left, right):  # type: ignore[misc]
        return (left or []) + (right or [])


class GoalState(TypedDict, total=False):
    # ── identity ──────────────────────────────────────────────────────
    run_id: str
    user_id: str

    # ── goal input ────────────────────────────────────────────────────
    raw_goal: str
    clarified_goal: Optional[str]
    goal_type: Optional[str]          # "exam" | "skill" | "career"
    preferred_language: Optional[str] # "Hindi" | "Hinglish" | "English"
    target_months: Optional[int]
    daily_hours: Optional[float]
    playlist_url: Optional[str]
    website_url: Optional[str]
    reference_text: Optional[str]

    # ── clarification loop ────────────────────────────────────────────
    needs_clarification: bool
    clarification_question: Optional[str]
    missing_slots: list[str]
    conversation_turns: int

    # ── entity & research ─────────────────────────────────────────────
    detected_entity: Optional[str]
    realtime_context: Optional[str]
    citations: list[str]

    # ── roadmap ───────────────────────────────────────────────────────
    draft_syllabus: Optional[list[dict]]
    verified_syllabus: Optional[list[dict]]
    verification_passed: bool
    repair_attempts: int              # NEW: caps the verify→repair loop

    # ── creator / youtube ─────────────────────────────────────────────
    selected_creator: Optional[str]
    selected_creator_name: Optional[str]

    # ── output / meta ─────────────────────────────────────────────────
    confidence_score: float
    resource_warnings: list[str]
    stage: str
    stage_index: int
    degraded: bool
    metrics: dict[str, float]
    error: Optional[str]
    execution_time_ms: float

    # ── LangChain Core message history (add_messages reducer) ─────────
    messages: Annotated[list[BaseMessage], add_messages]


def initial_state(
    user_id: str,
    raw_goal: str,
    run_id: str,
    *,
    clarification_reply: Optional[str] = None,
    preferred_language: Optional[str] = None,
    target_months: Optional[int] = None,
    daily_hours: Optional[float] = None,
    playlist_url: Optional[str] = None,
    website_url: Optional[str] = None,
    reference_text: Optional[str] = None,
) -> GoalState:
    """Build the initial GoalState dict for a fresh pipeline run."""
    return GoalState(
        run_id=run_id,
        user_id=user_id,
        raw_goal=raw_goal,
        goal_type=None,
        clarified_goal=None,
        needs_clarification=False,
        clarification_question=None,
        missing_slots=[],
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
        conversation_turns=0,
        execution_time_ms=0.0,
        stage="init",
        stage_index=0,
        degraded=False,
        metrics={},
        error=None,
        messages=[],
    )
