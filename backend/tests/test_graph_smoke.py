"""
tests/test_graph_smoke.py — CI smoke test for the LangGraph goal pipeline.

Runs one full pipeline invocation against a mocked LLM so that:
  1. Graph wiring errors fail the build rather than the deploy.
  2. The verify→repair loop is exercised (repair_attempts cap).
  3. No real Groq API calls are made (hermetic, $0).

Usage:
  cd backend && python -m pytest tests/test_graph_smoke.py -v
"""

from __future__ import annotations

import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_llm_response(content: str):
    """Return a mock ChatGroq that always replies with *content*."""
    ai_msg = MagicMock()
    ai_msg.content = content
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=ai_msg)
    return llm


def _make_mock_gateway(json_payload: str = '{"modules": [{"title": "Mock Module 1", "parts": ["topic A", "topic B"]}]}'):
    """Return a mock LLMGateway that returns structured JSON."""
    gw = MagicMock()
    resp = MagicMock()
    resp.text = json_payload
    resp.model_used = "mock"
    resp.degraded = False
    resp.repaired = False
    resp.error = None
    gw.call = AsyncMock(return_value=resp)
    return gw


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_graph_wiring():
    """
    Verify the graph compiles without errors.
    Does NOT require LangGraph to be installed in the test env —
    if import fails, the test is skipped with a clear message.
    """
    try:
        from services.graph.builder import build_goal_graph
    except ImportError as exc:
        pytest.skip(f"LangGraph not installed — skipping graph wiring test: {exc}")

    # Use MemorySaver regardless of REDIS_URL in CI
    with patch.dict(os.environ, {"REDIS_URL": ""}):
        graph = build_goal_graph()
    assert graph is not None, "build_goal_graph() returned None"


@pytest.mark.asyncio
async def test_graph_full_run_skill_goal():
    """
    Run a full skill-goal pipeline with mocked LLM and services.
    Asserts: stages execute, final state has syllabus and confidence > 0.
    """
    try:
        from services.graph.builder import build_goal_graph
        from services.graph.state import initial_state
    except ImportError as exc:
        pytest.skip(f"LangGraph not installed: {exc}")

    mock_gw = _make_mock_gateway()
    mock_llm = _mock_llm_response("SKILL")

    with (
        patch("services.graph.nodes._get_chat_model", return_value=mock_llm),
        patch("services.graph.nodes.get_gateway", return_value=mock_gw),
        patch("services.groq.generate_exam_roadmap_structured", new_callable=AsyncMock, side_effect=Exception("Instructor disabled in test")),
        patch("services.groq.generate_roadmap_grounded", new_callable=AsyncMock, return_value=[{"title": "Module 1", "parts": ["topic A"]}]),
        patch("services.exam_resolver.resolve_exam") as mock_resolve,
        patch("services.graph.nodes.enrich_with_youtube_node", new_callable=AsyncMock, return_value={"verified_syllabus": [{"title": "Module 1", "parts": ["topic A || vid123"]}], "stage": "youtube_enrich", "stage_index": 6}),
        patch("services.youtube.search_youtube_video_id", new_callable=AsyncMock, return_value=None),
        patch.dict(os.environ, {"REDIS_URL": ""}),
    ):
        mock_res = MagicMock()
        mock_res.matched = False
        mock_res.entity = None
        mock_res.candidates = []
        mock_res.confidence = 0.0
        mock_resolve.return_value = mock_res

        graph = build_goal_graph()
        state = initial_state(
            user_id="test-user",
            raw_goal="Learn Python in 3 months beginner level",
            run_id="smoke-test-001",
        )
        config = {"configurable": {"thread_id": "smoke-test-001"}}
        final = await graph.ainvoke(state, config)

    assert final is not None, "Graph returned None"
    assert final.get("stage") == "done", f"Expected stage='done', got '{final.get('stage')}'"
    assert final.get("confidence_score", 0) > 0, "confidence_score should be > 0"
    syllabus = final.get("verified_syllabus") or final.get("draft_syllabus")
    assert syllabus, "Final state has no syllabus"


@pytest.mark.asyncio
async def test_repair_loop_cap():
    """
    Verify route_after_verify returns 'fail' once repair_attempts >= MAX_REPAIR,
    not 'repair', so the loop cannot run indefinitely.
    """
    from services.graph.edges import route_after_verify, MAX_REPAIR

    exhausted_state = {
        "run_id": "test",
        "verification_passed": False,
        "repair_attempts": MAX_REPAIR,
    }
    result = route_after_verify(exhausted_state)
    assert result == "fail", (
        f"Expected 'fail' when repair_attempts={MAX_REPAIR}, got '{result}'"
    )

    # One below cap → should still try repair
    one_below = {**exhausted_state, "repair_attempts": MAX_REPAIR - 1}
    result = route_after_verify(one_below)
    assert result == "repair", (
        f"Expected 'repair' when attempts={MAX_REPAIR - 1}, got '{result}'"
    )


@pytest.mark.asyncio
async def test_clarification_pause():
    """
    Verify that a goal without timeline or level info causes the graph
    to exit at the intake→clarify edge with needs_clarification=True.
    """
    try:
        from services.graph.builder import build_goal_graph
        from services.graph.state import initial_state
    except ImportError as exc:
        pytest.skip(f"LangGraph not installed: {exc}")

    mock_gw = _make_mock_gateway()
    mock_llm = _mock_llm_response("SKILL")

    with (
        patch("services.graph.nodes._get_chat_model", return_value=mock_llm),
        patch("services.graph.nodes.get_gateway", return_value=mock_gw),
        patch.dict(os.environ, {"REDIS_URL": ""}),
    ):
        graph = build_goal_graph()
        state = initial_state(
            user_id="test-user",
            raw_goal="Learn Python",   # missing: no timeline, no level
            run_id="smoke-clarify-001",
        )
        config = {"configurable": {"thread_id": "smoke-clarify-001"}}
        final = await graph.ainvoke(state, config)

    assert final.get("needs_clarification") is True, (
        "Expected needs_clarification=True for underspecified goal"
    )
    assert final.get("clarification_question"), "Expected a clarification question"
