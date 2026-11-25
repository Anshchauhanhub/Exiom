import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.learning_context import LearningContext, build_goal_context_string


def test_agent_context_marks_paused_goals_as_on_hold():
    context = LearningContext(
        all_goal_summaries=[{
            "title": "Paused course",
            "status": "paused",
            "total_tasks": 2,
            "completed_tasks": 1,
            "current_module": "Module 2",
            "module_titles": ["Module 1", "Module 2"],
        }]
    )

    prompt_context = build_goal_context_string(context)

    assert "PAUSED goals as on hold" in prompt_context
    assert "'Paused course' — 50% done, currently paused" in prompt_context
    assert "ACTIVE GOALS:" not in prompt_context


def test_groq_client_rejects_leading_equals(monkeypatch):
    pytest.importorskip("groq")
    from services.groq import GroqConfigurationError, get_groq_client
    import services.groq as groq_service

    groq_service._groq_client = None
    monkeypatch.setenv("GROQ_API_KEY", "=gsk_example")

    with pytest.raises(GroqConfigurationError, match="leading '='"):
        get_groq_client()


@pytest.mark.asyncio
async def test_calendar_tasks_exclude_incomplete_paused_goal_parts():
    pytest.importorskip("jose")
    from routers.goals import get_all_tasks

    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    db.execute.return_value = result

    await get_all_tasks(SimpleNamespace(id="11111111-1111-1111-1111-111111111111"), db)

    statement = db.execute.call_args.args[0]
    compiled = statement.compile()
    query = str(compiled)
    assert "goals.status =" in query
    assert "parts.status =" in query
    assert " OR " in query
    assert "active" in compiled.params.values()
    assert "passed" in compiled.params.values()
