"""
backend/services/graph — LangGraph-powered goal agent pipeline.

Modules:
  state.py   — GoalState TypedDict (single source of truth)
  nodes.py   — 8-stage async node functions (pure, return partial dicts)
  edges.py   — routing / conditional-edge functions
  builder.py — CompiledStateGraph factory + get_goal_graph() singleton
"""
