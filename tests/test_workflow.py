"""Structural tests for the LangGraph financial workflow.

These do not touch any external API. They only verify that the graph wiring
and the router/tool contracts stay in sync, which is the class of bug most
likely to slip in when a data source is added or renamed.
"""

from typing import get_args

from backend.core.rag.adaptive_rag import RouteQuery
from backend.core.rag.financial_workflow import app, tool_map


def _route_sources() -> set[str]:
    """The set of data-source names the router is allowed to emit."""
    return set(get_args(RouteQuery.model_fields["primary_datasource"].annotation))


def test_graph_compiles_with_expected_nodes():
    nodes = set(app.get_graph().nodes)
    assert {"router", "retriever", "quality_filter", "reconciler", "generator"} <= nodes


def test_every_route_source_has_a_tool():
    missing = _route_sources() - set(tool_map)
    assert not missing, f"router can select sources with no tool: {missing}"


def test_no_orphan_tools():
    orphan = set(tool_map) - _route_sources()
    assert not orphan, f"tool_map has entries the router can never select: {orphan}"


def test_all_tools_are_callable():
    assert all(callable(fn) for fn in tool_map.values())
