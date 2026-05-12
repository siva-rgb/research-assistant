# tests/test_executor_strategy.py
"""
Tests for _search_with_strategy and the updated executor_node.

Three core behaviours to verify:
  1. Vectorstore-sufficient path skips web search entirely
  2. Vectorstore-insufficient path runs web search and merges
  3. Time-sensitive queries always run web search regardless of VS coverage
  4. URL deduplication works correctly across sources
  5. Executor correctly passes is_time_sensitive from state
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from agent.state import ResearchPlan, SubTask, SearchReasult, create_itital_state
from agent.nodes import (
    executor_node,
)
from agent.tools import _search_with_strategy,get_tool

# ── Fixtures ──────────────────────────────────────────────────────────────────
VS_SCORE_THRESHOLD= 0.85
VS_MIN_COUNT=3
MAX_MERGED_RESULTS=8
def make_vs_results(
    count: int,
    score: float,
    url_prefix: str = "https://vectorstore.example.com/",
) -> list[SearchReasult]:
    """Create fake vectorstore results with a given score."""
    return [
        SearchReasult(
            query="test query",
            title=f"VS Result {i}",
            url=f"{url_prefix}doc-{i}",
            content=f"Vectorstore content {i}",
            score=score,
            source="vectorstore",
        )
        for i in range(count)
    ]


def make_web_results(
    count: int,
    score: float = 0.78,
    url_prefix: str = "https://web.example.com/",
) -> list[SearchReasult]:
    """Create fake web search results."""
    return [
        SearchReasult(
            query="test query",
            title=f"Web Result {i}",
            url=f"{url_prefix}article-{i}",
            content=f"Web content {i}",
            score=score,
            source="web",
        )
        for i in range(count)
    ]


def make_tool_result(success: bool, data: list) -> dict:
    return {
        "success": success,
        "tool_name": "test",
        "data": data,
        "error": "" if success else "failed",
        "metadata": {},
    }


def state_with_plan(is_time_sensitive: bool = False) -> dict:
    """State with a single search_web subtask ready to execute."""
    s = create_itital_state("What is LangGraph?", "test-session")
    s["intent"] = "Understand what LangGraph is."
    s["research_scope"] = "Moderate"
    s["is_time_sensitive"] = is_time_sensitive
    s["plan"] = ResearchPlan(
        subtasks=[
            SubTask(
                step_index=0,
                task_type="search_web",
                description="Overview of LangGraph",
                query="LangGraph overview",
                status="pending",
            )
        ],
        total_steps=1,
    )
    s["current_step"] = 0
    return s


# ── _is_time_sensitive tests ──────────────────────────────────────────────────

# def test_time_sensitive_detects_latest():
#     assert _is_time_sensitive("What is the latest LangGraph release?") is True

# def test_time_sensitive_detects_year():
#     assert _is_time_sensitive("What AI announcements happened in 2025?") is True

# def test_time_sensitive_detects_today():
#     assert _is_time_sensitive("What happened today in AI?") is True

# def test_time_sensitive_not_triggered_on_stable_query():
#     assert _is_time_sensitive("What is LangGraph and how does it work?") is False

# def test_time_sensitive_not_triggered_on_historical():
#     assert _is_time_sensitive("How does cosine similarity work?") is False

# def test_time_sensitive_case_insensitive():
#     assert _is_time_sensitive("What are the LATEST updates to GPT-4?") is True


# ── _search_with_strategy tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_strategy_uses_vectorstore_only_when_sufficient():
    """
    When VS returns >= VS_MIN_COUNT results at >= VS_SCORE_THRESHOLD,
    web search must not be called.
    """
    vs_results = make_vs_results(count=4, score=0.90)
    mock_vs = AsyncMock(return_value=make_tool_result(True, vs_results))
    mock_web = AsyncMock()   # should never be called

    with patch("agent.tools.get_tool") as mock_get_tool:
        def tool_dispatch(name):
            return mock_vs if name == "search_vectorstore" else mock_web
        mock_get_tool.side_effect = tool_dispatch

        results = await _search_with_strategy(
            query="LangGraph overview",
            session_id="test",
            is_time_sensitive=False,
        )

    # Web tool must not have been called
    mock_web.assert_not_called()
    assert len(results) == 4
    assert all(r["source"] == "vectorstore" for r in results)


@pytest.mark.asyncio
async def test_strategy_falls_back_to_web_when_vs_insufficient():
    """
    When VS returns fewer than VS_MIN_COUNT high-confidence results,
    web search must run and results must be merged.
    """
    vs_results = make_vs_results(count=1, score=0.85)  # only 1 — not enough
    web_results = make_web_results(count=5, score=0.79)

    mock_vs = AsyncMock(return_value=make_tool_result(True, vs_results))
    mock_web = AsyncMock(return_value=make_tool_result(True, web_results))

    with patch("agent.tools.get_tool") as mock_get_tool:
        def tool_dispatch(name):
            return mock_vs if name == "search_vectorstore" else mock_web
        mock_get_tool.side_effect = tool_dispatch

        results = await _search_with_strategy(
            query="LangGraph overview",
            session_id="test",
            is_time_sensitive=False,
        )

    mock_web.assert_called_once()
    assert len(results) == 6        # 1 VS + 5 web
    sources = {r["source"] for r in results}
    assert "vectorstore" in sources
    assert "web" in sources


@pytest.mark.asyncio
async def test_strategy_always_uses_web_when_time_sensitive():
    """
    Time-sensitive queries must always run web search even when VS
    has sufficient high-confidence coverage.
    """
    # VS has excellent coverage — 5 results at 0.95
    vs_results = make_vs_results(count=5, score=0.95)
    web_results = make_web_results(count=3, score=0.80)

    mock_vs = AsyncMock(return_value=make_tool_result(True, vs_results))
    mock_web = AsyncMock(return_value=make_tool_result(True, web_results))

    with patch("agent.tools.get_tool") as mock_get_tool:
        def tool_dispatch(name):
            return mock_vs if name == "search_vectorstore" else mock_web
        mock_get_tool.side_effect = tool_dispatch

        results = await _search_with_strategy(
            query="latest LangGraph updates",
            session_id="test",
            is_time_sensitive=True,    # ← forces web regardless
        )

    # Web MUST have been called despite good VS coverage
    mock_web.assert_called_once()
    # Results include both sources
    sources = {r["source"] for r in results}
    assert "web" in sources


@pytest.mark.asyncio
async def test_strategy_deduplicates_by_url():
    """
    If the same URL appears in both VS and web results,
    it must appear only once in the merged output.
    """
    shared_url = "https://example.com/shared-article"

    vs_results = [SearchReasult(
        query="q", title="Shared", url=shared_url,
        content="vs content", score=0.88, source="vectorstore",
    )]
    web_results = [
        SearchReasult(
            query="q", title="Shared", url=shared_url,
            content="web content", score=0.82, source="web",
        ),
        SearchReasult(
            query="q", title="Unique web", url="https://example.com/unique",
            content="unique", score=0.75, source="web",
        ),
    ]

    mock_vs = AsyncMock(return_value=make_tool_result(True, vs_results))
    mock_web = AsyncMock(return_value=make_tool_result(True, web_results))

    with patch("agent.tools.get_tool") as mock_get_tool:
        def tool_dispatch(name):
            return mock_vs if name == "search_vectorstore" else mock_web
        mock_get_tool.side_effect = tool_dispatch

        results = await _search_with_strategy(
            query="q",
            session_id="test",
            is_time_sensitive=False,
        )

    # shared_url appears exactly once despite being in both sources
    result_urls = [r["url"] for r in results]
    assert result_urls.count(shared_url) == 1
    assert len(results) == 2     # shared + unique


@pytest.mark.asyncio
async def test_strategy_respects_max_merged_results():
    """Output is capped at MAX_MERGED_RESULTS regardless of input size."""
    vs_results = make_vs_results(count=2, score=0.75)   # below threshold
    web_results = make_web_results(count=20, score=0.80)

    mock_vs = AsyncMock(return_value=make_tool_result(True, vs_results))
    mock_web = AsyncMock(return_value=make_tool_result(True, web_results))

    with patch("agent.tools.get_tool") as mock_get_tool:
        def tool_dispatch(name):
            return mock_vs if name == "search_vectorstore" else mock_web
        mock_get_tool.side_effect = tool_dispatch

        results = await _search_with_strategy(
            query="q",
            session_id="test",
            is_time_sensitive=False,
        )

    assert len(results) <= MAX_MERGED_RESULTS


@pytest.mark.asyncio
async def test_strategy_returns_empty_when_both_tools_fail():
    """If both tools fail, return empty list — never raise."""
    mock_vs = AsyncMock(return_value=make_tool_result(False, []))
    mock_web = AsyncMock(return_value=make_tool_result(False, []))

    with patch("agent.tools.get_tool") as mock_get_tool:
        def tool_dispatch(name):
            return mock_vs if name == "search_vectorstore" else mock_web
        mock_get_tool.side_effect = tool_dispatch

        results = await _search_with_strategy(
            query="q",
            session_id="test",
            is_time_sensitive=False,
        )

    assert results == []


@pytest.mark.asyncio
async def test_strategy_results_sorted_by_score_descending():
    """Merged results must always be sorted best-first."""
    vs_results = make_vs_results(count=2, score=0.70)
    web_results = [
        SearchReasult(query="q", title="High", url="https://a.com",
                     content="c", score=0.92, source="web"),
        SearchReasult(query="q", title="Low", url="https://b.com",
                     content="c", score=0.61, source="web"),
    ]

    mock_vs = AsyncMock(return_value=make_tool_result(True, vs_results))
    mock_web = AsyncMock(return_value=make_tool_result(True, web_results))

    with patch("agent.nodes.get_tool") as mock_get_tool:
        def tool_dispatch(name):
            return mock_vs if name == "search_vectorstore" else mock_web
        mock_get_tool.side_effect = tool_dispatch

        results = await _search_with_strategy(
            query="q",
            session_id="test",
            is_time_sensitive=False,
        )

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ── executor_node integration tests ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_executor_passes_time_sensitive_flag():
    """
    executor_node must read is_time_sensitive from state and pass it
    to _search_with_strategy. Verify the flag flows correctly.
    """
    state = state_with_plan(is_time_sensitive=True)
    captured_flags = []

    async def mock_strategy(query, session_id, is_time_sensitive):
        captured_flags.append(is_time_sensitive)
        return []

    with patch("agent.nodes._search_with_strategy", side_effect=mock_strategy), \
         patch("agent.nodes.refine_query", AsyncMock(return_value="LangGraph overview")):
        await executor_node(state)

    assert captured_flags == [True]


@pytest.mark.asyncio
async def test_executor_advances_step_on_success():
    """current_step must increment by 1 after each executor call."""
    state = state_with_plan()
    web_results = make_web_results(count=3)

    with patch("agent.nodes._search_with_strategy",
               AsyncMock(return_value=web_results)), \
         patch("agent.nodes.refine_query",
               AsyncMock(return_value="LangGraph overview")):
        result = await executor_node(state)

    assert result["current_step"] == 1
    assert result["execution_iteration_count"] == 1


@pytest.mark.asyncio
async def test_executor_advances_step_even_when_no_results():
    """
    If search returns nothing, current_step still advances.
    We don't stall on empty results — reflection handles coverage.
    """
    state = state_with_plan()

    with patch("agent.nodes._search_with_strategy", AsyncMock(return_value=[])), \
         patch("agent.nodes.refine_query",
               AsyncMock(return_value="LangGraph overview")):
        result = await executor_node(state)

    assert result["current_step"] == 1
    # search_results should NOT be in updates (nothing to append)
    assert "search_results" not in result


@pytest.mark.asyncio
async def test_executor_respects_iteration_limit():
    """When iteration limit is hit, executor returns without calling any tool."""
    state = state_with_plan()
    state["execution_iteration_count"] = 999

    search_called = []

    with patch("agent.nodes.settings") as mock_settings, \
         patch("agent.nodes._search_with_strategy",
               side_effect=lambda *a, **kw: search_called.append(1)):
        mock_settings.max_search_iteration = 3
        mock_settings.max_tool_retries = 2
        result = await executor_node(state)

    assert search_called == []     # tool never called
    assert result["execution_iteration_count"] == 1000


@pytest.mark.asyncio
async def test_executor_handles_read_document_subtask():
    """read_document branch writes to documents_read, not search_results."""
    state = create_itital_state("Read a document", "test-session")
    state["intent"] = "Read a specific document."
    state["research_scope"] = "Narrow"
    state["is_time_sensitive"] = False
    state["plan"] = ResearchPlan(
        subtasks=[SubTask(
            step_index=0,
            task_type="read_document",
            description="Read LangGraph docs",
            query="https://langchain-ai.github.io/langgraph/",
            status="pending",
        )],
        total_steps=1,
    )
    state["current_step"] = 0

    mock_read = AsyncMock(return_value={
        "success": True,
        "tool_name": "read_document",
        "data": "LangGraph is a library for...",
        "error": "",
        "metadata": {"chars": 100},
    })

    with patch("agent.tools.get_tool", return_value=mock_read):
        result = await executor_node(state)

    assert "documents_read" in result
    assert len(result["documents_read"]) == 1
    assert result["documents_read"][0]["content"] == "LangGraph is a library for..."
    assert "search_results" not in result