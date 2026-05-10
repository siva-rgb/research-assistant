import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from agent.tools import search_web, read_document, get_tool, ToolResult

def make_tavily_response(n_result:int =2)-> dict:
    """Build fake Tavily API response with n result"""
    return {
        "results": [
            {
                "title": f"Result Title {i}",
                "url": f"https://example.com/article-{i}",
                "content": f"Content snippet for result {i}. "*10,
                "score": round(0.9- i*0.1, 1),
            }
            for i in range(n_result)
        ]
    }

@pytest.mark.asyncio
async def test_search_web_success():
    """Happy path - Tavily return results, we get a clean ToolResult"""
    fake_response= make_tavily_response(2)
    with patch("agent.tools.AsyncTavlilyClient") as mock_client_class:
        mock_client= AsyncMock()
        mock_client.search.return_value= fake_response
        mock_client_class.return_value= mock_client

        result= await search_web("Extremely obscure query xyz123")
    
    assert result["success"] is True
    assert result["data"]==[]
    assert result["metadata"]["result-count"]== 0


@pytest.mark.asyncio
async def test_search_web_empty_query():
    """Empty query should fall immidiattely without calling the API"""
    result= await search_web("")
    assert result["success"]is False
    assert "empty" in result["error"].lower()

@pytest.mark.asyncio
async def test_search_web_no_result():
    """Tavily returning 0 result"""
    with patch("agent.tools.AsyncTavilyClient") as mock_client_class:
        mock_client= AsyncMock()
        mock_client.search.return_value= {"results":[]}
        mock_client_class.return_value= mock_client

@pytest.mark.asyncio
async def test_search_web_timeout():
    """Timeout should return success=False  with a clear error message"""
    with patch("agent.tools.AsyncTailyClient") as mock_client_class:
        mock_client= AsyncMock()
        mock_client.search.side_effect= asyncio.TimeoutError()
        mock_client_class.return_value= mock_client

        with patch("agent.tools.settings") as mock_settings:
            mock_settings.tavily_api_key="fake"
            mock_settings.max_tool_retries=0

            ressult= await search_web("test query")

    assert ressult["success"] is False
    assert "timed out" in ressult["error"].lower()

def test_get_tool_returns_correct_function():
    tool= get_tool("search_web")
    assert tool is search_web

    tool= get_tool("read_document")
    assert tool is read_document

def test_get_tool_raise_on_unknown():
    with pytest.raises(KeyError) as exc_info:
        get_tool("nonexisitent_tool")
    assert "nonexisitent_tool" in str(exc_info.value)
    assert "Avaialable tools" in str(exc_info.value)