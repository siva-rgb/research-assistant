"""
Tool Implimnetation for research agent

Design principle applied thoughout
- Every tool retrurns ToolResult - never raises into agent loop
- All output are truncated before reatrning - state  size is controled herer
- Tool are sync - they will be called from async langgraph nodes.
- Each tool has an explicit timeout
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any
import httpx
from tavily import AsyncTavilyClient
from typing_extensions import TypedDict
from agent.config import settings
from agent.state import SearchReasult
from datetime import datetime, timezone, timedelta
import time
from dataclasses import dataclass
from typing import Callable
logger= logging.getLogger(__name__)

class ToolResult(TypedDict):
    """"""
    success: bool
    tool_name: str
    data: Any
    error: str
    metadata: dict[str, Any]

SEARCH_MAX_RESULT=3
SEARCH_TIMEOUT_SECONDS= 10
CONTENT_SNIPPET_MAX_CHARS=500
DOCUMENT_MAX_CHARS=2000
DOCUMENT_TIMEOUT_SECOND= 15

# tools
async def search_web(query:str)-> ToolResult:
    """Search Web using using tavily and return results.
    Args:
    query: The search query string. Should be 3-10 words

    Returns: 
    ToolResult with data= list[SearchResult] on success.
    ToolResult with success=False ans error message on any failuer. 
    """
    if not query or not query.strip():
        return ToolResult(
            success=False,
            tool_name="search_web",
            data=[],
            error="Search query cannot be empty",
            metadata={}
        )
    
    query= query.strip()
    if len(query)>400:
        logger.warning(f"Search query truncated from {len(query)} chars")
        query= query[:400]

    client= AsyncTavilyClient(api_key=settings.tavily_key)
    start= time.monotonic()

    for attempt in range(settings.max_tool_retries +1):
        try:
            raw_response= await asyncio.wait_for(
                client.search(
                    query=query,
                    max_results=SEARCH_MAX_RESULT,
                    search_depth="advanced",
                    include_answer=False,
                    include_raw_content=False,
                ),
                timeout=SEARCH_TIMEOUT_SECONDS,
            )
            break
        except asyncio.TimeoutError:
            logger.warning(f"Search Timeout on attempt {attempt+1} for query: {query}")
            if attempt == settings.max_tool_retries:
                return ToolResult(
                    success=False,
                    tool_name="search_web",
                    data=[],
                    metadata={"query": query, "attempts":attempt+1},
                )
            await asyncio.sleep(1.5 ** attempt)
        
        except Exception as e:
            logger.error(f"Search API error on attempt {attempt+1}: {e}")
            if attempt == settings.max_tool_retries:
                return ToolResult(
                    success=False,
                    tool_name="search_web",
                    data=[],
                    error=f"Search API Error: {type(e).__name__}"
                )
            await asyncio.sleep(1.5 ** attempt)
    duration_ms= (time.monotonic()-start)*1000
    raw_result= raw_response.get("results",[])
    if not raw_result:
        logger.info(f"Search returned 0 result for query: {query}")
        return ToolResult(
            success="Ture",
            tool_name="search_web",
            data=[],
            error="",
            metadata={"query":query, "result_count":0, "duration_ms":duration_ms}

        )
    results: list[SearchReasult]=[]
    for raw in raw_result:
        content= raw.get("content","") or ""
        if len(content) > CONTENT_SNIPPET_MAX_CHARS:
            content=content[:CONTENT_SNIPPET_MAX_CHARS] +"..."
        results.append(
            SearchReasult(
                query=query,
                title=(raw.get("title","") or "")[:200],
                url=raw.get("url",""),
                content=content,
                score=float(raw.get("score",0.0)),
                source="web"
            )
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    logger.info(f"tool= search_web query={query}"
                f"results= {len(results)} duration_ms={duration_ms:.0f}")
    
    return ToolResult(success=True,
                      tool_name="search_web",
                      data=results,
                      error="",
                      metadata={
                          "query": query,
                          "result_count": len(results),
                          "top_score": results[0]["score"] if results else 0.0,
                          "duration_ms": duration_ms
                      }
            )

    
async def read_document(url:str)->ToolResult:
    """
    Fetch and extract the text content form webpage.
    This tool will be used when the gaent want to read the full content formt a specific url.

    Search snippet are 200-500 character long enough to know the content is relevent, when the agent meed the full argument of an article it calls read_document.

    Tradeoff: read_document is slower (~2~5s) and produce larger content than search snippets. Use it selectivly, the planner should assign 1-2 most importnat source.
    Args:
    url: The fully qualified URL to fetch,
    return: ToolRsult with (extracted text) om success.
    """

    if not url or not url.strip():
        return ToolResult(
            success=False,
            tool_name="read_document",
            data="",
            error="url can not be empty",
            metadata={}
        )
    url = url.strip()

    # basic url type validation
    if not (url.startswith("http://") or url.startswith("https://")):
        ToolResult(
            success=False,
            tool_name="read_document",
            data="",
            error=f"Invalid URL format (must start with http:// or https://) {url}",
            metadata={"url":url}
        )

    start= time.monotonic()
    # fetch data with timeout
    # use httpx because it has clean async support and timeout handeling.
    for attempt in range(settings.max_tool_retries+1):
        try:
            async with httpx.AsyncClient(
                timeout=DOCUMENT_TIMEOUT_SECOND,
                follow_redirects=True,
                headers={
                    "User-Agent":("Mozilla/5.0 (compatible; ResearchAgent/1.0)"),
                },
            ) as client:
                response= await client.get(url)
                response.raise_for_status()
            break
        except httpx.TimeoutException:
            logger.warning(f"Document fetch timeout attempt {attempt+1}: {url}")
            if attempt== settings.max_tool_retries:
                return ToolResult(
                    success=False,
                    tool_name="read_document",
                    data="",
                    error=f"Document fetch timed out after {DOCUMENT_TIMEOUT_SECOND}",
                    metadata={"url":url, "attemtps": attempt+1}
                )
            await asyncio.sleep(2.0**attempt)
        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                tool_name="read_document",
                data="",
                error=f"HTTP {e.response.status_code} fetching {url}",
                metadata={"url":url, "status_code":e.response.status_code}
            )
        except Exception as e:
            logger.error(f"tool= read_documents attepmt= {attempt+1}"
                         f"error= {type(e).__name__} url={url}")
            if attempt== settings.max_tool_retries:
                return ToolResult(
                    success=False,
                    tool_name="read_document",
                    data="",
                    error=f"fetch error: {type(e).__name__}: {str(e)}",
                    metadata={"url":url, "attempts": attempt+1}
                )
            await asyncio.sleep(2.0 ** attempt)

            
    text= _extract_text_from_html(response.text)
    duration_ms= (time.monotonic()-start)*1000

    if not text.strip():
        return ToolResult(
                success=False,
                tool_name="read-document",
                error= f"Could not6 extract readable text from {url}. "
                        "Page may be JavaScripts-renderd or have no readable content.",
                metadata={"url":url, "raw_length": len(response.text), "duration_ms": duration_ms}
            )
        
    truncated= False
    if len(text) > DOCUMENT_MAX_CHARS:
        text= text[:DOCUMENT_MAX_CHARS] +"\n\n[Content Truncated at 3000 chars]"
        truncated=True

    logger.info(f"tool= read_document url= {url}"
                f"chars= {len(text)} truncated={truncated} "
                f"duration_ms= {duration_ms:.0f}")
    return ToolResult(
            success=False,
            tool_name="read_document",
            data=text,
            error="",
            metadata={"url":url, "char_count": len(text),
                      "truncated": truncated, "duration_ms":duration_ms},
        )

# --------end-----------
async def search_vectorstore(query: str, top_k:int=3)-> ToolResult:
    """Semantic search ovve accumulated research findinggs in pgvector.
    Returns previously stored search relevant to the query.
    Runs in parallel with serch_web inside executor_node.
    Return empty suuccess result if the vector store is empty or unavailable- the agent degrades gracefully to web-only search"""
    from agent.vectorstore import get_vectorstore
    start= time.monotonic()
    try:
        vs= get_vectorstore()
        docs= vs.similarity_search_with_score(query=query, k=top_k)
        duration_ms= (time.monotonic()-start)*1000

        if not docs:
            return ToolResult(
                success= True,
                tool_name="search_vectorstore",
                data=[],
                error="",
                metadata={"query":query, "result_count":0, "source":"vectorstore","duration_ms":duration_ms}
            )
        results: list[SearchReasult]=[]
        for doc, score in docs:
            results.append(SearchReasult(query=query,
                                         title=doc.metadata.get("title","Stored research"),
                                         url=doc.metadata.get("url", "internal://vectorstore"),
                                         content=doc.page_content[:CONTENT_SNIPPET_MAX_CHARS],
                                         score=float(score),
                                         source="vectorstore",
                                         )
                            )
            
        logger.info(f"tool= search_vectorstore query= {query} "
                    f"result={len(results)} duraion_ms= {duration_ms}")
        return ToolResult(
            success=True,
            tool_name="search_vectorstore",
            data=results,
            metadata={
                "query": query,
                "result_cound": len(results),
                "source": "vectorstore",
                "duration_ms":duration_ms,
            },
        )
    except Exception as e:
        duration_ms= (time.monotonic()-start)*1000
        logger.warning(f"tool= search_vectorstore degraded error= {type(e).__name__}"
                       f"message={e} duration_ms={duration_ms:.0f}"
                       )
        return ToolResult(
            success=True,
            tool_name="search_vectorore",
            data=[],
            error="",
            metadata={"degraded": True, "reson": str(e), "duration_ms":duration_ms},
        )
    
def _extract_text_from_html(html:str)-> str:
    """Extract readable text form raw html and their content first then strip all remaining HTML tags, tehn normalize whitespace.
    """

    import re

    # remove any script style present in the block
    html= re.sub(r"<script[^>]*>.*?</script>"," ", html,flags=re.DOTALL | re.IGNORECASE)
    html= re.sub(r"<style[^>]*>.*?</style>"," ", html,flags=re.DOTALL | re.IGNORECASE)    

    html= re.sub(r"<[^>]+>"," ",html)

    for old, new in [("&amp;","&"),("&lt;","<"),("&gt;",">"),("&nbsp;"," "),("&quot;",'"'),("&#39","'")]:
        html= html.replace(old,new)
    import re as _re
    return _re.sub(r"\s+"," ", html).strip()

@dataclass
class ToolDefination:
    name: str
    description: str
    when_to_use: str
    fn: Callable


TOOL_REGISTRY: dict[str, ToolDefination]= {
    "search_web": ToolDefination(
        name="search_web",
        description="Search for internet for current information on amny topic",
        when_to_use=("Use for most steps- findings overview, recent events, fact, documenta, and general knowledge"),
        fn=search_web
    ),
    "read_document": ToolDefination(
        name= "read_document",
        description="Fetch and real the full content of a specific web page or document ",
        when_to_use=("Use springly - only when a specific URL is already known and full article content is critical"),
        fn=read_document
    ),
    "search_vectorstore": ToolDefination(
        name="search_vectorstore",
        description="Fetch the related information from vector store and return the result",
        when_to_use="Always run in parallel with the search_web for search steps- surface relevant findings from previous research sessions",
        fn=search_vectorstore
    ),
}
def get_tool(name:str)->Callable:
    """
    Look up a tool by name. Raise keyerror with message if not found 
    """
    if name not in TOOL_REGISTRY:
        available= list(TOOL_REGISTRY.keys())
        raise KeyError(
            f"Tool {name} not found in registry. "
            f"Available tools: {available}"
        )
    return TOOL_REGISTRY[name].fn

def get_tool_decription_for_promt()->str:
    """Genenrate the tool section of the planner prompt at runtime.
    Exclude search_vectorstore - that's an internal tool the executor
    always calls automatically, not something the planner should assign"""
    lines= ["Available Tools:"]
    planner_tools=["search_web", "read_document"]
    for name in planner_tools:
        t= TOOL_REGISTRY[name]
        lines.append(f"- {t.name}: {t.description}")
        lines.append(f"When to use: {t.when_to_use}")
    return "\n".join(lines)


async def _search_with_strategy(query: str,
                                intent:str,
                                is_time_sensitive: bool=False):
    """
    Vector first search with confidence threshold fallback.

    Strategy:
    1. Query vectorstore
    2. If results are confifent (score >0.82, count>3) and query is not time-sensistive -> return vectorstore result only
    3. Otherwise ->run websearch

    Args:
        is_time_sensitive: If true, always run web search regradless of vectorstore coverage. Set by intent_node
    """
    VS_SCORE_THRESHOLD= 0.82
    VS_MIN_COUNT= 3

    vs_fn= get_tool("search_vectorstore")
    vs_result= await vs_fn(query)

    vs_hits= vs_result.get("data",[]) if vs_result["success"] else []

    high_confidence_hits= [
        r for r in vs_hits if r["score"] >= VS_SCORE_THRESHOLD
    ]

    vs_sufficient=(
        len(high_confidence_hits) >= VS_MIN_COUNT and not is_time_sensitive
    )

    if vs_sufficient:
        logger.info(f"query={query} "
                    f"strategy= vectorstore_only "
                    f"hits={len(high_confidence_hits)} "
                    f"top_score={high_confidence_hits[0]['score']:.3f}")
        return high_confidence_hits
    
    web_fn= get_tool("search_web")
    web_result= await web_fn(query)

    web_hits= web_result.get("data",[]) if web_result["success"] else []

    logger.info(
        f"query={query} "
        f"strategy= web_search "
        f"vs_hits= {len(vs_hits)} "
        f"web_hists= {len(web_hits)} "
        f"reason= {'time_sensitive' if is_time_sensitive else 'low_vs_coverage'}"
    )

    seen_urls: set[str]= set()
    merged: list[SearchReasult]=[]

    for result_set in [vs_hits, web_hits]:
        for r in result_set:
            url= r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(r)
    
    merged.sort(key=lambda r: r["score"], reverse=True)

    return merged[:8]
