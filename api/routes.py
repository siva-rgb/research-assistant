"""All route handlers
Routes are thin - they validate the request, call the appropriate service function and return response
"""

from __future__ import annotations
import logging
import uuid
from datetime import datetime, timezone
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from api.schema import (ResearchRequest, ResearchResponse,
                        FeedbackRequest, FeedbackResponse,
                        SessionStatusResponse, HealthResponse)

from api.session import session_store
from api.streaming import steam_research_session

logger= logging.getLogger(__name__)
router= APIRouter()

def now_iso()->str:
    return datetime.now(timezone.utc).isoformat()

@router.post("/research", summary="Start a research session",
             description=("Accepts a research query. If stream= True (default), return an "
                          "SSE Stream of typed events. If stream=False, waits for completion "
                          "and return full JSON result"),
                          responses={
                              200:{"description": "SSE stream (stream= True) or JSON result (stream=False)"},
                              422:{"description": "Validatin error - query too short or too long"},
                              500:{"description": "Agent Unavailable"}
                          })
async def post_research(request: Request, body: ResearchRequest):
    """The primary endpoint. Create a session and starts the agent.
    Stream mode (defaut):
        Returnt Streaming response with Content-Type text/event-stream.
        Client reads events unitl it recives type= "done"
    Non-stream mode:
        Awaits graph completion and return ResearchResponse JSON.
        useful for testing.

    """

    session_id= str(uuid.uuid4())
    research_graph= request.app.state.research_graph

    logger.info(f"session_id={session_id} "
                f"route= POST/research "
                f"query={body.query} "
                f"stream= {body.stream}")
    
    if body.stream:
        # SSE streaming reponse
        # Streaming Response with media_type text/event-stream is the standard SSE setup .The client EventSource API
        return StreamingResponse(
            steam_research_session(query=body.query,
                                   session_id=session_id,
                                   research_graph=research_graph),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-Id": session_id,
            }
        )
    else:
        # Non streaming resposne: Run the grph to completion, return JSON
        from agent.state import create_itital_state

        initial_state= create_itital_state(query=body.query,
                                           session_id=session_id)
        await session_store.create(session_id=session_id, query=body.query)
        config= {"configurable": {"thread_id":session_id}}

        try:
            await research_graph.ainvoke(initial_state, config=config)
            checkpoint= await research_graph.aget_state(config)
            final_state= checkpoint.values if checkpoint else {}

        except Exception as e:
            logger.error(
                f"session_id= {session_id} "
                f"non_stream_error= {type(e).__name__}: {e}"
            )
            raise HTTPException(status_code=503, detail=str(e))
        
        await session_store.update_status(
            session_id=session_id,
            status=("error" if final_state.get("error") else "complete"),
            final_state=final_state
        )

        return ResearchResponse(
            session_id=session_id,
            query=body.query,
            report= final_state.get("final_response",""),
            followup_questions= final_state.get("followup_questions", []),
            search_result_count=len(final_state.get("search_results",[])),
            reflection_count=len(final_state.get("reflections",[])),
            is_time_sensitive=final_state.get("is_time_sensitive",False),
            created_at= final_state.get("created_at", now_iso()),
            completed_at= now_iso(),
            error=final_state.get("error",""),
        )
    
# post/feedback

@router.post("/feedback",response_model=FeedbackResponse,
             summary="Submmit Feedback for a research session",)
async def post_feedback(body: FeedbackRequest):
    """
    Record a thmbsup (ratting=1) or thumbs down (ratting=-1) for a session.
    The session must exist. the Feedback is stored in session record
    """

    recorded= await session_store.record_feedback(
        session_id=body.session_id,
        ratting= body.rating,
        comment= body.comment
    )

    if not recorded:
        raise HTTPException(
            status_code=404,
            detail=(f"Session {body.session_id} not found Feedback can only submitted for the session that exisit"),
        )
    
    logger.info(f"session_id={ body.session_id} "
                f"feedback_recorded rating= {body.rating}")
    return FeedbackResponse(
        session_id=body.session_id,
        rating=body.rating, 
        record_at=now_iso(),
    )

# GET/sessions/session_id
@router.get("/sessions/{session_id}",
            response_model=SessionStatusResponse,
            summary="Get status of a research session")
async def get_session(session_id:str):
    """Return current status of a session"""
    sessions= await session_store.get(session_id=session_id)

    if not sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    return SessionStatusResponse(
        session_id=sessions.session_id,
        query= sessions.query,
        status=sessions.status,
        created_at=sessions.created_at,
        updated_at=sessions.updated_at,
        is_complete=sessions.status in {"complete","error"},
        error=sessions.final_state.get("error", "") if sessions.final_state else "",
    )

@router.get("/sessions",
           summary="List of recent sessions",
           )
async def list_sessions(limit: int=20):
    """Return the most recent session newest first."""
    sessions= await session_store.list_recent(limit=min(limit,100))
    return {"sessions": sessions, "count": len(sessions)}

@router.get("/health",
            response_model=HealthResponse,
            summary="Health Check")
async def get_health(request: Request):
    """Rerturn the helth checks :
    Graph is compiled and accessible
    Pinecone is reachable
    """

    checks: dict[str, str]={}
    try:
        graph= request.app.state.research_graph

        checks["langgraph"]= "ok" if graph else "not_initialized"
    except Exception as e:
        checks["langgraph"]= f"error:{e}"

    # check pinecone
    try:
        from agent.vectorstore import _get_pinecone_client
        from agent.config import settings
        client= _get_pinecone_client()
        indexes=[i.name for i in client.list_indexes()]
        if settings.pinecone_index_name in indexes:
            checks["pinecone"]="ok"
        else:
            checks["pinecone"]= "index_not_found"

    except Exception as e:
        checks["pinecone"]= f"error: {e}"

    overall= ("ok" if all(v=="ok" for v in checks.values()) else "degraded")

    return HealthResponse(status=overall, checks=checks)