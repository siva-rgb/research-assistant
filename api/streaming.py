"""SSE Streaming logic
This module connects between langgraph astream_events and the sse events types the client consumes.

Design
    one function: steam_research_event_session() called by routes.
    It yield formatted SSE strings never touch fastapi directly
    Every agent maps to exactly one SSE event types.
    Error at any point yield as an ErrorEvent then a DoneEvennt
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import AsyncIterator
from agent.state import ResearchAgentState, create_itital_state
from api.schema import (NodeStartEvent, NodeCompletEvent, PlanReadyEvent, 
                        SearchResultEvent, ReflectionEvent, TokenEvent,
                        ResponseCompleteEvent, RunMetricsEvent, ErrorEvent, DoneEvent)
from api.session import session_store

logger= logging.getLogger(__name__)

_VISIBLE_NODES= {"intent_node", "planner_node", "executor_node", "reflection_node", "synthesis_node", "followup_node", "memory_update_node"}

_NODES_LABELS= {
    "intent_node": "Understand the intent of user qury",
    "planner_node": "Creating research plan",
    "executor_node": "Gathering information",
    "reflection_node": "Evaluate findings",
    "synthesis_node": "Writing Report",
    "followup_node": "Generating folloup questions",
    "memory_update_node": "Saving to knowledge base",
}

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def _format_sse(event_data: dict)->str:
    """
    format a dict as an SSE message string.
    SSE format:
    data: </json>\n\n
    The double new line terminate the event. The client Eventsource fires one message event
    per double-newline terminate block
    """

    return f"data: {json.dumps(event_data)}\n\n"

def compute_cost(node_metrics: list[dict])-> tuple[int, float]:
    """Return total token and estimate_cost_usd"""

    total_prompts= sum(m.get("prompt_tokens",0) for m in node_metrics)
    total_compltetion= sum(m.get("completion_tokens",0) for m in node_metrics)
    total_tokens= total_compltetion+total_prompts
    cost= (total_prompts/1_000_000*0.25)+ (total_compltetion/1_000_000*10.00)

    return total_tokens, round(cost,6)

async def steam_research_session(
        query: str,
        session_id: str,
        research_graph)-> AsyncIterator[str]:
    
    """Run the Research agent and yield SSE-formatted strings.
    
    Called by resaech route. Uses astream_event to receive fine-grained events from Langgraph
    map them to SSE events, Yield firmatted strings that fastAPI's StreamingResponse send to the client"""

    run_start= time.monotonic()

    initial_state= create_itital_state(query=query, session_id=session_id)
    await session_store.create(session_id=session_id, query=query)

    config= {"configurable": {"thread_id":session_id}}

    node_start_time: dict[str, float]= {}

    current_state: dict={}
    logger.info(f"session_id={session_id} "
                f"stream_start "
                f"query= {query}")
    
    try:
        async for event in research_graph.astream_events(initial_state, config= config,version="v2"):
            event_kind= event.get("event","")
            event_name= event.get("name","")
            event_data= event.get("data",{})
            event_meta= event.get("metadata",{})

            if (event_kind== "on_chain_start" and event_name in _VISIBLE_NODES):
                node_start_time[event_name]=  time.monotonic()
                yield _format_sse(NodeStartEvent(
                    session_id=session_id,
                    node_name= event_name,
                    timestamp=now_iso()
                ).model_dump())
            elif (event_kind== "on_chain_end" and event_name in _VISIBLE_NODES):
                duration_ms= (time.monotonic() - node_start_time.get(event_name, run_start)) * 1000

                output=event_data.get("output",{})
                if isinstance(output, dict):
                    current_state.update(output)

                yield _format_sse(NodeCompletEvent(
                    session_id=session_id,
                    node_name= event_name,
                    duration_ms=round(duration_ms,1),
                    timestamp=now_iso()
                ).model_dump())

                # plan ready after plan node
                if event_name== "planner_node" and isinstance(output, dict):
                    plan= output.get("plan",{})
                    if plan and plan.get("subtasks"):
                        yield _format_sse(PlanReadyEvent(
                            session_id= session_id,
                            total_steps=plan["total_steps"],
                            sub_tasks=[{
                                "step": t["step_index"],
                                "type": t["task_type"],
                                "description": t["description"],
                            }for t in plan["subtasks"]
                            ],
                            timestamp=now_iso()
                        ).model_dump())
                
                # Search resul;t after each executor_node pass
                elif event_name== "executor_node" and isinstance(output, dict):
                    new_results= output.get("search_results", [])
                    if new_results:
                        source= {r.get("source", "web") for r in new_results}
                        if source=={"vectorstore"}:
                            strategy= "vectorstore_only"
                        elif source== {"web"}:
                            strategy= "web_fallback"
                        else:
                            strategy= "merged"
                        
                        total_so_far= len(current_state.get("search_results",[]))

                        yield _format_sse(SearchResultEvent(
                            session_id=session_id,
                            step= output.get("current_step",0),
                            total_steps= current_state.get("plan",{}).get("total_steps",0),
                            result_added=len(new_results),
                            total_results= total_so_far,
                            strategy_used=strategy,
                            timestamp= now_iso()
                        ).model_dump())

                elif event_name== "reflection_node" and isinstance(output, dict):
                    reflections= output.get("reflections",[])
                    if reflections:
                        latest= reflections[-1]
                        yield _format_sse(ReflectionEvent(
                            session_id=session_id,
                            decision=latest.get("decision","accept"),
                            critique= latest.get("critique","")[:200],
                            timestamp=now_iso(),
                        ).model_dump())
                
                # Response complete after synthesis node
                elif event_name== "synthesis_node" and isinstance(output, dict):
                    report= output.get("final_response","")

                    if report:
                        yield _format_sse(ResponseCompleteEvent(
                            session_id=session_id,
                            report=report,
                            followup_questions=[],
                            timestamp=now_iso()
                        ).model_dump())

                # Followup question - after followup_node
                elif (event_name=="followup_node" and isinstance(output,dict) and output.get("followup_questions")):
                    yield _format_sse({
                        "type": "followup_questions",
                        "session_id": session_id,
                        "questions": output["followup_questions"],
                        "timestamp": now_iso(),
                    })
            elif event_kind== "on_chat_model_stream":
                # check if we are inside synthesis by inspecting run heirarchy
                tags= event.get("tags",[])
                run_id= event.get("run_id","")

                in_synthesis= ("synthesis_node" in str(tags) or "synthesis" in str(event_meta.get("langgraph_node",""))
                               )
                if in_synthesis:
                    chunk= event_data.get("chunk")
                    if chunk and hasattr(chunk,"content") and chunk.content:
                        yield _format_sse(TokenEvent(
                            session_id=session_id,
                            token=chunk.content,
                            timestamp=now_iso()
                        ).model_dump())
            # retrive final state from checkpointer for metrics
        try:
            checkpoint= await research_graph.aget_state(config)
            final_state= checkpoint.values if checkpoint else current_state

        except Exception as e:
            final_state= current_state

            # update session record
        is_complete= final_state.get("is_complete",False)
        has_error= bool(final_state.get("error",""))
        status= "error" if has_error else "complete" if is_complete else "running"

        await session_store.update_status(session_id=session_id,
                                              status=status,
                                              final_state=final_state)
        node_metrics= final_state.get("node_metrics",[])
        total_token, cost= compute_cost(node_metrics=node_metrics)
        total_ms= (time.monotonic()- run_start)*1000

        yield _format_sse(RunMetricsEvent(
                session_id=session_id,
                total_tokens=total_token,
                estimated_cost_usd=cost,
                total_duration_ms=round(total_ms,1),
                search_result_count= len(final_state.get("search_results",[])),
                reflection_count= len(final_state.get("reflections",[])),
                timestamp= now_iso(),
            ).model_dump())

        logger.info(f"session_di= {session_id} "
                        f"stram_complete "
                        f"status={status} "
                        f"tokens={total_token} "
                        f"cost_usd= {cost} "
                        f"duration_ms= {total_ms:.0f}"
                    )
    except Exception as e:
        logger.error(
            f"session_id= {session_id} "
            f"stream_error "
            f"error= {type(e).__name__} "
            f"message={str(e)} "
        )

        await session_store.update_status(
            session_id=session_id,
            status="error"
        )
        yield _format_sse(ErrorEvent(session_id=session_id,
                                     message= f"Agent error: {type(e).__name__}: {str(e)}",
                                     timestamp=now_iso()).model_dump()
                                     )
        
    finally:
        yield _format_sse(DoneEvent(
            session_id=session_id,
            timestamp= now_iso(),
        ).model_dump())
                        


