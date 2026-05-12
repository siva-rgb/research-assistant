"""
Langgraph node functions for research Agent.

Each node:
- Receives the full ResearchAgentState return partial dic of updates only.
- uses invoke_llm() from agent.llm - never imposrts ChatOpenAI directly
- Writes Nodemetrics to track token usage and timing.
- Never raises- error are written to state.error
- Emits structured log fields (session_id=, node=, etc.) on every log call
- 
Node execution order (set by graph edge):
intent_node -> planner_node -> executor_node -> reflection_node

"""

from __future__ import annotations
import json
import logging
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import asyncio
from agent.config import settings
from agent.state import (ResearchAgentState,
                         ResearchPlan,
                         SubTask,
                         ReflectionNote,
                         NodeMetrics,
                         SearchReasult)
from agent.prompts import (SYSTEM_PROMPT, 
                           INTENT_EXTRACTION_PROMPT,
                           PLLANER_PROMPT,
                           QUERY_REFINEMENT_PROMPT,
                           REFLECTION_PROMPT,
                           SYNTHESIS_PROMPT,
                           FOLLOWUP_PROMPT,
                           format_findings_for_prompt,
                           format_findings_summary)
from agent.utils import refine_query, now_iso, parse_json_response
from agent.tools import get_tool, get_tool_decription_for_promt, _search_with_strategy
from agent.llm import invoke_llm
import time
logger= logging.getLogger(__name__)

_llm= ChatOpenAI(api_key=settings.model_api_key,
                 base_url=settings.openai_base_url,
                 model=settings.openai_model,
                 temperature=0.0,
                 )
_synthesis_llm= ChatOpenAI(api_key=settings.model_api_key,
                 base_url=settings.openai_base_url,
                 model=settings.openai_model,
                 temperature=0.0,
                 )
def make_metrics(node_name: str,
                 prompt_tokens: int,
                 completion_tokens: int,
                 duration_ms:float)-> NodeMetrics:
    return NodeMetrics(
        node_name=node_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_ms=duration_ms
    )
def _sid(state: ResearchAgentState)->str:
    """Convinence: extract session_id for log lines."""
    return state.get("session_id", "unknown")


# intent Node 
async def intent_node(state: ResearchAgentState)-> dict:
    """Extract structured intent from user's raw query
    reads: intnet.query
    writes: state.intent, state.research_scope, state.messages, state.error,
            state.node_metrics, state.updated_at
    """
    sid= _sid(state)
    logger.info(f"sesion_id={sid} node_name= intent_node query={state['query']}")

    message= [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=INTENT_EXTRACTION_PROMPT.format(query=state['query'])) 
    ]

    try:
        response= await invoke_llm(messages=message,
                                   temprature=0.0,
                                   node_name="intent_node")
        
        metrics= make_metrics(
            "intent_node",
            response.prompt_token,
            response.completion_toknes,
            response.duration_ms,
        )

        if not response.content:
            return{
                "error": "Intent Node: llm call failed - empty response.",
                "is_complete": True,
                "node_metrics":[metrics],
                "updated_at": now_iso(),
            }
            
        raw= response.content
        logger.info(f"session_id= {sid} node= intent_node\n"
                    f"llm_reponse= {raw}")
        parsed= parse_json_response(raw, "intent_node")

        if parsed is None:
            return {
                "error": "Intnent Extraction failed: Model return invalid JSON",
                "is_complete": True,
                "node_metrics":[metrics],
                "updated_at": now_iso()
            }
        if "intent" not in parsed or "research_scope" not in parsed:
            return{
                "error": f"Intent extraction returend incomplete data: {list(parsed.keys())}",
                "is_complete":True,
                "node_metrics": [metrics],
                "updated_at": now_iso(),
            }
        valid_scope= {"Narrow", "Moderate", "Broad"}

        scope= parsed.get("research_scope","Moderate")

        if scope not in valid_scope:
            logger.warning(f"session_id= {state['session_id']} node= intent_node"
                           f"invalid_scope={scope} defaulting= Moderate")
            scope= "Moderate"

        logger.info(f"session_id= {state['session_id']} node= intent_node"
                    f"scope= {scope}"
                    f"prompt_token= {response.prompt_token}"
                    f"completion_token= {response.completion_toknes}")
        

        return {
            "intent": parsed["intent"],
            "research_scope": scope,
            "messages":[HumanMessage(content=state["query"]),
                        AIMessage(content=f"Research scope: {scope}. Starting Research..."),
                        ],
            "node_metrics":[metrics],
            "updated_at": now_iso(),
            "is_time_sensistive": parsed["is_time_sensitive"]
        }
    except Exception as e:
        logger.error(f"[intent_node] Unexpected error: {e}")
        metrics= make_metrics("intent_node", 0, 0, 0.0)
        return {
            "error": f"Intent node failed: {type(e).__name__}: {str(e)}",
            "is_complete": True,
            "node_metrics":[metrics],
            "updated_at": now_iso()
        }
    
# planner node
async def planner_node(state: ResearchAgentState)->dict:
    """Generate a structured plan
    read: state.intent, state.reseasrch_scope
    writes: state.plan, state.node_matrics, state.updated_at, state.error
    """
    logger.info(f"[planner_node] Planning for scope {state['research_scope']}")

    sid= _sid(state)
    logger.info(f"session_id= {sid} node= planner_node"
                f"scope= {state['research_scope']}")
    
    messages= [SystemMessage(content=SYSTEM_PROMPT),
               HumanMessage(content= PLLANER_PROMPT.format(intent=state['intent'],
                                                           research_scope=state['research_scope'],
                                                           key_concepts="(Extracted from Query)",
                                                           tool_descriptions= get_tool_decription_for_promt()
                                                           )
                                                           )
               ]
    try:
        response= await invoke_llm(messages=messages,
                                   temprature="0.0",
                                   node_name="planner_node")
        
        metrics= make_metrics(
            "planner_node",
            response.prompt_token,
            response.completion_toknes,
            response.duration_ms
        )
        if not response.content:
            return{
                "error": "Planner node: Model return invalid JSON.",
                "is_complete": True,
                "node_metrics": [metrics],
                "updated_at": now_iso(),
            }

        raw= response.content
        parse= parse_json_response(raw, "planner_node")

        if parse is None:
            return {
                "error": "Planning eturned no subtasks",
                "is_complete":True,
                "node_metrics": [metrics],
                "updated_at": now_iso()
            }
        if "subtasks" not in parse or not parse["subtasks"]:
            return {
                "error": f"Intent extraction return incopmplete data {list(parse.keys())}",
                "is_complete":True, 
                "node_metrics": [metrics],
                "updated_at": now_iso(),
            }
        
        # build typed Subtask objects, validating each one
        subtasks: list[SubTask]=[]

        for i, raw_task in enumerate(parse["subtasks"]):
            subtasks.append(
                SubTask(step_index=i,
                        task_type=raw_task.get("task_type","search"),
                        description=raw_task.get("description",f"Step {i}"),
                        query= raw_task.get("query",""),
                        status="pending"
                )
            )
        plan= ResearchPlan(subtasks=subtasks,total_steps=len(subtasks))
        logger.info(f"session_id= {sid} node= planner_node"
                    f"total_step= {plan['total_steps']}"
                    f"prompt_status={response.prompt_token}")

        for t in subtasks:
            logger.debug(f"session_id= {sid} step={t['step_index']} "
                         f"type= {t['task_type']} query= {t['query']}")
        
        return {"plan":plan,
                "node_metrics":[metrics],
                "updated_at": now_iso()}
    except Exception as e:
        logger.error(f"[planner_node] Unexpected Error: {type(e).__name__}: {str(e)}")
        metrics= make_metrics("planner_node", 0, 0, 0.0)
        return {
            "error": f"Planner Node failed: {type(e).__name__}: {str(e)}",
            "is_complete": True,
            "node_metrics":[metrics],
            "updated_at":now_iso(),
        }
    
# executor node

async def executor_node(state: ResearchAgentState)->dict:
    """
    Execute a single subtask from the plan.

    Each call to this node execute excatly one subtask at a time, graphs loop back to this node until all the subtask are complete

    reads: state.plan, state.current_step, state.execution_iteration_count, state.intent, state.search_results(for every refinement context)
    writes: state.search_results or state.documents_read (depending on the tool), state.current_step, state.execution_iteration_count, state.error
    """
    sid= _sid(state)
    step_idx= state["current_step"]
    iteration= state['execution_iteration_count']
    is_time_sensitive= state.get("is_time_sensitive",False)

    # sefty check for iteration limit
    if iteration >= settings.max_search_iteration:
        logger.warning(f"session_id= {sid} node= executor_node"
                       f"iteration_limit_reached= {iteration}")
        return {
            "execution_iteration_count": iteration+1,
            "updated_at": now_iso()
        }
    #sefty check for: no plan or endo of plan
    plan= state.get("plan")
    if not plan or step_idx >= len(plan["subtasks"]):
        logger.info(f"session_id= {sid} node= executor_node"
                    f"no_more_subtask step= {step_idx}"
                )
        
        return {"execution_iteration_count": iteration+1,
                "updated_at": now_iso()}
    
    
    subtask= plan["subtasks"][step_idx]

    logger.info(f"session_id= {sid} node= executor_node"
                f"step= {step_idx}/{plan['total_steps']}"
                f"type= {subtask['task_type']} - {subtask['description']}"
                f"query= {subtask['query']}"
                )
    # optionaly refine the query based on findin so far
    # only refine if it has prior findings (not 1st step)
    # and the subtask is a search (not a read_document)
    start= time.monotonic()
    updates: dict[str, Any]={
        "current_step": step_idx+1,
        "execution_iteration_count": iteration +1,
        "updated_at": now_iso(),
    }

    if subtask["task_type"]in ["search_web","search"]:
        query= await refine_query(
            intent= state["intent"],
            subtask_description= subtask["description"],
            planned_query= subtask["query"],
            search_results=state["search_results"]
        )

        results= await _search_with_strategy(query=query,session_id= sid, is_time_sensitive=is_time_sensitive)
        # web_fn= get_tool("search_web")
        # vs_fn= get_tool("search_vectorstore")

        # web_result, vs_result= await asyncio.gather(web_fn(query), vs_fn(query))

        # merge result duplicarte by URL preserve source tag

        # seen_urls: set[str]= set()
        # merged: list[SearchReasult]= []

        # for result_set in [vs_result, web_result]:
        #     if result_set["success"]:
        #         for r in result_set["data"]:
        #             url= r.get("url","")
        #             if url and url not in seen_urls:
        #                 seen_urls.add(url)
        #                 merged.append(r)
        # merged.sort(key=lambda r: r["score"], reverse=True)
        # merged= merged[:8]

        if results:
            updates["search_results"]= results

            logger.info(f"session_id= {sid} node= executor_node "
                        f"step={step_idx} "
                        f"search_result_added= {len(results)} "
                        f"cummulative= {len(state['search_results']) + len(results)} "
                        f"duration_ms={(time.monotonic()-start)*1000:.0f} ")
        else:
            logger.warning(f"session_id= {sid} node= executor_node "
                           f"step={step_idx} no result from eighter source")
            
    elif subtask["task_type"]== "read_document":
        read_fn= await read_fn(subtask["read_document"])
        result= await read_fn(subtask["query"])

        if result["success"] and result["data"]:
            updates["documents_read"]= [{
                "url": subtask["query"],
                "title": subtask["description"],
                "content": result["data"],
            }]

            logger.info(f"session_id= {sid} node= executor_node "
                        f"step= {step_idx} "
                        f"doc_chars= {len(result['data'])} "
                        f"duration_ms= {(time.monotonic()-start) * 1000:.0f}")
        else:
            logger.warning(f"session_id= {sid} node=executor_node "
                           f"step= {step_idx}"
                           f"read_document_failed_error= {result['error']}")
    else:
        logger.warning(f"session_id={sid} node= executor_node "
                       f"unknown_task_type= {subtask['task_type']}")
        
    updates["node_metrics"]= [make_metrics("executor_node",0,0, (time.monotonic()-start)*1000
                                           )]
    return updates


async def reflection_node(state: ResearchAgentState)->dict:
    """
    Evaluate the quality and completeness of current findings.
    Decides whether to accept,m revise or abort

    Reads: state.intent, state.plan, state.search)result, state.documents_read, state.reflection_iteration_count,

    Writes: state.reflections, state.reflection_decision, state.reflection_iteration_count, state.plan
    """

    sid= _sid(state)

    iteration= state["reflection_iteration_count"]
    logger.info(f"session_id={sid} node= relection_node"
                f"iteration= {iteration+1}"
                f"finding_count={len(state['search_results'])}")

    # Enfirce iteration reflection limit
    if iteration >= settings.max_reflection_teration:
        logger.warning(f"session_id={sid} node= reflection_node"
                       f"max_iteration_reached= {iteration} forcing= accept")

        note= ReflectionNote(iteration= iteration+1,
                             critique="Maximum reflection iteration reached -accepting current findings.",
                             decision="accept",
                             additional_queries=[])
        return {
            "reflections":[note],
            "reflection_decision":"accept",
            "reflection_iteration_count": iteration+1,
            "node_metrics": [make_metrics("reflection_node", 0, 0, 0.0)],
            "updated_at": now_iso(),
        }
    
    
    # build finding summary for the prompt
    finding_summary= format_findings_summary(state["search_results"])
    steps_completed= state["current_step"]
    total_steps= state["plan"]["total_steps"] if state.get("plan") else 0

    messages=[
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=REFLECTION_PROMPT.format(
            intent=state["intent"],
            total_steps= total_steps,
            steps_completed=steps_completed,
            findings_summary= finding_summary,
        ))
    ]

    try:
        response= await invoke_llm(messages=messages,
                                   temprature=0.0,
                                   node_name="reflection_node")
        metrics= make_metrics("reflection_node",
                              response.prompt_token,
                              response.completion_toknes,
                              response.duration_ms)
        if not response.content:
            logger.warning(f"session_id= {sid} node= refelction_node"
                           f"llm_cal= llm call failed to give response focing=accept")
            note= ReflectionNote(
                iteration=iteration+1,
                critique="LLM call failed - defaulting to accept.",
                decision= "accept",
                additional_queries=[],
            )
            return{
                "reflections":[note],
                "reflection_decision": "accept",
                "reflection_iteration_count": iteration+1,
                "node_metrics": [metrics],
                "updated_at": now_iso()
            }
        parsed= parse_json_response(response.content, "reflection_node")

        if parsed is None:
            logger.warning(f"session_id={sid} node= reflection_node"
                           f"parsed_reult= parse failed forcing= accept")

            note= ReflectionNote(
                iteration=iteration+1,
                critique= "Reflection parse failed",
                decision="accept",
                additional_queries=[]
            )
            return{
                "reflections":[note],
                "reflection_decision":"accept",
                "reflection_iteration_count": iteration+1,
                "node_metrics": [metrics],
                "updated_at": now_iso()
            }
        
        decision= parsed.get("decision", "accept")

        if decision not in {"accept", "revise", "abort"}:
            logger.warning(f"session_id= {sid} node= reflection_node"
                           f"invalid_decession={decision} defaulting= accept")
            
            decision="accept"

        additional_queries= parsed.get("additonal_query", [])

        note= ReflectionNote(
            iteration= iteration+1,
            critique= parsed.get("critique"),
            decision=decision,
            additional_queries=additional_queries,
        )

        logger.info(f"session_id={sid} node= refelction_node"
                    f"decision= {decision}"
                    f"confidence= {parsed.get('confidence_score','?')} "
                    f"additional_queries= {len(additional_queries)} "
                    f"prompt_tokens= {response.prompt_token}")

        updates: dict[str, Any]= {
            "reflections": [note],
            "reflection_decision": decision,
            "reflection_iteration_count": iteration+1,
            "node_metrics": [metrics],
            "updated_at": now_iso(),
        }

        if decision =="revise" and additional_queries:
            current_plan= state.get("plan",ResearchPlan(subtasks=[], total_steps=0))
            current_subtask= list(current_plan["subtasks"])
            next_index= len(current_subtask)

            for q in additional_queries[:2]:
                current_subtask.append(
                    SubTask(step_index=next_index,
                            task_type="search",
                            description=f"Additional Search: {q}",
                            query=q,
                            status="pending",
                            )
                )
                next_index+=1

            updates["plan"]=ResearchPlan(
                subtasks=current_subtask,
                total_steps=len(current_subtask)
            )

            logger.info(f"session_id= {sid} node= reflection_node "
                        f"added_subtask= {min(2, len(additional_queries))} "
                        f"new_total= {len(current_subtask)}"
                        )

        return updates
    except Exception as e:
        logger.error(f"session_id= {state['session_id']} node= reflection_node"
                     f"error= {type(e).__name__} msg= {str(e)}")
        

        note= ReflectionNote(
            iteration= iteration+1,
            critique=f"Reflection Error: {str(e)}",
            decision="accept",
            additional_queries=[],

        )
        metrics= make_metrics("reflection_node", 0, 0, 0.0)

        return{
            "reflections": [note],
            "reflection_decision": "accept",
            "reflection_iteration_count": iteration+1,
            "node_metrics": [metrics],
            "updated_at": now_iso()
        }
    

async def synthesis_node(state:ResearchAgentState)->dict:
    """
    Synthesize all collected findings into a final research report.
    Reads: state.intent, state.search_results, state.documents_read
    Writes: state.final_response, state.is_complete, stae.messages
    """
    sid= _sid(state)
    logger.info(f"session_id={sid} node= synthesis_node"
                f"search_results= {len(state['search_results'])}"
                f"document_read= {len(state['documents_read'])}")

    formatted_findings= format_findings_for_prompt(search_results=state['search_results'],
                                                   document_read=state['documents_read'])
    
    messages= [SystemMessage(content=SYSTEM_PROMPT),
               HumanMessage(content=SYNTHESIS_PROMPT.format(
                   intent= state['intent'],
                   formatted_findings= formatted_findings,)
               ),
    ]

    try:
        response= await invoke_llm(messages=messages,
                                  temprature=0.3,
                                  node_name="synthesis_node")
        metrics= make_metrics("synthesis_node",
                              response.prompt_token,
                              response.completion_toknes,
                              response.duration_ms)
        
        if not response.content:
            return{
                "error": "Synthesis node- LLM call failed - empty response.",
                "is_complete": True,
                "node_metrics":[metrics],
                "updated_at": now_iso(),
            }
        report= response.content.strip()

        logger.info(f"session_id= {sid} node= synthesis_node"
                    f"report_chars= {len(report)}"
                    f"prompt_tokens={response.prompt_token}"
                    f"completion_token= {response.completion_toknes}")

        return{
            "final_response": report,
            "is_complete":True,
            "messages": [AIMessage(content=report)],
            "node_metrics":[metrics],
            "updated_at": now_iso(),
        }
    except Exception as e:
        logger.error(f"session_id={sid} node= synthesis_node"
                     f"error={type(e).__name__} msg={str(e)}")
        metrics= make_metrics("synthesis_node", 0,0,0.0)
        return{
            "error": f"Synthesis failed: {type(e).__name__}: {str(e)}",
            "is_complete": True,
            "node_metrics":[metrics],
            "updated_at": now_iso(),
        }

async def followup_node(state: ResearchAgentState):
    """After the research agent generate the response create some followup question for the user to answer
      try to write it from scratch"""
    
    sid= _sid(state=state)
    query= state.get("query","")
    finding_summary= state.get("final_response","")
    finding_summary= finding_summary[:500] if len(finding_summary)>500 else finding_summary

    # form message
    messages= [SystemMessage(content="You are a helpful research assistance"),
               HumanMessage(content=FOLLOWUP_PROMPT.format(query=query,
                                                           report_summary=finding_summary))]
    
    #call llm
    try:

        response= await invoke_llm(messages=messages, temprature=0.7, node_name="followup_node")
        parsed= parse_json_response(response.content,"followup_node")

        questions= parsed.get("questions",[])
        questions= [q for q in questions if isinstance(q, str)][:3]
    except Exception as e:
        logger.warning(f"session_id= {sid} followup_node failed :{e}")
        questions=[]

    # logger.info(f"session_id= {sid} node= followup_node failed:{e}")
    return{
        "followup_questions": questions,
        "updated_at": now_iso()
    }



async def error_node(state: ResearchAgentState)-> dict:
    """
    Terminal node fir unrecoverable errors.
    Produce a user-facing error message and marks the task complete.

    Read: state.error
    Write: state.final_response, state.is_complete
    """

    sid= _sid(state)


    error_msg= state.get("error","An unknown error ocoured.")

    logger.error(f"session_id= {sid} node= error_node "
                 f"terminal_error= {error_msg}")
    


    return{
        "final_response":(
            f"I was unable to complete your research request.\n\n"
            f"Reason: {error_msg}\n\n"
            f"Please try rephreasing your query or try again later."
        ),
        "is_complete":True,
        "updated_at": now_iso()
    }

async def memory_update_node(state: ResearchAgentState)->dict:
    """
    Store research findings in the vector store after successfull synthesis.
    This make Future research sessio faster- the agent find relevant past findings via search vector store 
    before hitting the the web.

    Reads: state.rearch_result, state.session_if, state.query
    Writes: nothing to state (write directly tov ector store)
    """

    from agent.vectorstore import store_research_findings

    sid= _sid(state)
    logger.info(f"session_id= {sid} node= memeory_update_node"
                f"findings_tostore= {len(state['search_results'])}")
    start= time.monotonic()

    try:
        stored= await store_research_findings(findings=state["search_results"],
                                      session_id=state["session_id"],
                                      query=state["query"],
                                      )
        duration_ms= (time.monotonic()- start)*1000

        logger.info(f"session_id= {sid} node= memory_node"
                    f"stored= {stored} duration_ms= {duration_ms:.0f}")
        

    except Exception as e:
        duration_ms= (time.monotonic()-start)*1000
        logger.error(f"session_id= {sid} node= memory_update_node "
                     f"error= {type(e).__name__}"
                     f"message= {str(e)} duration_ms= {duration_ms}")

    return {"node_metrics": [make_metrics("memory_update_node", 
                                          0, 0, 
                                          (time.monotonic()- start)*1000)],
            "updated_at": now_iso()
                
            }

