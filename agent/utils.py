"""Shared utilites acccross the function """
from __future__ import annotations
import json
import logging
from typing import Optional
from datetime import datetime, timezone, timedelta
logger= logging.getLogger(__name__)


def parse_json_response(raw:str, node_name: str)->dict|None:
    """Parse a json string response
    Models sometimes wrap JSON in markdown fence despite being told not to 
    this handels such cases
    """

    raw= raw.strip()

    if raw.startswith("```"):
        lines= raw.split("\n")
        # remove 1st line (```json or ```) and last line (````)
        raw= "\n".join(lines[1:-1] if lines[-1].strip()== "```"else lines[1:])

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"[{node_name}] jSon parse failed {e}\nRaw ressponse: {raw[:500]}")

        return None
    
def now_iso():
    return datetime.now(timezone.utc).isoformat()

async def refine_query(intent: str,
                       subtask_description: str,
                       planned_query: str,
                       search_results: list[dict])->str:
    """
    Optionally improve a plan query based on accumuated findings.
    Retruns planned_query unchanges if refinement failed
    """

    if not search_results:
        return planned_query
    from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
    from agent.prompts import QUERY_REFINEMENT_PROMPT, format_findings_summary
    from agent.llm import invoke_llm

    try:
        message= [SystemMessage(content="You are a precise assistant"), HumanMessage(content=QUERY_REFINEMENT_PROMPT.format(
            intent= intent,
            subtask_description= subtask_description,
            planned_query= planned_query,
            findings_summary= format_findings_summary(search_result=search_results)
        ))]
        response= await invoke_llm(messages=message,
                                   temprature=0.4,
                                   node_name="intent_node")



        if not response.content:
            return planned_query
        logger.info(f"llm refinement response: {response.content}")
        parsed= parse_json_response(response.content, "refine_query")
        if parsed and "query" in parsed:
            refined=parsed["query"].strip()
            if refined and refined != planned_query:
                logger.debug(
                    f"query_refined original={planned_query}"
                    f"refined= {refined}"
                )
            return  refined or planned_query     
    except Exception as e:
        logger.warning(f"[refine_query] Failed, using planned query: {e}")
    return planned_query

