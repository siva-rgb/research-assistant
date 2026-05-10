from __future__ import annotations

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import ResearchAgentState
from agent.nodes import (
    intent_node,
    planner_node,
    executor_node,
    reflection_node,
    synthesis_node,
    error_node,
    memory_update_node,
)

logger= logging.getLogger(__name__)

def route_after_intent(state: ResearchAgentState)->str:
    """After intent extraction: go to planner, or error if extraction failed"""
    if state.get("error"):
        return "error_node"
    if not state.get("intent"):
        return "error_node"
    return "planner_node"

def route_after_planner(state:ResearchAgentState)->str:
    """After Plannin: go to executor or error if planning failed"""
    if state.get("error"):
        return "error_node"
    plan= state.get("plan")

    if not plan or not plan.get("subtasks"):
        return "error_node"
    return "executor_node"

def route_after_executor(state: ResearchAgentState)->str:

    """After each executor step:
    - If there are more subtasks rmaining->  loop back to executor
    - If all subtasks are done -> go to reflection
    - If there's an error-> go to error node
    
    This is the core execution loop routing.
    """

    if state.get("error"):
        return "error_node"
    
    plan= state.get("plan")
    if not plan:
        return "reflection_node"
    current_step= state.get("current_step",0)
    total_steps= plan.get("total_steps",0)

    if current_step< total_steps and state.get("execution_iteration_count", 0) < 5:
        logger.debug(f"[route_after_executor] More Steps: {current_step}/total_steps")

        return "executor_node"
    logger.debug(f"[route_after_executor] All stes done -> reflection")
    return "reflection_node"
    
def route_after_reflection(state: ResearchAgentState)->str:
    """
    After reflection:
    - accept -> Synthesize
    - revise -> back to excutor (to run the additional subtask added by reflection)
    - abort -> error node with explanation
    """

    decision= state.get("reflection_decision","accept")

    if decision =="accept":
        return "synthesis_node"
    elif decision=="revise":
        return "executor_node"
    elif decision =="abort":
        return "error_node"
    else:
        logger.warning(f"[route_after_decession] unexpected decession: {decision}")
        return "synthesis_node"
    
def build_graph()->StateGraph:
    """
    Build and return the compiled reasearch agent graph.

    Graph topology
    START-> intent_node
    intent_node -> [conditional]-> planner_node | error_node
    planner_node -> [conditional] -> executor_node | error_node
    reflection_node -> [conditional]-> synthesis_node | executro_node | error_node
    systhesis_node -> END
    error_node -> END
    """

    graph= StateGraph(ResearchAgentState)

    graph.add_node("intent_node", intent_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("executor_node",executor_node)    
    graph.add_node("reflection_node",reflection_node)
    graph.add_node("synthesis_node",synthesis_node)
    graph.add_node("memory_update_node", memory_update_node)
    graph.add_node("error_node",error_node)

    graph.set_entry_point("intent_node")

    # edges
    # After intent: conditional or error presence
    graph.add_conditional_edges(
        "intent_node",
        route_after_intent,
        {"planner_node":"planner_node",
         "error_node": "error_node",
         },
    )

    # After planner: conditional or prsence
    graph.add_conditional_edges(
        "planner_node",
        route_after_planner,
        {
            "executor_node": "executor_node",
            "error_node": "error_node",
        },
    )

    # After executor: loop or advance - the core execution cycle
    graph.add_conditional_edges(
        "executor_node",
        route_after_executor,
        {
            "executor_node":"executor_node",
            "reflection_node": "reflection_node",
            "error_node": "error_node",
        },
    )

    # After reflection: accept/revise/abort

    graph.add_conditional_edges(
        "reflection_node",
        route_after_reflection,
        {
            "synthesis_node": "synthesis_node",
            "executor_node": "executor_node",
            "error_node": "error_node"
        },

    )
    
    # terminal nodes -> END
    graph.add_edge("synthesis_node", "memory_update_node")
    graph.add_edge("memory_update_node",END)
    graph.add_edge("error_node", END)

    return graph

def build_research_graph():
    checkpoinetr= MemorySaver()
    compiled= build_graph().compile(checkpointer=checkpoinetr)
    logger.info("Research graph compiled with Memopry Saver checkpointer")
    return compiled
# research_graph= build_research_graph()


