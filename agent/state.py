"""
State Schema for research agent
"""
from __future__ import annotations
from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator
from datetime import datetime, timezone

class ResearchPlan(TypedDict):
    """The class keep the structured plan created by the planner node. A list of subtask, each with a type and a description.
    The executor works through subtask in order
    """
    subtasks: list[SubTask]
    total_steps: int

class SubTask(TypedDict):
    """
    A single unit of work within the research plan.
    Filed:
    step_index: Position on the plan (0-bases)
    task_type: 'search' | 'read' | 'synthesize' | 'verify'
    description: Natural language description of waht to do
    query: The concret queryt or url to use (filled during the execution)
    status: 'pending' | 'complete' | 'failed'
    """
    step_index: int
    task_type: str
    description: str
    query: str
    status: str

class SearchReasult(TypedDict):
    """Single Result from a tool call.
    kept flat and serializable - no nested object"""
    query: str
    title: str
    url: str
    content: str
    score: float
    source: str #'web' | 'vectorstore'

class ReflectionNote(TypedDict):
    """ A single reflection pass- the model's self evaluation of current findings. Multiple reflection loop
    """
    iteration: int
    critique: str # waht is missing or waek
    decision: str # 'accept' | 'revise' | 'abort'
    additional_queries: list[str] # suggested queries if decision== revise

class NodeMetrics(TypedDict):
    node_name: str
    prompt_tokens:int
    completion_tokens: int
    duration_ms: float


class ResearchAgentState(TypedDict):
    """Complete working memory of the Research Agent
    Lifecycle of the fields
    Initialize at graph start:
    - query, session_id, messages
    writtent by intent note
    -intent, research_scope 
    written by the planner node:
    - plan
    written by executor_node (each iteration):
    - search_results, document_read, current_step, execution_iteratin_coutn
    written by reflection_node:
    - reflections, reflection_decisions, reflection iteration_count
    writtent by synthesis_node:
    - final_response
    writtent by any node on error:
    - error, is_complete (set to True on abort)
    written at end:
    - is_complete 
    """

    query: str # the raw question submitted by the user
    session_id: str # unique id for session resarch
    """the conversational turn history. uses append reducers- each node that produce a message append it.
    we keep this spearate from findings so conversationa; turns and researchs don't get mixed together,
    """
    messages: Annotated[list[BaseMessage], operator.add] 
    intent:str # A structured one-paragraph restatement of the research goal as understood by teh intent extraction node

    created_at:str
    updated_at:str

    """Narrorw | Modrate | Broad the complexity classification used by the pallner to decide how many subtask to generate and by the router to select which planner variant to use"""
    research_scope: str 

    """The structured execution plan a list of su"task
    written by planner, read by Executor loop.
    """
    plan: ResearchPlan 

    current_step:int # index of current plan being executed, incriment by the executor by each successful step.

    search_results: Annotated[list[SearchReasult], operator.add]
    
    """Full document content fetched by document reader tool.
    Apen reducer - document accumulate accross step
    Important document content is trucated
    """
    documents_read: Annotated[list[dict[str,str]], operator.add]

    execution_iteration_count: int # how many time executor has run

    # Reflection
    """All reflection passes - each refelection cycle append a new note
    Append reducer- keep full reflection history.
    """
    reflections: Annotated[list[ReflectionNote], operator.add]
    reflection_decision: str # most resent relect decision 'accept' | 'revise' | 'abort'
    reflection_iteration_count: int
    
    """The final synthesize research report ready to diliver to the user"""
    final_response: str
    node_metrics: Annotated[list[NodeMetrics],operator.add]

    """set to true when the agent reaches a terminal state 
    successful synthesis, reflection abort, or unrecoverable error"""
    is_complete: bool
    is_time_sensitive: bool
    error: str

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def create_itital_state(query:str, session_id:str)-> ResearchAgentState:
    """
    Returnns a fully initialized state dict for a new research session.
    Every field must be present, Fields with accumulating reducers start as empty list.
    Counter stat at 0, booleans start at False.

    This function is a factory function, Typeddict does not support default values,
    This fuction is the single source of truth
    """
    now= _now_iso()

    return ResearchAgentState(
        query=query,
        session_id=session_id,
        messages=[],
        created_at=now,
        updated_at=now,
        intent="",
        research_scope="",
        plan= ResearchPlan(subtasks=[], total_steps=0),
        current_step=0,
        search_results= [],
        documents_read=[],
        execution_iteration_count=0,
        reflections=[],
        reflection_decision="",
        reflection_iteration_count=0,
        final_response="",
        node_metrics=[],
        is_complete=False,
        is_time_sensitive=False,
        error="",
    )
    









