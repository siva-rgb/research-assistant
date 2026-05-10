"""State state for schema correctness
Focus: Reducer behaviour
"""

import operator
import pytest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from agent.state import(
    ResearchAgentState,
    ResearchPlan,
    SubTask,
    SearchReasult,
    ReflectionNote,
    create_itital_state
)

def test_ititial_state_has_all_fields():
    """Every field must be present in initial state"""
    state= create_itital_state(query="test query", session_id="session-001")

    assert state['query']== "test query"
    assert state['session_id']== "session-001"
    assert state['messages']== []
    assert state['search_reasult']== []
    assert state['documents_read']== []
    assert state['reflection']== []
    assert state['current_step']== 0 
    assert state['execution_iteration_count'] == 0
    assert state['reflection_iteration_count'] == 0
    assert state['is_complete'] is False
    assert state['error'] == ""
    assert state['final_response'] == ""

def test_search_results_use_append_reducer():
    """Verify that the search_reasult accumulate correctly"""

    state= create_itital_state("test", 's1')
    first_batch: list[SearchReasult]= [
        SearchReasult(qury="q1", title="T1", url="u1", content="c1", score=0.9),
        SearchReasult(qury="q1", title="T2", url="u2", content="c2", score=0.8)
    ]
    state['search_reasult']= operator.add(state['search_reasult'], first_batch)
    assert len(state['search_reasult'])==2

    second_batch: list[SearchReasult]=[
        SearchReasult(qury="q2", title="T3", url="u3", content="c3", score=0.9)
    ]

    state['search_reasult']= operator.add(state['search_reasult'], second_batch)

    assert len(state['search_reasult'])==3
    assert state['search_reasult'][0]['title']== "T1"
    assert state['search_reasult'][2]['title']== "T3"

def test_result_acumulate():
    """Reflection note must build up accross cycles, not overwrite."""
    state= create_itital_state("test", "s1")

    note1= ReflectionNote(iteration=1,
                          critique="Missing Recent data",
                          decession="revise",
                          additional_queries=["latest 2024 data"])
    state['reflection']= operator.add(state["reflection"],[note1])

    note2= ReflectionNote(iteration=2,
                          critique="Good coverage now",
                          decession="accept",
                          additional_queries=[])
    state['reflection']= operator.add(state['reflection'],[note2])

    assert len(state['reflection'])==2
    assert state['reflection'][0]['decession']=="revise"
    assert state['reflection'][1]['decession']=="accept"

def test_scaler_filds_overwrite():
    """scaler fileds like current_step must over write not accumulate"""
    state= create_itital_state("test","s1")
    assert state['current_step']==0

    state['current_step']=1
    assert state['current_step']==1

    state['current_step']=2
    assert state['current_step']==2

def test_plan_structure():
    state= create_itital_state("test","s1")
    plan= ResearchPlan(
        subtask=[
            SubTask(
                step_index=0,
                task_type="search",
                description= "find overview",
                query= "Langgraph Overview 2024",
                status="pending",
            ),
            SubTask(
                step_index=1,
                task_type="search",
                description= "find updates",
                query= "Langgraph latest updates",
                status="pending",
            )
        ],
        total_step=2
    )
    state["plan"]=plan
    assert state["plan"]["total_step"]==2
    assert state["plan"]["subtask"][0]["task_type"]=="search"
    assert state["plan"]["subtask"][1]["status"]=="pending"


