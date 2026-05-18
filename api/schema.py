"""
Pydantic model fora all API request and reponse bodies.

These are the HTTP contract - completely speprate from ResearchAgentState which is the internal agent contract.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field

class ResearchRequest(BaseModel):
    query: str= Field(...,
                      min_length=3, 
                      max_length=2000,
                      description="Ther research question to investigate", 
                      examples=["What is the difference between langchain and langgraph"])
    stream: bool= Field(default=True,
                        description=("If True, response is strimed vis SSE."
                                     "If False, wait for completion and return JSON."))
    model_config= {
        "json_schema_extra":{
            "examples":[{"query": "What is Langgraph and how does it work?",
                         "Stream":True}]
        }
    }

class FeedbackRequest(BaseModel):
    """Body of post feedback"""
    session_id:str= Field(..., description="Session ID from the reseach response.")
    rating: Literal[1,-1]= Field(..., description="1 for thumbs up (helpful) -1 for thumbs down (not helpful)")
    comment: str= Field(default="",
                        max_length=1000,
                        description="Optional free text comment")
    

class ResearchResponse(BaseModel):
    """Non-streaming response body from POST/researh when stream=False. Containes the complte result after the agent finishes"""
    session_id: str
    query:str
    report: str
    followup_questions: list[str]
    search_result_count: int
    reflection_count: int
    is_time_sensitive: bool
    created_at: str
    completed_at:str
    error: str=""

class FeedbackResponse(BaseModel):
    """Response body Post / feedback"""
    session_id:str
    rating: int
    record_at: str
    message: str= "Feedback recorded. Thank you"

class SessionStatusResponse(BaseModel):
    """Response body GET /sessions/{session_id}"""
    session_id: str
    query:str
    status: Literal["running","complete","error"]
    created_at: str
    updated_at: str
    is_complete: bool
    error:str= ""

class HealthResponse(BaseModel):
    """Response from body from GET/health"""
    status: Literal["ok", "degraded"]
    version: str= "1.0.0"
    checks: dict[str, str]

class SSEEvent(BaseModel):
    """Base class for all SSE event payloads."""
    type: str
    session_id: str
    timestamp: str= Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class NodeStartEvent(SSEEvent):
    type: Literal["node_start"]= "node_start"
    node_name:str

class NodeCompletEvent(SSEEvent):
    type: Literal["node_complete"]= "node_complete"
    node_name: str
    duration_ms: float

class PlanReadyEvent(SSEEvent):
    """Emmited adter planner_node completes -shows the reseach plan."""
    type: Literal["plan_ready"]= "plan_ready"
    total_steps: int
    sub_tasks: list[dict[str, Any]]

class SearchResultEvent(SSEEvent):
    type: Literal["search_results"]= "search_results"
    step: int
    total_steps: int
    result_added: int
    total_results: int
    strategy_used: str

class ReflectionEvent(SSEEvent):
    type: Literal["reflection"]= "reflection"
    decision: str # 'accept' | 'revise' | 'abort'
    critique: str

class TokenEvent(SSEEvent):
    """Emitted for each token during synthesis"""
    type: Literal["token"]= "token"
    token: str

class ResponseCompleteEvent(SSEEvent):
    """Emitted when syntehsis_node finished - contains all the full report"""
    type: Literal["response_completed"]= "response_completed"
    report: str
    followup_questions: list[str]

class RunMetricsEvent(SSEEvent):
    """Emmited at the veryend with cost and performace summary."""
    type: Literal["run_metrics"]= "run_metrics"
    total_tokens: int
    estimated_cost_usd: float
    total_duration_ms: float
    search_result_count: int
    reflection_count: int

class ErrorEvent(SSEEvent):
    """Emitted if the agent encounters an unrecoverab;e error"""

    type: Literal["error"]= "error"
    message: str

class DoneEvent(SSEEvent):
    type: Literal["done"]= "done"
