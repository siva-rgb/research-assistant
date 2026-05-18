from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Literal

logger= logging.getLogger(__name__)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

class Session:
    """Lighweight session record
    Holds metadata and a snapshott of the final agent state.
    Does not hold the full langgraph checkpoint- that lives in MemorySaver"""

    __slots__=("session_id", "query", "status",
               "created_at", "updated_at", "final_state", "feedback")
    
    def __init__(self, session_id: str, query: str):
        self.session_id=session_id
        self.query= query
        self.status: Literal["running","complete", "error"]="running"
        self.created_at: str= now_iso()
        self.updated_at: str= now_iso()
        self.final_state:dict[str,Any]={}
        self.feedback: dict[str, Any]={}

    def to_dict(self)-> dict[str,Any]:
        return{
            "session_id": self.session_id,
            "query": self.query,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "has_feedback": bool(self.feedback)
        }
class SessionStore:
    """Thread safe in-memory session stor"""

    def __init__(self, max_session=500):
        self._session: dict[str, Session]={}
        self._lock= asyncio.Lock()
        self._max_session= max_session
    
    # create a session and store it
    async def create(self, session_id: str, query: str)-> Session:
        async with self._lock:
            # delete the oldes session key if at capacity
            if len(self._session)>= self._max_session:
                oldest_key= next(iter(self._session))
                del self._session[oldest_key]

                logger.warning(f"Session_store= evicted session_id={oldest_key}")

            session= Session(session_id=session_id, query=query)
            self._session[session_id]= session
            logger.debug(f"session_stroe= created session_id={session_id}"
                         f"total= {len(self._session)}")
            
            return session
    
    async def get(self, session_id: str)-> Session | None:
        """get a preticular session by its session id"""
        async with self._lock:
            return self._session.get(session_id)
        
    async def update_status(self, session_id: str, 
                            status: Literal["Running","complete", "error"],
                            final_state: dict[str, Any]| None= None)-> None:
        async with self._lock:
            session= self._session.get(session_id)
            if session:
                session.status= status
                session.updated_at= now_iso()
                if final_state:
                    session.final_state= final_state

    async def record_feedback(self, session_id: str,
                              ratting: int,
                              comment:str)->bool:
        """Returns true if session found and feedback is recorded"""
        async with self._lock:
            session= self._session.get(session_id)
            if not session:
                return False
            session.feedback= {
                "rating": ratting,
                "comment": comment,
                "recorded_at": now_iso(),
            }
            logger.info(f"session_id= {session_id}"
                        f"feedback_ratting={ratting}")
            
            return True
        
    async def list_recent(self, limit: int=20)-> list[dict]:
        async with self._lock:
            # list of all the session available
            sessions= list(self._session.values())
            sessions.sort(key=lambda x: x.created_at, reverse=True)

            return [s.to_dict() for s in sessions [:limit]] 
    
session_store= SessionStore()