"""FastAPI application entrypoint
Responsibilities:
    - Application lifecycle (startup/shutdown via lifespan)
    - Middleware (CORS, request logging)
    - Router Regression
    - App-level state

The research graph is built once and attached to app.state.
Route access it via request.app.state.research_graph.
This is the fastapi-idomatic way to share a singleton accross requests without using module level globals 
that are hard to override in state
"""

from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.routes import router

logger= logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup logic before the first request and shutdown logic after the last request
    Startup:
        -Build the langgraph (compile the graph with Memory Servie checkpointer)
        -Verify Pinecone connectivity
        -Attach graph to the app.state so routes can access it
    Shutdown:
        -Log final session count
        -Any cleanup (connection pools, etc) goes here in project 2
    """

    logger.info("startup: building research graph")
    start= time.monotonic()
    
    from agent.graph import build_research_graph
    research_graph= build_research_graph()
    app.state.research_graph= research_graph

    logger.info(f"startup: research graph ready "
                f"duration_ms={(time.monotonic()-start)*1000:.0f}")
    
    #verify pinecone is reachable
    try:
        from agent.vectorstore import get_vectorstore
        get_vectorstore()
        logger.info("startup: pinecone connection verified")
    
    except Exception as e:
        logger.warning(f"startup: pineocne unavailable ({type(e).__name__}: {e}) "
                       f"agent will only use web search")
        
    logger.info("startup: complete - ready to serve the request")
    try:
        yield
    finally:
        from api.session import session_store
        sessions= await session_store.list_recent(limit=1000)
        logger.info(f"shutdown: total_sessions={len(sessions)}")

def create_app()->FastAPI:
    """
    Return the configured fastapi application.
    Using a factory function (not a module level instance) makes the app easier to test
    """

    app= FastAPI(title="Research Agent API",
                 description=(
                     "Research agent System Submit a research query and received a streamed cited report"
                 ),
                 version="1.0.0",
                 docs_url="/docs",
                 redoc_url="/redoc",
                 lifespan=lifespan)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins= ["*"],
        allow_credentials= True,
        allow_methods= ["GET","POST","OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Session-Id"],
    )

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        start= time.monotonic()
        response= await call_next(request)
        duration_ms= (time.monotonic()-start)*1000

        logger.info(f"methods= {request.method} "
                    f"path={request.url.path} "
                    f"status={response.status_code} "
                    f"duration_ms= {duration_ms:.0f}")
        return response
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"unhandled_exception "
                     f"path= {request.url.path} "
                     f"error= {type(exc).__name__}: {exc}")
        
        return JSONResponse(
            status_code=500,
            content={"details": "Internal Server error check Server logs"}
        )
    
    #register routes
    app.include_router(router=router, prefix="/api/v1")
    return app

app= create_app()