"""
Vector store setup and access.
Using pgvefctor via Langchain's PG vector ingeration 
"""
from __future__ import annotations
from functools import lru_cache
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from agent.config import settings 
from agent.state import SearchReasult
import logging

logger= logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _get_pinecone_client()-> Pinecone:
    """Singletone Pinecone Client validated the API on first call
    falg immidiately if ivalid"""
    logger.info("initializing Pinecone Client")
    return Pinecone(settings.pinecone_api_key)

@lru_cache(maxsize=1)
def _get_embeddings()->OpenAIEmbeddings:

    """Singleton embeddings model 
    text-embeddign-small: 1536 dimession, fast, cheap """
    return OpenAIEmbeddings(api_key=settings.embedding_api_key,
                                 base_url=settings.openai_base_url,
                                 model="azure.text-embedding-3-small",
                                 dimensions=512
                                 )

  

@lru_cache(maxsize=1)
def get_vectorstore()->PineconeVectorStore:
    """
    Return a singletone PGVector instance lru_cahe ensure we create the connecion once per process.
    PGVector creates the table automatically on first use.
    """
    client= _get_pinecone_client()
    # verify index exist before calling
    existing_index = [idx["name"] if isinstance(idx, dict) else idx.name 
                  for idx in client.list_indexes()]

    if settings.pinecone_index_name not in existing_index:
        raise RuntimeError(f"Pinecone index {settings.pinecone_index_name} does not exist"
                           f"Create it in the Pinecone\nExisting index: {existing_index}")
    index= client.Index(settings.pinecone_index_name)

    return PineconeVectorStore(
        index=index,
        embedding=_get_embeddings(),
        text_key="text",
    )

async def store_research_findings(
        findings:list[SearchReasult],
        session_id: str,
        query: str
):
    """Stores research findings as vector embeddings after a session completes. Called by memory_updated_nodes.

    Args:
    findings: The search results form stae.search_results.
    session_id: Used to tagvectors- lets filterby session
    query: The Original research query- stored as metadata
    
    Returns:
        Number of document stored, 0 if no valid findings 
    
    Each SearchResult becomes one document in the ector store, tagges with each session_id and original query for future filtering"""

    if not findings:
        logger.info(f"session_id= {session_id} no findings to store")
        return 0
    new_findings= [f for f in findings if f.get("source","web") !="vectorstore"]

    if not new_findings:
        logger.info(f"session_id{session_id}"
                    f"All findings are from vector store nothing new to store")
        return 0
    vs= get_vectorstore()
    docs = [
        Document(
            page_content=f"{f['title']}\n{f['content']}",
            metadata={
                "url": f["url"],
                "title": f["title"],
                "session_id": session_id,
                "original_query": query,
                "score": f["score"],
            },
        )
        for f in new_findings if f.get("content")
    ]

    if not docs:
        logger.info(f"session_id= {session_id} "
                    f"no document with content after filtering")
        return 0
    await vs.aadd_documents(docs)
    
    logger.info(f"session_id={session_id} stored_docs={len(docs)}"
                f"collection= {settings.vector_store_collection}")
    
    return len(docs)
        
    
