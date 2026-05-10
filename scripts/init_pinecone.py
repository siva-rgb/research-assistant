"""
Verify the Pinecone setup and connectivity
Run once befor first use
"""

import asyncio
import sys
from langchain_core.documents import Document
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

async def verify_response()->None:
    print("\n====== Pinecone Setup Verification ======")
    try:
        from agent.config import settings
        print(f"Setting loaded")
        print(f"Pinecone Key: {settings.pinecone_api_key[:5]}")
        print(f"Pinecone Index Name: {settings.pinecone_index_name}")
        print(f"Pinecone Model: {settings.embedding_model}")

    except Exception as e:
        print(f"Seting failed {e}")
        print("Check the .env and config file")

    # Pinecone client + index exist
    try:
        from agent.vectorstore import _get_pinecone_client

        client= _get_pinecone_client()
        indexes = [idx["name"] if isinstance(idx, dict) else idx.name for idx in client.list_indexes()]
        print(f"Pinecone connected .Indexes: {indexes}")

        if settings.pinecone_index_name not in indexes:
            print(f"Index {settings.pinecone_index_name} not found.\n "
                  f"Create it in pinecone console \n")
            sys.exit(1)

        index= client.Index(settings.pinecone_index_name)
        stats= index.describe_index_stats()
        print(f"Index {settings.pinecone_index_name} found"
              f"Total Vector: {stats.get('total_vector_count',0)}\n"
              f"Dimenssion: {stats.get('dimension','unknown')}")
        if stats.get("dimension") and stats["dimension"] != 512:
            print(f"Index dimension is {stats['dimension']} expected 1536")
            sys.exit(1)

    except Exception as e:
        print(f"Pinecone connection failed: {e}")

    try:
        from agent.vectorstore import get_vectorstore, store_research_findings
        from agent.state import SearchReasult

        print(f"\nTesting Updert +query round trip")

        test_finding= SearchReasult(query="pinecone test",
                                    title="Verification document",
                                    url="https://example.com/verify-test",
                                    content="This is a verification document to test Pinecone connectivity",
                                    score=0.1,
                                    source="web")
        stored= await store_research_findings(
            findings=[test_finding],
            session_id="verify-session",
            query="pinecone connectivity test"
        )

        print(f"Upsert {stored} test documents")

        print(" Waiting 3s for idex consistancy ")

        await asyncio.sleep(3)

        vs= get_vectorstore()

        results= await vs.asimilarity_search_with_score("Pinecone connectivity test",k=1)
        if results:
            doc, score= results[0]

            print(f"Query returned {len(results)} results")
            print(f"Top results: {doc.page_content[:60]}")
            print(f"score: {score:.4f}")
        else:
            print(f"Query return 0 resutls")

    except Exception as e:
        print(f"Vector store failed: {e}")
        sys.exit(1)

    print("All pinecone check passed")

if __name__=="__main__":
    asyncio.run(verify_response())