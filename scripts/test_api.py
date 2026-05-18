"""End to end API test script Run while the srever is running. uvicorn api.main:app --reload --port 8000

Test:
-Heath Check
-Non streaming research Request
-Streaming Research Request 
-Feedback Submission
-Session status retrieval
"""

from __future__ import annotations
import asyncio
import json
import sys
import time
import httpx

BASE_URL= "http://localhost:8000/api/v1"

async def test_health():
    print("\n--- Health Check ---\n")
    async with httpx.AsyncClient() as client:
        r= await client.get(f"{BASE_URL}/health")

        assert r.status_code==200, f"Expected 200, got {r.status_code}"
        data= r.json()
        print(data)
        print(f"status: {data['status']}")
        print(f"checks: {data['checks']}")

    print(" Health Check Passed")

async def test_non_streaming():
    print(f"\n--- Non Streaming research ---\n")
    print(f"This will take 20-30s while agent run")
    start= time.monotonic()

    async with httpx.AsyncClient(timeout=120) as client:
        r= await client.post(f"{BASE_URL}/research",
                             json={
                                 "query": "What is Langgraph explain in 100 words?",
                                 "stream": False,
                             })
        assert r.status_code==200, f"Expected 200, got {r.status_code}: {r.text}"
        data= r.json()

    elapsed= time.monotonic()-start
    print(f" session_id: {data['session_id']}")
    print(f" ellapsed:   {elapsed:.1f}s")
    print(f" report:     {data['report']}")
    print(f" followups:  {data['followup_questions']}")
    print(f" error:      {data['error'] or '(None)'}")
    assert data['report'], "Report should not be empty"
    print(" Non streaming passed ")
    return data["session_id"]

async def test_streaming(session_id_for_feedback:list):
    print("\n--- streaming research ---\n")
    print(" Connecting to SSE stream...")

    event_received= []
    report_received= False
    done_received= False
    start= time.monotonic()


    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/research",
            json={
                "query": "What are the three main componenet of Langgraph?",
                "stream": True,
            },
        ) as response:
            assert response.status_code==200
            session_id= response.headers.get("x-session-id","unknown")
            session_id_for_feedback.append(session_id)
            print(f" session_id: {session_id}")

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw= line[len("data:")].strip()

                if not raw:
                    continue

                try:
                    event= json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_type= event.get("type","unknown")
                event_received.append(event_type)
                if event_type == "node_start":
                    print(f" -> node_start: {event.get('node_name')}")

                elif event_type== "plan_ready":
                    step= event.get("total_steps",0)
                    print(f" ->plan_ready: {step} steps")
                elif event_type== "search_results":
                    print(f" -> search_results: "
                          f"+{event.get('result_added')} "
                          f"(total= {event.get('total_results')}) "
                          f"strategy= {event.get('strategy_used')}"
                          )
                
                elif event_type=="reflection":
                    print(f" -> reflection: {event.get('decision')}")
                elif event_type=="token":
                    pass

                elif event_type=="response_completed":
                    report= event.get("report","")
                    print(f" -> response_completed: {len(report)} chars")
                    report_received= True
                elif event_type== "followup_questions":
                    print(f" -> folowup_questions: {event.get('questions')}")

                elif event_type == "run_metrics":
                    print(f" -> run_metrics "
                          f"tokens= {event.get('total_tokens')} "
                          f"cost=${event.get('estimated_cost_usd'):.4f} "
                          f"duration={event.get('total_duration_ms'):.0f}ms"
                         )
                elif event_type=="error":
                    print(f" error: {event.get('message')}")

                elif event_type=="done":
                    done_received=True
                    print(f" -> done ({time.monotonic()-start:.1f})s total")
                    break

    assert report_received, "Should have received response_completed event"
    assert done_received, "Should have received done event"
    assert "node_start" in event_received
    assert "plan_ready" in event_received
    print(" streaming passed")

async def test_feedback(session_id:str):
    print(f"\n--- Feedback submission (session: {session_id}) ---\n")
    async with httpx.AsyncClient() as client:
        r= await client.post(f"{BASE_URL}/feedback",
                             json={
                                 "session_id": session_id,
                                 "rating":1,
                                 "comment": "Very helpfull well-cited.",
                             })
        assert r.status_code==200, f"Expected 200 got {r.status_code}: {r.text}"
        data= r.json()
        print(f" record_at: {data['record_at']}")
        print(f" message: {data['message']}")
    print(" feedback passed")
async def test_session_status(session_id: str):
    print(f"\n--- Session status (session: {session_id}) ---")
    async with httpx.AsyncClient() as client:
        r= await client.get(f"{BASE_URL}/sessions/{session_id}")
        assert r.status_code==200
        data= r.json()
        print(f" status: {data['status']}")
        print(f" is_complete: {data['is_complete']}")
    print(" Session status completed")

async def main():
    print("\n---- Research Agent API - END-TO-END Test")

    try:
        await test_health()
        session_id= await test_non_streaming()

        streaming_session_ids= []
        await test_streaming(streaming_session_ids)

        if streaming_session_ids:
            sid= streaming_session_ids[0]
            await test_feedback(sid)
            await test_session_status(sid)

        print("--- All test passed ---")
    
    except httpx.ConnectError:
        print("\n Cannot Connect to the server Start it first ")
        sys.exit(1)

    except Exception as e:
        print(f"\n Test failed: {e}")
        sys.exit(1)


if __name__=="__main__":
    asyncio.run(main())
                    
                    
