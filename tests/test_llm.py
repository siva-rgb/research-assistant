import operator
import pytest
import sys
import asyncio
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from agent.llm import invoke_llm
message= [
        SystemMessage(content="This is a test call only reply with one word"),
        HumanMessage(content="Reply with good morning if you read this") 
    ]

async def test_esponse():
    response= await invoke_llm(messages=message,
                               temprature=0.0,
                               node_name='test_node')
    result= response.content

    print(result)


asyncio.run(test_esponse())