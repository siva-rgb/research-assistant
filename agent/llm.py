from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from agent.config import settings
import time
import logging
from dataclasses import dataclass, field
from typing import Any

logger= logging.getLogger(__name__)

@dataclass 
class LLMResponse:
    content: str
    prompt_token: int=0
    completion_toknes: int=0
    duration_ms: float=0.0
    raw: Any= field(default=None, repr=False)


def get_llm(mode_name:str=settings.openai_model, termprature: float=0.0):
    return ChatOpenAI(
        api_key=settings.model_api_key,
        base_url=settings.openai_base_url,
        model=mode_name,
        temperature=termprature,
    )

async def invoke_llm(messages: list[BaseMessage],
                     temprature: float=0.1,
                     node_name:str='unknow'):
    
    """Async call that captures the token usage and timing use this in every node
    for doing llm call.
    """

    llm= get_llm(settings.openai_model,termprature=temprature)
    start= time.monotonic()
    try: 
        response= await llm.ainvoke(messages)

        duration_ms= (time.monotonic()-start)*1000

        #extract token usage from token metadata
        usage= getattr(response,"usage_metadata", None) or {}
        prompt_token= usage.get("input_tokens",0)
        completion_token= usage.get("output_tokens", 0)

        logger.debug(
            f"node={node_name} "
            f"prompt_tokens={prompt_token} "
            f"completion_tokens={completion_token} "
            f"duration_ms={duration_ms:.0f}"
        )

        return LLMResponse(content=response.content,
                           prompt_token=prompt_token,
                           completion_toknes=completion_token,
                           duration_ms= duration_ms,
                           raw=response)
    except Exception as e:
        duration_ms= (time.monotonic()- start)*1000

        logger.error(
            f"node= {node_name} llm_error= {type(e).__name__}"
            f"message= {str(e)} duration_ms= {duration_ms}"
        )

        return LLMResponse(
            content="",
            prompt_token=0,
            completion_toknes=0,
            duration_ms=duration_ms
        )
    
