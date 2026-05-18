from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import logging
import logging.config
class Settings(BaseSettings):
    model_api_key: str
    openai_base_url:str=""
    openai_model: str="azure.gpt-4.1"
    embedding_api_key: str
    embedding_model: str= "azure.text-embedding-3-small"
    openai_max_retries: int=2
    openai_request_timeout: int=30
    vector_store_collection:str ="research_findings"
    vector_top_k: int=3

    secret_key:int
    access_token_expire_minutes:int
    refresh_token_expire_days:int
    tavily_key:str
    #---------------Postgre DB-----------------
    pinecone_api_key: str
    pinecone_index_name: str= "research_findings"


    max_search_iteration: int= 3
    max_reflection_teration: int=2
    max_tool_retries: int=2

    app_env: str="development"
    log_level: str= "DEBUG"

    model_config=SettingsConfigDict(env_file=".env",
                                    env_file_encoding="utf-8",
                                    case_sensitive=False)

@lru_cache(maxsize=1)
def get_setting()->Settings:
    """Return the singletone Settings instance"""
    settings= Settings()
    return settings

settings= get_setting()

def configure_logging()->None:
    log_level= settings.log_level.upper()

    logging.config.dictConfig({
        "version":1,
        "disable_existing_loggers":False,
        "formatters":{
            "structured": {
                "format":(
                    "%(asctime)s level=%(levelname)s"
                    "logger=%(name)s %(message)s"
                ),
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "handlers":{
            "console":{
                "class": "logging.StreamHandler",
                "formatter": "structured",
                "stream": "ext://sys.stdout",
            },
        },
        "root":{"level":log_level,"handlers":["console"]},
        "loggers":{
            "httpx":{"level":"WARNING"},
            "httpcore":{"level":"WARNING"},
            "openai":{"level":"WARNING"},
            "pinecone": {"level":"WARNING"}
        }
    })

configure_logging()

