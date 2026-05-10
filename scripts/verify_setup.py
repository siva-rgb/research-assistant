import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


def check_python_version():
    major, minor = sys.version_info[:2]

    assert major == 3 and minor >= 11, (
        f"Need python 3.11+, got {major}.{minor}. "
        "Create a new venv with the right python version"
    )
    print(f"✅ Python {major}.{minor}")


def check_package():
    packages = {
        "langgraph",
        "langchain",
        "langchain_openai",
        "tavily",
        "fastapi",
        "pydantic",
        "dotenv"
    }

    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError as e:
            print(f"❌ {pkg} - {e}")
            print("Run: pip install -r requirements.txt")
            sys.exit(1)


def check_setting():
    try:
        from agent.config import settings
        print(f"✅ Settings loaded (env={settings.app_env})")
        print(f"OpenAI api key: {settings.model_api_key[:8]}...")
        print(f"Tavily api key: {settings.tavily_key[:8]}...")
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        print("Check your .env file")
        sys.exit(1)


def check_openai():
    from langchain_openai import ChatOpenAI
    from agent.config import settings

    llm = ChatOpenAI(
        api_key=settings.model_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.0
    )

    response = llm.invoke("Reply with exactly one word: ready")
    print(f"✅ LLM responding: {response.content.strip()}")


if __name__ == "__main__":
    print("======= Research Agent Verification =======\n")
    check_python_version()
    check_package()
    check_setting()
    check_openai()
    print("\n✅ All tests completed successfully")