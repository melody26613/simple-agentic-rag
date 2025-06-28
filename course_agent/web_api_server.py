import argparse
import os
import uvicorn

from urllib.parse import urlparse

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from pydantic import BaseModel

from tools.chat_ollama_tools import ChatOllamaTools
from course_agent.custom_logger import logger


class InputRequest(BaseModel):
    input: str
    model: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/course")


async def invoke(data: InputRequest, client: MultiServerMCPClient) -> str:
    model = ChatOllamaTools(base_url=ollama_uri, model=data.model)

    try:
        tools = await client.get_tools()
        agent = create_react_agent(
            model=model,
            tools=tools,
        )
        response = await agent.ainvoke({"messages": data.input})

        last_message = response["messages"][-1]
        print(f"\n\n[invoke] {last_message=}")

        return last_message.content

    except Exception as e:
        print(f"[Error] {e}")
        return e


@router.post("/search", response_class=PlainTextResponse)
async def search(request_data: InputRequest):
    client = MultiServerMCPClient(
        {
            "milvus_db_for_course": {
                "command": "python",
                "args": [
                    "-m",
                    "course_agent.milvus_mcp_server",
                ],
                "env": {
                    "COLLECTION": "course_collection",
                    "DB_IP": target_db_ip,
                    "DB_PORT": str(target_db_port),
                    "TEXT_EMBEDDER_URL": os.getenv(
                        "TEXT_EMBEDDER_URL", "http://localhost:11434/api/embed"
                    ),
                    "TEXT_EMBEDDER_MODEL": os.getenv(
                        "TEXT_EMBEDDER_MODEL", "qllama/multilingual-e5-base:latest"
                    ),
                },
                "transport": "stdio",
            },
        }
    )

    response = await invoke(request_data, client=client)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20000)
    parser.add_argument("--db_uri", type=str, default="http://localhost:19530")
    parser.add_argument("--ollama_uri", type=str, default="http://localhost:11434")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    parsed = urlparse(args.db_uri)
    target_db_ip = parsed.hostname
    target_db_port = parsed.port

    ollama_uri = args.ollama_uri

    app.include_router(router)

    uvicorn.run(app, host=args.host, port=args.port)
