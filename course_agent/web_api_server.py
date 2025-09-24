import argparse
import os
import uvicorn

from urllib.parse import urlparse
from typing import Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse

from pydantic import BaseModel

from tools.chat_ollama_tools import ChatOllamaTools
from course_agent.custom_logger import logger


class InputRequest(BaseModel):
    input: str
    model: str
    stream: Optional[bool] = True


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
        logger.info(f"invoke() {last_message=}")

        return last_message.content

    except Exception as e:
        logger.error(f"invoke() {e}")
        raise HTTPException(status_code=500, detail="Internal error")


async def invoke_stream(data: InputRequest, client: MultiServerMCPClient):
    model = ChatOllamaTools(base_url=ollama_uri, model=data.model)

    try:
        tools = await client.get_tools()
        agent = create_react_agent(
            model=model,
            tools=tools,
        )

        async for event in agent.astream(
            {"messages": data.input}, stream_mode="messages"
        ):
            message = event[0]
            if message.type != "AIMessageChunk":
                continue

            if not message.content or len(message.content) == 0:
                continue

            yield message.content

    except Exception as e:
        logger.error(f"invoke_stream() {e}")
        raise HTTPException(status_code=500, detail="Internal error")


async def generate_stream(data: InputRequest, client: MultiServerMCPClient):
    whole_content = ""
    async for chunk in invoke_stream(data, client):
        whole_content += chunk
        yield f"data: {chunk}\n\n"

    logger.info(f"generate_stream() {whole_content=}")


@router.post("/search")
async def search(request_data: InputRequest):
    logger.info(f"search() {request_data=}")

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
                    "DB_NAME": target_db_name,
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

    if request_data.stream:
        return StreamingResponse(
            generate_stream(data=request_data, client=client),
            media_type="text/event-stream",
        )
    else:
        response = await invoke(data=request_data, client=client)
        return PlainTextResponse(content=response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20000)
    parser.add_argument("--db_uri", type=str, default="http://localhost:19530")
    parser.add_argument("--db_name", type=str, default="default")
    parser.add_argument("--ollama_uri", type=str, default="http://localhost:11434")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    parsed = urlparse(args.db_uri)
    target_db_ip = parsed.hostname
    target_db_port = parsed.port
    target_db_name = args.db_name

    ollama_uri = args.ollama_uri

    app.include_router(router)

    uvicorn.run(app, host=args.host, port=args.port)
