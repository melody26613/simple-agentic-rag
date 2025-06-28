import asyncio
import argparse
import os

from urllib.parse import urlparse

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from tools.chat_ollama_tools import ChatOllamaTools
from course_agent.custom_logger import logger


async def main():
    model = ChatOllamaTools()
    model.base_url = ollama_uri
    model.model = ollama_model

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
                    "TEXT_EMBEDDER_URL": os.getenv("TEXT_EMBEDDER_URL", "http://localhost:11434/api/embed"),
                    "TEXT_EMBEDDER_MODEL": os.getenv("TEXT_EMBEDDER_MODEL", "qllama/multilingual-e5-base:latest"),
                },
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(
        model,
        tools,
    )
    response = await agent.ainvoke({"messages": "Recommend a NCHU course that can improve my android development skills."})

    last_message = response["messages"][-1]
    logger.info(f"\n\n{last_message=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_uri", type=str, default="http://localhost:19530")
    parser.add_argument("--ollama_uri", type=str, default="http://localhost:11434")
    parser.add_argument("--ollama_model", type=str, default="qwen3:1.7b")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    parsed = urlparse(args.db_uri)
    target_db_ip = parsed.hostname
    target_db_port = parsed.port

    ollama_uri = args.ollama_uri
    ollama_model = args.ollama_model

    asyncio.run(main())
