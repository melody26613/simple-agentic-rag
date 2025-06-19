import asyncio
import argparse
import traceback
import sys
import os

from typing import (
    Callable,
    Sequence,
    Union,
)

from urllib.parse import urlparse
from typing import Annotated, Sequence, TypedDict

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.tools import BaseTool

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_mcp_adapters.client import MultiServerMCPClient

from tools.chat_ollama_tools import ChatOllamaTools
from course_agent.custom_logger import logger

MAX_RECURSION_LIMIT = 10


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    messages = state["messages"]

    last_message = messages[-1]

    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


def try_except_continue(state, func):
    count = 0
    while count < MAX_RECURSION_LIMIT:
        count += 1
        try:
            ret = func(state)
            return ret
        except Exception as e:
            print("I crashed trying to run:", func)
            print("Here is my error")
            print(e)
            traceback.print_exception(*sys.exc_info())


class CourseDbGraph:
    __MAX_DB_AGENT_CALL_COUNT = 3

    def __init__(
        self,
        llm_db_agent: ChatOllamaTools,
        llm_generator: ChatOllamaTools,
        tools: Sequence[Union[BaseTool, Callable]],
    ):
        self.llm_db_agent = llm_db_agent
        self.llm_generator = llm_generator
        self.tool_list = tools
        self.tool_definitions = [
            convert_to_openai_function(t) for t in self.tool_list]

        self.agent_call_count = 0

        self.build_graph()

    # Nodes

    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        messages = state["messages"]

        model = self.llm_db_agent

        if self.agent_call_count < self.__MAX_DB_AGENT_CALL_COUNT:
            model = model.bind_tools(tools=self.tool_definitions)
        else:
            pass

        agent_prompt = ChatPromptTemplate.from_messages(
            [
                # ("system", system_prompt),
                ("human", "{question}"),
                MessagesPlaceholder("chat_history"),
            ]
        )

        model = agent_prompt | model

        response = None
        try:
            question = f"""The question is: "{messages[0]}" """
            response = model.invoke(
                {"question": question, "chat_history": messages[:]})
        except Exception as e:
            return {"messages": f"Failed to call db agent, error: {e}"}

        self.agent_call_count += 1

        return {"messages": [response]}

    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        messages = state["messages"]

        question = messages[0].content

        gen_prompt = ChatPromptTemplate.from_messages(
            [
                # ("system", system_prompt),
                ("human", "{question}"),
                MessagesPlaceholder("chat_history"),
            ]
        )

        model = gen_prompt | self.llm_generator

        response = None

        try:
            response = model.invoke(
                {"question": question, "chat_history": messages[1:]}
            )
        except Exception as e:
            return {"messages": f"Failed to generate answer, error: {e}"}

        self.agent_call_count = 0

        return {"messages": response.content}

    def build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node(
            "agent", lambda state: try_except_continue(state, self.agent))

        tool_node = ToolNode(self.tool_list)
        workflow.add_node("action", tool_node)

        workflow.add_node(
            "generate", lambda state: try_except_continue(state, self.generate)
        )

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": "generate",
            },
        )

        workflow.add_edge("action", "generate")
        workflow.add_edge("generate", END)

        self.graph = workflow.compile()

    async def query(self, question: str):
        inputs = {
            "messages": [
                (("user", question)),
            ]
        }

        out = await self.graph.ainvoke(inputs)
        response = out["messages"][-1]
        response = "".join(response.content.splitlines())

        return response


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_uri", type=str, default="http://localhost:19530")
    parser.add_argument("--ollama_uri", type=str,
                        default="http://localhost:11434")
    parser.add_argument("--llm_db_agent", type=str, default="qwen3:1.7b")
    parser.add_argument("--llm_generator", type=str, default="qwen3:1.7b")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    llm_db_agent = ChatOllamaTools()
    llm_db_agent.base_url = args.ollama_uri
    llm_db_agent.model = args.llm_db_agent

    llm_generator = ChatOllamaTools()
    llm_generator.base_url = args.ollama_uri
    llm_generator.model = args.llm_generator

    parsed = urlparse(args.db_uri)
    target_db_ip = parsed.hostname
    target_db_port = parsed.port

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
                    "TEXT_EMBEDDER_URL": os.getenv("TEXT_EMBEDDER_URL", None),
                    "TEXT_EMBEDDER_MODEL": os.getenv("TEXT_EMBEDDER_MODEL", None),
                },
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()

    graph = CourseDbGraph(
        llm_db_agent=llm_db_agent, llm_generator=llm_generator, tools=tools
    )
    response = await graph.query(
        "Recommend a NCHU course that can improve my android development skills."
    )
    print(f"\n\n{response}")


if __name__ == "__main__":
    asyncio.run(main())
