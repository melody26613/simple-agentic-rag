import json
import requests

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

from pydantic import BaseModel

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from langchain_core.messages.tool import ToolCall


class ChatOllamaTools(BaseChatModel):
    base_url: str = "http://localhost:11434"

    model: str = "qwen3:1.7b"

    chat_stream: bool = False

    timeout: int = 300

    tool_choice: str = "auto"

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "remote ollama"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:

        url = f"{self.base_url}/api/chat"

        messages = self.__convert_messages(messages=messages)

        if "functions" in kwargs:
            available_tools = self.__convert_tool_input(functions=kwargs["functions"])
        else:
            available_tools = None

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "tools": available_tools,
                "tool_choice": self.tool_choice,
                "stream": self.chat_stream,
            },
            ensure_ascii=False
        )
        headers = {
            "Content-Type": "application/json",
        }

        print(f"[ChatOllamaTools] _generate {url=}, {payload=}, {headers=}")

        try:
            response = requests.post(
                url=url,
                headers=headers,
                data=payload.encode("utf-8"),
                timeout=self.timeout,
                stream=self.chat_stream,
            )
        except Exception as e:
            print(f"Exception {str(e)} happened")
            return None

        if response.status_code != 200:
            print(f"Error when generate, resposne status code={response.status_code}, response={response.text}")
            return None

        print(f"[ChatOllamaTools] _generate {response.text=}")

        
        full_text = ""
        tool_calls: list[ToolCall] = []

        for line in response.iter_lines():
            print(f"{line=}")
            if line:
                chunk = line

                if self.chat_stream:
                    if line.startswith(b'data: '):
                        chunk = line[len(b'data: '):]
                        print(f"{chunk=}")
                        if chunk == b'[DONE]':
                            break

                try:
                    data = json.loads(chunk.decode('utf-8'))

                    message = data.get("message", {})
                    content = message.get("content", "")
                    full_text += content

                    tool_calls_data = message.get("tool_calls")
                    if isinstance(tool_calls_data, list):
                        for i, raw_tool in enumerate(tool_calls_data):
                            func = raw_tool.get("function", {})
                            tool_call: ToolCall = {
                                "name": func.get("name"),
                                "args": func.get("arguments", {}),
                                "id": raw_tool.get("id", f"tool-{i}"),
                            }
                            tool_calls.append(tool_call)

                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

        print(f"[ChatOllamaTools] _generate {full_text=}, {tool_calls=}")

        chat_generation = ChatGeneration(
            message=AIMessage(
                content=full_text,
                tool_calls=tool_calls,
            ),
            generation_info=None,
        )

        print(f"{chat_generation=}")
        return ChatResult(generations=[chat_generation])


    def __convert_messages(
            self, messages: List[BaseMessage]) -> List[Dict[str, Union[str, List[str]]]]:
        result: List = []

        for message in messages:
            role = ""
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, ToolMessage):
                role = "tool"
            else:
                print(f"[ChatOllamaTools] __convert_messages {message=}")
                raise ValueError("Received unsupported message type for ChatOllamaTools.")

            if isinstance(message.content, str):
                content = message.content
            else:
                content = ""
                for content_part in cast(List[Dict], message.content):
                    if content_part.get("type") == "text":
                        content += f"\n{content_part['text']}"
                    else:
                        raise ValueError("Unsupported message content type.")

            content = content.strip()

            msg_dict = {
                "role": role,
                "content": content
            }

            print(f"{message=}")

            if getattr(message, "name", None):
                msg_dict["name"] = message.name
            if getattr(message, "id", None):
                msg_dict["id"] = message.id
            if getattr(message, "tool_call_id", None):
                msg_dict["tool_call_id"] = message.tool_call_id
            if getattr(message, "tool_calls", None):
                msg_dict["tool_calls"] = self.__convert_ai_message_with_tools(message.tool_calls)

            result.append(msg_dict)

        return result
    
    def __convert_ai_message_with_tools(self, tool_calls: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "arguments": tool["args"]
                }
            }
            for tool in tool_calls
        ]
    
    def __convert_tool_input(self, functions: dict):
        return [
            {
                "type": "function",
                "function": func
            }
            for func in functions
        ]

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool)["function"] for tool in tools]
        return super().bind(functions=formatted_tools, **kwargs)


if __name__ == "__main__":
    model = ChatOllamaTools()

    messages = [
        SystemMessage(
            content="Please answer user's question in English"
        ),
        HumanMessage(
            content="Can you speak English?"
        )
    ]

    result = model._generate(messages=messages)
    print(result)
