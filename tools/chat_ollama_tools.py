import json
import requests
import hashlib

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
    Iterator,
)

from datetime import datetime

from pydantic import BaseModel, Field

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from langchain_core.messages.tool import ToolCall


class ChatOllamaTools(BaseChatModel):
    _DEFAULT_BASE_URL = "http://localhost:11434"
    _DEFAULT_CHAT_MODEL = "qwen3:1.7b"
    _DEFAULT_CHAT_STREAM = True
    _DEFAULT_CHAT_TIMEOUT = 300
    _DEFAULT_TOOL_CHOICE = "auto"
    _DEFAULT_ENABLE_THINK = False

    base_url: str = Field(
        default=_DEFAULT_BASE_URL, description="The base url that hosts ollama"
    )

    model: str = Field(default=_DEFAULT_CHAT_MODEL, description="The chat model name")

    chat_stream: bool = Field(
        default=_DEFAULT_CHAT_STREAM,
        description="Enable the response in streaming mode or not",
    )

    timeout: int = Field(
        default=_DEFAULT_CHAT_TIMEOUT,
        description="The timeout in second for ollama chat api",
    )

    tool_choice: str = Field(
        default=_DEFAULT_TOOL_CHOICE, description="Enable the llm to use tools or not"
    )

    think_mode: bool = Field(
        default=_DEFAULT_ENABLE_THINK, description="Enable think mode or not"
    )

    def __init__(
        self,
        base_url: Optional[str] = _DEFAULT_BASE_URL,
        model: Optional[str] = _DEFAULT_CHAT_MODEL,
        chat_stream: Optional[bool] = _DEFAULT_CHAT_STREAM,
        timeout: Optional[int] = _DEFAULT_CHAT_TIMEOUT,
        tool_choice: Optional[str] = _DEFAULT_TOOL_CHOICE,
        think_mode: Optional[bool] = _DEFAULT_ENABLE_THINK,
    ):
        super().__init__()

        self.base_url = base_url
        self.model = model
        self.chat_stream = chat_stream
        self.timeout = timeout
        self.tool_choice = tool_choice
        self.think_mode = think_mode

        print(f"[ChatOllamaTools] __init__() variables={self.__dict__.items()}")

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "chat ollama tools"

    def _create_chat_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        url = f"{self.base_url}/api/chat"

        print(f"[ChatOllamaTools] _create_chat_stream {messages=}")
        messages = self.__convert_messages(messages=messages)

        available_tools = None
        if "functions" in kwargs:
            available_tools = self.__convert_tool_input(functions=kwargs["functions"])

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "tools": available_tools,
                "tool_choice": self.tool_choice,
                "stream": self.chat_stream,
                "think": self.think_mode,
            },
            ensure_ascii=False,
        )
        headers = {
            "Content-Type": "application/json",
        }

        print(f"[ChatOllamaTools] _create_chat_stream {payload=}, {headers=}")

        response = requests.post(
            url=url,
            headers=headers,
            data=payload.encode("utf-8"),
            timeout=self.timeout,
            stream=self.chat_stream,
        )

        if response.status_code != 200:
            raise Exception(
                f"[ChatOllamaTools] error when generate, resposne status code={response.status_code}, response={response.text}"
            )

        return response.iter_lines(decode_unicode=True)

    def _chat_stream_with_aggregation(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk: Optional[ChatGenerationChunk] = None
        for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
            if stream_resp:
                chunk = self._chat_stream_response_to_chat_generation_chunk(stream_resp)
                if chunk is None:
                    continue

                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    def _chat_stream_response_to_chat_generation_chunk(
        self,
        stream_response: Any,
    ) -> ChatGenerationChunk:
        """Convert a stream response to a generation chunk."""
        if isinstance(stream_response, bytes):
            stream_response = stream_response.decode("utf-8")

        if stream_response.startswith("data: "):
            stream_response = stream_response[len("data: ") :]

        parsed_response = json.loads(stream_response, strict=False)
        generation_info = (
            parsed_response if parsed_response.get("done", False) is True else None
        )
        additional_kwargs = {}

        message = parsed_response.get("message", None)
        if message is None:
            return None

        content = message.get("content", "")

        tool_calls = message.get("tool_calls", None)
        additional_kwargs["tool_calls"] = self.__convert_tool_calls_output(tool_calls)

        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content=content,
                additional_kwargs=additional_kwargs,
            ),
            generation_info=generation_info,
        )
        return chunk

    def __convert_tool_calls_output(self, tool_calls: List[dict]) -> List[dict]:
        if tool_calls is None:
            return None

        modified_data = []

        for item in tool_calls:
            if "function" in item and "arguments" in item["function"]:
                arguments_dict = item["function"]["arguments"]
                arguments = json.dumps(arguments_dict, ensure_ascii=False)

                new_item = item.copy()
                new_item["id"] = f"tool-{self.__gen_id()}"
                new_item["type"] = "function"
                new_item["function"] = new_item["function"].copy()
                new_item["function"]["arguments"] = arguments

                modified_data.append(new_item)
            else:
                modified_data.append(item)

        return modified_data

    def __gen_id(self) -> str:
        seed = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        hash_value = hashlib.sha1(seed.encode()).hexdigest()
        return hash_value

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
            if stream_resp:
                chunk = self._chat_stream_response_to_chat_generation_chunk(stream_resp)
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=self.verbose,
                    )
                yield chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        final_chunk = self._chat_stream_with_aggregation(
            messages,
            stop=stop,
            run_manager=run_manager,
            verbose=self.verbose,
            **kwargs,
        )
        final_tool_calls = final_chunk.message.tool_calls

        chat_generation = ChatGeneration(
            message=AIMessage(
                content=final_chunk.text,
                tool_calls=final_tool_calls,
            ),
            generation_info=final_chunk.generation_info,
        )
        print(f"[ChatOllamaTools] _generate {chat_generation=}")
        return ChatResult(generations=[chat_generation])

    def __convert_messages(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Union[str, List[str]]]]:
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
                raise ValueError(
                    "Received unsupported message type for ChatOllamaTools."
                )

            if isinstance(message.content, str):
                content = message.content
            else:
                content = ""
                for content_part in cast(List[Dict], message.content):
                    if content_part.get("type") == "text":
                        content += f"""\n{content_part["text"]}"""
                    else:
                        raise ValueError("Unsupported message content type.")

            content = content.strip()

            msg_dict = {"role": role, "content": content}

            if getattr(message, "name", None):
                msg_dict["name"] = message.name
            if getattr(message, "id", None):
                msg_dict["id"] = message.id
            if getattr(message, "tool_call_id", None):
                msg_dict["tool_call_id"] = message.tool_call_id
            if getattr(message, "tool_calls", None):
                msg_dict["tool_calls"] = self.__convert_ai_message_with_tools(
                    message.tool_calls
                )

            result.append(msg_dict)

        return result

    def __convert_ai_message_with_tools(self, tool_calls: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {"name": tool["name"], "arguments": tool["args"]},
            }
            for tool in tool_calls
        ]

    def __convert_tool_input(self, functions: dict):
        return [{"type": "function", "function": func} for func in functions]

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
        SystemMessage(content="Please answer user's question in English"),
        HumanMessage(content="Can you speak English?"),
    ]

    result = ""
    for chunk in model._stream(messages=messages):
        if chunk is not None:
            print(chunk)
            result += chunk.text

    print(f"{result=}")
