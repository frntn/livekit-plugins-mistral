# Copyright 2025 Matthieu FRONTON
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import json 
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

# pip install mistralai
from mistralai import Mistral # Updated import

# Import types directly from mistralai - structure changed in v1.8.0+
try:
    from mistralai import ChatMessage, ToolCall, CompletionEvent, ToolChoice as MistralToolChoiceModel, FunctionName as MistralFunctionName
except ImportError:
    # Fallback for different API structure
    try:
        from mistralai.models import ChatMessage, ToolCall, CompletionEvent, ToolChoice as MistralToolChoiceModel, FunctionName as MistralFunctionName
    except ImportError:
        # Create minimal compatible classes if imports fail
        from typing import Dict, Any, Optional, List
        
        class ChatMessage:
            def __init__(self, role: str, content: str, **kwargs):
                self.role = role
                self.content = content
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class ToolCall:
            def __init__(self, id: str, function: Dict[str, Any], **kwargs):
                self.id = id
                self.function = function
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class CompletionEvent:
            def __init__(self, data: Any):
                self.data = data
        
        class MistralToolChoiceModel:
            def __init__(self, function: Any, type: str):
                self.function = function
                self.type = type
        
        class FunctionName:
            def __init__(self, name: str):
                self.name = name
        
        MistralFunctionName = FunctionName

from livekit.agents import llm, utils
from livekit.agents.llm import ChatContext, FunctionTool, ToolChoice as LiveKitToolChoice # Renamed to avoid clash

# Try different import paths for livekit-agents types
try:
    from livekit.agents.llm.common_types import ChatChunkDelta, ChoiceDelta, LLMError, APIConnectionError
except ImportError:
    try:
        from livekit.agents.llm import ChatChunkDelta, ChoiceDelta, LLMError, APIConnectionError
    except ImportError:
        from livekit.agents import APIError as LLMError, APIConnectionError
        from livekit.agents.llm import ChoiceDelta
        # Create minimal ChatChunkDelta if not available
        class ChatChunkDelta:
            def __init__(self, content: str = "", role: str = "assistant", **kwargs):
                self.content = content
                self.role = role
                for k, v in kwargs.items():
                    setattr(self, k, v)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import ChatModels


class AgentError(LLMError):
    """Agent-specific errors with descriptive prefixes"""
    pass


@dataclass
class _LLMOptions:
    model: str | ChatModels
    mode: Literal["normal", "agent"] = field(default="normal")
    temperature: NotGivenOr[float] = field(default=NOT_GIVEN)
    tool_choice: NotGivenOr[LiveKitToolChoice] = field(default=NOT_GIVEN) 
    max_tokens: NotGivenOr[int] = field(default=NOT_GIVEN)
    top_p: NotGivenOr[float] = field(default=NOT_GIVEN)
    random_seed: NotGivenOr[int] = field(default=NOT_GIVEN)
    safe_prompt: NotGivenOr[bool] = field(default=NOT_GIVEN)


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "mistral-small-latest",
        mode: Literal["normal", "agent"] = "normal",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN, 
        client: Mistral | None = None, # Updated type hint
        temperature: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[LiveKitToolChoice] = NOT_GIVEN, 
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        random_seed: NotGivenOr[int] = NOT_GIVEN,
        safe_prompt: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__()
        self._opts = _LLMOptions(
            model=model,
            mode=mode,
            temperature=temperature,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            top_p=top_p,
            random_seed=random_seed,
            safe_prompt=safe_prompt,
        )

        _api_key = api_key if is_given(api_key) else os.environ.get("MISTRAL_API_KEY")
        if not _api_key:
            raise ValueError(
                "MISTRAL_API_KEY is required, either as an argument or an environment variable."
            )

        _server_url = base_url if is_given(base_url) else None 

        # Create client with appropriate parameters based on what's provided
        if _server_url:
            self._client = client or Mistral(api_key=_api_key, server_url=_server_url)
        else:
            self._client = client or Mistral(api_key=_api_key)

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS, 
        tool_choice: NotGivenOr[LiveKitToolChoice] = NOT_GIVEN, 
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        random_seed: NotGivenOr[int] = NOT_GIVEN,
        safe_prompt: NotGivenOr[bool] = NOT_GIVEN,
        **kwargs, 
    ) -> LLMStream:
        if self._opts.mode == "normal":
            return self._chat_normal(
                chat_ctx=chat_ctx,
                tools=tools,
                conn_options=conn_options,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                random_seed=random_seed,
                safe_prompt=safe_prompt,
                **kwargs
            )
        else:  # mode == "agent"
            return self._chat_agent(
                chat_ctx=chat_ctx,
                tools=tools,
                conn_options=conn_options,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                random_seed=random_seed,
                safe_prompt=safe_prompt,
                **kwargs
            )

    def _chat_normal(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS, 
        tool_choice: NotGivenOr[LiveKitToolChoice] = NOT_GIVEN, 
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        random_seed: NotGivenOr[int] = NOT_GIVEN,
        safe_prompt: NotGivenOr[bool] = NOT_GIVEN,
        **kwargs, 
    ) -> LLMStream:
        
        mistral_messages: list[dict] = []
        for item in chat_ctx.items:
            if item.type == "message":
                role = item.role
                if role == "developer": 
                    role = "system"
                elif role == "function_call_output": 
                    role = "tool" 
                    if hasattr(item, 'call_id') and hasattr(item, 'output') and hasattr(item, 'name'):
                         mistral_messages.append({
                             "role": "tool",
                             "content": item.output,
                             "tool_call_id": item.call_id,
                             "name": item.name
                         })
                         continue
                    else:
                        logger.warning(f"Skipping function_call_output item due to missing attributes: {item}")
                        continue

                content_str = item.text_content or ""
                mistral_messages.append({"role": role, "content": content_str})

            elif item.type == "function_call":
                tool_calls_payload = []
                for call in item.tool_calls:
                    arguments_payload = call.arguments
                    if isinstance(arguments_payload, dict): 
                        arguments_payload = json.dumps(arguments_payload)

                    tool_calls_payload.append({
                        "id": call.call_id,
                        "function": {"name": call.name, "arguments": arguments_payload}
                    })
                mistral_messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls_payload})


        mistral_tools_payload = None
        if tools:
            mistral_tools_payload = []
            for tool in tools:
                # Handle different tool types
                if hasattr(tool, 'parameters') and hasattr(tool, 'name') and hasattr(tool, 'description'):
                    # This is a FunctionTool object
                    parameters_schema = tool.parameters.model_json_schema() if tool.parameters else {"type": "object", "properties": {}}
                    mistral_tools_payload.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": parameters_schema
                        }
                    })
                else:
                    # Skip tools that don't have the expected structure
                    logger.warning(f"Skipping tool with unexpected structure: {tool}")
                    continue
        
        final_tool_choice: str | MistralToolChoiceModel | None = None
        _tool_choice_opt = tool_choice if is_given(tool_choice) else self._opts.tool_choice
        if is_given(_tool_choice_opt):
            if _tool_choice_opt in ["auto", "any", "none", "required"]: # Valid ToolChoiceEnum values
                final_tool_choice = _tool_choice_opt # type: ignore
            elif isinstance(_tool_choice_opt, dict) and _tool_choice_opt.get("type") == "function":
                func_name = _tool_choice_opt["function"]["name"]
                final_tool_choice = MistralToolChoiceModel(function=MistralFunctionName(name=func_name), type="function") # type: ignore
            else:
                logger.warning(f"Unsupported tool_choice format: {_tool_choice_opt}. Defaulting to 'auto'.")
                final_tool_choice = "auto"


        _temperature = temperature if is_given(temperature) else self._opts.temperature
        _max_tokens = max_tokens if is_given(max_tokens) else self._opts.max_tokens
        _top_p = top_p if is_given(top_p) else self._opts.top_p
        _random_seed = random_seed if is_given(random_seed) else self._opts.random_seed
        _safe_prompt = safe_prompt if is_given(safe_prompt) else self._opts.safe_prompt

        final_temperature = _temperature if is_given(_temperature) else None
        final_max_tokens = _max_tokens if is_given(_max_tokens) else None
        final_top_p = _top_p if is_given(_top_p) else None
        final_random_seed = _random_seed if is_given(_random_seed) else None
        final_safe_prompt = _safe_prompt if is_given(_safe_prompt) else False 

        # Call the new SDK structure: self._client.chat.stream_async()
        mistral_stream = self._client.chat.stream_async( # Updated call
            model=self._opts.model, # type: ignore
            messages=mistral_messages, # type: ignore
            tools=mistral_tools_payload, # type: ignore
            tool_choice=final_tool_choice, # type: ignore
            temperature=final_temperature, 
            max_tokens=final_max_tokens,
            top_p=final_top_p,
            random_seed=final_random_seed,
            safe_prompt=final_safe_prompt, 
            **kwargs
        )

        return LLMStream(
            self,
            mistral_stream=mistral_stream, # type: ignore
            chat_ctx=chat_ctx, 
            tools=tools or [],
            conn_options=conn_options,
        )

    def _chat_agent(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS, 
        tool_choice: NotGivenOr[LiveKitToolChoice] = NOT_GIVEN, 
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        random_seed: NotGivenOr[int] = NOT_GIVEN,
        safe_prompt: NotGivenOr[bool] = NOT_GIVEN,
        **kwargs, 
    ) -> LLMStream:
        try:
            # Convert ChatContext to agent conversation inputs
            agent_inputs = []
            for item in chat_ctx.items:
                if item.type == "message":
                    role = item.role
                    if role == "developer": 
                        role = "system"
                    # Skip function_call_output for agent mode - let agent handle tools internally
                    elif role == "function_call_output":
                        continue
                    
                    content_str = item.text_content or ""
                    agent_inputs.append({"role": role, "content": content_str})
                # Skip function_call items for agent mode - let agent handle tools internally
                elif item.type == "function_call":
                    continue

            # Get completion parameters
            _temperature = temperature if is_given(temperature) else self._opts.temperature
            _max_tokens = max_tokens if is_given(max_tokens) else self._opts.max_tokens
            _top_p = top_p if is_given(top_p) else self._opts.top_p
            _random_seed = random_seed if is_given(random_seed) else self._opts.random_seed
            _safe_prompt = safe_prompt if is_given(safe_prompt) else self._opts.safe_prompt

            final_temperature = _temperature if is_given(_temperature) else None
            final_max_tokens = _max_tokens if is_given(_max_tokens) else None
            final_top_p = _top_p if is_given(_top_p) else None
            final_random_seed = _random_seed if is_given(_random_seed) else None
            final_safe_prompt = _safe_prompt if is_given(_safe_prompt) else False

            # Build completion_args for agent API
            completion_args = {}
            if final_temperature is not None:
                completion_args["temperature"] = final_temperature
            if final_max_tokens is not None:
                completion_args["max_tokens"] = final_max_tokens
            if final_top_p is not None:
                completion_args["top_p"] = final_top_p
            if final_random_seed is not None:
                completion_args["random_seed"] = final_random_seed
            if final_safe_prompt is not None:
                completion_args["safe_prompt"] = final_safe_prompt

            # Use agent API - always start new conversation (stateless like normal mode)
            mistral_stream = self._client.beta.conversations.start_stream(
                agent_id=self._opts.model,  # agent_id instead of model
                inputs=agent_inputs,        # Full conversation history
                store=False,               # Don't store server-side (stateless)
                completion_args=completion_args if completion_args else None,
                **kwargs
            )

            return LLMStream(
                self,
                mistral_stream=mistral_stream, # type: ignore
                chat_ctx=chat_ctx, 
                tools=tools or [],
                conn_options=conn_options,
            )

        except Exception as e:
            logger.error(f"Error during Mistral Agent stream: {e}", exc_info=e)
            raise AgentError(f"AgentError: {str(e)}") from e


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm_instance: LLM, 
        *,
        mistral_stream: AsyncIterator[CompletionEvent], # Updated type hint
        chat_ctx: ChatContext,
        tools: list[FunctionTool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(llm_instance, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._mistral_stream = mistral_stream 
        self._request_id: str | None = None 
        self._current_tool_calls_agg: dict[str, dict[str, Any]] = {}


    async def _run(self) -> None:
        try:
            # Await the coroutine to get the async iterator
            stream_iterator = await self._mistral_stream
            async for event_chunk in stream_iterator: # event_chunk is CompletionEvent
                # actual data is in event_chunk.data which is CompletionChunk
                chunk_data = event_chunk.data 
                
                if not self._request_id and hasattr(chunk_data, 'id') and chunk_data.id:
                    self._request_id = chunk_data.id

                if chunk_data.choices:
                    choice = chunk_data.choices[0] # This is CompletionResponseStreamChoice
                    delta = choice.delta # This is DeltaMessage
                    
                    delta_content = delta.content
                    delta_role = delta.role or "assistant" 
                    
                    current_chunk_tool_calls: list[llm.FunctionToolCall] = []

                    if delta.tool_calls: # delta.tool_calls is List[ToolCall]
                        for tc_item in delta.tool_calls: # tc_item is ToolCall
                            call_id = tc_item.id
                            # Ensure func_name and func_args_delta are handled if tc_item.function might be None
                            func_name = tc_item.function.name if tc_item.function else None
                            func_args_delta = tc_item.function.arguments if tc_item.function else None


                            if call_id not in self._current_tool_calls_agg:
                                self._current_tool_calls_agg[call_id] = { # type: ignore
                                    "name": func_name or "", 
                                    "arguments": func_args_delta or "",
                                    "id": call_id 
                                }
                            else:
                                if func_args_delta:
                                     self._current_tool_calls_agg[call_id]["arguments"] += func_args_delta # type: ignore
                            
                    finish_reason = choice.finish_reason
                    if finish_reason == "tool_calls":
                        for call_id, aggregated_call in self._current_tool_calls_agg.items():
                            current_chunk_tool_calls.append(
                                llm.FunctionToolCall(
                                    call_id=call_id, # type: ignore
                                    name=aggregated_call["name"],
                                    arguments=aggregated_call["arguments"]
                                )
                            )
                        self._current_tool_calls_agg = {} 

                    livekit_delta = ChoiceDelta(
                        content=delta_content or "", 
                        role=delta_role, # type: ignore
                        tool_calls=current_chunk_tool_calls if current_chunk_tool_calls else [],
                    )

                    livekit_chunk = llm.ChatChunk(
                        id=self._request_id or utils.shortuuid(), 
                        delta=livekit_delta,
                    )
                    self._event_ch.send_nowait(livekit_chunk)

                if hasattr(chunk_data, 'usage') and chunk_data.usage:
                    usage = chunk_data.usage 
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            id=self._request_id or utils.shortuuid(),
                            usage=llm.CompletionUsage(
                                completion_tokens=usage.completion_tokens, # type: ignore
                                prompt_tokens=usage.prompt_tokens, # type: ignore
                                total_tokens=usage.total_tokens, # type: ignore
                            )
                        )
                    )

        except Exception as e:
            logger.error(f"Error during Mistral LLM stream: {e}", exc_info=e)
            # Create a simple exception without the problematic body parameter
            try:
                raise LLMError(f"Mistral stream error: {e}")
            except TypeError:
                # If LLMError constructor fails, use a basic Exception
                raise Exception(f"Mistral stream error: {e}") from e
        finally:
            if self._current_tool_calls_agg:
                final_tool_calls = []
                for call_id, aggregated_call in self._current_tool_calls_agg.items():
                    final_tool_calls.append(
                        llm.FunctionToolCall(
                            call_id=call_id, # type: ignore
                            name=aggregated_call["name"],
                            arguments=aggregated_call["arguments"]
                        )
                    )
                if final_tool_calls:
                    final_tool_chunk = llm.ChatChunk(
                        id=self._request_id or utils.shortuuid(),
                        delta=ChoiceDelta(
                            content=None,
                            role="assistant",
                            tool_calls=final_tool_calls
                        )
                    )
                    self._event_ch.send_nowait(final_tool_chunk)
                self._current_tool_calls_agg = {}

            self._event_ch.close()
