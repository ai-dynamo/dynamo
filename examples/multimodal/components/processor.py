# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import copy
import json
import logging
import os
import signal
import sys
import uuid
from enum import Enum
from typing import AsyncIterator, Optional, Tuple, Union

import uvloop
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import FlexibleArgumentParser

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, register_llm
from dynamo.runtime import Client, DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo._core import parse_tool_calls_py

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.args import Config, base_parse_args, parse_endpoint
from utils.chat_processor import ChatProcessor, CompletionsProcessor, ProcessMixIn
from utils.protocol import (
    MultiModalInput,
    MultiModalRequest,
    MyRequestOutput,
    vLLMMultimodalRequest,
)

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


class Processor(ProcessMixIn):
    """
    vLLM pre and post processing
    """

    @classmethod
    def parse_args(cls) -> Tuple[argparse.Namespace, Config]:
        DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
        DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.processor.generate"
        DEFAULT_DOWNSTREAM_ENDPOINT = f"dyn://{DYN_NAMESPACE}.llm.generate"

        parser = FlexibleArgumentParser(
            description="vLLM based processor for Dynamo LLM."
        )
        parser.add_argument(
            "--prompt-template",
            type=str,
            required=True,
            help=(
                "Different multi-modal models expect the prompt to contain different special media prompts. "
                "The processor will use this argument to construct the final prompt. "
                "User prompt will replace '<prompt>' in the provided template. "
                "For example, if the user prompt is 'please describe the image' and the prompt template is "
                "'USER: <image> <prompt> ASSISTANT:', the resulting prompt is "
                "'USER: <image> please describe the image ASSISTANT:'."
            ),
        )
        parser.add_argument(
            "--endpoint",
            type=str,
            default=DEFAULT_ENDPOINT,
            help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: '{DEFAULT_ENDPOINT}'",
        )
        parser.add_argument(
            "--downstream-endpoint",
            type=str,
            default=DEFAULT_DOWNSTREAM_ENDPOINT,
            help=f"The endpoint string of the downstream LLM worker in 'dyn://namespace.component.endpoint' format. Default: '{DEFAULT_DOWNSTREAM_ENDPOINT}'",
        )

        args, config = base_parse_args(parser)

        return args, config

    def __init__(
        self,
        args: argparse.Namespace,
        engine_args: AsyncEngineArgs,
        llm_worker_client: Client,
        custom_template_path: Optional[str] = None,
        tool_call_parser: Optional[str] = None,
    ):
        self.llm_worker_client = llm_worker_client
        self.prompt_template = args.prompt_template
        self.engine_args = engine_args
        self.model_config = self.engine_args.create_model_config()
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        self.tokenizer = self._create_tokenizer(self.engine_args, custom_template_path)
        self.chat_processor = ChatProcessor(self.tokenizer, self.model_config)
        self.completions_processor = CompletionsProcessor(
            self.tokenizer, self.model_config
        )
        self.tool_call_parser = tool_call_parser

    def cleanup(self):
        pass

    def _create_tokenizer(self, engine_args: AsyncEngineArgs, custom_template_path: Optional[str] = None) -> AnyTokenizer:
        """Create a TokenizerGroup using engine arguments similar to VLLM's approach"""
        model_path = engine_args.model

        # Create the base tokenizer with VLLM's typical settings
        base_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            truncation_side="left",
            use_fast=True,  # VLLM might use the fast tokenizer for efficiency
        )
        # Store custom template path but DON'T set it as default on the tokenizer
        # We'll apply it conditionally using shallow copy (thread-safe)
        # The template itself handles whether tools are present or not with {% if tools %} logic
        if custom_template_path:
            logger.info(f"Custom chat template path: {custom_template_path}")
            with open(custom_template_path, 'r') as f:
                self.custom_tool_template = f.read()
            logger.info("Custom chat template loaded (will be used for all requests when specified)")
        else:
            self.custom_tool_template = None
        return base_tokenizer

    # Main method to parse the request and send the request to the vllm worker.
    async def _generate(
        self,
        raw_request: Union[CompletionRequest, ChatCompletionRequest],
        multimodal_input: MultiModalInput,
        request_type: RequestType,
    ):
        request_id = str(uuid.uuid4().hex)
        logger.info(f"Got raw request: {raw_request}")
        (
            request,
            conversation,
            prompt,
            engine_prompt,
            sampling_params,
        ) = await self._parse_raw_request(raw_request)

        worker_request = vLLMMultimodalRequest(
            engine_prompt=engine_prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            multimodal_input=multimodal_input,
        )

        # model_dump_json() serializes the request to JSON string
        # This API could accept Pydantic class, but SamplingParams
        # in vLLMMultimodalRequest is not a Pydantic class and will
        # cause TypeError: unsupported type SamplingParams
        response_generator = await self.llm_worker_client.round_robin(
            worker_request.model_dump_json()
        )

        output = self._generate_responses(response_generator, request_type)

        # Stream the processed responses
        async for response in await self._stream_response(
            request, output, request_id, conversation
        ):
            yield response

    # This method is used to process the responses from the engine generator.
    async def _generate_responses(
        self,
        response_generator: AsyncIterator[RequestOutput],
        request_type: RequestType,
    ):
        async for resp in response_generator:
            # Deserialize the response from the engine
            # Creates correct vLLM objects for each field
            output = MyRequestOutput.model_validate_json(resp.data())

            # OpenAIServingChat.chat_completion_stream_generator() method expects a RequestOutput object
            request_output = RequestOutput(
                request_id=output.request_id,
                prompt=output.prompt,
                prompt_token_ids=output.prompt_token_ids,
                prompt_logprobs=output.prompt_logprobs,
                outputs=output.outputs,
                finished=output.finished,
                metrics=output.metrics,
            )

            if request_type == RequestType.CHAT:
                # For chat requests, yield the request_output directly.
                yield request_output
            else:
                raise NotImplementedError(
                    f"Request type {request_type} not implemented"
                )

    # The generate endpoint will be used by the frontend to handle incoming requests.
    async def generate(self, raw_request: MultiModalRequest):
        logger.debug(f"Got raw request: {raw_request}")
        if not isinstance(raw_request, MultiModalRequest):
            # If the request is not MultiModalRequest, convert it to MultiModalRequest
            raw_request = MultiModalRequest.model_validate(raw_request)

        # If tools are provided, apply the chat template with tools
        # We need to apply the Jinja template but NOT process images (keep them as URLs)
        if raw_request.tools and len(raw_request.tools) > 0:
            # Convert messages and tools to dicts for template rendering
            messages_for_template = []
            for msg in raw_request.messages:
                # Flatten multi-part content into a single text string for template
                content_parts = []
                for content in msg.content:
                    if content.type == "text":
                        content_parts.append(content.text)
                    elif content.type == "image_url":
                        # Use Qwen's vision tokens
                        content_parts.append("<|vision_start|><|image_pad|><|vision_end|>")
                    elif content.type == "video_url":
                        # Use similar format for video if needed
                        content_parts.append("<|vision_start|><|video_pad|><|vision_end|>")
                
                messages_for_template.append({
                    "role": msg.role,
                    "content": "".join(content_parts)  # Join without spaces to keep tokens together
                })
            
            # Convert tools to dicts
            tools_dicts = [tool.model_dump() if hasattr(tool, 'model_dump') else tool for tool in raw_request.tools]
            
            # Apply the custom tool calling template
            if self.custom_tool_template:
                # THREAD-SAFE: Create a shallow copy to avoid race conditions when mutating chat_template
                # Shallow copy is fast (~1-5 µs) and only copies attribute references, not large objects like vocab
                temp_tokenizer = copy.copy(self.tokenizer)
                temp_tokenizer.chat_template = self.custom_tool_template
                prompt = temp_tokenizer.apply_chat_template(
                    messages_for_template,
                    tools=tools_dicts,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                # Use default tokenizer template with tools
                prompt = self.tokenizer.apply_chat_template(
                    messages_for_template,
                    tools=tools_dicts,
                    add_generation_prompt=True,
                    tokenize=False
                )
            
            # Create a simple message with the formatted prompt (like the original path)
            msg = {
                "role": "user",
                "content": prompt,
            }
            
            chat_request = ChatCompletionRequest(
                model=raw_request.model,
                messages=[msg],
                stream=raw_request.stream,
                max_tokens=raw_request.max_tokens,
                temperature=raw_request.temperature,
                request_id=str(uuid.uuid4()),
                tools=raw_request.tools,
                tool_choice=raw_request.tool_choice,
            )
        else:
            # Check if this is a multimodal request (has images/video)
            has_multimodal = any(
                item.type in ["image_url", "video_url"] 
                for msg in raw_request.messages 
                for item in msg.content
            )
            
            if has_multimodal:
                # Original path: manual template replacement for multimodal non-tool calls
                # Ensure the configured template includes the placeholder
                template = self.prompt_template
                if "<prompt>" not in template:
                    raise ValueError("prompt_template must contain '<prompt>' placeholder")

                # Extract all text from content items (handles image-only, text-only, or mixed)
                text_parts = []
                for item in raw_request.messages[0].content:
                    if item.type == "text":
                        text_parts.append(item.text)
                
                # Use empty string if no text (image-only case)
                user_text = " ".join(text_parts) if text_parts else ""

                prompt = template.replace("<prompt>", user_text)

                msg = {
                    "role": "user",
                    "content": prompt,
                }

                chat_request = ChatCompletionRequest(
                    model=raw_request.model,
                    messages=[msg],
                    stream=raw_request.stream,
                    max_tokens=raw_request.max_tokens,
                    temperature=raw_request.temperature,
                    request_id=str(uuid.uuid4()),
                    tools=raw_request.tools,
                    tool_choice=raw_request.tool_choice,
                )
            else:
                # Text-only chat: use tokenizer's chat template
                messages_for_template = []
                for msg in raw_request.messages:
                    # Flatten content to string
                    content_text = " ".join([item.text for item in msg.content if item.type == "text"])
                    messages_for_template.append({
                        "role": msg.role,
                        "content": content_text
                    })
                
                # Apply chat template
                prompt = self.tokenizer.apply_chat_template(
                    messages_for_template,
                    add_generation_prompt=True,
                    tokenize=False
                )
                
                msg = {
                    "role": "user",
                    "content": prompt,
                }

                chat_request = ChatCompletionRequest(
                    model=raw_request.model,
                    messages=[msg],
                    stream=raw_request.stream,
                    max_tokens=raw_request.max_tokens,
                    temperature=raw_request.temperature,
                    request_id=str(uuid.uuid4()),
                    tools=raw_request.tools,
                    tool_choice=raw_request.tool_choice,
                )
        multimodal_input = MultiModalInput()

        for message in raw_request.messages:
            for item in message.content:
                if item.type == "image_url":
                    multimodal_input.image_url = item.image_url.url
                elif item.type == "video_url":
                    if multimodal_input.image_url is not None:
                        raise ValueError("Cannot provide both image and video URLs")
                    multimodal_input.video_url = item.video_url.url

        # Allow text-only messages (no image/video required)
        # This enables both pure text chat and multimodal use cases

        # Buffer chunks when tool calling is enabled to clear content after parsing
        accumulated_content = ""
        buffered_chunks = []
        
        async for response in self._generate(
            chat_request, multimodal_input, RequestType.CHAT
        ):
            logger.debug(
                f"Generated response type {type(response)}, content: {response}"
            )
            # reconstructing back the OpenAI chat response as dynamo egress expects it
            if response.startswith("data: [DONE]"):
                break
            
            # Handle both streaming (with "data: " prefix) and non-streaming responses
            if response.startswith("data: "):
                response = json.loads(response.lstrip("data: "))
            else:
                response = json.loads(response)
                # Convert non-streaming format (message) to streaming format (delta)
                if "choices" in response and "message" in response["choices"][0]:
                    message_content = response["choices"][0]["message"]["content"]
                    response["choices"][0]["delta"] = {"content": message_content, "role": "assistant"}
                    del response["choices"][0]["message"]
                    response["object"] = "chat.completion.chunk"
            
            # Buffer chunks and accumulate content when tool calling is configured
            if (
                self.tool_call_parser
                and raw_request.tools
                and "choices" in response
                and len(response["choices"]) > 0
            ):
                choice = response["choices"][0]
                
                # Buffer this chunk
                buffered_chunks.append(response)
                
                # Accumulate delta content
                if "delta" in choice and choice["delta"].get("content"):
                    accumulated_content += choice["delta"]["content"]
                
                # Parse when we hit the end (finish_reason is set)
                finish_reason = choice.get("finish_reason")
                if finish_reason == "stop":
                    if accumulated_content:
                        logger.info(f"Attempting to parse accumulated tool calls (length={len(accumulated_content)}) with parser: {self.tool_call_parser}")
                        try:
                            tool_calls, normal_text = parse_tool_calls_py(accumulated_content, self.tool_call_parser)
                            logger.info(f"Parse result: {len(tool_calls) if tool_calls else 0} tool calls found")
                            
                            if tool_calls:
                                # Convert tool calls to OpenAI format
                                tool_call_chunks = []
                                for idx, tc in enumerate(tool_calls):
                                    tool_call_chunks.append({
                                        "index": idx,
                                        "id": tc["id"],
                                        "type": tc["type"],
                                        "function": {
                                            "name": tc["function"]["name"],
                                            "arguments": tc["function"]["arguments"]
                                        }
                                    })
                                
                                # Clear content from ALL buffered chunks (per OpenAI spec)
                                for buffered_chunk in buffered_chunks:
                                    if "choices" in buffered_chunk and len(buffered_chunk["choices"]) > 0:
                                        buffered_choice = buffered_chunk["choices"][0]
                                        if "delta" in buffered_choice:
                                            buffered_choice["delta"]["content"] = ""
                                        elif "message" in buffered_choice:
                                            buffered_choice["message"]["content"] = ""
                                
                                # Add tool_calls to the final chunk
                                if "delta" in choice:
                                    choice["delta"]["tool_calls"] = tool_call_chunks
                                elif "message" in choice:
                                    choice["message"]["tool_calls"] = tool_call_chunks
                                
                                choice["finish_reason"] = "tool_calls"
                                logger.info(f"Cleared content from {len(buffered_chunks)} chunks and added {len(tool_calls)} tool call(s) to final chunk")
                        except Exception as e:
                            logger.warning(f"Failed to parse tool calls: {e}", exc_info=True)
                            # Continue with original response if parsing fails
                    
                    # Yield all buffered chunks now that we've processed them
                    for chunk in buffered_chunks:
                        yield chunk
                    buffered_chunks = []
            else:
                # No tool calling, yield immediately
                yield response


async def graceful_shutdown(runtime):
    """
    By calling `runtime.shutdown()`, the endpoints will immediately be unavailable.
    However, in-flight requests will still be processed until they are finished.
    After all in-flight requests are finished, the `serve_endpoint` functions will return
    and the engine will be shutdown by Python's garbage collector.
    """
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    # Runtime setup
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    # worker setup
    args, config = Processor.parse_args()
    await init(runtime, args, config)


async def init(runtime: DistributedRuntime, args: argparse.Namespace, config: Config):
    """
    Instantiate and serve
    """

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        args.downstream_endpoint
    )
    llm_worker_client = (
        await runtime.namespace(parsed_namespace)
        .component(parsed_component_name)
        .endpoint(parsed_endpoint_name)
        .client()
    )

    handler = Processor(
        args,
        config.engine_args,
        llm_worker_client,
        config.custom_jinja_template,
        config.tool_call_parser,
    )

    logger.info("Waiting for LLM Worker Instances ...")
    await llm_worker_client.wait_for_instances()

    # Register the endpoint as entrypoint to a model
    logger.info(f"Config: {config.tool_call_parser}, {config.reasoning_parser}, {config.custom_jinja_template}")
    runtime_config = ModelRuntimeConfig()
    runtime_config.tool_call_parser = config.tool_call_parser
    runtime_config.reasoning_parser = config.reasoning_parser

    await register_llm(
        ModelInput.Text,
        ModelType.Chat,
        generate_endpoint,
        config.model,
        config.served_model_name,
        kv_cache_block_size=config.engine_args.block_size,
        runtime_config=runtime_config,
        custom_template_path=config.custom_jinja_template,
    )

    logger.info(f"Starting to serve the {args.endpoint} endpoint...")

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=[("model", config.model)]
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
