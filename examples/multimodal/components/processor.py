# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import uuid
from enum import Enum
from typing import AsyncIterator, Tuple, Union

import uvloop
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import FlexibleArgumentParser

from dynamo.llm import ModelInput, ModelType, register_llm
from dynamo.runtime import Client, DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

# To import example local module
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
        DEFAULT_DOWNSTREAM_ENDPOINT = f"dyn://{DYN_NAMESPACE}.encoder.generate"

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
            help=f"The endpoint string of the downstream encoder in 'dyn://namespace.component.endpoint' format. Default: '{DEFAULT_DOWNSTREAM_ENDPOINT}'",
        )

        args, config = base_parse_args(parser)

        return args, config

    def __init__(
        self,
        args: argparse.Namespace,
        engine_args: AsyncEngineArgs,
        encode_worker_client: Client,
    ):
        self.encode_worker_client = encode_worker_client
        self.prompt_template = args.prompt_template
        self.engine_args = engine_args
        self.model_config = self.engine_args.create_model_config()
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        self.tokenizer = self._create_tokenizer(self.engine_args)
        self.chat_processor = ChatProcessor(self.tokenizer, self.model_config)
        self.completions_processor = CompletionsProcessor(
            self.tokenizer, self.model_config
        )

    def cleanup(self):
        pass

    def _create_tokenizer(self, engine_args: AsyncEngineArgs) -> AnyTokenizer:
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
        return base_tokenizer

    # Main method to parse the request and send the request to the vllm worker.
    async def _generate(
        self,
        raw_request: Union[CompletionRequest, ChatCompletionRequest],
        multimodal_input: MultiModalInput,
        request_type: RequestType,
    ):
        request_id = str(uuid.uuid4().hex)
        logger.debug(f"Got raw request: {raw_request}")
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
        response_generator = await self.encode_worker_client.round_robin(
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

        # Ensure the configured template includes the placeholder
        template = self.prompt_template
        if "<prompt>" not in template:
            raise ValueError("prompt_template must contain '<prompt>' placeholder")

        # Safely extract user text
        try:
            user_text = raw_request.messages[0].content[0].text
        except (IndexError, AttributeError) as e:
            raise ValueError(f"Invalid message structure: {e}")

        prompt = template.replace("<prompt>", user_text)

        msg = {
            "role": "user",
            "content": prompt,
        }

        # Set stream=True - the http frontend will handle aggregation of
        # streamed chunks into a single http response, or stream them
        # back as SSE responses based on the stream flag in the request.
        chat_request = ChatCompletionRequest(
            model=raw_request.model,
            messages=[msg],
            stream=True,
            max_tokens=raw_request.max_tokens,
            temperature=raw_request.temperature,
            request_id=str(uuid.uuid4()),
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
                elif item.type == "audio_url":
                    if (
                        multimodal_input.image_url is not None
                        or multimodal_input.video_url is not None
                    ):
                        raise ValueError("Cannot mix image, video and audio URLs")
                    multimodal_input.audio_url = item.audio_url.url

        if (
            multimodal_input.image_url is None
            and multimodal_input.video_url is None
            and multimodal_input.audio_url is None
        ):
            raise ValueError("Either image URL or video URL or audio URL is required")

        async for response in self._generate(
            chat_request, multimodal_input, RequestType.CHAT
        ):
            logger.debug(
                f"Generated response type {type(response)}, content: {response}"
            )
            # reconstructing back the OpenAI chat response as dynamo egress expects it
            if response.startswith("data: [DONE]"):
                break
            response = json.loads(response.lstrip("data: "))
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


@dynamo_worker()
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

    generate_endpoint = component.endpoint(config.endpoint)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        args.downstream_endpoint
    )
    encode_worker_client = (
        await runtime.namespace(parsed_namespace)
        .component(parsed_component_name)
        .endpoint(parsed_endpoint_name)
        .client()
    )

    handler = Processor(args, config.engine_args, encode_worker_client)

    logger.info("Waiting for Encoder Worker Instances ...")
    await encode_worker_client.wait_for_instances()

    # Register the endpoint as entrypoint to a model
    await register_llm(
        ModelInput.Text,  # Custom processor is used and this type bypasses SDK processor
        ModelType.Chat,
        generate_endpoint,
        config.model,
        config.served_model_name,
        kv_cache_block_size=config.engine_args.block_size,
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
