# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import uuid
from enum import Enum
from typing import AsyncIterator, Union

from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from vllm.outputs import RequestOutput
from vllm.tokenizers import TokenizerLike as AnyTokenizer

from dynamo.runtime import Client

from ..multimodal_utils import (
    ChatProcessor,
    CompletionsProcessor,
    MultiModalInput,
    MultiModalRequest,
    MyRequestOutput,
    ProcessMixIn,
    vLLMMultimodalRequest,
)

logger = logging.getLogger(__name__)


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


class ProcessorHandler(ProcessMixIn):
    """
    vLLM pre and post processing for multimodal requests
    """

    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        encode_worker_client: Client,
        prompt_template: str,
    ):
        self.encode_worker_client = encode_worker_client
        self.prompt_template = prompt_template
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
        context,
    ):
        request_id = str(uuid.uuid4().hex)
        logger.debug(f"Got raw request: {raw_request}")
        (
            request,
            conversation,
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
    async def generate(self, raw_request: MultiModalRequest, context):
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

        if multimodal_input.image_url is None and multimodal_input.video_url is None:
            raise ValueError("Either image URL or video URL is required")

        async for response in self._generate(
            chat_request, multimodal_input, RequestType.CHAT, context
        ):
            logger.debug(
                f"Generated response type {type(response)}, content: {response}"
            )
            # reconstructing back the OpenAI chat response as dynamo egress expects it
            if response.startswith("data: [DONE]"):
                break
            response = json.loads(response.lstrip("data: "))
            yield response
