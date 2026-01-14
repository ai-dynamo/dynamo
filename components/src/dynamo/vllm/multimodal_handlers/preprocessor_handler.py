# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import uuid
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Union

from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from vllm.outputs import RequestOutput
from vllm.tokenizers import TokenizerLike as AnyTokenizer

from dynamo.runtime import Client

from ..multimodal_utils import (
    ChatProcessor,
    CompletionsProcessor,
    MultiModalGroup,
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


class ECProcessorHandler(ProcessorHandler):
    """
    Processor handler for ECConnector-based encoder
    """

    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        encoder_worker_client: Client,
        pd_worker_client: Client,
        prompt_template: str,
    ):
        """
        Initialize the ECConnector processor.

        Args:
            engine_args: vLLM engine arguments for model config
            encoder_worker_client: Client for encoder worker endpoints
            pd_worker_client: Client for PD worker endpoints
            prompt_template: Multimodal prompt template
        """
        # Initialize base class with encoder client
        super().__init__(engine_args, encoder_worker_client, prompt_template)

        # Store additional PD client for disaggregated architecture
        self.encoder_client = encoder_worker_client
        self.pd_client = pd_worker_client

        logger.info("ECProcessorHandler initialized with disaggregated architecture")

    @staticmethod
    def _extract_multimodal_items(request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all multimodal items (images/videos) from the request messages.

        Args:
            request_data: The request dictionary

        Returns:
            List of multimodal content items
        """
        items = []
        messages = request_data.get("messages", [])

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for item in content:
                item_type = item.get("type")
                if item_type in ("image_url", "video_url"):
                    items.append(item)

        return items

    @staticmethod
    def _create_encoder_request(
        prompt: str,
        mm_item: Dict[str, Any],
        model: str,
        request_id: str,
    ) -> Dict[str, Any]:
        # Create MultiModalInput from the item
        multimodal_input = {}
        modality = None

        if mm_item.get("type") == "image_url":
            multimodal_input["image_url"] = mm_item["image_url"]["url"]
            modality = "image"
        elif mm_item.get("type") == "video_url":
            multimodal_input["video_url"] = mm_item["video_url"]["url"]
            modality = "video"
        else:
            raise ValueError(f"Unsupported multimodal type: {mm_item.get('type')}")

        return {
            "prompt": prompt,
            "request_id": request_id,
            "multimodal_input": multimodal_input,
            "modality": modality,
        }

    async def _encode_multimodal_items(
        self,
        prompt: str,
        mm_items: List[Dict[str, Any]],
        model: str,
        request_id: str,
    ) -> None:
        """
        Send all multimodal items to encoder workers concurrently.

        Each item is sent as a separate request to an encoder worker.
        The encoder processes the item and stores embeddings to shared storage.
        """
        if not mm_items:
            logger.debug(f"[{request_id}] No multimodal items to encode")
            return

        logger.info(f"[{request_id}] Encoding {len(mm_items)} multimodal item(s)")

        tasks = []
        for idx, mm_item in enumerate(mm_items):
            # Create unique request ID for each item
            item_request_id = f"{request_id}_mm_{idx}"

            # Build encoder request
            encoder_request = self._create_encoder_request(
                prompt=prompt,
                mm_item=mm_item,
                model=model,
                request_id=item_request_id,
            )

            # Create task for this encoder request
            task = self._send_to_encoder(encoder_request, item_request_id)
            tasks.append(task)

        # Wait for all encoder requests to complete
        try:
            await asyncio.gather(*tasks)
            logger.info(f"[{request_id}] All encoders completed successfully")
        except Exception as e:
            logger.error(f"[{request_id}] Encoder encoding failed: {e}")
            raise

    async def _send_to_encoder(
        self,
        encoder_request: Dict[str, Any],
        request_id: str,
    ) -> None:
        """
        Send a single request to an encoder worker and wait for completion.
        """
        try:
            # Convert to JSON
            request_json = json.dumps(encoder_request)

            # Send to encoder worker (round-robin)
            response_stream = await self.encoder_client.round_robin(request_json)

            # Consume the response stream
            async for chunk in response_stream:
                pass

            logger.debug(f"[{request_id}] Encoder completed successfully")

        except Exception as e:
            logger.error(f"[{request_id}] Encoder request failed: {e}")
            raise

    async def generate(self, raw_request: MultiModalRequest, context):
        """
        Main endpoint handler for chat completion requests with ECConnector.
        """
        logger.debug(f"ECProcessor received request: {raw_request}")

        if not isinstance(raw_request, MultiModalRequest):
            raw_request = MultiModalRequest.model_validate(raw_request)

        # Ensure the configured template includes the placeholder
        template = self.prompt_template
        if "<prompt>" not in template:
            raise ValueError("prompt_template must contain '<prompt>' placeholder")

        # Safely extract user text
        user_text = None
        for message in raw_request.messages:
            for item in message.content:
                if item.type == "text":
                    user_text = item.text
                    break
        if not user_text:
            raise ValueError("No text content found in request")

        prompt = template.replace("<prompt>", user_text)

        msg = {
            "role": "user",
            "content": prompt,
        }

        # Generate single request ID for entire flow
        request_id = str(uuid.uuid4().hex)

        # Create chat request for preprocessing
        chat_request = ChatCompletionRequest(
            model=raw_request.model,
            messages=[msg],
            stream=True,
            max_tokens=raw_request.max_tokens,
            temperature=raw_request.temperature,
            request_id=request_id,
        )

        # Step 1: Extract multimodal input (needed for PD worker to generate mm_hash)
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

        # Step 2: Send multimodal items to encoder (ECConnector producer)
        mm_items = self._extract_multimodal_items(raw_request.model_dump())

        if mm_items:
            logger.info(
                f"[{request_id}] Encoding {len(mm_items)} multimodal item(s) via encoder..."
            )
            try:
                await self._encode_multimodal_items(
                    prompt=prompt,
                    mm_items=mm_items,
                    model=raw_request.model,
                    request_id=request_id,
                )
            except Exception as e:
                logger.error(f"[{request_id}] Encoder processing failed: {e}")
                error_response = {
                    "error": {
                        "message": f"Encoder processing failed: {str(e)}",
                        "type": "encoder_error",
                        "code": 500,
                    }
                }
                yield error_response
                return

        # Step 2: Preprocess request (parse chat, tokenize, create engine prompt)
        logger.debug(f"[{request_id}] Preprocessing request...")
        (
            request,
            conversation,
            engine_prompt,
            sampling_params,
        ) = await self._parse_raw_request(chat_request)

        # Step 3: Create worker request for PD worker (WITH multimodal_input)
        # PD worker needs multimodal_input to generate mm_hash and lookup EC cache
        # vLLM will see the multimodal items, generate mm_hash, and load from cache
        worker_request = vLLMMultimodalRequest(
            engine_prompt=engine_prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            multimodal_inputs=[
                MultiModalGroup(multimodal_input=multimodal_input)
            ],  # ✓ Keep this so vLLM can generate mm_hash
        )

        logger.debug(
            f"[{request_id}] Forwarding to PD worker (vLLM will load from ECConnector cache using mm_hash)..."
        )

        # Step 4: Send to PD worker (ECConnector consumer - will load from storage)
        response_generator = await self.pd_client.round_robin(
            worker_request.model_dump_json()
        )

        # Step 5: Generate and stream responses (reuse base class method)
        output = self._generate_responses(response_generator, RequestType.CHAT)

        async for response in await self._stream_response(
            request, output, request_id, conversation
        ):
            logger.debug(f"[{request_id}] Generated response: {type(response)}")
            # Reconstruct OpenAI chat response
            if response.startswith("data: [DONE]"):
                break
            response = json.loads(response.lstrip("data: "))
            yield response
