# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import uuid
from collections import defaultdict
from enum import Enum
from typing import AsyncIterator, Final

from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams as VllmSamplingParams
from vllm.tokenizers import TokenizerLike as AnyTokenizer

from dynamo.runtime import Client

from ..handlers import BaseWorkerHandler, build_sampling_params
from ..multimodal_utils import (
    ChatProcessor,
    CompletionsProcessor,
    MultiModalGroup,
    MultiModalInput,
    MyRequestOutput,
    PatchedTokensPrompt,
    ProcessMixIn,
    vLLMMultimodalRequest,
)

logger = logging.getLogger(__name__)

# Multimodal data dictionary keys
IMAGE_URL_KEY: Final = "image_url"
VIDEO_URL_KEY: Final = "video_url"
URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"


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
        pd_worker_client: Client,
        prompt_template: str,
    ):
        self.encode_worker_client = encode_worker_client
        self.pd_worker_client = pd_worker_client
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
        raw_request,
        multimodal_inputs,
        context,
    ):
        request_id = str(uuid.uuid4().hex)

        # Build sampling params from request using shared utility
        sampling_params = build_sampling_params(
            raw_request, self.default_sampling_params
        )

        # [gluo WIP] encoder doesn't really need any of this
        encode_request = vLLMMultimodalRequest(
            engine_prompt=PatchedTokensPrompt(prompt_token_ids=[]),
            sampling_params=VllmSamplingParams(),
            # sampling_params=sampling_params,
            request_id=request_id,
            multimodal_inputs=[],
        )

        for mm_type, urls in multimodal_inputs.items():
            for url in urls:
                multimodal_input = MultiModalInput()
                if mm_type == IMAGE_URL_KEY:
                    multimodal_input.image_url = url
                elif mm_type == VIDEO_URL_KEY:
                    multimodal_input.video_url = url
                    # [gluo NOTE] panic for now as encoder here is for image only
                    raise ValueError("Video URL not supported in encode worker yet")
                encode_request.multimodal_inputs.append(
                    MultiModalGroup(multimodal_input=multimodal_input)
                )

        # model_dump_json() serializes the request to JSON string
        # This API could accept Pydantic class, but SamplingParams
        # in vLLMMultimodalRequest is not a Pydantic class and will
        # cause TypeError: unsupported type SamplingParams
        response_generator = await self.encode_worker_client.round_robin(
            encode_request.model_dump_json()
        )
        # Gather transformed requests
        worker_request = vLLMMultimodalRequest(
            engine_prompt=PatchedTokensPrompt(
                prompt_token_ids=raw_request["token_ids"]
            ),
            sampling_params=sampling_params,
            request_id=request_id,
            multimodal_inputs=[],  # will be filled in next
        )
        async for response in response_generator:
            logger.debug(f"Received response from encode worker: {response}")
            output = vLLMMultimodalRequest.model_validate_json(response.data())
            worker_request.multimodal_inputs.extend(output.multimodal_inputs)

        response_generator = await self.pd_worker_client.round_robin(
            worker_request.model_dump_json(), context=context
        )

        # [gluo FIXME] <im_end> being returned
        async for output in self._generate_responses(response_generator):
            yield output

    # This method is used to process the responses from the engine generator.
    async def _generate_responses(
        self,
        response_generator: AsyncIterator[RequestOutput],
    ):
        # [gluo WIP] modified from handler.py (BaseWorkerHandler.generate_tokens)
        num_output_tokens_so_far = 0
        try:
            async for resp in response_generator:
                # Deserialize the response from the engine
                # Creates correct vLLM objects for each field
                output = MyRequestOutput.model_validate_json(resp.data())

                # OpenAIServingChat.chat_completion_stream_generator() method expects a RequestOutput object
                res = RequestOutput(
                    request_id=output.request_id,
                    prompt=output.prompt,
                    prompt_token_ids=output.prompt_token_ids,
                    prompt_logprobs=output.prompt_logprobs,
                    outputs=output.outputs,
                    finished=output.finished,
                    metrics=output.metrics,
                )

                output = res.outputs[0]
                next_total_toks = len(output.token_ids)
                out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}

                # Extract logprobs for new tokens if available
                log_probs, top_logprobs = BaseWorkerHandler._extract_logprobs(
                    output, num_output_tokens_so_far
                )
                if log_probs is not None:
                    out["log_probs"] = log_probs
                if top_logprobs is not None:
                    out["top_logprobs"] = top_logprobs

                if output.finish_reason:
                    out["finish_reason"] = output.finish_reason
                    out["completion_usage"] = BaseWorkerHandler._build_completion_usage(
                        request_output=res
                    )
                if output.stop_reason:
                    out["stop_reason"] = output.stop_reason
                yield out
                num_output_tokens_so_far = next_total_toks
        except asyncio.CancelledError:
            # raise EngineShGeneratorExit when engine exits so that frontend can migrate the request
            raise GeneratorExit(
                "Decode engine was shut down during token generation"
            ) from None

    def _extract_multimodal_data(self, request):
        """
        Extract and decode multimodal data from PreprocessedRequest.
        """
        # [gluo NOTE] modified from components/src/dynamo/vllm/handlers.py
        if "multi_modal_data" not in request or request["multi_modal_data"] is None:
            return {}

        # [gluo FIXME] add this security option
        # Security check: reject multimodal data if not explicitly enabled
        # if not self.enable_multimodal:
        #     raise ValueError(
        #         "Received multimodal data but multimodal processing is not enabled. "
        #         "Use --enable-multimodal flag to enable multimodal processing."
        #     )

        mm_map = request["multi_modal_data"]
        multimodal_inputs = defaultdict(list)

        for mm_type in [IMAGE_URL_KEY, VIDEO_URL_KEY]:
            for item in mm_map.get(mm_type, []):
                if isinstance(item, dict) and URL_VARIANT_KEY in item:
                    multimodal_inputs[mm_type].append(item[URL_VARIANT_KEY])
                elif isinstance(item, dict) and DECODED_VARIANT_KEY in item:
                    # Decoded support from PRs #3971/#3988 (frontend decoding + NIXL transfer)
                    # Will contain NIXL metadata for direct memory access
                    # TODO: Implement NIXL read when PRs merge
                    logger.warning(
                        "Decoded multimodal data not yet supported in standard worker"
                    )

        return multimodal_inputs

    # The generate endpoint will be used by the frontend to handle incoming requests.
    async def generate(self, request, context):
        logger.debug(f"Got preprocessed request: {request}")

        # Extract multimodal inputs for dispatching to encode worker
        multimodal_inputs = self._extract_multimodal_data(request)

        if not multimodal_inputs:
            raise ValueError("Either image URL or video URL is required")
        elif len(multimodal_inputs) > 1:
            raise ValueError(
                "Only one of image URL or video URL is supported per request"
            )

        async for response in self._generate(request, multimodal_inputs, context):
            yield response
