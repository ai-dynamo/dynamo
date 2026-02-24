# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from vllm.inputs.data import TokensPrompt

import dynamo.nixl_connect as connect
from dynamo.runtime import DistributedRuntime

from ..args import Config
from ..constants import DisaggregationMode
from ..handlers import BaseWorkerHandler
from ..multimodal_utils import MyRequestOutput, vLLMMultimodalRequest
from ..multimodal_utils.model import construct_qwen_decode_mm_data, is_qwen_vl_model

logger = logging.getLogger(__name__)


class MultimodalDecodeWorkerHandler(BaseWorkerHandler):
    """Decode worker for disaggregated multimodal serving"""

    def __init__(
        self,
        runtime,
        engine_client,
        config: Config,
        shutdown_event=None,
        generate_endpoint=None,
    ):
        # Get default_sampling_params from config
        default_sampling_params = (
            config.engine_args.create_model_config().get_diff_sampling_param()
        )

        # Call BaseWorkerHandler.__init__ with proper parameters
        super().__init__(
            runtime,
            engine_client,
            default_sampling_params,
            enable_multimodal=config.enable_multimodal,
            generate_endpoint=generate_endpoint,
            config=config,
            shutdown_event=shutdown_event,
        )

        self.config = config
        self.enable_disagg = config.disaggregation_mode == DisaggregationMode.PREFILL

    async def async_init(self, runtime: DistributedRuntime):
        """Async initialization - connector needs async setup"""
        self._connector = connect.Connector()
        logger.info("Multimodal Decode Worker async initialization completed.")

    async def generate(self, request: vLLMMultimodalRequest, context):
        logger.debug(f"Got raw request: {request}")
        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received decode request: {{ id: {request.request_id} }}.")

        # For Qwen VL models with mRoPE, we need to pass multi_modal_data containing
        # image_grid_thw for position embeddings calculation. The decode worker
        # receives the ORIGINAL unexpanded prompt (with placeholders), and vLLM
        # will expand it using the multi_modal_data, ensuring the block count
        # matches what prefill computed.
        #
        # We pass unique placeholder embeddings (seeded by request_id) since the
        # actual embeddings are already in the KV cache from prefill. The unique
        # values prevent incorrect prefix cache matches between different images.
        multi_modal_data = None
        if is_qwen_vl_model(self.config.model):
            image_grid_thw = getattr(request, "image_grid_thw", None)
            embeddings_shape = getattr(request, "embeddings_shape", None)
            if image_grid_thw is None or embeddings_shape is None:
                logger.warning(
                    "Missing Qwen VL decode fields (image_grid_thw/embeddings_shape); "
                    "skipping multi_modal_data construction."
                )
            else:
                multi_modal_data = construct_qwen_decode_mm_data(
                    image_grid_thw, embeddings_shape, request.request_id
                )

        lora_request = self._resolve_lora_request(request.model)
        gen = self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=request.engine_prompt["prompt_token_ids"],
                multi_modal_data=multi_modal_data,
            ),
            sampling_params=request.sampling_params,
            request_id=request.request_id,
            lora_request=lora_request,
        )

        async for response in gen:
            logger.debug(f"Response kv_transfer_params: {response.kv_transfer_params}")
            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
                metrics=response.metrics,
                kv_transfer_params=response.kv_transfer_params,
            ).model_dump_json()
