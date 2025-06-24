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

import asyncio
import logging
import os
import signal
from typing import Optional, Union

import connect
import torch
from components.disagg_router import PyDisaggregatedRouter
from components.video_encode_worker import VllmEncodeWorker
from components.video_prefill_worker import VllmPrefillWorker
from transformers import AutoProcessor
from utils.logging import check_required_workers
from utils.nixl import NixlMetadataStore
from utils.prefill_queue import PrefillQueue
from utils.protocol import EncodeRequest, MyRequestOutput, vLLMMultimodalRequest
from utils.vllm import parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest
from vllm.sampling_params import RequestOutputKind

from dynamo.sdk import async_on_start, depends, dynamo_context, endpoint, service

logger = logging.getLogger(__name__)

# Constants for the shape and dtype of the INCOMING FRAMES tensor from EncodeWorker.
# IMPORTANT ASSUMPTION: EncodeWorker must provide frames of this fixed shape and dtype.
INCOMING_FRAMES_DTYPE = torch.uint8
INCOMING_FRAMES_DEVICE = "cpu"


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmDecodeWorker:
    # For disaggregated serving, we need to link the prefill worker to the vllm worker
    prefill_worker = depends(VllmPrefillWorker)
    # For aggregated serving, we need to link the encode worker to the vllm worker.
    encode_worker = depends(VllmEncodeWorker)

    def _expand_video_tokens_in_prompt(
        self,
        original_tokens: list[int],
        num_frames_to_expand_to: int,
        image_token_id: int,  # This should be the ID from hf_processor.tokenizer
        add_dummy_tokens: bool,
        dummy_token_id: int = 0,
        num_dummy_tokens_per_frame: int = 0,
    ) -> list[int]:
        """
        Expands the first occurrence of image_token_id in original_tokens
        to num_frames_to_expand_to occurrences. Optionally adds dummy tokens.
        """
        expanded_prompt_list = []
        token_expanded_successfully = False
        for token_id_val in original_tokens:
            if token_id_val == image_token_id and not token_expanded_successfully:
                for _ in range(num_frames_to_expand_to):
                    expanded_prompt_list.append(image_token_id)
                    if add_dummy_tokens:
                        dummy_tokens_to_add = [
                            dummy_token_id
                        ] * num_dummy_tokens_per_frame
                        expanded_prompt_list.extend(dummy_tokens_to_add)

                token_expanded_successfully = True
            else:
                expanded_prompt_list.append(token_id_val)

        if not token_expanded_successfully:
            # If the specific video token ID isn't found (e.g. prompt had no video placeholder),
            # it implies the original prompt didn't intend for video.
            # This might be an issue if video data is expected.
            logger.warning(
                f"Image token ID {image_token_id} for expansion not found in prompt tokenized by hf_processor. Prompt: {original_tokens}. This might be okay if no video was intended in this specific prompt structure."
            )
            return list(original_tokens)  # Return original if no video token to expand

        return expanded_prompt_list

    def __init__(self):
        self.client = None
        self.min_workers = 1
        self.disaggregated_router: Optional[PyDisaggregatedRouter] = None
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.model_path = self.engine_args.model
        self.num_sampled_frames = getattr(self.engine_args, "num_sampled_frames", 8)
        self.frame_height = getattr(self.engine_args, "frame_height", 336)
        self.frame_width = getattr(self.engine_args, "frame_width", 336)
        self.frame_channels = getattr(self.engine_args, "frame_channels", 3)
        self.dummy_token_id = getattr(self.engine_args, "dummy_token_id", 0)
        self.video_token_id = getattr(self.engine_args, "video_token_id", 32000)
        self.dummy_tokens_per_frame = getattr(
            self.engine_args, "dummy_tokens_per_frame", 144
        )
        self.do_remote_prefill = self.engine_args.remote_prefill
        self.model_name = (
            self.engine_args.served_model_name
            if self.engine_args.served_model_name is not None
            else "vllm"
        )
        self._prefill_queue_nats_server = os.getenv(
            "NATS_SERVER", "nats://localhost:4222"
        )
        self._prefill_queue_stream_name = self.model_name
        logger.info(
            f"Prefill queue: {self._prefill_queue_nats_server}:{self._prefill_queue_stream_name}"
        )

        if self.engine_args.remote_prefill:
            if self.engine_args.enable_chunked_prefill is not False:
                logger.info("Chunked prefill is not supported yet, setting to False")
                self.engine_args.enable_chunked_prefill = False

            if self.engine_args.preemption_mode != "swap":
                logger.info("Preemption mode is not supported yet, setting to swap")
                self.engine_args.preemption_mode = "swap"

            if self.engine_args.pipeline_parallel_size != 1:
                logger.info("Pipeline parallel size is not supported yet, setting to 1")
                self.engine_args.pipeline_parallel_size = 1

        if self.engine_args.router == "kv":
            raise NotImplementedError(
                "Multimodal requests are not supported for kv router mode"
            )

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

    @async_on_start
    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")

        if self.engine_args.router == "kv":
            raise NotImplementedError(
                "Multimodal requests are not supported for kv router mode"
            )

        # Load the Hugging Face processor
        try:
            self.hf_processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            logger.info(f"Successfully loaded AutoProcessor from: {self.model_path}")
            if (
                not hasattr(self.hf_processor, "tokenizer")
                or self.hf_processor.tokenizer is None
            ):
                logger.warning(
                    f"Loaded AutoProcessor from {self.model_path} but it does not have a 'tokenizer' attribute or it is None."
                )
        except Exception as e:
            logger.error(
                f"Failed to load AutoProcessor from {self.model_path}: {e}",
                exc_info=True,
            )
            # Depending on the desired behavior, you might want to raise the error
            # or allow the worker to start without a processor if it's optional for some paths.
            # For this change, processor is critical.
            raise RuntimeError(f"Failed to initialize AutoProcessor: {e}") from e

        runtime = dynamo_context["runtime"]

        # Common setup for interacting with EncodeWorker (NIXL, client)
        # This is needed for aggregated mode OR for local prefill in disaggregated mode.
        enc_comp_ns, enc_comp_name = VllmEncodeWorker.dynamo_address()  # type: ignore
        self.encode_worker_client = (
            await runtime.namespace(enc_comp_ns)
            .component(enc_comp_name)
            .endpoint("encode")
            .client()
        )
        # Initialize the connector for RDMA transfers within the specified namespace.
        self._connector = connect.Connector(runtime=runtime, namespace=enc_comp_ns)
        await self._connector.initialize()

        # NIXL buffer for receiving raw video frames.
        incoming_frames_shape = (
            self.num_sampled_frames,
            self.frame_height,
            self.frame_width,
            self.frame_channels,
        )
        raw_frames_tensor = torch.empty(
            incoming_frames_shape,
            dtype=INCOMING_FRAMES_DTYPE,
            device=INCOMING_FRAMES_DEVICE,
        )
        # Create a descriptor for the tensor to make it available for remote access.
        descriptor = connect.Descriptor(raw_frames_tensor)
        # Register the memory with the connector, making it discoverable.
        descriptor.register_memory(self._connector)
        self._frames_descriptor = (raw_frames_tensor, descriptor)

        await check_required_workers(self.encode_worker_client, self.min_workers)

        if self.do_remote_prefill:  # Disaggregated mode specific setup
            metadata = self.engine_client.nixl_metadata
            metadata_store = NixlMetadataStore("dynamo", runtime)
            await metadata_store.put(metadata.engine_id, metadata)

            if self.engine_args.conditional_disagg:
                self.disaggregated_router = PyDisaggregatedRouter(
                    runtime,
                    self.model_name,
                    max_local_prefill_length=self.engine_args.max_local_prefill_length,
                    max_prefill_queue_size=self.engine_args.max_prefill_queue_size,
                )
                await self.disaggregated_router.async_init()
            else:
                self.disaggregated_router = (
                    None  # Always remote prefill if not conditional_disagg
                )

            # embedding_size is used for dummy token calculation in remote prefill case.
            # For LLaVA-NeXT-Video, the model architecture processes each frame into a 12x12 grid
            # of visual tokens, resulting in 144 tokens per frame. This is a fixed architectural
            # constant. For more details on the vision tower architecture, see the LLaVA-1.5 paper
            # which LLaVA-NeXT is based on: https://arxiv.org/abs/2310.03744
            self.embedding_size = 144
            logger.info(
                f"Disaggregated mode: Using LLaVA-NeXT-Video embedding size: {self.embedding_size}"
            )

        else:  # Aggregated mode specific setup
            self.disaggregated_router = (
                None  # No disaggregated router in aggregated mode
            )
            logger.info(
                "Aggregated mode: VllmDecodeWorker will handle multimodal data directly via NIXL."
            )

        logger.info("Initialization complete.")

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Received signal {signum}, shutting down")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.close()
            logger.info("Shutdown complete.")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()

    def get_remote_prefill_request_callback(self):
        async def callback(request: RemotePrefillRequest):
            try:
                async with PrefillQueue.get_instance(
                    nats_server=self._prefill_queue_nats_server,
                    stream_name=self._prefill_queue_stream_name,
                ) as prefill_queue:
                    await prefill_queue.enqueue_prefill_request(request)
                logger.info(
                    f"DecodeWorker {request.request_id}: Successfully enqueued remote prefill request."
                )
            except Exception as e:
                logger.error(
                    f"DecodeWorker {request.request_id}: Failed to enqueue remote prefill request: {e}",
                    exc_info=True,
                )

        return callback

    @endpoint()
    async def generate(self, request: vLLMMultimodalRequest):
        request_id = request.request_id
        video_url = request.video_url  # Video path for EncodeWorker
        # TODO: Fix existing tokenizer <video> not found error and remove this.
        user_text_prompt = request.engine_prompt.get(
            "text_prompt", "Describe the video."
        )
        logger.info(
            f"Received multimodal request {{ id: {request_id} }} with user text: '{user_text_prompt}'."
        )

        # Constants for token manipulation
        # For LLaVA-NeXT-Video models, the video token ID is 32000, not 32001
        # 32001 is for image tokens in LLaVA-NeXT-Video, 32000 is for video tokens
        VIDEO_TOKEN_ID_FOR_EXPANSION = 32000
        DUMMY_TOKEN_ID = 0

        # Variables to be set based on processing path
        prompt_argument_for_vllm: Union[str, TokensPrompt]
        current_received_multimodal_data_tensor: Optional[torch.Tensor] = None
        current_remote_prefill_params: Optional[RemotePrefillParams] = None
        multi_modal_data_for_engine: Optional[dict] = None

        if self.do_remote_prefill:
            logger.info(f"Disaggregated mode: request {{ id: {request_id} }}.")
            # Tokenize the prompt string to get base IDs for router length check and potential remote prefill manipulation
            base_prompt_ids_for_router = request.engine_prompt["prompt_token_ids"]
            if (
                isinstance(base_prompt_ids_for_router, list)
                and len(base_prompt_ids_for_router) > 0
                and isinstance(base_prompt_ids_for_router[0], list)
                and len(base_prompt_ids_for_router) == 1
            ):
                base_prompt_ids_for_router = base_prompt_ids_for_router[0]

            should_prefill_remotely_decision = True
            if self.disaggregated_router:
                async with PrefillQueue.get_instance(
                    nats_server=self._prefill_queue_nats_server,
                    stream_name=self._prefill_queue_stream_name,
                ) as prefill_queue:
                    prefill_queue_size = await prefill_queue.get_queue_size()
                should_prefill_remotely_decision = (
                    await self.disaggregated_router.prefill_remote(
                        len(base_prompt_ids_for_router),
                        request.prefix_hit_rate,
                        prefill_queue_size,
                    )
                )

            if should_prefill_remotely_decision:
                logger.info(
                    f"Disaggregated: Prefilling REMOTELY for request {{ id: {request_id} }} (orig prompt len {len(base_prompt_ids_for_router)})"
                )
                current_remote_prefill_params = RemotePrefillParams(
                    is_remote_prefill=True,
                    remote_prefill_request_callback=self.get_remote_prefill_request_callback(),
                    multimodal_data_source={"video_url": video_url},
                )
                num_dummies = self.embedding_size - 1
                # For remote prefill, expand the *single* video token from base_prompt_ids and add dummies
                expanded_and_dummied_ids = self._expand_video_tokens_in_prompt(
                    base_prompt_ids_for_router,  # Use the tokenized output of chat_template
                    self.num_sampled_frames,
                    VIDEO_TOKEN_ID_FOR_EXPANSION,
                    add_dummy_tokens=True,
                    dummy_token_id=DUMMY_TOKEN_ID,
                    num_dummy_tokens_per_frame=num_dummies,
                )
                prompt_argument_for_vllm = TokensPrompt(
                    prompt_token_ids=expanded_and_dummied_ids, multi_modal_data=None
                )
                multi_modal_data_for_engine = None  # Handled by prefill worker
            else:  # Local prefill in disaggregated mode
                logger.info(
                    f"Disaggregated: Prefilling LOCALLY for request {{ id: {request_id} }} (orig prompt len {len(base_prompt_ids_for_router)})"
                )
                raw_frames_tensor_from_nixl, desc = self._frames_descriptor
                # Create a writable operation handle for the remote EncodeWorker.
                # This allows the EncodeWorker to write directly into this worker's `raw_frames_tensor_from_nixl`.
                with self._connector.create_writable(desc) as writable:
                    enc_req = EncodeRequest(
                        request_id=request_id,
                        video_url=video_url,
                        # Serialize the writable handle to send it to the EncodeWorker.
                        serialized_request=writable.to_serialized(),
                    )
                    async for _ in await self.encode_worker_client.round_robin(
                        enc_req.model_dump_json()
                    ):
                        pass
                    # Wait for the remote write from the EncodeWorker to complete.
                    await writable.wait_for_completion()
                current_received_multimodal_data_tensor = raw_frames_tensor_from_nixl
                # The vLLM engine's processor for raw visual data expects a CPU-based NumPy array.
                # Therefore, we must first move the tensor from the GPU to the CPU memory
                # before converting it to a NumPy array.
                # See vLLM's official example for raw image inputs: https://github.com/vllm-project/vllm/blob/main/examples/llava_example.py
                video_numpy = current_received_multimodal_data_tensor.numpy()
                multi_modal_data_for_engine = {"video": video_numpy}
                prompt_argument_for_vllm = request.engine_prompt[
                    "prompt_token_ids"
                ]  # Pass raw string to vLLM
                current_remote_prefill_params = None
        else:  # AGGREGATED MODE
            logger.info(
                f"Aggregated mode: request {{ id: {request_id} }}. Fetching frames directly."
            )
            raw_frames_tensor_from_nixl, desc = self._frames_descriptor
            # Create a writable operation handle for the remote EncodeWorker.
            # This allows the EncodeWorker to write directly into this worker's `raw_frames_tensor_from_nixl`.
            with self._connector.create_writable(desc) as writable:
                enc_req = EncodeRequest(
                    request_id=request_id,
                    video_url=video_url,
                    # Serialize the writable handle to send it to the EncodeWorker.
                    serialized_request=writable.to_serialized(),
                )
                async for _ in await self.encode_worker_client.round_robin(
                    enc_req.model_dump_json()
                ):
                    pass
                # Wait for the remote write from the EncodeWorker to complete.
                await writable.wait_for_completion()
            current_received_multimodal_data_tensor = raw_frames_tensor_from_nixl
            # The vLLM engine's processor for raw visual data expects a CPU-based NumPy array.
            # Therefore, we must first move the tensor from the GPU to the CPU memory
            # before converting it to a NumPy array.
            # See vLLM's official example for raw image inputs: https://github.com/vllm-project/vllm/blob/main/examples/llava_example.py
            video_numpy = current_received_multimodal_data_tensor.numpy()
            multi_modal_data_for_engine = {"video": video_numpy}
            prompt_argument_for_vllm = request.engine_prompt[
                "prompt_token_ids"
            ]  # Pass raw string to vLLM
            current_remote_prefill_params = None

        request.sampling_params.output_kind = RequestOutputKind.DELTA

        # Prepare the first argument for vLLM engine's generate call
        final_vllm_input: Union[str, dict]
        if isinstance(prompt_argument_for_vllm, dict):
            # This handles the remote prefill path where we have a TokensPrompt,
            # which is a dict-like object.
            final_vllm_input = prompt_argument_for_vllm
        elif isinstance(prompt_argument_for_vllm, list):
            # This handles the local prefill (aggregated or disaggregated) path
            # where we have a list of token IDs and raw video data.
            final_vllm_input = {
                "prompt_token_ids": prompt_argument_for_vllm,
                "multi_modal_data": multi_modal_data_for_engine,
            }
        else:
            logger.error(
                f"Unexpected type for prompt_argument_for_vllm: {type(prompt_argument_for_vllm)}"
            )
            raise TypeError("Invalid type for vLLM prompt argument.")

        async for response in self.engine_client.generate(
            final_vllm_input,  # This is now the prompts argument (dict)
            sampling_params=request.sampling_params,
            request_id=request.request_id,
            remote_prefill_params=current_remote_prefill_params,
        ):
            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
            ).model_dump_json()
