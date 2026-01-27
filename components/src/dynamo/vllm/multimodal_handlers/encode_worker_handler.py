# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import shutil
import time
from io import BytesIO
from queue import Queue
from typing import AsyncGenerator, AsyncIterator, Optional

import av
import safetensors
import torch
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.multimodal.hasher import MultiModalHasher
from vllm.sampling_params import SamplingParams

import dynamo.nixl_connect as connect
from dynamo.runtime import Client, DistributedRuntime

from ..multimodal_utils import (
    AudioLoader,
    ImageLoader,
    VLLMNativeEncoderRequest,
    VLLMNativeEncoderResponse,
    calculate_frame_sampling_indices,
    encode_image_embeddings,
    get_embedding_hash,
    get_encoder_components,
    get_video_metadata,
    load_video_content,
    load_vision_model,
    open_video_container,
    prepare_tensor_for_rdma,
    read_video_pyav,
    resize_video_frames,
    vLLMMultimodalRequest,
)

logger = logging.getLogger(__name__)

try:
    import cupy as array_module

    if not array_module.cuda.is_available():
        raise ImportError("CUDA is not available.")
    DEVICE = "cuda"
    logger.info("Using cupy for array operations (GPU mode).")
except ImportError as e:
    logger.warning(f"Failed to import cupy, falling back to numpy: {e}.")
    import numpy as array_module

    DEVICE = "cpu"

CACHE_SIZE_MAXIMUM = 8

TRANSFER_LOCAL = int(os.getenv("TRANSFER_LOCAL", 1))


class EncodeWorkerHandler:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        pd_worker_client: Client,
    ) -> None:
        self.pd_worker_client = pd_worker_client
        self.engine_args = engine_args
        self.model = self.engine_args.model

        self.image_loader = ImageLoader(cache_size=CACHE_SIZE_MAXIMUM)
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model, trust_remote_code=True
        )
        self.vision_model = load_vision_model(self.model)
        self.min_workers = 1

        # Get encoder components for the model
        self.vision_encoder, self.projector = get_encoder_components(
            self.model, self.vision_model
        )
        self._connector = None
        self._accumulated_time = 0.0
        self._processed_requests = 0
        self.readables = []
        self.cached_embeddings = {}

    def cleanup(self):
        pass

    async def async_init(self, runtime: DistributedRuntime):
        """Initialize the connector for RDMA transfers"""
        logger.info("Encode worker startup started.")
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()
        logger.info("Encode worker startup completed.")

    async def generate(
        self, request: vLLMMultimodalRequest, context
    ) -> AsyncIterator[str]:
        logger.debug(f"Got raw request: {request}")
        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received encode request: {{ id: {request.request_id} }}.")

        request_id = request.request_id

        # The following steps encode the requested image and provided useful embeddings.
        # 1. Open the image from the provided URL.
        # 2. Process the image using the image processor.
        # 3. Run the image through the vision model's vision tower.
        # 4. Run the results of the vision tower through the multi-modal projector.
        # 5. Create a descriptor for the embeddings.
        # 6. Create a write operation using the serialized request and the descriptor.
        # 7. Await for the write operation to complete.
        # 8. Yield the encode response.

        try:
            time_start = time.perf_counter()
            for idx in range(len(request.multimodal_inputs)):
                if not request.multimodal_inputs[idx].multimodal_input.image_url:
                    raise ValueError("image_url is required for the encode worker.")

                image_url = request.multimodal_inputs[idx].multimodal_input.image_url
                # see if we have local cache
                if image_url in self.cached_embeddings:
                    (
                        embedding_key,
                        image_grid_thw,
                        embeddings_shape,
                    ) = self.cached_embeddings[image_url]
                    # [gluo FIXME] need mechanism to clean up local files
                    request.multimodal_inputs[
                        idx
                    ].serialized_request = (
                        f"/tmp/encoder_cache.{embedding_key}.safetensors"
                    )
                    request.multimodal_inputs[idx].multimodal_input.image_url = None
                    request.multimodal_inputs[idx].image_grid_thw = image_grid_thw
                    request.multimodal_inputs[idx].embeddings_shape = embeddings_shape
                    continue

                image = await self.image_loader.load_image(image_url)

                logger.debug(
                    f"Processing image {image_url} for request: {{ id: {request_id} }}"
                )
                image_embeds = self.image_processor(images=image, return_tensors="pt")

                # Encode the image embeddings using model-specific encoder
                embeddings = encode_image_embeddings(
                    model_name=self.model,
                    image_embeds=image_embeds,
                    vision_encoder=self.vision_encoder,
                    projector=self.projector,
                )

                image_grid_thw = (
                    image_embeds["image_grid_thw"].tolist()
                    if "image_grid_thw" in image_embeds
                    else None
                )
                logger.debug(
                    f"Pixel values stats: mean={image_embeds['pixel_values'].mean().item()}, std={image_embeds['pixel_values'].std().item()}, min={image_embeds['pixel_values'].min().item()}, max={image_embeds['pixel_values'].max().item()}"
                )

                # Move embeddings to CPU for NIXL transfer to avoid UCX/InfiniBand issues
                embeddings_cpu = embeddings.cpu()

                request.multimodal_inputs[idx].image_grid_thw = image_grid_thw
                request.multimodal_inputs[idx].embeddings_shape = tuple(
                    embeddings.shape
                )

                if TRANSFER_LOCAL:
                    embedding_key = get_embedding_hash(image_url)
                    logger.debug(
                        f"ENCODER: saving local safetensors file with key {embedding_key}, {embeddings_cpu.numel()} * {embeddings_cpu.element_size()} bytes"
                    )
                    tensors = {"ec_cache": embeddings_cpu}
                    safetensors.torch.save_file(
                        tensors, f"/tmp/encoder_cache.{embedding_key}.safetensors"
                    )
                    # [gluo FIXME] need mechanism to clean up local files
                    request.multimodal_inputs[
                        idx
                    ].serialized_request = (
                        f"/tmp/encoder_cache.{embedding_key}.safetensors"
                    )
                    self.cached_embeddings[image_url] = (
                        embedding_key,
                        request.multimodal_inputs[idx].image_grid_thw,
                        request.multimodal_inputs[idx].embeddings_shape,
                    )
                else:
                    # [gluo FIXME] nixl_connector path needs to be update to handle multiple embeddings
                    descriptor = connect.Descriptor(embeddings_cpu)
                    self.readables.append(
                        await self._connector.create_readable(descriptor)
                    )
                    request.multimodal_inputs[idx].serialized_request = self.readables[
                        -1
                    ].metadata()

                # Clear the image URL as hint that the image is passed as embeddings.
                request.multimodal_inputs[idx].multimodal_input.image_url = None

            logger.debug(f"Request: {request.model_dump_json()}")

            time_end = time.perf_counter()
            self._accumulated_time += time_end - time_start
            self._processed_requests += 1
            logger.debug(
                f"Encoded image(s) for request {{ id: {request_id} }} in {time_end - time_start:.4f} seconds. "
                f"Average encoding time: {self._accumulated_time / self._processed_requests:.4f} seconds over {self._processed_requests} requests."
            )

            # Yield transformed request back
            yield request.model_dump_json()

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            raise


class AudioEncodeWorkerHandler:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        pd_worker_client: Client,
    ) -> None:
        self.pd_worker_client = pd_worker_client
        self.engine_args = engine_args
        self.model = self.engine_args.model

        self.audio_loader = AudioLoader(cache_size=CACHE_SIZE_MAXIMUM)
        self.audio_processor = AutoProcessor.from_pretrained(
            self.model, trust_remote_code=True
        )
        self.audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model, device_map="auto", torch_dtype=torch.float16
        ).eval()

        self._connector = None
        self.readables = []

    def _get_audio_embeddings(self, audio_features):
        input_features, feature_attention_mask = (
            audio_features.input_features,
            audio_features.feature_attention_mask,
        )
        with torch.no_grad():
            (
                audio_feat_lengths,
                audio_output_lengths,
            ) = self.audio_model.audio_tower._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            batch_size, _, max_mel_seq_len = input_features.shape
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            seq_range = (
                torch.arange(
                    0,
                    max_seq_len,
                    dtype=audio_feat_lengths.dtype,
                    device=audio_feat_lengths.device,
                )
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(
                batch_size, max_seq_len
            )
            padding_mask = seq_range >= lengths_expand

            audio_attention_mask_ = padding_mask.view(
                batch_size, 1, 1, max_seq_len
            ).expand(batch_size, 1, max_seq_len, max_seq_len)
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.audio_model.audio_tower.conv1.weight.dtype,
                device=self.audio_model.audio_tower.conv1.weight.device,
            )
            audio_attention_mask[audio_attention_mask_] = float("-inf")

            audio_outputs = self.audio_model.audio_tower(
                input_features, attention_mask=audio_attention_mask
            )
            selected_audio_feature = audio_outputs.last_hidden_state
            audio_features = self.audio_model.multi_modal_projector(
                selected_audio_feature
            )

            num_audios, max_audio_tokens, embed_dim = audio_features.shape
            audio_features_mask = torch.arange(
                max_audio_tokens, device=audio_output_lengths.device
            )[None, :]
            audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
            audio_features = audio_features[audio_features_mask]

            return audio_features

    def cleanup(self):
        pass

    async def async_init(self, runtime: DistributedRuntime):
        logger.info("Audio encode worker startup started.")
        self._connector = connect.Connector()
        logger.info("Audio encode worker startup completed.")

    async def generate(
        self, request: vLLMMultimodalRequest, context
    ) -> AsyncIterator[str]:
        logger.debug(f"Got raw request: {request}")
        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received audio encode request: {{ id: {request.request_id} }}.")

        request_id = request.request_id

        try:
            for idx, mm_group in enumerate(request.multimodal_inputs):
                if not mm_group.multimodal_input.audio_url:
                    raise ValueError(
                        "audio_url is required for the audio encode worker."
                    )

                audio, _ = await self.audio_loader.load_audio(
                    mm_group.multimodal_input.audio_url
                )
                audio_features = self.audio_processor(
                    text="test<|AUDIO|>",
                    audio=audio,
                    return_tensors="pt",
                    padding=False,
                )
                audio_embeddings = self._get_audio_embeddings(audio_features)
                audio_embeddings = audio_embeddings.cpu()

                descriptor = connect.Descriptor(audio_embeddings)
                self.readables.append(await self._connector.create_readable(descriptor))

                request.multimodal_inputs[idx].serialized_request = self.readables[
                    -1
                ].metadata()
                request.multimodal_inputs[idx].multimodal_input.audio_url = None
                request.multimodal_inputs[idx].embeddings_shape = tuple(
                    audio_embeddings.shape
                )

            yield request.model_dump_json()

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            raise


class VideoEncodeWorkerHandler:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        pd_worker_client: Client,
        num_frames_to_sample: int = 8,
    ) -> None:
        self.pd_worker_client = pd_worker_client
        self.engine_args = engine_args
        self.model = self.engine_args.model
        self.min_workers = 1

        self.num_frames_to_sample = num_frames_to_sample
        self.frame_height = 336
        self.frame_width = 336
        self.frame_channels = 3
        self._video_content_cache: dict[str, BytesIO] = {}
        self._cache_queue: Queue[str] = Queue(maxsize=CACHE_SIZE_MAXIMUM)

        self._http_timeout = 60.0
        self._connector = None
        self.readables = []

    def cleanup(self):
        pass

    async def async_init(self, runtime: DistributedRuntime):
        logger.info("Video encode worker startup started.")
        self._connector = connect.Connector()
        logger.info("Video encode worker startup completed.")

    async def generate(
        self, request: vLLMMultimodalRequest, context
    ) -> AsyncIterator[str]:
        logger.debug(f"Got raw request: {request}")
        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received video encode request: {{ id: {request.request_id} }}.")

        request_id = request.request_id

        for idx, mm_group in enumerate(request.multimodal_inputs):
            video_url = mm_group.multimodal_input.video_url
            if video_url is None:
                raise ValueError("video_url is required for the video encode worker.")

            container: Optional[av.container.InputContainer] = None
            try:
                video_content_stream = await load_video_content(
                    video_url,
                    self._video_content_cache,
                    self._cache_queue,
                    self._http_timeout,
                )

                container = await open_video_container(video_content_stream, video_url)
                if not container or not container.streams.video:
                    raise ValueError(f"No video stream in {video_url}.")

                total_frames, duration_sec = get_video_metadata(container)
                indices = calculate_frame_sampling_indices(
                    total_frames, self.num_frames_to_sample, duration_sec, video_url
                )

                clip_np = await read_video_pyav(container, indices)
                if clip_np.size == 0:
                    raise ValueError(
                        f"Failed to extract any video frames from {video_url} for indices {indices.tolist()}."
                    )

                frames_tensor_orig_res = torch.from_numpy(clip_np)
                resized_frames_tensor_hwc = resize_video_frames(
                    frames_tensor_orig_res, self.frame_height, self.frame_width
                )
                tensor_for_descriptor = prepare_tensor_for_rdma(
                    resized_frames_tensor_hwc, request_id
                )

                request.multimodal_inputs[idx].embeddings_shape = tuple(
                    tensor_for_descriptor.shape
                )
                descriptor = connect.Descriptor(tensor_for_descriptor)
                self.readables.append(await self._connector.create_readable(descriptor))
                request.multimodal_inputs[idx].serialized_request = self.readables[
                    -1
                ].metadata()
                request.multimodal_inputs[idx].multimodal_input.video_url = None
            finally:
                if container:
                    await asyncio.to_thread(container.close)

        yield request.model_dump_json()


class VLLMEncodeWorkerHandler:
    """
    Handler for vLLM-native encoder worker using ECConnector.
    """

    def __init__(self, runtime, component, engine_client, config):
        """
        Initialize the handler.

        Args:
            runtime: Dynamo distributed runtime
            component: Dynamo component instance
            engine_client: vLLM AsyncLLM instance
            config: Dynamo Config object with CLI arguments
        """
        self.runtime = runtime
        self.component = component
        self.engine_client = engine_client
        self.config = config
        self.temp_dirs = []
        self.image_loader = ImageLoader()

        logger.info(
            f"VLLMNativeEncoderWorkerHandler initialized with "
            f"backend={config.ec_connector_backend}, "
            f"storage_path={config.ec_storage_path}"
        )

    def add_temp_dir(self, temp_dir):
        """Add temporary directory for cleanup."""
        if temp_dir:
            self.temp_dirs.append(temp_dir)

    async def generate(self, request, context) -> AsyncGenerator[str, None]:
        """
        Process encoder request and trigger vLLM encoder execution.

        Args:
            request: VLLMNativeEncoderRequest with multimodal_inputs (list of MultiModalGroup)
            context: Request context from Dynamo runtime

        Yields:
            JSON-encoded VLLMNativeEncoderResponse for each processed item
        """
        # Parse request
        if not isinstance(request, VLLMNativeEncoderRequest):
            if isinstance(request, str):
                request = VLLMNativeEncoderRequest.model_validate_json(request)
            else:
                request = VLLMNativeEncoderRequest.model_validate(request)

        if not request.multimodal_inputs:
            raise ValueError("No multimodal inputs provided in request")

        logger.info(
            f"Processing {len(request.multimodal_inputs)} multimodal item(s) "
            f"for request_id={request.request_id}"
        )

        # Load all images
        # TODO: support video and audio encoding later
        media_list = []
        modality = "image"
        for idx, mm_group in enumerate(request.multimodal_inputs):
            mm_input = mm_group.multimodal_input
            if mm_input.image_url:
                media = await self.image_loader.load_image(mm_input.image_url)
                media_list.append(media)
            elif mm_input.video_url:
                raise NotImplementedError("Video encoding not yet supported")
            else:
                raise ValueError(
                    f"No media URL provided in multimodal_input[{idx}]. "
                    "Specify image_url or video_url."
                )

        # Process all images in one vLLM request
        prompt_dict = TokensPrompt(
            prompt_token_ids=request.token_ids,
            multi_modal_data={"image": media_list},
        )

        try:
            gen = self.engine_client.generate(
                prompt=prompt_dict,
                sampling_params=SamplingParams(max_tokens=1, min_tokens=0),
                request_id=request.request_id,
            )

            # Consume generator to trigger encoder execution
            async for _ in gen:
                pass

            logger.info(
                f"[{request.request_id}] Encoder execution completed for all {len(media_list)} image(s)"
            )

        except Exception as e:
            logger.error(f"[{request.request_id}] Encoder execution failed: {e}")
            raise

        # Compute mm_hash for each image and yield responses
        for idx, media in enumerate(media_list):
            item_request_id = f"{request.request_id}_mm_{idx}"

            try:
                mm_hash = MultiModalHasher.hash_kwargs(
                    model_id=self.config.model, image=media
                )
                logger.debug(f"[{item_request_id}] Computed mm_hash: {mm_hash}")
            except Exception as e:
                logger.error(f"[{item_request_id}] Failed to compute mm_hash: {e}")
                raise

            response = VLLMNativeEncoderResponse(
                request_id=item_request_id,
                mm_hash=mm_hash,
                modality=modality,
                connector_metadata={
                    "ec_connector": self.config.ec_connector_backend,
                    "storage_path": self.config.ec_storage_path,
                },
            )

            logger.debug(f"[{item_request_id}] Returning response: {response}")
            yield response.model_dump_json()

        logger.info(
            f"All {len(request.multimodal_inputs)} multimodal items processed "
            f"for request_id={request.request_id}"
        )

    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up VLLMNativeEncoderWorkerHandler")

        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_dir}: {e}")
