# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import torch
from transformers import AutoImageProcessor, AutoProcessor
from transformers.video_utils import VideoMetadata
from vllm.engine.arg_utils import AsyncEngineArgs

import dynamo.nixl_connect as connect
from dynamo.common.multimodal import (
    LocalEmbeddingSender,
    NixlReadEmbeddingSender,
    NixlWriteEmbeddingSender,
)
from dynamo.common.multimodal.embedding_transfer import AbstractEmbeddingSender
from dynamo.common.multimodal.video_loader import VideoLoader
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.common.utils.time_section import time_and_log_code_section
from dynamo.runtime import DistributedRuntime

from ..constants import EmbeddingTransferMode
from ..multimodal_utils import (
    ImageLoader,
    encode_image_embeddings,
    get_encoder_components,
    load_vision_model,
    vLLMMultimodalRequest,
)
from ..multimodal_utils.embedding_cache import EmbeddingCache
from ..multimodal_utils.model import is_qwen_vl_model

logger = logging.getLogger(__name__)

CACHE_SIZE_MAXIMUM = 8

# [gluo WIP] now it's time to revisit
# Both embedding transfer suffers from increasing latency as
# number of concurrent requests increases, NixlPersistentEmbedding transfers
# scale worse than local. Need to investigate why.
# [gluo NOTE] default off to benchmark standalone encoder
ENABLE_ENCODER_CACHE = int(os.getenv("ENABLE_ENCODER_CACHE", 1))


@dataclass
class EmbeddingItem:
    key: str
    modality: str
    grid_thw: list
    embeddings: torch.Tensor
    timestamps: list[list[float]] | None = None


class EncodeWorkerHandler:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        embedding_transfer_mode: EmbeddingTransferMode,
    ) -> None:
        self.engine_args = engine_args
        self.model = self.engine_args.model

        self.image_loader = ImageLoader(cache_size=CACHE_SIZE_MAXIMUM)
        self.video_loader = VideoLoader()
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model, trust_remote_code=True
        )
        self.video_processor = None
        if is_qwen_vl_model(self.model):
            self.video_processor = AutoProcessor.from_pretrained(
                self.model, trust_remote_code=True
            ).video_processor
        self.vision_model = load_vision_model(
            self.model, enforce_eager=self.engine_args.enforce_eager
        )
        hidden_size = getattr(self.vision_model, "out_hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(
                getattr(self.vision_model, "config", None), "hidden_size", "unknown"
            )
        logger.debug(f"embedding hidden dim: {hidden_size}")
        self.min_workers = 1

        # Get encoder components for the model
        self.vision_encoder, self.projector = get_encoder_components(
            self.model, self.vision_model
        )
        self._connector: connect.Connector | None = None
        self._accumulated_time = 0.0
        self._processed_requests = 0
        self.readables: list[Any] = []
        self.embedding_cache = EmbeddingCache() if ENABLE_ENCODER_CACHE else None
        self.embedding_sender: AbstractEmbeddingSender
        if embedding_transfer_mode == EmbeddingTransferMode.LOCAL:
            self.embedding_sender = LocalEmbeddingSender()
        elif embedding_transfer_mode == EmbeddingTransferMode.NIXL_WRITE:
            self.embedding_sender = NixlWriteEmbeddingSender()
        elif embedding_transfer_mode == EmbeddingTransferMode.NIXL_READ:
            self.embedding_sender = NixlReadEmbeddingSender()
        else:
            raise ValueError(
                f"Invalid embedding transfer mode: {embedding_transfer_mode}"
            )

        self.send_complete_queue: asyncio.Queue[tuple[Any, Any]] = asyncio.Queue()
        self.send_complete_checker_task = asyncio.create_task(
            self.check_complete(self.send_complete_queue)
        )

    async def check_complete(self, queue):
        while True:
            transfer_future, embedding = await queue.get()
            if transfer_future is None:  # Sentinel value to stop the checker
                queue.task_done()
                break
            await transfer_future
            queue.task_done()

    def cleanup(self):
        self.send_complete_queue.put_nowait(
            (None, None)
        )  # Send sentinel value to stop the checker

    async def async_init(self, runtime: DistributedRuntime):
        """Initialize the connector for RDMA transfers"""
        logger.info("Encode worker startup started.")
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()
        logger.info("Encode worker startup completed.")

    def _get_request_modality(self, request: vLLMMultimodalRequest) -> str:
        modalities = set()
        for group in request.multimodal_inputs or []:
            group_input = group.multimodal_input
            if group_input is None:
                continue
            if group_input.image_url is not None:
                modalities.add("image")
            if group_input.video_url is not None:
                modalities.add("video")
        if not modalities:
            raise ValueError("Encode worker requires image_url or video_url inputs.")
        if len(modalities) > 1:
            # TODO: Support mixed image+video batches once the request fanout
            # path becomes fully modality-aware.
            raise ValueError(
                "Mixed image_url and video_url batches are not supported by the encode worker."
            )
        return modalities.pop()

    def _get_embedding_key(self, url: str, modality: str) -> str:
        if modality == "video":
            # TODO: Include sampling/processor knobs if those become
            # request-specific.
            return EmbeddingCache.generate_hash_key("video", url)
        return EmbeddingCache.generate_hash_key(url)

    def _calculate_video_timestamps(self, metadata: dict[str, Any]) -> list[float]:
        assert self.video_processor is not None
        merge_size = self.video_processor.merge_size
        indices = list(metadata["frames_indices"])
        if len(indices) % merge_size != 0:
            indices = indices + [indices[-1]] * (merge_size - len(indices) % merge_size)
        timestamps = [idx / metadata["fps"] for idx in indices]
        return [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]

    async def _encode_missing_images(
        self,
        request: vLLMMultimodalRequest,
        need_encode_indexes: list[tuple[int, str]],
        embedding_lists: list[EmbeddingItem | None],
        request_id: str,
    ) -> None:
        with _nvtx.annotate(
            "mm:enc:image_load", color="green"
        ), time_and_log_code_section(f"[ENCODE] request: {request_id} image loading"):
            image_tasks = []
            image_to_load = []
            for idx, _ in need_encode_indexes:
                group_mm_input = request.multimodal_inputs[idx].multimodal_input
                assert group_mm_input is not None
                assert group_mm_input.image_url is not None
                url: str = group_mm_input.image_url
                image_tasks.append(
                    asyncio.create_task(self.image_loader.load_image(url))
                )
                image_to_load.append(url)
            results = await asyncio.gather(*image_tasks, return_exceptions=True)
            loaded_images = []
            collective_exceptions = ""
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    url = image_to_load[i]
                    logger.error(f"Failed to load image from {url[:80]}...: {result}")
                    collective_exceptions += (
                        f"Failed to load image from {url[:80]}...: {result}\n"
                    )
                    continue
                loaded_images.append(result)
            if collective_exceptions:
                raise ValueError(
                    f"Errors occurred during image loading:\n{collective_exceptions}"
                )

        with _nvtx.annotate(
            "mm:enc:image_preprocess", color="yellow"
        ), time_and_log_code_section(
            f"[ENCODE] request: {request_id} image processing"
        ):
            image_embeds = await asyncio.to_thread(
                self.image_processor, images=loaded_images, return_tensors="pt"
            )

        with _nvtx.annotate(
            "mm:enc:vision_encode", color="red"
        ), time_and_log_code_section(f"[ENCODE] request: {request_id} encoding"):
            embeddings = await asyncio.to_thread(
                encode_image_embeddings,
                model_name=self.model,
                image_embeds=image_embeds,
                vision_encoder=self.vision_encoder,
                projector=self.projector,
            )

        with _nvtx.annotate("mm:enc:split_embeddings", color="orange"):
            if is_qwen_vl_model(self.model):
                merge_size = self.vision_encoder.spatial_merge_size
                sizes = (
                    image_embeds["image_grid_thw"].prod(-1) // merge_size // merge_size
                ).tolist()
                splitted_embeddings = embeddings.squeeze(0).split(sizes)
                logger.debug(
                    f"Splitted embeddings lengths: {[e.shape for e in splitted_embeddings]}"
                )
            else:
                logger.debug(f"image embedding shape: {embeddings.shape}")
                splitted_embeddings = embeddings

            image_grid_thw = (
                image_embeds["image_grid_thw"].tolist()
                if "image_grid_thw" in image_embeds
                else None
            )

        for split_idx, (list_idx, key) in enumerate(need_encode_indexes):
            embedding_item = EmbeddingItem(
                key=key,
                modality="image",
                grid_thw=[image_grid_thw[split_idx]] if image_grid_thw else [],
                embeddings=splitted_embeddings[split_idx].unsqueeze(0),
            )
            embedding_lists[list_idx] = embedding_item
            if self.embedding_cache is not None:
                self.embedding_cache.set(key, embedding_item)

    async def _encode_missing_videos(
        self,
        request: vLLMMultimodalRequest,
        need_encode_indexes: list[tuple[int, str]],
        embedding_lists: list[EmbeddingItem | None],
        request_id: str,
    ) -> None:
        if self.video_processor is None:
            raise ValueError(
                f"Video encode is only supported for Qwen VL models, got {self.model}."
            )

        with _nvtx.annotate(
            "mm:enc:video_load", color="green"
        ), time_and_log_code_section(f"[ENCODE] request: {request_id} video loading"):
            video_tasks = []
            video_to_load = []
            for idx, _ in need_encode_indexes:
                group_mm_input = request.multimodal_inputs[idx].multimodal_input
                assert group_mm_input is not None
                assert group_mm_input.video_url is not None
                url: str = group_mm_input.video_url
                video_tasks.append(
                    asyncio.create_task(self.video_loader.load_video(url))
                )
                video_to_load.append(url)
            results = await asyncio.gather(*video_tasks, return_exceptions=True)
            loaded_videos = []
            collective_exceptions = ""
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    url = video_to_load[i]
                    logger.error(f"Failed to load video from {url[:80]}...: {result}")
                    collective_exceptions += (
                        f"Failed to load video from {url[:80]}...: {result}\n"
                    )
                    continue
                loaded_videos.append(result)
            if collective_exceptions:
                raise ValueError(
                    f"Errors occurred during video loading:\n{collective_exceptions}"
                )

        with _nvtx.annotate(
            "mm:enc:video_preprocess", color="yellow"
        ), time_and_log_code_section(
            f"[ENCODE] request: {request_id} video processing"
        ):
            video_arrays = [frames for frames, _ in loaded_videos]
            video_metadata = [
                VideoMetadata(
                    **{k: metadata[k] for k in metadata if k != "do_sample_frames"}
                )
                for _, metadata in loaded_videos
            ]
            video_inputs = await asyncio.to_thread(
                self.video_processor,
                videos=video_arrays,
                video_metadata=video_metadata,
                do_sample_frames=False,
                fps=None,
                return_tensors="pt",
            )
            del video_arrays

        with _nvtx.annotate(
            "mm:enc:vision_encode", color="red"
        ), time_and_log_code_section(f"[ENCODE] request: {request_id} encoding"):
            embeddings = await asyncio.to_thread(
                self.vision_encoder,
                video_inputs["pixel_values_videos"].to(self.vision_encoder.device),
                grid_thw=video_inputs["video_grid_thw"],
            )
            if isinstance(embeddings, (tuple, list)):
                embeddings = embeddings[0]

        with _nvtx.annotate("mm:enc:split_embeddings", color="orange"):
            merge_size = self.vision_encoder.spatial_merge_size
            sizes = (
                video_inputs["video_grid_thw"].prod(-1) // merge_size // merge_size
            ).tolist()
            splitted_embeddings = embeddings.split(sizes)
            video_grid_thw = video_inputs["video_grid_thw"].tolist()
            timestamps_per_video = [
                self._calculate_video_timestamps(metadata)
                for _, metadata in loaded_videos
            ]

        for split_idx, (list_idx, key) in enumerate(need_encode_indexes):
            timestamps = [timestamps_per_video[split_idx]]
            assert (
                len(timestamps[0]) == video_grid_thw[split_idx][0]
            ), "timestamps length must match the temporal dimension of video_grid_thw"
            embedding_item = EmbeddingItem(
                key=key,
                modality="video",
                grid_thw=[video_grid_thw[split_idx]],
                embeddings=splitted_embeddings[split_idx].unsqueeze(0),
                timestamps=timestamps,
            )
            embedding_lists[list_idx] = embedding_item
            if self.embedding_cache is not None:
                self.embedding_cache.set(key, embedding_item)

    @_nvtx.range_decorator("mm:encode_worker_generate", color="blue")
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
        assert (
            request.multimodal_inputs is not None
        ), "multimodal_inputs must not be None for encode worker"

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
            modality = self._get_request_modality(request)

            with _nvtx.annotate("mm:enc:cache_check", color="cyan"):
                need_encode_indexes = []
                embedding_lists: list[EmbeddingItem | None] = [None] * len(
                    request.multimodal_inputs
                )
                for idx in range(len(request.multimodal_inputs)):
                    group_input = request.multimodal_inputs[idx].multimodal_input
                    if group_input is None:
                        raise ValueError(
                            "multimodal_input is required for the encode worker."
                        )

                    url = (
                        group_input.image_url
                        if modality == "image"
                        else group_input.video_url
                    )
                    if url is None:
                        raise ValueError(
                            f"{modality}_url is required for the encode worker."
                        )

                    embedding_key = self._get_embedding_key(url, modality)
                    if (
                        self.embedding_cache is not None
                        and self.embedding_cache.has_key(embedding_key)
                    ):
                        embedding_lists[idx] = self.embedding_cache.get(embedding_key)
                    else:
                        need_encode_indexes.append((idx, embedding_key))

            if need_encode_indexes:
                if modality == "image":
                    await self._encode_missing_images(
                        request, need_encode_indexes, embedding_lists, request_id
                    )
                else:
                    await self._encode_missing_videos(
                        request, need_encode_indexes, embedding_lists, request_id
                    )

            before_transfer_time = time.perf_counter()

            with _nvtx.annotate("mm:enc:embedding_transfer", color="purple"):
                # Prepare transfer
                send_tasks = [
                    asyncio.create_task(
                        self.embedding_sender.send_embeddings(
                            embedding_item.embeddings, stage_embeddings=True
                        )
                    )
                    for embedding_item in embedding_lists
                    if embedding_item is not None
                ]
                transfer_requests = await asyncio.gather(*send_tasks)

                after_transfer_time = time.perf_counter()

                for idx, item in enumerate(zip(embedding_lists, transfer_requests)):
                    embedding_item, transfer_request = item
                    assert embedding_item is not None
                    logger.debug(
                        f"{embedding_item.embeddings.shape} prepared for transfer."
                    )
                    # Update request for transfer metadata
                    group = request.multimodal_inputs[idx]
                    assert group.multimodal_input is not None
                    group.multimodal_input.image_url = None
                    group.multimodal_input.video_url = None
                    if embedding_item.modality == "image":
                        group.image_grid_thw = embedding_item.grid_thw
                    else:
                        group.video_grid_thw = embedding_item.grid_thw
                        group.timestamps = embedding_item.timestamps
                    group.embeddings_shape = tuple(embedding_item.embeddings.shape)  # type: ignore[assignment]
                    group.serialized_request = transfer_request[0]

                    # Keep a reference of the embedding and only drop reference when the transfer is done
                    self.send_complete_queue.put_nowait(
                        (transfer_request[1], embedding_item.embeddings)
                    )

            logger.debug(f"Request: {request.model_dump_json()}")

            time_end = time.perf_counter()
            self._accumulated_time += time_end - time_start
            self._processed_requests += 1
            logger.debug(
                f"received request {{ id: {request_id} }} at time {time_start:.4f}, processed in {time_end - time_start:.4f} seconds, break down: image loading and encoding time {(before_transfer_time - time_start):.4f} seconds, transfer preparation time {(after_transfer_time - before_transfer_time):.4f} seconds, after transfer time {(time_end - after_transfer_time):.4f} seconds."
            )
            logger.debug(
                f"Encoded {modality}(s) for request {{ id: {request_id} }} in {time_end - time_start:.4f} seconds. "
                f"Average encoding time: {self._accumulated_time / self._processed_requests:.4f} seconds over {self._processed_requests} requests."
            )

            # Yield transformed request back
            yield request.model_dump_json()

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            raise
