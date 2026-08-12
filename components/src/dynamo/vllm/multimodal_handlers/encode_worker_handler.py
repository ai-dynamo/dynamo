# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import torch
from transformers import AutoImageProcessor
from vllm.engine.arg_utils import AsyncEngineArgs

import dynamo.nixl_connect as connect
from dynamo.common.backend.multimodal import encoder_terminal_chunk
from dynamo.common.multimodal import EMBEDDING_SENDER_FACTORIES
from dynamo.common.multimodal.embedding_transfer import AbstractEmbeddingSender
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
from ..multimodal_utils.custom_encoder import (
    AsyncVisionEncoder,
    CustomEncoderAdapter,
    LinearEmbedsAdapter,
    extract_custom_encoder_image_urls,
    load_custom_encoder,
    stage_linear_visual_prompt,
)
from ..multimodal_utils.embedding_cache import EmbeddingCache
from ..multimodal_utils.model import ModelFamily, resolve_model_family

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
    image_grid_thw: list
    embeddings: torch.Tensor


class EncodeWorkerHandler:
    def __init__(
        self,
        config: Any,
        embedding_transfer_mode: EmbeddingTransferMode,
    ) -> None:
        self.config = config
        engine_args: AsyncEngineArgs = config.engine_args
        self.engine_args = engine_args
        self.model = self.engine_args.model
        self._custom_encoder: AsyncVisionEncoder | None = None
        self._custom_encoder_adapter: CustomEncoderAdapter | None = None
        self._custom_encoder_model_config: Any | None = None
        self.send_complete_queue: asyncio.Queue[tuple[Any, Any]] | None = None
        self.send_complete_checker_task: asyncio.Task | None = None
        self._cleanup_complete = False

        try:
            if config.custom_encoder_class:
                self._custom_encoder_model_config = engine_args.create_model_config()
                (
                    self._custom_encoder,
                    self._custom_encoder_adapter,
                ) = load_custom_encoder(
                    config,
                    self._custom_encoder_model_config,
                    actor_name="custom-encode-worker",
                )
                self.image_loader = None
                self.image_processor = None
                self.vision_model = None
                self.vision_encoder = None
                self.projector = None
            else:
                self.image_loader = ImageLoader(cache_size=CACHE_SIZE_MAXIMUM)
                self.image_processor = AutoImageProcessor.from_pretrained(
                    self.model, trust_remote_code=self.engine_args.trust_remote_code
                )
                self.vision_model = load_vision_model(
                    self.model,
                    enforce_eager=self.engine_args.enforce_eager,
                    trust_remote_code=self.engine_args.trust_remote_code,
                )
                hidden_size = getattr(self.vision_model, "out_hidden_size", None)
                if hidden_size is None:
                    hidden_size = getattr(
                        getattr(self.vision_model, "config", None),
                        "hidden_size",
                        "unknown",
                    )
                logger.debug(f"embedding hidden dim: {hidden_size}")
                self.vision_encoder, self.projector = get_encoder_components(
                    self.model, self.vision_model
                )
            self.min_workers = 1
            self._connector: connect.Connector | None = None
            self._accumulated_time = 0.0
            self._processed_requests = 0
            self.readables: list[Any] = []
            self.embedding_cache = EmbeddingCache() if ENABLE_ENCODER_CACHE else None
            self.embedding_sender: AbstractEmbeddingSender = EMBEDDING_SENDER_FACTORIES[
                embedding_transfer_mode
            ]()

            self.send_complete_queue = asyncio.Queue()
            self.send_complete_checker_task = asyncio.create_task(
                self.check_complete(self.send_complete_queue)
            )
        except BaseException:
            if self._custom_encoder is not None:
                try:
                    self._custom_encoder.shutdown()
                except Exception:
                    logger.exception(
                        "Failed to shut down custom encoder during startup rollback"
                    )
                self._custom_encoder = None
                self._custom_encoder_adapter = None
            raise

    async def check_complete(self, queue):
        while True:
            transfer_future, embedding = await queue.get()
            if transfer_future is None:  # Sentinel value to stop the checker
                queue.task_done()
                break
            try:
                await transfer_future
            except Exception:
                logger.exception("Encoder embedding transfer failed")
            queue.task_done()

    async def cleanup(self):
        if self._cleanup_complete:
            return
        self._cleanup_complete = True
        cleanup_error = None
        sender_drained = False

        try:
            embedding_sender = getattr(self, "embedding_sender", None)
            if embedding_sender is not None:
                await embedding_sender.aclose()
            sender_drained = True
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
            else:
                logger.exception("Embedding sender cleanup also failed")
        finally:
            if self._custom_encoder is not None:
                try:
                    self._custom_encoder.shutdown()
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                    else:
                        logger.exception("Custom encoder cleanup also failed")
                finally:
                    self._custom_encoder = None
                    self._custom_encoder_adapter = None

        checker_task = self.send_complete_checker_task
        self.send_complete_checker_task = None
        if checker_task is not None:
            try:
                if sender_drained and self.send_complete_queue is not None:
                    await self.send_complete_queue.join()
                    self.send_complete_queue.put_nowait((None, None))
                    await checker_task
                else:
                    checker_task.cancel()
                    try:
                        await checker_task
                    except asyncio.CancelledError:
                        pass
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                else:
                    logger.exception("Transfer completion checker cleanup also failed")

        self.send_complete_queue = None
        if cleanup_error is not None:
            raise cleanup_error

    async def async_init(self, runtime: DistributedRuntime):
        """Initialize the connector for RDMA transfers"""
        logger.info("Encode worker startup started.")
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()
        logger.info("Encode worker startup completed.")

    @_nvtx.range_decorator("mm:encode_worker_generate", color="blue")
    async def generate(self, request: Any, context) -> AsyncIterator[Any]:
        if self._custom_encoder is not None:
            async for chunk in self._generate_custom_encoder(request, context):
                yield chunk
            return

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

            with _nvtx.annotate("mm:enc:cache_check", color="cyan"):
                # Before batch process images, check cache first
                need_encode_indexes = []
                embedding_lists: list[EmbeddingItem | None] = [None] * len(
                    request.multimodal_inputs
                )
                for idx in range(len(request.multimodal_inputs)):
                    group_input = request.multimodal_inputs[idx].multimodal_input
                    if group_input is None or not group_input.image_url:
                        raise ValueError("image_url is required for the encode worker.")

                    image_url = group_input.image_url
                    # see if we have local cache
                    embedding_key = EmbeddingCache.generate_hash_key(image_url)
                    if (
                        self.embedding_cache is not None
                        and self.embedding_cache.has_key(embedding_key)
                    ):
                        (image_grid_thw, embeddings) = self.embedding_cache.get(
                            embedding_key
                        )
                        embedding_lists[idx] = EmbeddingItem(
                            embedding_key, image_grid_thw, embeddings
                        )
                    # compute
                    else:
                        # keep track of key to avoid recompute of it
                        need_encode_indexes.append((idx, embedding_key))

            with _nvtx.annotate(
                "mm:enc:image_load", color="green"
            ), time_and_log_code_section(
                f"[ENCODE] request: {request_id} image loading"
            ):
                # Load and generate image tensors
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
                        logger.error(
                            f"Failed to load image from {url[:80]}...: {result}"
                        )
                        collective_exceptions += (
                            f"Failed to load image from {url[:80]}...: {result}\n"
                        )
                        continue
                    loaded_images.append(result)
                if collective_exceptions:
                    raise ValueError(
                        f"Errors occurred during image loading:\n{collective_exceptions}"
                    )

            if loaded_images:
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
                ), time_and_log_code_section(
                    f"[ENCODE] request: {request_id} encoding"
                ):
                    # Encode the image embeddings using model-specific encoder
                    embeddings = await asyncio.to_thread(
                        encode_image_embeddings,
                        model_name=self.model,
                        image_embeds=image_embeds,
                        vision_encoder=self.vision_encoder,
                        projector=self.projector,
                    )
                    # Sync XPU to ensure kernels complete before NIXL transfer.
                    if embeddings.device.type == "xpu":
                        torch.xpu.synchronize()

                with _nvtx.annotate("mm:enc:split_embeddings", color="orange"):
                    # [gluo FIXME] This is specific to qwen vision processing..
                    # Split concatenated embeddings for each image item.
                    if resolve_model_family(self.model) is ModelFamily.QWEN_VL:
                        merge_size = self.vision_encoder.spatial_merge_size
                        sizes = (
                            image_embeds["image_grid_thw"].prod(-1)
                            // merge_size
                            // merge_size
                        ).tolist()
                        splitted_embeddings = embeddings.squeeze(0).split(sizes)
                        logger.debug(
                            f"Splitted embeddings lengths: {[e.shape for e in splitted_embeddings]}"
                        )
                    else:
                        # Validated on llava (NOTE need to double check on other models) that the
                        # embeddings already has batch dimension for images, so we can directly
                        # split by batch dimension
                        logger.debug(f"image embedding shape: {embeddings.shape}")
                        splitted_embeddings = embeddings

                    image_grid_thw = (
                        image_embeds["image_grid_thw"].tolist()
                        if "image_grid_thw" in image_embeds
                        else None
                    )

            # fill in the embedding_lists with new computed embeddings and cache them
            for split_idx, (list_idx, key) in enumerate(need_encode_indexes):
                embedding_lists[list_idx] = EmbeddingItem(
                    key,
                    [image_grid_thw[split_idx]] if image_grid_thw else [],
                    splitted_embeddings[split_idx].unsqueeze(0),
                )
                # Cache the computed value for future use
                if self.embedding_cache is not None:
                    self.embedding_cache.set(
                        embedding_lists[list_idx].key,  # type: ignore
                        (
                            embedding_lists[list_idx].image_grid_thw,  # type: ignore
                            embedding_lists[list_idx].embeddings,  # type: ignore
                        ),
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
                    group.image_grid_thw = embedding_item.image_grid_thw
                    group.embeddings_shape = tuple(embedding_item.embeddings.shape)  # type: ignore[assignment]
                    group.serialized_request = transfer_request[0]

                    # Keep a reference of the embedding and only drop reference when the transfer is done
                    self.send_complete_queue.put_nowait(
                        (transfer_request[1], embedding_item.embeddings)
                    )

            payload = request.model_dump_json()

            time_end = time.perf_counter()
            self._accumulated_time += time_end - time_start
            self._processed_requests += 1
            logger.debug(
                f"received request {{ id: {request_id} }} at time {time_start:.4f}, processed in {time_end - time_start:.4f} seconds, break down: image loading and encoding time {(before_transfer_time - time_start):.4f} seconds, transfer preparation time {(after_transfer_time - before_transfer_time):.4f} seconds, after transfer time {(time_end - after_transfer_time):.4f} seconds."
            )
            logger.debug(
                f"Encoded image(s) for request {{ id: {request_id} }} in {time_end - time_start:.4f} seconds. "
                f"Average encoding time: {self._accumulated_time / self._processed_requests:.4f} seconds over {self._processed_requests} requests."
            )

            # Yield transformed request back
            yield payload

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            raise

    async def _generate_custom_encoder(
        self,
        request: Any,
        context: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        if isinstance(request, str):
            request = json.loads(request)
        elif not isinstance(request, dict):
            request = request.model_dump(mode="python")

        request_id = context.id()
        if self._custom_encoder is None or self._custom_encoder_adapter is None:
            raise RuntimeError("custom encode worker was not initialized")
        if self._custom_encoder_model_config is None:
            raise RuntimeError("custom encode worker has no decoder model config")

        image_urls = extract_custom_encoder_image_urls(request)
        if not image_urls:
            raise ValueError("custom encode worker received no image inputs")
        token_ids = request.get("token_ids") or []
        encodings = await self._custom_encoder.encode(image_urls)
        if not isinstance(self._custom_encoder_adapter, LinearEmbedsAdapter):
            raise ValueError(
                "frontend custom-encoder routing currently requires a text-only "
                "decoder using LinearEmbedsAdapter"
            )
        prepared_prompt = self._custom_encoder_adapter.prepare_compact_prompt(
            token_ids,
            encodings,
        )
        handoff, transfer_future = await stage_linear_visual_prompt(
            prepared_prompt,
            self.embedding_sender,
            transfer_mode=self.config.embedding_transfer_mode.value,
            decoder_model=self.config.model,
            decoder_revision=self.config.engine_args.revision,
            model_config=self._custom_encoder_model_config,
        )
        self.send_complete_queue.put_nowait(
            (transfer_future, prepared_prompt.visual_embeds)
        )
        self._processed_requests += 1
        logger.debug(
            "Custom encode worker prepared request %s with %d image(s)",
            request_id,
            len(image_urls),
        )
        yield encoder_terminal_chunk(handoff)
