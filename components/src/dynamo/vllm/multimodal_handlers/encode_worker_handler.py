# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import torch
from transformers import AutoImageProcessor
from vllm.engine.arg_utils import AsyncEngineArgs

import dynamo.nixl_connect as connect
from dynamo.common.multimodal import (
    LocalEmbeddingSender,
    NixlReadEmbeddingSender,
    NixlWriteEmbeddingSender,
)
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
from ..multimodal_utils.custom_encoder import FullPromptEncoder
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


def _make_embedding_sender(
    embedding_transfer_mode: EmbeddingTransferMode,
) -> AbstractEmbeddingSender:
    if embedding_transfer_mode == EmbeddingTransferMode.LOCAL:
        return LocalEmbeddingSender()
    elif embedding_transfer_mode == EmbeddingTransferMode.NIXL_WRITE:
        return NixlWriteEmbeddingSender()
    elif embedding_transfer_mode == EmbeddingTransferMode.NIXL_READ:
        return NixlReadEmbeddingSender()
    raise ValueError(f"Invalid embedding transfer mode: {embedding_transfer_mode}")


class _BaseEncodeWorkerHandler:
    """Shared NIXL transfer infrastructure for both encode worker variants."""

    def __init__(self, embedding_transfer_mode: EmbeddingTransferMode) -> None:
        self._connector: connect.Connector | None = None
        self.embedding_sender: AbstractEmbeddingSender = _make_embedding_sender(
            embedding_transfer_mode
        )
        self.send_complete_queue: asyncio.Queue[tuple[Any, Any]] = asyncio.Queue()
        self.send_complete_checker_task = asyncio.create_task(
            self._check_complete(self.send_complete_queue)
        )

    async def _check_complete(self, queue: asyncio.Queue) -> None:
        while True:
            transfer_future, _embedding = await queue.get()
            if transfer_future is None:  # Sentinel value to stop the checker
                queue.task_done()
                break
            await transfer_future
            queue.task_done()

    def cleanup(self) -> None:
        self.send_complete_queue.put_nowait(
            (None, None)
        )  # Send sentinel value to stop the checker

    async def async_init(self, runtime: DistributedRuntime) -> None:
        logger.info("Encode worker startup started.")
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()
        logger.info("Encode worker startup completed.")


class EncodeWorkerHandler(_BaseEncodeWorkerHandler):
    """Standard encode worker: loads a built-in VLM, outputs image embeddings.

    The PD worker receives the image embeddings and handles token expansion
    and multimodal data assembly internally via vLLM's input processor.
    """

    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        embedding_transfer_mode: EmbeddingTransferMode,
    ) -> None:
        super().__init__(embedding_transfer_mode)
        self.engine_args = engine_args
        self.model = engine_args.model

        self.image_loader = ImageLoader(cache_size=CACHE_SIZE_MAXIMUM)
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model, trust_remote_code=True
        )
        self.vision_model = load_vision_model(
            self.model, enforce_eager=engine_args.enforce_eager
        )
        hidden_size = getattr(self.vision_model, "out_hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(
                getattr(self.vision_model, "config", None), "hidden_size", "unknown"
            )
        logger.debug(f"embedding hidden dim: {hidden_size}")
        # Get encoder components for the model
        self.vision_encoder, self.projector = get_encoder_components(
            self.model, self.vision_model
        )
        self.min_workers = 1
        self._accumulated_time = 0.0
        self._processed_requests = 0
        self.readables: list[Any] = []
        self.embedding_cache = EmbeddingCache() if ENABLE_ENCODER_CACHE else None

    @_nvtx.range_decorator("mm:encode_worker_generate", color="blue")
    async def generate(
        self, request: vLLMMultimodalRequest, context
    ) -> AsyncIterator[str]:
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
                f"received request {{ id: {request_id} }} at time {time_start:.4f}, "
                f"processed in {time_end - time_start:.4f} seconds, "
                f"image loading and encoding: {before_transfer_time - time_start:.4f}s, "
                f"transfer preparation: {after_transfer_time - before_transfer_time:.4f}s"
            )

            # Yield transformed request back
            yield payload

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            raise


class FullPromptEncodeWorkerHandler(_BaseEncodeWorkerHandler):
    """Custom-encoder worker: delegates to a FullPromptEncoder, outputs full prompt embeddings.

    The customer supplies a FullPromptEncoder subclass via --full-prompt-encoder-class.
    This handler calls encoder.encode(image_urls, lm_token_ids) and transfers the
    resulting (seq_len, lm_hidden_dim) tensor to the PD as EmbedsPrompt.
    The PD runs transformer layers on it directly — no token expansion, no multimodal
    data assembly.
    """

    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        embedding_transfer_mode: EmbeddingTransferMode,
        full_prompt_encoder_class: str,
    ) -> None:
        super().__init__(embedding_transfer_mode)
        self.model = engine_args.model

        # Load the customer-supplied FullPromptEncoder subclass.
        # --model is the checkpoint path passed to encoder.load().
        # The encoder is responsible for loading everything it needs
        # (ViT, projector, text embedding table, etc.).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        module_path, _, class_name = full_prompt_encoder_class.rpartition(".")
        module = importlib.import_module(module_path)
        self.encoder: FullPromptEncoder = getattr(module, class_name)()
        self.encoder.load(self.model, device)
        logger.info(
            "FullPromptEncodeWorkerHandler: loaded %s from %s on %s",
            full_prompt_encoder_class,
            self.model,
            device,
        )

    @_nvtx.range_decorator("mm:full_prompt_encode_worker_generate", color="blue")
    async def generate(
        self, request: vLLMMultimodalRequest, context
    ) -> AsyncIterator[str]:
        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)

        assert (
            request.multimodal_inputs
        ), "multimodal_inputs must not be None for FullPromptEncodeWorkerHandler"

        request_id = request.request_id
        logger.debug("FullPromptEncodeWorkerHandler: request %s", request_id)

        image_urls = [
            g.multimodal_input.image_url
            for g in request.multimodal_inputs
            if g.multimodal_input and g.multimodal_input.image_url
        ]
        token_ids: list[int] = (
            getattr(request.engine_prompt, "prompt_token_ids", None) or []
            if request.engine_prompt is not None
            else []
        )

        spliced = await asyncio.to_thread(self.encoder.encode, image_urls, token_ids)
        spliced = spliced.reshape(-1, spliced.shape[-1])  # ensure 2D

        spliced_batched = spliced.unsqueeze(0)  # (1, seq_len, lm_hidden_dim)
        (serialized, transfer_future) = await self.embedding_sender.send_embeddings(
            spliced_batched, stage_embeddings=True
        )

        group = request.multimodal_inputs[0]
        group.serialized_request = serialized
        group.embeddings_shape = tuple(spliced_batched.shape)
        group.image_grid_thw = None
        if group.multimodal_input is not None:
            group.multimodal_input.image_url = None

        await self.send_complete_queue.put((transfer_future, spliced))

        logger.info(
            "FullPromptEncodeWorkerHandler: %s — spliced shape=%s",
            request_id,
            tuple(spliced.shape),
        )
        yield request.model_dump_json()
