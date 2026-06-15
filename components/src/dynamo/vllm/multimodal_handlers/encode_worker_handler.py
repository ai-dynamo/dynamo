# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import torch
from transformers import AutoImageProcessor, AutoTokenizer
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
        engine_args: AsyncEngineArgs,
        embedding_transfer_mode: EmbeddingTransferMode,
    ) -> None:
        self.engine_args = engine_args
        self.model = self.engine_args.model

        self.image_loader = ImageLoader(cache_size=CACHE_SIZE_MAXIMUM)
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model, trust_remote_code=True
        )
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

        # EXTERNAL_PROMPT_EMBEDS mode: load the LM embed_tokens weight so the
        # encoder can produce a fully-spliced text+image embedding tensor.
        # Controlled by DYN_EXTERNAL_PROMPT_EMBEDS env var (not resolve_model_family,
        # since the encoder model itself is a VLM that resolves to QWEN_VL).
        self.lm_embed_tokens: Optional[torch.nn.Embedding] = None
        self._image_pad_token_id: Optional[int] = None
        self.vlm_tokenizer: Optional[AutoTokenizer] = None
        if int(os.getenv("DYN_EXTERNAL_PROMPT_EMBEDS", "0")):
            logger.info(
                "EXTERNAL_PROMPT_EMBEDS family detected — loading LM embed_tokens "
                "from %s (one-time CPU load, then GPU)",
                self.model,
            )
            from transformers import AutoModel  # lazy import — heavy

            # AutoModelForCausalLM doesn't support VLMs; use AutoModel with
            # trust_remote_code so Qwen2.5-VL loads correctly.
            # Use get_input_embeddings() rather than a hardcoded attribute path
            # because the embedding layer lives at different paths across models
            # (e.g. .model.embed_tokens for Qwen3, ._input_embed_layer for Qwen2.5-VL).
            hf = AutoModel.from_pretrained(
                self.model,
                device_map="cpu",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            weight = hf.get_input_embeddings().weight.detach().clone()
            del hf
            gc.collect()
            self.lm_embed_tokens = torch.nn.Embedding(
                weight.shape[0], weight.shape[1], _weight=weight
            )
            self.lm_embed_tokens.eval()
            self.lm_embed_tokens = self.lm_embed_tokens.to("cuda")
            logger.info(
                "Loaded embed_tokens: vocab=%d hidden=%d",
                weight.shape[0],
                weight.shape[1],
            )

            tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            self.vlm_tokenizer = tok
            self._image_pad_token_id = tok.convert_tokens_to_ids("<|image_pad|>")
            if self._image_pad_token_id == tok.unk_token_id:
                logger.warning(
                    "<|image_pad|> not found in tokenizer vocab for %s; "
                    "spliced embedding will fall back to image-only",
                    self.model,
                )
                self._image_pad_token_id = None
            else:
                logger.info("Image pad token ID: %d", self._image_pad_token_id)

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

        # EXTERNAL_PROMPT_EMBEDS: produce a fully-spliced text+image embedding.
        if int(os.getenv("DYN_EXTERNAL_PROMPT_EMBEDS", "0")):
            result = await self._generate_spliced_embeds(request)
            yield result
            return

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

    async def _generate_spliced_embeds(self, request: vLLMMultimodalRequest) -> str:
        """Produce a fully-spliced text+image embedding for EXTERNAL_PROMPT_EMBEDS.

        Embeds all text tokens via the LM's embed_tokens layer, encodes the
        image via the ViT, and splices the image embeddings at the position of
        the <|image_pad|> placeholder token.  The resulting [N_total, hidden]
        tensor is sent via the configured embedding sender and the modified
        request JSON is returned for the PD to receive.

        Contract: single image per request.
        """
        assert (
            request.multimodal_inputs is not None
            and len(request.multimodal_inputs) == 1
        ), "EXTERNAL_PROMPT_EMBEDS: exactly one image group expected per request"

        group = request.multimodal_inputs[0]
        if group.multimodal_input is None or not group.multimodal_input.image_url:
            raise ValueError(
                "image_url is required for EXTERNAL_PROMPT_EMBEDS encode worker"
            )

        request_id = request.request_id

        # 1. Load and preprocess the image
        image = await self.image_loader.load_image(group.multimodal_input.image_url)
        image_inputs = self.image_processor(images=[image], return_tensors="pt")

        # 2. Encode image through the vision encoder → [N_img, enc_hidden]
        with torch.no_grad():
            image_embeds = encode_image_embeddings(
                self.model, image_inputs, self.vision_encoder, self.projector
            )
        image_embeds = image_embeds.reshape(
            -1, image_embeds.shape[-1]
        )  # [N_img, hidden]
        logger.debug(
            "[spliced] %s: image_embeds shape=%s dtype=%s",
            request_id,
            tuple(image_embeds.shape),
            image_embeds.dtype,
        )

        # 3. Get token IDs with <|image_pad|> via VLM template re-tokenization.
        # The Dynamo frontend tokenizes using the PD model's text-only template,
        # which drops image content — so prompt_token_ids has no <|image_pad|>.
        # We re-tokenize here using the VLM tokenizer to get the correct structure.
        token_ids: list[int] = []
        if request.engine_prompt is not None:
            token_ids = getattr(request.engine_prompt, "prompt_token_ids", None) or []

        if (
            self.lm_embed_tokens is not None
            and self._image_pad_token_id is not None
            and self.vlm_tokenizer is not None
        ):
            import re

            try:
                # Decode text-only tokens to extract user message content
                decoded = self.vlm_tokenizer.decode(
                    token_ids, skip_special_tokens=False
                )
                user_match = re.search(
                    r"<\|im_start\|>user\n(.*?)<\|im_end\|>", decoded, re.DOTALL
                )
                user_text = user_match.group(1).strip() if user_match else ""

                # Re-render with VLM template, including image placeholder
                vlm_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": group.multimodal_input.image_url or "",
                            },
                            {"type": "text", "text": user_text},
                        ],
                    }
                ]
                vlm_result = self.vlm_tokenizer.apply_chat_template(
                    vlm_messages, tokenize=True, add_generation_prompt=True
                )
                # apply_chat_template returns BatchEncoding (not a plain dict),
                # plain list, or plain dict — normalize to a Python list.
                if hasattr(vlm_result, "input_ids"):
                    vlm_ids = list(vlm_result.input_ids)
                elif "input_ids" in vlm_result:
                    vlm_ids = list(vlm_result["input_ids"])
                else:
                    vlm_ids = list(vlm_result)
                if self._image_pad_token_id in vlm_ids:
                    token_ids = vlm_ids
                    logger.debug(
                        "[spliced] %s: re-tokenized with VLM template → %d tokens, "
                        "image_pad at pos %d",
                        request_id,
                        len(token_ids),
                        token_ids.index(self._image_pad_token_id),
                    )
                else:
                    logger.warning(
                        "[spliced] %s: VLM re-tokenize did not produce image_pad; "
                        "falling back to original tokens",
                        request_id,
                    )
            except Exception as retok_err:
                logger.warning(
                    "[spliced] %s: VLM re-tokenize failed (%s); using original tokens",
                    request_id,
                    retok_err,
                )

        if (
            self.lm_embed_tokens is not None
            and self._image_pad_token_id is not None
            and self._image_pad_token_id in token_ids
        ):
            # 4. Find the image placeholder position and embed all tokens
            pad_pos = token_ids.index(self._image_pad_token_id)
            ids_tensor = torch.tensor(token_ids, dtype=torch.long, device="cuda")
            with torch.no_grad():
                all_text_embs = self.lm_embed_tokens(ids_tensor)  # [N_tok, hidden]
            prefix = all_text_embs[:pad_pos]  # [N_pre, hidden]
            suffix = all_text_embs[pad_pos + 1 :]  # [N_suf, hidden]

            # 5. Splice: [prefix_text | image | suffix_text]
            image_embeds = image_embeds.to(prefix.dtype)
            spliced = torch.cat([prefix, image_embeds, suffix], dim=0)
            logger.info(
                "[spliced] %s: prefix=%d image=%d suffix=%d → total=%d",
                request_id,
                prefix.shape[0],
                image_embeds.shape[0],
                suffix.shape[0],
                spliced.shape[0],
            )
        else:
            # Fallback: no text (missing embed_tokens or no pad token in prompt)
            spliced = image_embeds
            logger.warning(
                "[spliced] %s: falling back to image-only embedding "
                "(lm_embed_tokens=%s, image_pad_token_id=%s, pad_in_tokens=%s)",
                request_id,
                self.lm_embed_tokens is not None,
                self._image_pad_token_id,
                self._image_pad_token_id in token_ids if token_ids else False,
            )

        # 6. Send the spliced tensor via the embedding sender (same NIXL/local path)
        spliced_batched = spliced.unsqueeze(0)  # [1, N_total, hidden]
        (serialized, transfer_future) = await self.embedding_sender.send_embeddings(
            spliced_batched,
            stage_embeddings=True,
        )
        group.serialized_request = serialized
        group.embeddings_shape = tuple(spliced_batched.shape)
        group.image_grid_thw = None  # not applicable for spliced sequence
        # Clear image URL so the PD doesn't try to re-download
        group.multimodal_input.image_url = None

        await self.send_complete_queue.put((transfer_future, spliced))

        return request.model_dump_json()
