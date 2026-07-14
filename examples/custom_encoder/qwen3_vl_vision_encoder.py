# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process Qwen3-VL vision tower for the CustomEncoder path.

This backend is intended to exercise a real, public vision encoder through
``AsyncVisionEncoder`` and ``ThreadedMicroBatcher``. Qwen3-VL also produces
DeepStack features that its native language model injects at intermediate
decoder layers. The current CustomEncoder contract carries only the primary
image embeddings, so this backend is suitable for integration and performance
testing but does not claim numerical parity with native Qwen3-VL inference.
"""

from __future__ import annotations

import base64
import functools
import gc
import hashlib
import io
import logging
import os
import time
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.vllm.multimodal_utils.vision_encoder_backend import Preprocessed
from examples.custom_encoder.qwen_vision_encoder import QwenVisionEncoderBackend

logger = logging.getLogger(__name__)


def _parse_graph_buckets() -> tuple[int, ...]:
    raw = os.environ.get("DYN_QWEN3_VL_GRAPH_BATCH_BUCKETS", "1,2,4,8")
    try:
        buckets = tuple(int(value) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError(
            "DYN_QWEN3_VL_GRAPH_BATCH_BUCKETS must be comma-separated integers"
        ) from exc
    if not buckets or tuple(sorted(set(buckets))) != buckets or buckets[0] < 1:
        raise ValueError(
            "DYN_QWEN3_VL_GRAPH_BATCH_BUCKETS must be strictly increasing "
            f"positive integers, got {raw!r}"
        )
    return buckets


def _parse_graph_image_sizes() -> tuple[tuple[int, int], ...]:
    raw = os.environ.get("DYN_QWEN3_VL_GRAPH_IMAGE_SIZES", "299x299,500x500")
    sizes: list[tuple[int, int]] = []
    try:
        for value in raw.split(","):
            width, height = value.lower().split("x", 1)
            sizes.append((int(width), int(height)))
    except ValueError as exc:
        raise ValueError(
            "DYN_QWEN3_VL_GRAPH_IMAGE_SIZES must look like 299x299,500x500"
        ) from exc
    if not sizes or any(width < 1 or height < 1 for width, height in sizes):
        raise ValueError(
            "DYN_QWEN3_VL_GRAPH_IMAGE_SIZES must contain positive dimensions"
        )
    return tuple(sizes)


_GRAPH_BATCH_BUCKETS = _parse_graph_buckets()
_GRAPH_IMAGE_SIZES = _parse_graph_image_sizes()
_DISABLE_CUDA_GRAPHS = os.environ.get(
    "DYN_QWEN3_VL_DISABLE_CUDA_GRAPHS", ""
).lower() in {"1", "true", "yes"}
_TIMING_ENABLED = os.environ.get("DYN_CUSTOM_ENCODER_TIMING", "").lower() in {
    "1",
    "true",
    "yes",
}


@dataclass(frozen=True)
class Qwen3VLImageInputs:
    """CPU tensors produced by the Qwen3-VL image processor for one image."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    content_digest: bytes | None = None


@dataclass
class _CapturedVisionGraph:
    graph: torch.cuda.CUDAGraph
    forward: torch.nn.Module
    host_pixel_values: torch.Tensor
    static_pixel_values: torch.Tensor
    static_output: torch.Tensor
    tokens_per_item: int


class _StaticQwen3VLVisionForward(torch.nn.Module):
    """Vision forward with grid-derived metadata fixed at capture time."""

    def __init__(
        self,
        visual: torch.nn.Module,
        grid_key: tuple[int, int, int],
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        # Transformers models expose dynamically-generated module attributes whose
        # stubs collapse to ``Tensor | Module``. Runtime validation happens when the
        # parent backend extracts the visual tower; keep these HF internals dynamic.
        self.visual: Any = visual
        dynamic_visual: Any = visual
        grid_cpu = torch.tensor([grid_key] * batch_size, dtype=torch.long, device="cpu")
        grid_device = grid_cpu.to(device)
        with torch.inference_mode():
            pos_embeds = dynamic_visual.fast_pos_embed_interpolate(grid_device)
            pos_embeds = pos_embeds.to(dtype=dynamic_visual.dtype)
            rotary_pos_emb = dynamic_visual.rot_pos_emb(grid_device)
            rotary_pos_emb = rotary_pos_emb.reshape(pos_embeds.shape[0], -1)
            rotary = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = rotary.cos(), rotary.sin()
        cu_seqlens = torch.repeat_interleave(
            grid_cpu[:, 1] * grid_cpu[:, 2], grid_cpu[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        self.register_buffer("pos_embeds", pos_embeds, persistent=False)
        self.register_buffer("position_cos", cos, persistent=False)
        self.register_buffer("position_sin", sin, persistent=False)
        # HF's SDPA path converts these fixed splits to a Python list. Keeping the
        # tensor on CPU avoids a forbidden device sync during CUDA graph capture.
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.visual.patch_embed(pixel_values)
        hidden_states = hidden_states + self.pos_embeds
        position_embeddings = (self.position_cos, self.position_sin)
        for block in self.visual.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=position_embeddings,
            )
        # DeepStack mergers are intentionally omitted: the CustomEncoder contract
        # currently consumes only the primary merger output.
        return self.visual.merger(hidden_states)


class Qwen3VLVisionEncoder(QwenVisionEncoderBackend):
    """Real Qwen3-VL ViT/projector behind Dynamo's custom encoder contract."""

    preprocess_concurrency = 4
    buckets = None if _DISABLE_CUDA_GRAPHS else _GRAPH_BATCH_BUCKETS
    max_batch_cost = int(
        os.environ.get(
            "DYN_QWEN3_VL_MAX_BATCH_COST",
            str(_GRAPH_BATCH_BUCKETS[-1]),
        )
    )

    def __init__(self) -> None:
        self._device: torch.device
        self._cached_preprocess: Any | None = None
        self._embedding_cache: MultimodalEmbeddingCacheManager | None = None
        self._embedding_cache_oversize = 0
        self._embedding_cache_peak_bytes = 0
        self._embedding_cache_coalesced = 0
        self._processor: Any | None = None
        self._visual: Any | None = None
        self._graphs: dict[tuple[tuple[int, int, int], int], _CapturedVisionGraph] = {}
        self._graph_pool: Any | None = None
        self._graph_grid_keys: frozenset[tuple[int, int, int]] = frozenset()
        self.tokenizer: Any = None

    def build(self, model_id: str) -> None:
        """Load the public checkpoint, retain only its vision module, and warm it."""
        self._cached_preprocess = None
        self._embedding_cache = None
        self._embedding_cache_oversize = 0
        self._embedding_cache_peak_bytes = 0
        self._embedding_cache_coalesced = 0
        self._processor = None
        self._visual = None
        self._graphs = {}
        self._graph_pool = None
        self._graph_grid_keys = frozenset()
        self.tokenizer = None
        try:
            self._build(model_id)
        except BaseException:
            self.close()
            raise

    def _build(self, model_id: str) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_size = int(os.environ.get("DYN_QWEN3_VL_PREPROCESS_CACHE_SIZE", "0"))
        self._configure_preprocess_cache(cache_size)
        embedding_cache_bytes = self._parse_embedding_cache_capacity()
        self._embedding_cache = (
            MultimodalEmbeddingCacheManager(embedding_cache_bytes)
            if embedding_cache_bytes
            else None
        )
        processor = AutoProcessor.from_pretrained(model_id)
        self._processor = processor
        self.tokenizer = processor.tokenizer

        logger.info(
            "[Qwen3VLVisionEncoder] loading %s vision tower on %s "
            "(preprocess_cache_size=%d embedding_cache_bytes=%d)",
            model_id,
            self._device,
            cache_size,
            embedding_cache_bytes,
        )
        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self._visual = full_model.model.visual.eval().to(self._device)

        # Detach the retained module before dropping the full checkpoint. This
        # releases the duplicate language-model weights before vLLM starts.
        full_model.model.visual = None
        del full_model
        gc.collect()

        self._graphs = {}
        self._graph_pool = torch.cuda.graph_pool_handle()
        if self._device.type == "cuda" and self.buckets:
            self._capture_cuda_graphs()
        else:
            warmup_image = Image.new("RGB", (500, 500), color=(127, 127, 127))
            warmup_item = self._process_image(warmup_image)
            outputs = self.forward_batch([warmup_item] * self.max_batch_cost)
            del outputs, warmup_item
        logger.info(
            "[Qwen3VLVisionEncoder] warmup complete: buckets=%s "
            "max_batch_cost=%d graphs=%d",
            self.buckets,
            self.max_batch_cost,
            len(self._graphs),
        )

    def _require_processor(self) -> Any:
        processor = self._processor
        if processor is None:
            raise RuntimeError("Qwen3VLVisionEncoder processor is not loaded")
        return processor

    def _require_visual(self) -> Any:
        visual = self._visual
        if visual is None:
            raise RuntimeError(
                "Qwen3VLVisionEncoder.forward_batch() called before build()"
            )
        return visual

    def preprocess(self, raw: str) -> Preprocessed[Qwen3VLImageInputs]:
        """Fetch/decode one image and run the CPU Qwen3-VL image processor."""
        started = time.monotonic()
        result = (
            self._cached_preprocess(raw)
            if self._cached_preprocess is not None
            else self._preprocess_uncached(raw)
        )
        if _TIMING_ENABLED:
            logger.info(
                "custom_encoder_timing stage=preprocess elapsed_ms=%.3f",
                (time.monotonic() - started) * 1000,
            )
        return result

    def _configure_preprocess_cache(self, cache_size: int) -> None:
        if cache_size < 0:
            raise ValueError("DYN_QWEN3_VL_PREPROCESS_CACHE_SIZE must be >= 0")
        self._cached_preprocess = (
            functools.lru_cache(maxsize=cache_size)(self._preprocess_uncached)
            if cache_size
            else None
        )

    def _preprocess_uncached(self, raw: str) -> Preprocessed[Qwen3VLImageInputs]:
        """Compute one read-only CPU input; optionally memoized by source string."""
        image, content_digest = self._load_image(
            raw, compute_digest=self._embedding_cache is not None
        )
        item = self._process_image(image, content_digest=content_digest)
        grid_key = self._grid_key(item)
        if self._graphs and grid_key not in self._graph_grid_keys:
            raise ValueError(
                f"image grid {grid_key} has no captured CUDA graph; configure "
                "DYN_QWEN3_VL_GRAPH_IMAGE_SIZES before startup"
            )
        return Preprocessed(item=item, cost=1, bucket_key=grid_key)

    def _process_image(
        self, image: Image.Image, *, content_digest: bytes | None = None
    ) -> Qwen3VLImageInputs:
        processor = self._require_processor()
        inputs = processor.image_processor(images=[image], return_tensors="pt")
        return Qwen3VLImageInputs(
            pixel_values=inputs["pixel_values"].contiguous(),
            image_grid_thw=inputs["image_grid_thw"].to(dtype=torch.long),
            content_digest=content_digest,
        )

    @staticmethod
    def _grid_key(item: Qwen3VLImageInputs) -> tuple[int, int, int]:
        if item.image_grid_thw.shape != (1, 3):
            raise ValueError(
                "Qwen3VLVisionEncoder expects one image per preprocessed item; "
                f"got image_grid_thw shape {tuple(item.image_grid_thw.shape)}"
            )
        temporal, height, width = item.image_grid_thw[0].tolist()
        return int(temporal), int(height), int(width)

    def _capture_cuda_graphs(self) -> None:
        if not self.buckets:
            raise RuntimeError("CUDA graph capture requires non-empty buckets")
        visual = self._require_visual()
        buckets = self.buckets
        free_before, _ = torch.cuda.mem_get_info(self._device)
        templates: dict[tuple[int, int, int], Qwen3VLImageInputs] = {}
        for width, height in _GRAPH_IMAGE_SIZES:
            image = Image.new("RGB", (width, height), color=(127, 127, 127))
            item = self._process_image(image)
            templates.setdefault(self._grid_key(item), item)
        self._graph_grid_keys = frozenset(templates)

        self._graph_pool = None
        for grid_key, item in templates.items():
            patches_per_item = int(item.pixel_values.shape[0])
            feature_size = int(item.pixel_values.shape[1])
            tokens_per_item = (
                grid_key[0]
                * grid_key[1]
                * grid_key[2]
                // visual.spatial_merge_size**2
            )
            for bucket in reversed(buckets):
                static_pixel_values = torch.zeros(
                    (bucket * patches_per_item, feature_size),
                    dtype=visual.dtype,
                    device=self._device,
                )
                # Reused pinned staging makes the following H2D copy genuinely
                # asynchronous. ``torch.cat`` otherwise allocates pageable host
                # memory, for which ``non_blocking=True`` cannot overlap the CPU.
                host_pixel_values = torch.empty(
                    (bucket * patches_per_item, feature_size),
                    dtype=item.pixel_values.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                forward = _StaticQwen3VLVisionForward(
                    visual, grid_key, bucket, self._device
                ).eval()

                warmup_stream = torch.cuda.Stream(device=self._device)
                warmup_stream.wait_stream(torch.cuda.current_stream(self._device))
                with torch.cuda.stream(warmup_stream), torch.inference_mode():
                    for _ in range(2):
                        warmup_output = forward(static_pixel_values)
                torch.cuda.current_stream(self._device).wait_stream(warmup_stream)
                torch.cuda.synchronize(self._device)
                static_output = torch.empty_like(warmup_output)

                graph = torch.cuda.CUDAGraph()
                with (
                    torch.inference_mode(),
                    torch.cuda.graph(graph, pool=self._graph_pool),
                ):
                    graph_output = forward(static_pixel_values)
                    static_output.copy_(graph_output)
                self._graphs[(grid_key, bucket)] = _CapturedVisionGraph(
                    graph=graph,
                    forward=forward,
                    host_pixel_values=host_pixel_values,
                    static_pixel_values=static_pixel_values,
                    static_output=static_output,
                    tokens_per_item=tokens_per_item,
                )
                logger.info(
                    "[Qwen3VLVisionEncoder] captured CUDA graph: "
                    "grid=%s bucket=%d input_patches=%d output_tokens=%d",
                    grid_key,
                    bucket,
                    bucket * patches_per_item,
                    bucket * tokens_per_item,
                )
        torch.cuda.synchronize(self._device)
        free_after, _ = torch.cuda.mem_get_info(self._device)
        logger.info(
            "[Qwen3VLVisionEncoder] CUDA graph capture complete: "
            "grids=%s buckets=%s graphs=%d device_memory_delta_gib=%.3f",
            sorted(templates),
            self.buckets,
            len(self._graphs),
            (free_before - free_after) / (1024**3),
        )

    def forward_batch(
        self,
        items: List[Qwen3VLImageInputs],
        target_bucket: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Run one eager batch or replay a same-grid padded CUDA graph."""
        if not items:
            raise ValueError("forward_batch requires at least one image")
        if len(items) > self.max_batch_cost:
            raise ValueError(
                f"batch size {len(items)} exceeds max_batch_cost={self.max_batch_cost}"
            )
        if target_bucket is not None and len(items) > target_bucket:
            raise ValueError(
                f"batch size {len(items)} exceeds target bucket {target_bucket}"
            )
        self._require_visual()

        cache = getattr(self, "_embedding_cache", None)
        if cache is None:
            return self._forward_uncached(items, target_bucket)

        outputs: list[torch.Tensor | None] = [None] * len(items)
        pending: OrderedDict[str, tuple[Qwen3VLImageInputs, list[int]]] = OrderedDict()
        uncached: list[tuple[Qwen3VLImageInputs, list[int]]] = []

        for index, item in enumerate(items):
            key = self._embedding_cache_key(item)
            if key is None:
                uncached.append((item, [index]))
                continue
            cached = cache.get(key)
            if cached is not None:
                outputs[index] = cached.tensor.clone()
                continue
            if key in pending:
                pending[key][1].append(index)
                self._embedding_cache_coalesced += 1
            else:
                pending[key] = (item, [index])

        misses = list(pending.items())
        misses.extend(("", value) for value in uncached)
        if misses:
            miss_items = [value[0] for _, value in misses]
            miss_bucket = (
                self._bucket_for_miss_count(len(miss_items))
                if target_bucket is not None
                else None
            )
            if (
                target_bucket is not None
                and miss_bucket is not None
                and miss_bucket > target_bucket
            ):
                raise RuntimeError(
                    f"cache miss bucket {miss_bucket} exceeds dispatched bucket "
                    f"{target_bucket}"
                )
            miss_outputs = self._forward_uncached(miss_items, miss_bucket)
            owned_outputs = self._validate_and_copy_embeddings(miss_items, miss_outputs)
            for (key, (_, indices)), output, owned in zip(
                misses, miss_outputs, owned_outputs
            ):
                if key:
                    if not cache.set(key, CachedEmbedding(tensor=owned)):
                        self._embedding_cache_oversize += 1
                    self._embedding_cache_peak_bytes = max(
                        self._embedding_cache_peak_bytes,
                        cache.stats["current_bytes"],
                    )
                for index in indices:
                    outputs[index] = output.clone() if key else output

        if any(output is None for output in outputs):
            raise RuntimeError("embedding cache scatter left an output unresolved")
        return [output for output in outputs if output is not None]

    def _forward_uncached(
        self,
        items: List[Qwen3VLImageInputs],
        target_bucket: Optional[int],
    ) -> List[torch.Tensor]:
        if target_bucket is not None:
            return self._forward_graph(items, target_bucket)
        return self._forward_eager(items)

    def _bucket_for_miss_count(self, miss_count: int) -> Optional[int]:
        if not self.buckets:
            return None
        for bucket in self.buckets:
            if miss_count <= bucket:
                return bucket
        raise ValueError(
            f"embedding cache miss count {miss_count} exceeds buckets {self.buckets}"
        )

    def _embedding_cache_key(self, item: Qwen3VLImageInputs) -> str | None:
        if item.content_digest is None:
            return None
        grid = self._grid_key(item)
        return f"{item.content_digest.hex()}:{grid[0]}:{grid[1]}:{grid[2]}"

    def _validate_and_copy_embeddings(
        self,
        items: List[Qwen3VLImageInputs],
        outputs: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        if len(outputs) != len(items):
            raise RuntimeError(
                f"vision forward returned {len(outputs)} outputs for "
                f"{len(items)} items"
            )
        owned: list[torch.Tensor] = []
        hidden_size: int | None = None
        merge_size = self._require_visual().spatial_merge_size
        for item, output in zip(items, outputs):
            expected_tokens = int(item.image_grid_thw.prod().item() // merge_size**2)
            if (
                output.device.type != "cpu"
                or output.dtype != torch.bfloat16
                or output.dim() != 2
                or output.shape[0] != expected_tokens
                or output.shape[1] < 1
            ):
                raise RuntimeError(
                    "vision forward returned an invalid embedding: "
                    f"shape={tuple(output.shape)} dtype={output.dtype} "
                    f"device={output.device}; expected "
                    f"({expected_tokens}, hidden) CPU bf16"
                )
            if hidden_size is None:
                hidden_size = output.shape[1]
            elif output.shape[1] != hidden_size:
                raise RuntimeError(
                    "vision forward returned inconsistent hidden sizes: "
                    f"{hidden_size} and {output.shape[1]}"
                )
            owned.append(
                output.detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .clone(memory_format=torch.contiguous_format)
            )
        return owned

    def _forward_eager(self, items: List[Qwen3VLImageInputs]) -> List[torch.Tensor]:
        visual = self._require_visual()
        events = self._timing_events()
        if events:
            events[0].record()
        pixel_values = torch.cat([item.pixel_values for item in items], dim=0).to(
            device=self._device, dtype=visual.dtype, non_blocking=True
        )
        image_grid_thw_cpu = torch.cat([item.image_grid_thw for item in items], dim=0)
        image_grid_thw = image_grid_thw_cpu.to(self._device, non_blocking=True)
        if events:
            events[1].record()
        split_sizes = (
            image_grid_thw_cpu.prod(dim=-1) // visual.spatial_merge_size**2
        ).tolist()

        with torch.inference_mode():
            visual_output = visual(pixel_values, grid_thw=image_grid_thw)
        # Transformers 4.x returns ``(merged, deepstack)``. Transformers 5.x
        # returns BaseModelOutputWithDeepstackFeatures, where the merged image
        # embeddings live in pooler_output (last_hidden_state is pre-merger).
        image_embeds = (
            visual_output.pooler_output
            if hasattr(visual_output, "pooler_output")
            else visual_output[0]
        )
        if events:
            events[2].record()
        # One batched D2H copy avoids a synchronization and allocation per image.
        # The returned splits are CPU views whose shared base owns independent
        # storage, so a later encoder call cannot overwrite them. Treat the views
        # as read-only: siblings intentionally alias that fresh base allocation.
        host_embeds = image_embeds.to(dtype=torch.bfloat16).cpu()
        outputs = list(torch.split(host_embeds, split_sizes))
        self._log_cuda_timings(events, len(items), None, len(items))
        logger.debug(
            "[Qwen3VLVisionEncoder] forward_batch n=%d tokens=%s",
            len(items),
            split_sizes,
        )
        return outputs

    def _forward_graph(
        self, items: List[Qwen3VLImageInputs], target_bucket: int
    ) -> List[torch.Tensor]:
        grid_keys = {self._grid_key(item) for item in items}
        if len(grid_keys) != 1:
            raise ValueError(
                f"graph batch mixed incompatible image grids: {sorted(grid_keys)!r}"
            )
        grid_key = next(iter(grid_keys))
        entry = getattr(self, "_graphs", {}).get((grid_key, target_bucket))
        if entry is None:
            raise ValueError(
                "target_bucket has no captured Qwen3-VL CUDA graph for "
                f"grid={grid_key}, bucket={target_bucket}; configure "
                "DYN_QWEN3_VL_GRAPH_IMAGE_SIZES before startup"
            )
        if len(items) > target_bucket:
            raise ValueError(
                f"batch size {len(items)} exceeds target bucket {target_bucket}"
            )

        events = self._timing_events()
        if events:
            events[0].record()
        input_rows = sum(item.pixel_values.shape[0] for item in items)
        host_pixel_values = entry.host_pixel_values[:input_rows]
        torch.cat(
            [item.pixel_values for item in items],
            dim=0,
            out=host_pixel_values,
        )
        entry.static_pixel_values[:input_rows].copy_(
            host_pixel_values, non_blocking=True
        )
        if events:
            events[1].record()
        entry.graph.replay()
        if events:
            events[2].record()
        real_output = entry.static_output[: len(items) * entry.tokens_per_item]
        host_output = real_output.to(dtype=torch.bfloat16).cpu()
        outputs = list(torch.split(host_output, entry.tokens_per_item))
        self._log_cuda_timings(events, len(items), target_bucket, len(items))
        logger.info(
            "[Qwen3VLVisionEncoder] replayed CUDA graph: "
            "grid=%s actual_batch=%d bucket=%d",
            grid_key,
            len(items),
            target_bucket,
        )
        return outputs

    def _timing_events(self) -> Optional[list[torch.cuda.Event]]:
        if not _TIMING_ENABLED or self._device.type != "cuda":
            return None
        return [torch.cuda.Event(enable_timing=True) for _ in range(4)]

    @staticmethod
    def _log_cuda_timings(
        events: Optional[list[torch.cuda.Event]],
        batch_size: int,
        bucket: Optional[int],
        cost: int,
    ) -> None:
        if not events:
            return
        events[3].record()
        events[3].synchronize()
        for stage, start, end in (
            ("h2d", events[0], events[1]),
            ("vit_forward", events[1], events[2]),
            ("d2h", events[2], events[3]),
        ):
            logger.info(
                "custom_encoder_timing stage=%s elapsed_ms=%.3f "
                "batch_size=%d bucket=%s cost=%d",
                stage,
                start.elapsed_time(end),
                batch_size,
                bucket,
                cost,
            )

    def close(self) -> None:
        """Release model and processor references on the actor thread."""
        if self._cached_preprocess is not None:
            cache_info = self._cached_preprocess.cache_info()
            logger.info(
                "[Qwen3VLVisionEncoder] preprocess cache: "
                "hits=%d misses=%d size=%d capacity=%d",
                cache_info.hits,
                cache_info.misses,
                cache_info.currsize,
                cache_info.maxsize,
            )
            self._cached_preprocess.cache_clear()
        self._cached_preprocess = None
        embedding_cache = getattr(self, "_embedding_cache", None)
        if embedding_cache is not None:
            stats = embedding_cache.stats
            logger.info(
                "[Qwen3VLVisionEncoder] embedding cache: "
                "hits=%d misses=%d entries=%d current_bytes=%d "
                "peak_bytes=%d capacity_bytes=%d evictions=%d "
                "oversize=%d coalesced=%d hit_rate=%.4f",
                stats["hits"],
                stats["misses"],
                stats["entries"],
                stats["current_bytes"],
                self._embedding_cache_peak_bytes,
                stats["capacity_bytes"],
                stats["evictions"],
                self._embedding_cache_oversize,
                self._embedding_cache_coalesced,
                stats["hit_rate"],
            )
        self._embedding_cache = None
        self._embedding_cache_oversize = 0
        self._embedding_cache_peak_bytes = 0
        self._embedding_cache_coalesced = 0
        self._visual = None
        self._processor = None
        self._graphs = {}
        self._graph_pool = None
        self._graph_grid_keys = frozenset()
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _parse_embedding_cache_capacity() -> int:
        raw = os.environ.get("DYN_QWEN3_VL_EMBEDDING_CACHE_BYTES", str(1024**3))
        try:
            capacity = int(raw)
        except ValueError as exc:
            raise ValueError(
                "DYN_QWEN3_VL_EMBEDDING_CACHE_BYTES must be a nonnegative integer, "
                f"got {raw!r}"
            ) from exc
        if capacity < 0:
            raise ValueError(
                "DYN_QWEN3_VL_EMBEDDING_CACHE_BYTES must be a nonnegative integer, "
                f"got {raw!r}"
            )
        return capacity

    @staticmethod
    def _load_image(
        source: str, *, compute_digest: bool = True
    ) -> tuple[Image.Image, bytes | None]:
        """Load RGB pixels and hash the encoded bytes used to decode them."""
        if source.startswith("data:"):
            try:
                header, payload = source.split(",", 1)
            except ValueError as exc:
                raise ValueError("invalid data URL: missing comma") from exc
            if ";base64" not in header:
                raise ValueError("only base64-encoded image data URLs are supported")
            raw = base64.b64decode(payload, validate=True)
        elif source.startswith(("http://", "https://")):
            with urllib.request.urlopen(source, timeout=15) as response:  # nosec B310
                raw = response.read()
        else:
            with open(source, "rb") as image_file:
                raw = image_file.read()
        digest = hashlib.sha256(raw).digest() if compute_digest else None
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB"), digest
