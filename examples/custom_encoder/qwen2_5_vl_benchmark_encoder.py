# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance-only Qwen2.5-VL vision tower for a text decoder benchmark."""

from __future__ import annotations

import base64
import functools
import gc
import io
import logging
import os
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.utils.torch_utils import set_default_torch_dtype

from dynamo.vllm.multimodal_utils.vision_encoder_backend import Preprocessed
from examples.custom_encoder.qwen_vision_encoder import QwenVisionEncoderBackend

logger = logging.getLogger(__name__)


def _parse_graph_buckets() -> tuple[int, ...]:
    raw = os.environ.get("DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS", "1,2,4,8,16,32,64")
    try:
        buckets = tuple(int(value) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError(
            "DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS must be comma-separated integers"
        ) from exc
    if not buckets or tuple(sorted(set(buckets))) != buckets or buckets[0] < 1:
        raise ValueError(
            "DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS must be strictly increasing "
            f"positive integers, got {raw!r}"
        )
    return buckets


def _parse_graph_image_sizes() -> tuple[tuple[int, int], ...]:
    raw = os.environ.get("DYN_QWEN2_VL_GRAPH_IMAGE_SIZES", "500x500")
    sizes: list[tuple[int, int]] = []
    try:
        for value in raw.split(","):
            width, height = value.lower().split("x", 1)
            sizes.append((int(width), int(height)))
    except ValueError as exc:
        raise ValueError(
            "DYN_QWEN2_VL_GRAPH_IMAGE_SIZES must look like 299x299,500x500"
        ) from exc
    if not sizes or any(width < 1 or height < 1 for width, height in sizes):
        raise ValueError(
            "DYN_QWEN2_VL_GRAPH_IMAGE_SIZES must contain positive dimensions"
        )
    return tuple(sizes)


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _decoder_hidden_size(config: Any) -> int:
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "hidden_size", None)
    if value is None:
        value = getattr(config, "hidden_size", None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(
            "Qwen2_5VLBenchmarkEncoder could not determine the decoder hidden size "
            f"from {type(config).__name__}"
        )
    return value


def _benchmark_output_hidden_size(
    native_hidden_size: int, decoder_hidden_size: int
) -> int:
    output_hidden_size = _parse_positive_int_env(
        "DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE", native_hidden_size
    )
    if output_hidden_size > native_hidden_size:
        raise ValueError(
            "DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE cannot exceed the vision "
            f"projector width {native_hidden_size}; got {output_hidden_size}"
        )
    if output_hidden_size != decoder_hidden_size:
        raise ValueError(
            "Qwen2_5VLBenchmarkEncoder output width must match the served decoder "
            f"hidden size; output={output_hidden_size}, decoder={decoder_hidden_size}"
        )
    return output_hidden_size


_GRAPH_BATCH_BUCKETS = _parse_graph_buckets()
_GRAPH_IMAGE_SIZES = _parse_graph_image_sizes()
_DISABLE_CUDA_GRAPHS = os.environ.get(
    "DYN_QWEN2_VL_DISABLE_CUDA_GRAPHS", ""
).lower() in {"1", "true", "yes"}
_TIMING_ENABLED = os.environ.get("DYN_CUSTOM_ENCODER_TIMING", "").lower() in {
    "1",
    "true",
    "yes",
}
_DISPATCH_LOG_ENABLED = os.environ.get(
    "DYN_CUSTOM_ENCODER_DISPATCH_LOG", ""
).lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Qwen2VLImageInputs:
    """CPU tensors produced by the Qwen2.5-VL image processor for one image."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


@dataclass
class _CapturedVisionGraph:
    graph: torch.cuda.CUDAGraph
    forward: torch.nn.Module
    host_pixel_values: torch.Tensor
    static_pixel_values: torch.Tensor
    static_output: torch.Tensor
    tokens_per_item: int


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    midpoint = hidden_states.shape[-1] // 2
    first = hidden_states[..., :midpoint]
    second = hidden_states[..., midpoint:]
    return torch.cat((-second, first), dim=-1)


class _VllmQwen2_5VisionAttention(torch.nn.Module):
    """HF projections with vLLM's variable-length vision attention kernel."""

    def __init__(self, attention: torch.nn.Module, prefix: str) -> None:
        super().__init__()
        dynamic_attention: Any = attention
        self.qkv = dynamic_attention.qkv
        self.proj = dynamic_attention.proj
        self.num_heads = int(dynamic_attention.num_heads)
        self.head_dim = int(dynamic_attention.head_dim)
        self.scaling = float(dynamic_attention.scaling)
        with (
            set_current_vllm_config(VllmConfig()),
            set_default_torch_dtype(self.qkv.weight.dtype),
        ):
            self.attn = MMEncoderAttention(
                num_heads=self.num_heads,
                head_size=self.head_dim,
                scale=self.scaling,
                prefix=prefix,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del rotary_pos_emb, kwargs
        if position_embeddings is None:
            raise ValueError("Qwen2.5-VL vision attention requires rotary embeddings")
        if cu_seqlens.device != hidden_states.device:
            raise ValueError("vision attention sequence boundaries must be on-device")

        sequence_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(sequence_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_dtype = query.dtype
        key_dtype = key.dtype
        cos = cos.unsqueeze(-2).float()
        sin = sin.unsqueeze(-2).float()
        query_float = query.float()
        key_float = key.float()
        query = ((query_float * cos) + (_rotate_half(query_float) * sin)).to(
            query_dtype
        )
        key = ((key_float * cos) + (_rotate_half(key_float) * sin)).to(key_dtype)

        if max_seqlen is None:
            # Eager callers outside this backend do not precompute the host scalar.
            # The custom eager and graph paths always pass it to avoid this sync.
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().cpu()
        output = self.attn(
            query=query.unsqueeze(0),
            key=key.unsqueeze(0),
            value=value.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return self.proj(output.reshape(sequence_length, -1).contiguous())


class _StaticQwen2VLVisionForward(torch.nn.Module):
    """Qwen2.5-VL vision forward with all grid metadata fixed for capture."""

    def __init__(
        self,
        visual: torch.nn.Module,
        grid_key: tuple[int, int, int],
        batch_size: int,
        device: torch.device,
        output_hidden_size: int,
    ) -> None:
        super().__init__()
        # Transformers models expose dynamically-generated module attributes whose
        # stubs collapse to ``Tensor | Module``. Runtime validation happens when the
        # parent backend extracts the visual tower; keep these HF internals dynamic.
        self.visual: Any = visual
        dynamic_visual: Any = visual
        grid_cpu = torch.tensor([grid_key] * batch_size, dtype=torch.long, device="cpu")
        with torch.inference_mode():
            grid_device = grid_cpu.to(device)
            rotary_pos_emb = dynamic_visual.rot_pos_emb(grid_device)
            window_index_cpu, cu_window_seqlens_list = dynamic_visual.get_window_index(
                grid_cpu
            )
            window_index = window_index_cpu.to(device)
            sequence_length = int(grid_cpu.prod(dim=-1).sum().item())
            merge_unit = int(dynamic_visual.spatial_merge_unit)
            rotary_pos_emb = rotary_pos_emb.reshape(
                sequence_length // merge_unit, merge_unit, -1
            )
            rotary_pos_emb = rotary_pos_emb[window_index].reshape(sequence_length, -1)
            rotary = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = rotary.cos(), rotary.sin()
            reverse_indices = torch.argsort(window_index)

        cu_window_seqlens_cpu = torch.tensor(
            cu_window_seqlens_list,
            dtype=torch.int32,
        ).unique_consecutive()
        cu_seqlens_cpu = torch.repeat_interleave(
            grid_cpu[:, 1] * grid_cpu[:, 2], grid_cpu[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens_cpu = torch.nn.functional.pad(cu_seqlens_cpu, (1, 0), value=0)
        self.sequence_length = sequence_length
        self.merge_unit = merge_unit
        self.output_hidden_size = output_hidden_size
        self.fullatt_block_indexes = frozenset(dynamic_visual.fullatt_block_indexes)
        self.register_buffer("window_index", window_index, persistent=False)
        self.register_buffer("reverse_indices", reverse_indices, persistent=False)
        self.register_buffer("position_cos", cos, persistent=False)
        self.register_buffer("position_sin", sin, persistent=False)
        # FlashAttention consumes device boundaries and reads max_seqlen from the
        # host. Keeping that scalar on CPU avoids a sync during graph capture.
        self.register_buffer("cu_seqlens", cu_seqlens_cpu.to(device), persistent=False)
        self.register_buffer(
            "cu_window_seqlens", cu_window_seqlens_cpu.to(device), persistent=False
        )
        self.register_buffer(
            "full_max_seqlen",
            (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max(),
            persistent=False,
        )
        self.register_buffer(
            "window_max_seqlen",
            (cu_window_seqlens_cpu[1:] - cu_window_seqlens_cpu[:-1]).max(),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.visual.patch_embed(pixel_values)
        hidden_states = hidden_states.reshape(
            self.sequence_length // self.merge_unit,
            self.merge_unit,
            -1,
        )
        hidden_states = hidden_states[self.window_index].reshape(
            self.sequence_length, -1
        )
        position_embeddings = (self.position_cos, self.position_sin)
        for layer_num, block in enumerate(self.visual.blocks):
            full_attention = layer_num in self.fullatt_block_indexes
            cu_seqlens = self.cu_seqlens if full_attention else self.cu_window_seqlens
            max_seqlen = (
                self.full_max_seqlen if full_attention else self.window_max_seqlen
            )
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                max_seqlen=max_seqlen,
            )
        hidden_states = self.visual.merger(hidden_states)
        return hidden_states[self.reverse_indices][:, : self.output_hidden_size]


def _forward_vllm_vision_attention(
    visual: Any,
    pixel_values: torch.Tensor,
    image_grid_thw_cpu: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Run variable-grid eager vision attention with host max-sequence metadata."""
    hidden_states = visual.patch_embed(pixel_values)
    rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
    window_index_cpu, cu_window_seqlens_list = visual.get_window_index(
        image_grid_thw_cpu
    )
    window_index = window_index_cpu.to(pixel_values.device)
    sequence_length = int(image_grid_thw_cpu.prod(dim=-1).sum().item())
    merge_unit = int(visual.spatial_merge_unit)

    hidden_states = hidden_states.reshape(sequence_length // merge_unit, merge_unit, -1)
    hidden_states = hidden_states[window_index].reshape(sequence_length, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(
        sequence_length // merge_unit, merge_unit, -1
    )
    rotary_pos_emb = rotary_pos_emb[window_index].reshape(sequence_length, -1)
    rotary = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (rotary.cos(), rotary.sin())

    cu_window_seqlens_cpu = torch.tensor(
        cu_window_seqlens_list, dtype=torch.int32
    ).unique_consecutive()
    cu_seqlens_cpu = torch.repeat_interleave(
        image_grid_thw_cpu[:, 1] * image_grid_thw_cpu[:, 2],
        image_grid_thw_cpu[:, 0],
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens_cpu = torch.nn.functional.pad(cu_seqlens_cpu, (1, 0), value=0)
    cu_seqlens = cu_seqlens_cpu.to(pixel_values.device)
    cu_window_seqlens = cu_window_seqlens_cpu.to(pixel_values.device)
    full_max_seqlen = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max()
    window_max_seqlen = (cu_window_seqlens_cpu[1:] - cu_window_seqlens_cpu[:-1]).max()

    fullatt_block_indexes = frozenset(visual.fullatt_block_indexes)
    for layer_num, block in enumerate(visual.blocks):
        full_attention = layer_num in fullatt_block_indexes
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens if full_attention else cu_window_seqlens,
            position_embeddings=position_embeddings,
            max_seqlen=full_max_seqlen if full_attention else window_max_seqlen,
        )
    hidden_states = visual.merger(hidden_states)
    return hidden_states[torch.argsort(window_index)]


class Qwen2_5VLBenchmarkEncoder(QwenVisionEncoderBackend):
    """Real Qwen2.5-VL ViT/projector behind Dynamo's custom encoder contract."""

    preprocess_concurrency = _parse_positive_int_env(
        "DYN_QWEN2_VL_PREPROCESS_CONCURRENCY", 64
    )
    buckets = None if _DISABLE_CUDA_GRAPHS else _GRAPH_BATCH_BUCKETS
    max_batch_cost = _parse_positive_int_env(
        "DYN_QWEN2_VL_MAX_BATCH_COST", _GRAPH_BATCH_BUCKETS[-1]
    )

    def __init__(self) -> None:
        self._device: torch.device
        self._cached_preprocess: Any | None = None
        self._processor: Any | None = None
        self._visual: Any | None = None
        self._dispatch_counts: Counter[tuple[str, int, int | None]] = Counter()
        self._graphs: dict[tuple[tuple[int, int, int], int], _CapturedVisionGraph] = {}
        self._graph_pool: Any | None = None
        self._graph_grid_keys: frozenset[tuple[int, int, int]] = frozenset()
        self._encoder_model_id: str | None = None
        self._output_hidden_size: int | None = None
        self.tokenizer: Any = None

    def build(self, model_id: str) -> None:
        """Load the public checkpoint, retain only its vision module, and warm it."""
        self._cached_preprocess = None
        self._processor = None
        self._visual = None
        self._dispatch_counts.clear()
        self._graphs = {}
        self._graph_pool = None
        self._graph_grid_keys = frozenset()
        self._encoder_model_id = None
        self._output_hidden_size = None
        self.tokenizer = None
        try:
            self._build(model_id)
        except BaseException:
            self.close()
            raise

    def _build(self, model_id: str) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_size = int(os.environ.get("DYN_QWEN2_VL_PREPROCESS_CACHE_SIZE", "0"))
        self._configure_preprocess_cache(cache_size)
        encoder_model_id = os.environ.get("DYN_QWEN2_VL_ENCODER_MODEL", model_id)
        decoder_config = AutoConfig.from_pretrained(model_id)
        decoder_hidden_size = _decoder_hidden_size(decoder_config)
        processor = AutoProcessor.from_pretrained(encoder_model_id)
        self._processor = processor
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._encoder_model_id = encoder_model_id

        logger.info(
            "[Qwen2_5VLBenchmarkEncoder] loading encoder=%s for decoder=%s on %s "
            "(preprocess_cache_size=%d preprocess_concurrency=%d)",
            encoder_model_id,
            model_id,
            self._device,
            cache_size,
            self.preprocess_concurrency,
        )
        full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            encoder_model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self._visual = full_model.model.visual.eval().to(self._device)
        self._validate_visual_architecture(self._visual)
        native_hidden_size = int(self._visual.config.out_hidden_size)
        output_hidden_size = _benchmark_output_hidden_size(
            native_hidden_size, decoder_hidden_size
        )
        self._output_hidden_size = output_hidden_size
        if output_hidden_size != native_hidden_size:
            logger.warning(
                "PERFORMANCE-ONLY Qwen2.5 adapter enabled: truncating the native "
                "vision output from %d to %d columns for decoder %s. This is not "
                "a trained projection and has no quality or model-parity claim.",
                native_hidden_size,
                output_hidden_size,
                model_id,
            )
        self._install_vllm_attention(self._visual)

        # Detach the retained module before dropping the full checkpoint. This
        # releases the duplicate language-model weights before vLLM starts.
        full_model.model.visual = None
        del full_model
        gc.collect()

        self._graphs = {}
        if self._device.type == "cuda" and self.buckets:
            self._capture_cuda_graphs()
        else:
            warmup_image = Image.new("RGB", (500, 500), color=(127, 127, 127))
            warmup_item = self._process_image(warmup_image)
            outputs = self.forward_batch([warmup_item] * self.max_batch_cost)
            del outputs, warmup_item
        logger.info(
            "[Qwen2_5VLBenchmarkEncoder] warmup complete: buckets=%s "
            "max_batch_cost=%d graphs=%d",
            self.buckets,
            self.max_batch_cost,
            len(self._graphs),
        )

    @staticmethod
    def _validate_visual_architecture(visual: Any) -> None:
        config = visual.config
        expected = {
            "depth": 32,
            "out_hidden_size": 2048,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "window_size": 112,
            "fullatt_block_indexes": [7, 15, 23, 31],
        }
        actual = {name: getattr(config, name, None) for name in expected}
        if actual != expected:
            raise ValueError(
                "Qwen2_5VLBenchmarkEncoder requires the validated Qwen2.5-VL "
                f"window-attention architecture; expected={expected}, actual={actual}"
            )

    @staticmethod
    def _install_vllm_attention(visual: Any) -> None:
        for layer_index, block in enumerate(visual.blocks):
            block.attn = _VllmQwen2_5VisionAttention(
                block.attn,
                prefix=f"custom_encoder.visual.blocks.{layer_index}.attn",
            ).to(device=visual.device, dtype=visual.dtype)

    def _require_processor(self) -> Any:
        processor = self._processor
        if processor is None:
            raise RuntimeError("Qwen2_5VLBenchmarkEncoder processor is not loaded")
        return processor

    def _require_visual(self) -> Any:
        visual = self._visual
        if visual is None:
            raise RuntimeError(
                "Qwen2_5VLBenchmarkEncoder.forward_batch() called before build()"
            )
        return visual

    def _require_output_hidden_size(self) -> int:
        output_hidden_size = self._output_hidden_size
        if output_hidden_size is None:
            raise RuntimeError(
                "Qwen2_5VLBenchmarkEncoder output width is not configured"
            )
        return output_hidden_size

    def preprocess(self, raw: str) -> Preprocessed[Qwen2VLImageInputs]:
        """Fetch/decode one image and run the CPU Qwen2.5-VL image processor."""
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
            raise ValueError("DYN_QWEN2_VL_PREPROCESS_CACHE_SIZE must be >= 0")
        self._cached_preprocess = (
            functools.lru_cache(maxsize=cache_size)(self._preprocess_uncached)
            if cache_size
            else None
        )

    def _preprocess_uncached(self, raw: str) -> Preprocessed[Qwen2VLImageInputs]:
        """Compute one read-only CPU input; optionally memoized by source string."""
        image = self._load_image(raw)
        item = self._process_image(image)
        grid_key = self._grid_key(item)
        if self._graphs and grid_key not in self._graph_grid_keys:
            raise ValueError(
                f"image grid {grid_key} has no captured CUDA graph; configure "
                "DYN_QWEN2_VL_GRAPH_IMAGE_SIZES before startup"
            )
        return Preprocessed(item=item, cost=1)

    def _process_image(self, image: Image.Image) -> Qwen2VLImageInputs:
        processor = self._require_processor()
        inputs = processor.image_processor(images=[image], return_tensors="pt")
        return Qwen2VLImageInputs(
            pixel_values=inputs["pixel_values"].contiguous(),
            image_grid_thw=inputs["image_grid_thw"].to(dtype=torch.long),
        )

    @staticmethod
    def _grid_key(item: Qwen2VLImageInputs) -> tuple[int, int, int]:
        if item.image_grid_thw.shape != (1, 3):
            raise ValueError(
                "Qwen2_5VLBenchmarkEncoder expects one image per preprocessed item; "
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
        templates: dict[tuple[int, int, int], Qwen2VLImageInputs] = {}
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
                forward = _StaticQwen2VLVisionForward(
                    visual,
                    grid_key,
                    bucket,
                    self._device,
                    self._require_output_hidden_size(),
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
                    "[Qwen2_5VLBenchmarkEncoder] captured CUDA graph: "
                    "grid=%s bucket=%d input_patches=%d output_tokens=%d",
                    grid_key,
                    bucket,
                    bucket * patches_per_item,
                    bucket * tokens_per_item,
                )
        torch.cuda.synchronize(self._device)
        free_after, _ = torch.cuda.mem_get_info(self._device)
        logger.info(
            "[Qwen2_5VLBenchmarkEncoder] CUDA graph capture complete: "
            "grids=%s buckets=%s graphs=%d device_memory_delta_gib=%.3f",
            sorted(templates),
            self.buckets,
            len(self._graphs),
            (free_before - free_after) / (1024**3),
        )

    def forward_batch(
        self,
        items: List[Qwen2VLImageInputs],
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
        return self._forward_uncached(items, target_bucket)

    def _forward_uncached(
        self,
        items: List[Qwen2VLImageInputs],
        target_bucket: Optional[int],
    ) -> List[torch.Tensor]:
        if target_bucket is not None:
            return self._forward_graph(items, target_bucket)
        if self.buckets:
            grid_keys = {self._grid_key(item) for item in items}
            if len(grid_keys) == 1:
                grid_key = next(iter(grid_keys))
                bucket = next(
                    (value for value in self.buckets if value >= len(items)),
                    None,
                )
                if bucket is not None and (grid_key, bucket) in self._graphs:
                    return self._forward_graph(items, bucket)
        return self._forward_eager(items)

    def _forward_eager(self, items: List[Qwen2VLImageInputs]) -> List[torch.Tensor]:
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
            first_block = next(iter(getattr(visual, "blocks", ())), None)
            if first_block is not None and isinstance(
                first_block.attn, _VllmQwen2_5VisionAttention
            ):
                image_embeds = _forward_vllm_vision_attention(
                    visual,
                    pixel_values,
                    image_grid_thw_cpu,
                    image_grid_thw,
                )
            else:
                image_embeds = visual(pixel_values, grid_thw=image_grid_thw)
            # Transformers 5.x wraps the merged vision tokens in
            # BaseModelOutputWithPooling; 4.x returns the tensor directly.
            image_embeds = getattr(image_embeds, "pooler_output", image_embeds)
            image_embeds = image_embeds[:, : self._require_output_hidden_size()]
        if events:
            events[2].record()
        # One batched D2H copy avoids a synchronization and allocation per image.
        # The returned splits are CPU views whose shared base owns independent
        # storage, so a later encoder call cannot overwrite them. Treat the views
        # as read-only: siblings intentionally alias that fresh base allocation.
        host_embeds = image_embeds.to(dtype=torch.bfloat16).cpu()
        outputs = list(torch.split(host_embeds, split_sizes))
        self._dispatch_counts[("eager", len(items), None)] += 1
        if _DISPATCH_LOG_ENABLED:
            logger.info(
                "custom_encoder_dispatch mode=eager batch_size=%d bucket=None",
                len(items),
            )
        self._log_cuda_timings(events, len(items), None, len(items))
        logger.debug(
            "[Qwen2_5VLBenchmarkEncoder] forward_batch n=%d tokens=%s",
            len(items),
            split_sizes,
        )
        return outputs

    def _forward_graph(
        self, items: List[Qwen2VLImageInputs], target_bucket: int
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
                "target_bucket has no captured Qwen2.5-VL CUDA graph for "
                f"grid={grid_key}, bucket={target_bucket}; configure "
                "DYN_QWEN2_VL_GRAPH_IMAGE_SIZES before startup"
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
        self._dispatch_counts[("graph", len(items), target_bucket)] += 1
        if _DISPATCH_LOG_ENABLED:
            logger.info(
                "custom_encoder_dispatch mode=graph batch_size=%d bucket=%d "
                "grid=%dx%dx%d",
                len(items),
                target_bucket,
                *grid_key,
            )
        self._log_cuda_timings(events, len(items), target_bucket, len(items))
        logger.debug(
            "[Qwen2_5VLBenchmarkEncoder] replayed CUDA graph: "
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
                "[Qwen2_5VLBenchmarkEncoder] preprocess cache: "
                "hits=%d misses=%d size=%d capacity=%d",
                cache_info.hits,
                cache_info.misses,
                cache_info.currsize,
                cache_info.maxsize,
            )
            self._cached_preprocess.cache_clear()
        self._cached_preprocess = None
        for (mode, batch_size, bucket), calls in sorted(
            self._dispatch_counts.items(),
            key=lambda item: (item[0][0], item[0][1], item[0][2] or 0),
        ):
            logger.info(
                "custom_encoder_dispatch_summary mode=%s batch_size=%d "
                "bucket=%s calls=%d",
                mode,
                batch_size,
                bucket,
                calls,
            )
        self._dispatch_counts.clear()
        self._visual = None
        self._processor = None
        self._graphs = {}
        self._graph_pool = None
        self._graph_grid_keys = frozenset()
        self._encoder_model_id = None
        self._output_hidden_size = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _load_image(source: str) -> Image.Image:
        """Load local, inline, or remote image bytes as owned RGB pixels."""
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
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB")
