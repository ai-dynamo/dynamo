# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-aware adapters for in-process custom vision encoders."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Sequence

import torch

from dynamo.vllm.multimodal_utils.embed_assembler import build_mixed_embeds
from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    BackendEncodingSpecV1,
    EncodedMediaResultV1,
    EncodedMediaV1,
    ForwardItemV1,
    LinearRowsV1,
    Qwen2VLImageEncodingV1,
    VisionEncoderBackend,
)

_QWEN2_ABI = "vllm-qwen2-vl-external-v1"
_QWEN2_VLLM_VERSIONS = frozenset({"0.24.0", "0.25.1"})
_QWEN2_ARCHITECTURES = frozenset(
    {
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
    }
)
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class MixedEmbedsPlan:
    """Worker-private plan for the legacy mixed prompt-embeddings route."""

    prompt_embeds: torch.Tensor
    prompt_token_ids: list[int]
    prompt_is_token_ids: list[bool]


@dataclass(frozen=True)
class NativeMMPlan:
    """Worker-private plan for vLLM's native external multimodal route."""

    multi_modal_data: dict[str, Any]


PromptPlan = MixedEmbedsPlan | NativeMMPlan


def _model_architectures(model_config: Any) -> tuple[str, ...]:
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None)
    if architectures is None:
        architectures = getattr(model_config, "architectures", None)
    return tuple(str(architecture) for architecture in (architectures or ()))


def _hidden_size(model_config: Any) -> int:
    getter = getattr(model_config, "get_hidden_size", None)
    value = getter() if callable(getter) else None
    if value is None:
        hf_config = getattr(model_config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", None)
        value = getattr(text_config, "hidden_size", None)
        if value is None:
            value = getattr(hf_config, "hidden_size", None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("CustomEncoder could not resolve the decoder hidden size")
    return value


def _spatial_merge_size(model_config: Any) -> int:
    hf_config = getattr(model_config, "hf_config", None)
    vision_config = getattr(hf_config, "vision_config", None)
    value = getattr(vision_config, "spatial_merge_size", None)
    if value is None:
        value = getattr(hf_config, "spatial_merge_size", None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("CustomEncoder could not resolve Qwen spatial_merge_size")
    return value


def _required_token_id(model_config: Any, name: str) -> int:
    hf_config = getattr(model_config, "hf_config", None)
    value = getattr(hf_config, name, None)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"CustomEncoder could not resolve Qwen {name}")
    return value


def _decoder_fingerprint(
    model_config: Any,
    architecture: str,
    hidden_size: int,
    spatial_merge_size: int,
    dtype: torch.dtype,
) -> str:
    return (
        f"model={getattr(model_config, 'model', None)}:"
        f"revision={getattr(model_config, 'revision', None)}:{architecture}:"
        f"hidden={hidden_size}:merge={spatial_merge_size}:dtype={dtype}"
    )


class BoundCustomEncoderAdapter:
    """A setup-time binding from one backend contract to one engine model."""

    def __init__(
        self,
        backend: VisionEncoderBackend,
        model_config: Any,
        engine_args: Any,
        vllm_config: Any | None = None,
    ) -> None:
        self.spec = getattr(backend, "encoding_spec", None)
        self._image_token_id: int | None = None
        self._vision_start_token_id: int | None = None
        self._vision_end_token_id: int | None = None
        self._video_token_id: int | None = None
        if self.spec is not None and not isinstance(self.spec, BackendEncodingSpecV1):
            raise TypeError(
                "VisionEncoderBackend.encoding_spec must be a "
                f"BackendEncodingSpecV1 or None, got {self.spec!r}"
            )

        if self.spec is None or self.spec.adapter_abi == "linear-rows-v1":
            qwen_architectures = set(_model_architectures(model_config)) & set(
                _QWEN2_ARCHITECTURES
            )
            if qwen_architectures:
                raise ValueError(
                    "Qwen2/2.5-VL CustomEncoder output requires the native "
                    "vllm-qwen2-vl-external-v1 adapter, not linear-rows-v1"
                )
            if not getattr(engine_args, "enable_prompt_embeds", False):
                raise ValueError(
                    "linear CustomEncoder output requires --enable-prompt-embeds"
                )
            token_id = getattr(backend, "image_token_id", None)
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError(
                    "linear CustomEncoder output requires an integer image_token_id"
                )
            self._image_token_id = token_id
            if self.spec is not None:
                self._validate_common_spec(self.spec)
            return

        self._validate_qwen2_binding(model_config, engine_args, vllm_config)

    @property
    def uses_typed_results(self) -> bool:
        return self.spec is not None

    @property
    def uses_native_multimodal(self) -> bool:
        return self.spec is not None and self.spec.adapter_abi == _QWEN2_ABI

    def _validate_common_spec(self, spec: BackendEncodingSpecV1) -> None:
        if not spec.producer_fingerprint.strip():
            raise ValueError("CustomEncoder producer_fingerprint must not be empty")
        if spec.output_dtype not in _DTYPES:
            raise ValueError(
                f"CustomEncoder output_dtype {spec.output_dtype!r} is unsupported"
            )
        if spec.hidden_size < 1:
            raise ValueError("CustomEncoder hidden_size must be positive")

    def _validate_qwen2_binding(
        self, model_config: Any, engine_args: Any, vllm_config: Any | None
    ) -> None:
        spec = self.spec
        if spec is None or spec.adapter_abi != _QWEN2_ABI:
            raise ValueError(
                f"Unsupported CustomEncoder adapter ABI: {getattr(spec, 'adapter_abi', None)!r}"
            )
        self._validate_common_spec(spec)
        vllm_version = version("vllm").split("+", 1)[0]
        if vllm_version not in _QWEN2_VLLM_VERSIONS:
            raise ValueError(
                "Qwen CustomEncoder has no validated adapter for vLLM "
                f"{vllm_version}; supported versions are "
                f"{sorted(_QWEN2_VLLM_VERSIONS)}"
            )
        if model_config is None:
            raise ValueError(
                "Qwen CustomEncoder requires the resolved vLLM ModelConfig"
            )
        if not getattr(engine_args, "enable_mm_embeds", False):
            raise ValueError("Qwen CustomEncoder output requires --enable-mm-embeds")
        if getattr(engine_args, "language_model_only", False):
            raise ValueError(
                "Qwen CustomEncoder MVP does not yet support --language-model-only"
            )
        if getattr(engine_args, "tensor_parallel_size", 1) != 1:
            raise ValueError("Qwen CustomEncoder MVP requires tensor_parallel_size=1")
        if getattr(engine_args, "pipeline_parallel_size", 1) != 1:
            raise ValueError("Qwen CustomEncoder MVP requires pipeline_parallel_size=1")
        if getattr(engine_args, "data_parallel_size", 1) != 1:
            raise ValueError("Qwen CustomEncoder MVP requires data_parallel_size=1")
        compilation_config = getattr(engine_args, "compilation_config", None)
        encoder_cudagraphs = (
            compilation_config.get("cudagraph_mm_encoder", False)
            if isinstance(compilation_config, dict)
            else getattr(compilation_config, "cudagraph_mm_encoder", False)
        )
        if encoder_cudagraphs:
            raise ValueError(
                "Qwen CustomEncoder external embeddings are incompatible with "
                "vLLM multimodal encoder CUDA graphs"
            )
        cache_config = getattr(vllm_config, "cache_config", None)
        prefix_caching = getattr(
            cache_config,
            "enable_prefix_caching",
            getattr(engine_args, "enable_prefix_caching", False),
        )
        if prefix_caching:
            raise ValueError(
                "Qwen CustomEncoder MVP requires --no-enable-prefix-caching"
            )
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        chunked_prefill = getattr(
            scheduler_config,
            "enable_chunked_prefill",
            getattr(engine_args, "enable_chunked_prefill", False),
        )
        if chunked_prefill:
            raise ValueError(
                "Qwen CustomEncoder MVP requires --no-enable-chunked-prefill"
            )

        architectures = _model_architectures(model_config)
        matches = [arch for arch in architectures if arch in _QWEN2_ARCHITECTURES]
        if len(matches) != 1:
            raise ValueError(
                "Qwen CustomEncoder requires exactly one supported architecture "
                f"from {sorted(_QWEN2_ARCHITECTURES)}, got {architectures}"
            )
        hidden_size = _hidden_size(model_config)
        if hidden_size != spec.hidden_size:
            raise ValueError(
                "CustomEncoder hidden size does not match the decoder: "
                f"encoder={spec.hidden_size}, decoder={hidden_size}"
            )
        merge_size = _spatial_merge_size(model_config)
        if spec.spatial_merge_size != merge_size:
            raise ValueError(
                "CustomEncoder spatial merge size does not match the decoder: "
                f"encoder={spec.spatial_merge_size}, decoder={merge_size}"
            )
        model_dtype = getattr(model_config, "dtype", None)
        expected_dtype = _DTYPES[spec.output_dtype]
        if model_dtype != expected_dtype:
            raise ValueError(
                "CustomEncoder output dtype does not match the decoder: "
                f"encoder={expected_dtype}, decoder={model_dtype}"
            )
        actual_fingerprint = _decoder_fingerprint(
            model_config, matches[0], hidden_size, merge_size, model_dtype
        )
        if spec.expected_decoder_config_fingerprint is None:
            raise ValueError(
                "Qwen CustomEncoder requires expected_decoder_config_fingerprint"
            )
        if spec.expected_decoder_config_fingerprint != actual_fingerprint:
            raise ValueError(
                "CustomEncoder decoder fingerprint mismatch: "
                f"expected={spec.expected_decoder_config_fingerprint!r}, "
                f"actual={actual_fingerprint!r}"
            )

        self._image_token_id = _required_token_id(model_config, "image_token_id")
        self._vision_start_token_id = _required_token_id(
            model_config, "vision_start_token_id"
        )
        self._vision_end_token_id = _required_token_id(
            model_config, "vision_end_token_id"
        )
        self._video_token_id = _required_token_id(model_config, "video_token_id")

    def prepare_prompt_plan(
        self,
        token_ids: list[int],
        encodings: Sequence[torch.Tensor | EncodedMediaV1],
    ) -> PromptPlan:
        """Convert canonical encoder results to a closed worker prompt plan."""
        if not self.uses_native_multimodal:
            rows: list[torch.Tensor] = []
            for encoding in encodings:
                if isinstance(encoding, torch.Tensor):
                    rows.append(encoding)
                elif isinstance(encoding, LinearRowsV1):
                    rows.append(encoding.rows)
                else:
                    raise TypeError(
                        "linear CustomEncoder returned a non-linear media result"
                    )
            if self._image_token_id is None:
                raise RuntimeError("linear CustomEncoder adapter is not bound")
            prompt_embeds, mixed_token_ids, is_token_ids = build_mixed_embeds(
                token_ids, rows, self._image_token_id
            )
            return MixedEmbedsPlan(
                prompt_embeds=prompt_embeds,
                prompt_token_ids=mixed_token_ids,
                prompt_is_token_ids=is_token_ids,
            )

        qwen_encodings: list[Qwen2VLImageEncodingV1] = []
        for encoding in encodings:
            if not isinstance(encoding, Qwen2VLImageEncodingV1):
                raise TypeError("Qwen CustomEncoder returned a non-Qwen media result")
            qwen_encodings.append(encoding)
        self._validate_qwen_placeholders(token_ids, len(qwen_encodings))
        return NativeMMPlan(
            multi_modal_data={
                "image": {
                    "image_embeds": torch.cat(
                        [encoding.projected for encoding in qwen_encodings], dim=0
                    ),
                    "image_grid_thw": torch.tensor(
                        [encoding.grid_thw for encoding in qwen_encodings],
                        dtype=torch.int64,
                        device="cpu",
                    ),
                }
            }
        )

    def _validate_qwen_placeholders(
        self, token_ids: list[int], image_count: int
    ) -> None:
        image_token_id = self._image_token_id
        start_token_id = self._vision_start_token_id
        end_token_id = self._vision_end_token_id
        video_token_id = self._video_token_id
        if None in (image_token_id, start_token_id, end_token_id, video_token_id):
            raise RuntimeError("Qwen CustomEncoder adapter is not bound")
        if video_token_id in token_ids:
            raise ValueError("Qwen CustomEncoder image MVP rejects video placeholders")
        start_positions = [
            index
            for index, token_id in enumerate(token_ids)
            if token_id == start_token_id
        ]
        image_positions = [
            index
            for index, token_id in enumerate(token_ids)
            if token_id == image_token_id
        ]
        end_positions = [
            index
            for index, token_id in enumerate(token_ids)
            if token_id == end_token_id
        ]
        if not (
            len(start_positions)
            == len(image_positions)
            == len(end_positions)
            == image_count
        ):
            raise ValueError(
                "Qwen CustomEncoder requires exactly one canonical vision triple "
                f"per image: images={image_count}, starts={len(start_positions)}, "
                f"placeholders={len(image_positions)}, ends={len(end_positions)}"
            )
        for index in start_positions:
            if token_ids[index : index + 3] != [
                start_token_id,
                image_token_id,
                end_token_id,
            ]:
                raise ValueError(
                    "Qwen CustomEncoder requires one canonical unexpanded "
                    "<vision_start><image_pad><vision_end> group per image"
                )


def reconcile_and_canonicalize(
    spec: BackendEncodingSpecV1,
    submitted: Sequence[ForwardItemV1[Any]],
    returned: Sequence[EncodedMediaResultV1],
) -> list[EncodedMediaV1]:
    """Validate correlation IDs, restore input order, and own result storage."""
    expected_ids = [item.correlation_id for item in submitted]
    if len(returned) != len(expected_ids):
        raise ValueError(
            "CustomEncoder returned the wrong number of results: "
            f"expected={len(expected_ids)}, actual={len(returned)}"
        )
    by_id: dict[bytes, EncodedMediaV1] = {}
    expected_set = set(expected_ids)
    for result in returned:
        if not isinstance(result, EncodedMediaResultV1):
            raise TypeError(
                "typed CustomEncoder must return EncodedMediaResultV1 values"
            )
        correlation_id = result.correlation_id
        if correlation_id not in expected_set:
            raise ValueError("CustomEncoder returned an unknown correlation_id")
        if correlation_id in by_id:
            raise ValueError("CustomEncoder returned a duplicate correlation_id")
        by_id[correlation_id] = result.media
    if by_id.keys() != expected_set:
        raise ValueError("CustomEncoder omitted one or more correlation IDs")
    return [
        _canonicalize_media(spec, by_id[correlation_id])
        for correlation_id in expected_ids
    ]


def _canonicalize_media(
    spec: BackendEncodingSpecV1, media: EncodedMediaV1
) -> EncodedMediaV1:
    if spec.adapter_abi == "linear-rows-v1":
        if not isinstance(media, LinearRowsV1):
            raise TypeError("linear-rows-v1 requires LinearRowsV1 results")
        return LinearRowsV1(rows=_canonicalize_tensor(spec, media.rows))
    if spec.adapter_abi != _QWEN2_ABI:
        raise ValueError(f"Unsupported CustomEncoder adapter ABI: {spec.adapter_abi!r}")
    if not isinstance(media, Qwen2VLImageEncodingV1):
        raise TypeError(
            "vllm-qwen2-vl-external-v1 requires Qwen2VLImageEncodingV1 results"
        )
    grid = media.grid_thw
    if (
        not isinstance(grid, tuple)
        or len(grid) != 3
        or any(not isinstance(value, int) or isinstance(value, bool) for value in grid)
        or any(value < 1 for value in grid)
    ):
        raise ValueError(
            f"Qwen image grid_thw must be three positive integers, got {grid!r}"
        )
    temporal, height, width = grid
    if temporal != 1:
        raise ValueError("Qwen CustomEncoder MVP supports image T == 1 only")
    merge_size = spec.spatial_merge_size
    if merge_size is None or merge_size < 1:
        raise ValueError("Qwen CustomEncoder requires a positive spatial_merge_size")
    if height % merge_size or width % merge_size:
        raise ValueError(
            "Qwen image grid height and width must each be divisible by "
            f"spatial_merge_size={merge_size}; got {grid}"
        )
    projected = _canonicalize_tensor(spec, media.projected)
    expected_rows = temporal * height * width // (merge_size**2)
    if projected.shape[0] != expected_rows:
        raise ValueError(
            "Qwen projected row count does not match image_grid_thw: "
            f"expected={expected_rows}, actual={projected.shape[0]}, grid={grid}"
        )
    return Qwen2VLImageEncodingV1(projected=projected, grid_thw=grid)


def _canonicalize_tensor(
    spec: BackendEncodingSpecV1, tensor: torch.Tensor
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("CustomEncoder media rows must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise ValueError("CustomEncoder media rows must be on CPU before return")
    if tensor.layout is not torch.strided or not tensor.is_contiguous():
        raise ValueError("CustomEncoder media rows must be dense and contiguous")
    if tensor.requires_grad:
        raise ValueError("CustomEncoder media rows must have requires_grad=False")
    if tensor.ndim != 2 or tensor.shape[0] < 1:
        raise ValueError(
            "CustomEncoder media rows must have shape [positive_tokens, hidden_size]"
        )
    if tensor.shape[1] != spec.hidden_size:
        raise ValueError(
            "CustomEncoder media hidden width mismatch: "
            f"expected={spec.hidden_size}, actual={tensor.shape[1]}"
        )
    expected_dtype = _DTYPES[spec.output_dtype]
    if tensor.dtype != expected_dtype:
        raise ValueError(
            "CustomEncoder media dtype mismatch: "
            f"expected={expected_dtype}, actual={tensor.dtype}"
        )
    if not torch.isfinite(tensor).all().item():
        raise ValueError("CustomEncoder media rows contain NaN or Inf")
    return tensor.detach().clone(memory_format=torch.contiguous_format)
