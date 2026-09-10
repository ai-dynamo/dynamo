# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM producer for the strict ``generation_artifact_v1`` format."""

from __future__ import annotations

import asyncio
import os
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from dynamo.artifacts.format_v1 import (
    GenerationArtifactChoice,
    GenerationArtifactView,
    encode_generation_artifact,
)
from dynamo.artifacts.storage import (
    ArtifactStorageError,
    ArtifactTarget,
    PresignedHttpPutTarget,
    put_artifact,
    target_from_settings,
)

_SUPPORTED_CONTENTS = frozenset({"moe_routes", "selected_logprobs"})
_DEFAULT_MAX_DECODED_BYTES = 64 << 20
_DEFAULT_PIPELINE_CONCURRENCY = 2
_PIPELINE_SEMAPHORES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class ArtifactCaptureError(ValueError):
    """Raised when a backend cannot satisfy an artifact request exactly."""


def generation_artifact_settings(
    request: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    extra_args = request.get("extra_args")
    for source in (
        request.get("nvext"),
        extra_args.get("nvext") if isinstance(extra_args, dict) else None,
    ):
        if not isinstance(source, dict) or "generation_artifact" not in source:
            continue
        settings = source["generation_artifact"]
        if not isinstance(settings, dict):
            raise ArtifactCaptureError("generation_artifact must be an object")
        return settings
    return None


def generation_artifact_contents(request: Mapping[str, Any]) -> frozenset[str]:
    settings = generation_artifact_settings(request)
    if settings is None:
        return frozenset()
    contents = settings.get("contents")
    if not isinstance(contents, list) or any(
        not isinstance(item, str) for item in contents
    ):
        raise ArtifactCaptureError(
            "generation_artifact.contents must be an array of strings"
        )
    if len(contents) != len(set(contents)):
        raise ArtifactCaptureError(
            "generation_artifact.contents must not contain duplicates"
        )
    unsupported = set(contents) - _SUPPORTED_CONTENTS
    if unsupported:
        raise ArtifactCaptureError(
            f"generation artifact content {min(unsupported)!r} is not supported by vLLM"
        )
    return frozenset(contents)


def _validate_generation_artifact_codec(settings: Mapping[str, Any]) -> None:
    codec = settings.get("codec", "zstd")
    if codec != "zstd":
        raise ArtifactCaptureError(
            "generation_artifact.codec must be the string 'zstd'"
        )


@dataclass(frozen=True)
class _ChoiceCapture:
    prompt_token_ids: tuple[int, ...] = ()
    completion_token_ids: tuple[int, ...] = ()
    selected_logprobs: tuple[float, ...] = ()
    routed_experts: Any | None = None
    previous: _ChoiceCapture | None = None


class VllmGenerationArtifactSession:
    """Captures one vLLM choice and delivers its final version 1 artifact."""

    def __init__(
        self,
        *,
        contents: frozenset[str],
        target: ArtifactTarget,
        model_config: Any,
    ) -> None:
        self.contents = contents
        self._target = target
        self._model_config = model_config
        self._choices: dict[int, _ChoiceCapture] = {}
        self._admitted = False

    @classmethod
    def from_backend_request(
        cls,
        request: Mapping[str, Any],
        *,
        model_config: Any,
        enable_rl: bool,
        route_capture_enabled: bool,
        choice_count: int,
    ) -> VllmGenerationArtifactSession | None:
        settings = generation_artifact_settings(request)
        if settings is None:
            return None
        if settings.get("format") != "generation_artifact_v1":
            raise ArtifactCaptureError("only generation_artifact_v1 is supported")
        if choice_count != 1:
            raise ArtifactCaptureError("generation artifacts currently require n=1")
        _validate_generation_artifact_codec(settings)
        contents = generation_artifact_contents(request)
        if "moe_routes" in contents and not enable_rl:
            raise ArtifactCaptureError(
                "moe_routes artifacts require Dynamo --enable-rl"
            )
        if "moe_routes" in contents and not route_capture_enabled:
            raise ArtifactCaptureError(
                "moe_routes artifacts require vLLM --enable-return-routed-experts"
            )
        try:
            target = target_from_settings(settings)
        except ArtifactStorageError as exc:
            raise ArtifactCaptureError(str(exc)) from exc
        return cls(contents=contents, target=target, model_config=model_config)

    def validate_route_start(self, token_start: int) -> None:
        if "moe_routes" in self.contents and token_start != 0:
            raise ArtifactCaptureError(
                "generation_artifact_v1 requires routed_experts_prompt_start=0"
            )

    async def admit(self, *, prompt_token_count: int, max_tokens: int) -> None:
        if self._admitted:
            raise ArtifactCaptureError(
                "generation artifact session was already admitted"
            )
        if prompt_token_count <= 0 or max_tokens <= 0:
            raise ArtifactCaptureError("generation artifact token bounds are invalid")
        sequence_length = prompt_token_count + max_tokens
        estimated_payload_bytes = sequence_length * np.dtype(np.int64).itemsize
        if "selected_logprobs" in self.contents:
            estimated_payload_bytes += max_tokens * np.dtype(np.float32).itemsize
        if "moe_routes" in self.contents:
            router_ids = _router_ids_from_config(self._model_config)
            expert_count = _expert_count_from_config(self._model_config)
            experts_per_token = _config_value(
                self._model_config,
                ("num_experts_per_tok", "num_experts_per_token", "moe_top_k"),
            )
            if (
                isinstance(experts_per_token, bool)
                or not isinstance(experts_per_token, int)
                or experts_per_token <= 0
            ):
                raise ArtifactCaptureError(
                    "model configuration has no trustworthy experts-per-token count"
                )
            estimated_payload_bytes += (
                max(0, sequence_length - 1)
                * len(router_ids)
                * experts_per_token
                * _canonical_integer_itemsize(expert_count - 1)
            )
        decoded_limit = _decoded_byte_limit()
        if estimated_payload_bytes > decoded_limit:
            raise ArtifactCaptureError(
                "generation artifact exceeds the decoded byte limit"
            )
        if (
            isinstance(self._target, PresignedHttpPutTarget)
            and estimated_payload_bytes + (64 << 10) > self._target.max_bytes
        ):
            raise ArtifactCaptureError(
                "generation artifact may exceed presigned target max_bytes"
            )
        _pipeline_concurrency()
        self._admitted = True

    def release(self) -> None:
        if self._admitted:
            self._admitted = False

    def record_chunk(
        self,
        *,
        choice_index: int,
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        selected_logprobs: list[float] | None,
        routed_experts: Any | None,
    ) -> None:
        previous = self._choices.get(choice_index)
        incoming_prompt = (
            tuple(int(token) for token in prompt_token_ids) if prompt_token_ids else ()
        )
        if (
            incoming_prompt
            and previous is not None
            and previous.prompt_token_ids
            and incoming_prompt != previous.prompt_token_ids
        ):
            raise ArtifactCaptureError("prompt token IDs changed during generation")
        prompt = (
            previous.prompt_token_ids
            if previous is not None and previous.prompt_token_ids
            else incoming_prompt
        )
        logprobs: tuple[float, ...] = ()
        if "selected_logprobs" in self.contents:
            if completion_token_ids and (
                selected_logprobs is None
                or len(selected_logprobs) != len(completion_token_ids)
            ):
                raise ArtifactCaptureError(
                    "selected logprobs are not aligned with completion tokens"
                )
            logprobs = tuple(float(value) for value in (selected_logprobs or ()))
        routes = (
            routed_experts
            if routed_experts is not None
            else previous.routed_experts
            if previous is not None
            else None
        )
        self._choices[choice_index] = _ChoiceCapture(
            prompt_token_ids=prompt,
            completion_token_ids=tuple(int(token) for token in completion_token_ids),
            selected_logprobs=logprobs,
            routed_experts=routes,
            previous=previous,
        )

    async def finalize_choice(
        self, *, choice_index: int, token_start: int
    ) -> dict[str, Any]:
        if choice_index != 0:
            raise ArtifactCaptureError(
                "generation artifacts currently require choice index 0"
            )
        self.validate_route_start(token_start)
        captured = self._choices.pop(choice_index, None)
        if captured is None or not captured.prompt_token_ids:
            raise ArtifactCaptureError(
                "generation artifact is missing prompt token IDs"
            )
        segments: list[_ChoiceCapture] = []
        segment: _ChoiceCapture | None = captured
        while segment is not None:
            segments.append(segment)
            segment = segment.previous
        segments.reverse()
        completion = tuple(
            token for segment in segments for token in segment.completion_token_ids
        )
        sequence = captured.prompt_token_ids + completion
        decoded_limit = _decoded_byte_limit()
        routes = None
        router_ids: tuple[int, ...] = ()
        expert_counts: tuple[int, ...] = ()
        if "moe_routes" in self.contents:
            if captured.routed_experts is None:
                raise ArtifactCaptureError(
                    "vLLM did not return requested routed experts"
                )
            routes = np.asarray(captured.routed_experts)
            expected_rows = max(0, len(sequence) - 1)
            if routes.ndim != 3 or routes.shape[0] != expected_rows:
                raise ArtifactCaptureError(
                    "vLLM routed experts are not aligned to sequence_token_ids"
                )
            router_ids, expert_counts = _resolve_router_layout(
                self._model_config, routes.shape[1]
            )
        selected = (
            np.asarray(
                [value for segment in segments for value in segment.selected_logprobs],
                dtype=np.float32,
            )
            if "selected_logprobs" in self.contents
            else None
        )
        estimated_payload_bytes = len(sequence) * np.dtype(np.int64).itemsize
        estimated_payload_bytes += routes.nbytes if routes is not None else 0
        estimated_payload_bytes += selected.nbytes if selected is not None else 0
        if decoded_limit <= 0 or estimated_payload_bytes > decoded_limit:
            raise ArtifactCaptureError(
                "generation artifact exceeds the decoded byte limit"
            )
        view = GenerationArtifactView(
            choices=(
                GenerationArtifactChoice(
                    choice_index=choice_index,
                    prompt_token_count=len(captured.prompt_token_ids),
                    sequence_token_ids=np.asarray(sequence, dtype=np.int64),
                    routed_experts=routes,
                    router_ids=router_ids,
                    expert_counts=expert_counts,
                    selected_logprobs=selected,
                    selected_logprobs_token_start=len(captured.prompt_token_ids),
                ),
            )
        )
        async with _pipeline_semaphore():
            encoded = await asyncio.to_thread(encode_generation_artifact, view)
            receipt = await put_artifact(encoded.data, self._target)
        return {
            "format": "generation_artifact_v1",
            "contents": sorted(self.contents),
            "state": "ready",
            "actual_bytes": receipt.actual_bytes,
            "sha256": receipt.sha256,
            "object_id": receipt.object_id,
        }


def _pipeline_concurrency() -> int:
    try:
        concurrency = int(
            os.environ.get(
                "DYN_GENERATION_ARTIFACT_PIPELINE_CONCURRENCY",
                str(_DEFAULT_PIPELINE_CONCURRENCY),
            )
        )
    except ValueError as exc:
        raise ArtifactCaptureError(
            "generation artifact pipeline concurrency is invalid"
        ) from exc
    if concurrency <= 0:
        raise ArtifactCaptureError(
            "generation artifact pipeline concurrency is invalid"
        )
    return concurrency


def _pipeline_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    concurrency = _pipeline_concurrency()
    cached = _PIPELINE_SEMAPHORES.get(loop)
    if cached is None or cached[0] != concurrency:
        semaphore = asyncio.Semaphore(concurrency)
        _PIPELINE_SEMAPHORES[loop] = (concurrency, semaphore)
        return semaphore
    return cached[1]


def _canonical_integer_itemsize(maximum: int) -> int:
    if maximum <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8).itemsize
    if maximum <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16).itemsize
    if maximum <= np.iinfo(np.int32).max:
        return np.dtype(np.int32).itemsize
    return np.dtype(np.int64).itemsize


def _config_value(config: Any, names: tuple[str, ...]) -> Any:
    hf_config = getattr(config, "hf_config", None)
    sources = [
        config,
        getattr(config, "hf_text_config", None),
        hf_config,
        getattr(hf_config, "text_config", None),
    ]
    values: list[Any] = []
    for source in sources:
        if source is None:
            continue
        for name in names:
            value = getattr(source, name, None)
            if value is not None:
                values.append(value)
    if not values:
        return None
    first = values[0]
    if any(type(value) is not type(first) or value != first for value in values[1:]):
        raise ArtifactCaptureError(
            f"model configuration has conflicting values for {names[0]}"
        )
    return first


def _expert_count_from_config(model_config: Any) -> int:
    expert_count = _config_value(
        model_config, ("num_experts", "n_routed_experts", "num_local_experts")
    )
    if (
        isinstance(expert_count, bool)
        or not isinstance(expert_count, int)
        or expert_count <= 0
        or expert_count - 1 > np.iinfo(np.int64).max
    ):
        raise ArtifactCaptureError(
            "model configuration has no trustworthy expert count"
        )
    return expert_count


def _decoded_byte_limit() -> int:
    try:
        limit = int(
            os.environ.get(
                "DYN_GENERATION_ARTIFACT_MAX_DECODED_BYTES",
                str(_DEFAULT_MAX_DECODED_BYTES),
            )
        )
    except ValueError as exc:
        raise ArtifactCaptureError(
            "generation artifact decoded byte limit is invalid"
        ) from exc
    if limit <= 0:
        raise ArtifactCaptureError("generation artifact decoded byte limit is invalid")
    return limit


def _router_ids_from_config(model_config: Any) -> tuple[int, ...]:
    layers = _config_value(model_config, ("num_hidden_layers", "n_layer"))
    first_dense = _config_value(model_config, ("first_k_dense_replace",)) or 0
    if (
        isinstance(layers, bool)
        or not isinstance(layers, int)
        or layers <= 0
        or isinstance(first_dense, bool)
        or not isinstance(first_dense, int)
        or not 0 <= first_dense <= layers
    ):
        raise ArtifactCaptureError(
            "model configuration has no trustworthy router layout"
        )

    candidates: list[tuple[int, ...]] = []
    frequency = _config_value(model_config, ("moe_layer_freq",))
    if isinstance(frequency, (list, tuple)):
        if len(frequency) != layers or any(
            not (
                isinstance(enabled, bool)
                or isinstance(enabled, int)
                and enabled in (0, 1)
            )
            for enabled in frequency
        ):
            raise ArtifactCaptureError(
                "model configuration has no trustworthy router layout"
            )
        candidates.append(
            tuple(index for index, enabled in enumerate(frequency) if enabled)
        )
    elif frequency is not None:
        if (
            isinstance(frequency, bool)
            or not isinstance(frequency, int)
            or frequency <= 0
        ):
            raise ArtifactCaptureError(
                "model configuration has no trustworthy router layout"
            )
        candidates.append(
            tuple(
                layer
                for layer in range(layers)
                if layer >= first_dense and layer % frequency == 0
            )
        )
    sparse_step = _config_value(model_config, ("decoder_sparse_step",))
    mlp_only_layers = _config_value(model_config, ("mlp_only_layers",)) or ()
    if sparse_step is not None:
        if (
            isinstance(sparse_step, bool)
            or not isinstance(sparse_step, int)
            or sparse_step <= 0
            or not isinstance(mlp_only_layers, (list, tuple, set))
            or any(
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or not 0 <= layer < layers
                for layer in mlp_only_layers
            )
        ):
            raise ArtifactCaptureError(
                "model configuration has no trustworthy router layout"
            )
        dense_layers = set(mlp_only_layers)
        candidates.append(
            tuple(
                layer
                for layer in range(layers)
                if layer >= first_dense
                and (layer + 1) % sparse_step == 0
                and layer not in dense_layers
            )
        )
    if not candidates:
        return tuple(range(first_dense, layers))
    if any(candidate != candidates[0] for candidate in candidates[1:]):
        raise ArtifactCaptureError("model router layout metadata is ambiguous")
    return candidates[0]


def _resolve_router_layout(
    model_config: Any, router_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if router_count <= 0:
        raise ArtifactCaptureError("routed experts must contain at least one router")
    expert_count = _expert_count_from_config(model_config)
    router_ids = _router_ids_from_config(model_config)
    if len(router_ids) != router_count:
        raise ArtifactCaptureError(
            "model router layout does not match vLLM route tensor"
        )
    return router_ids, (expert_count,) * router_count
