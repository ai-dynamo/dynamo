# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External handoff from a custom vision encoder to aggregated vLLM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from types import MappingProxyType
from typing import Any, cast

import torch

from dynamo.common.backend import GenerateRequest
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.multimodal_utils.custom_encoder.async_encoder import AsyncVisionEncoder
from dynamo.vllm.multimodal_utils.custom_encoder.backend import VisionEncoderBackend

EXTERNAL_ENCODER_RESULT_SCHEMA = "dynamo.external_encoder_result"
EXTERNAL_ENCODER_RESULT_VERSION = 0
LINEAR_EMBEDDINGS_FORMAT = "linear_embeddings"
REQUEST_PLANE_TRANSPORT = "request_plane_msgpack"

_IMAGE_URL = "image_url"
_URL = "Url"

EXTERNAL_ENCODER_REQUEST_FIELDS = (
    "multi_modal_data",
    "multi_modal_uuids",
    "prompt_embeds",
    "mm_processor_kwargs",
    "mm_routing_info",
    "media_io_kwargs",
)

EXTERNAL_ENCODER_EXTRA_ARG_FIELDS = (
    "mm_processor_kwargs",
    "mm_kwargs_shm",
    "mm_kwargs_nixl",
    "mm_hashes",
    "mm_hashes_by_modality",
    "mm_placeholders",
    "mm_placeholders_by_modality",
    "expanded_token_ids",
)

_SUPPORTED_DTYPES: Mapping[str, torch.dtype] = MappingProxyType(
    {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
)


def _check_keys(data: Mapping[str, Any], required: set[str], kind: str) -> None:
    missing = required - set(data)
    unknown = set(data) - required
    if missing:
        raise ValueError(f"{kind} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{kind} has unknown fields: {sorted(unknown)}")


def encode_request_plane_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Copy one CPU tensor into a MsgPack request-plane payload."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "external encoder features must be a torch.Tensor; "
            f"got {type(tensor).__name__}"
        )
    if tensor.device.type != "cpu":
        raise ValueError(f"external encoder tensor is on {tensor.device}; expected CPU")
    if not tensor.is_contiguous():
        raise ValueError("external encoder tensor must be contiguous")
    if tensor.dim() != 2 or any(dimension <= 0 for dimension in tensor.shape):
        raise ValueError(
            "external encoder tensor must be a non-empty 2D tensor; "
            f"got shape {tuple(tensor.shape)}"
        )
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    if dtype_name not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported external encoder tensor dtype {tensor.dtype}")

    raw = tensor.detach().view(torch.uint8).numpy().tobytes()
    return {
        "transport": REQUEST_PLANE_TRANSPORT,
        "shape": list(tensor.shape),
        "dtype": dtype_name,
        "data": raw,
    }


def _validate_request_plane_tensor_payload(
    payload: Mapping[str, Any],
) -> tuple[tuple[int, int], torch.dtype, bytes]:
    """Validate request-plane tensor metadata without allocating a tensor."""

    if not isinstance(payload, Mapping):
        raise ValueError("external encoder features must be an object")
    _check_keys(
        payload,
        {"transport", "shape", "dtype", "data"},
        "external encoder features",
    )
    if payload["transport"] != REQUEST_PLANE_TRANSPORT:
        raise ValueError(
            f"unsupported external encoder transport {payload['transport']!r}"
        )

    shape = payload["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape
        )
    ):
        raise ValueError(
            "external encoder tensor shape must contain two positive integers"
        )
    dtype_name = payload["dtype"]
    if not isinstance(dtype_name, str) or dtype_name not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported external encoder tensor dtype {dtype_name!r}")
    data = payload["data"]
    if not isinstance(data, bytes):
        raise ValueError("external encoder tensor data must be bytes")

    dtype = _SUPPORTED_DTYPES[dtype_name]
    expected_bytes = prod(shape) * torch.empty((), dtype=dtype).element_size()
    if len(data) != expected_bytes:
        raise ValueError(
            "external encoder tensor byte count does not match shape and dtype; "
            f"expected {expected_bytes}, got {len(data)}"
        )

    return (shape[0], shape[1]), dtype, data


def decode_request_plane_tensor(payload: Mapping[str, Any]) -> torch.Tensor:
    """Reconstruct an owned CPU tensor from a MsgPack request payload."""

    shape, dtype, data = _validate_request_plane_tensor_payload(payload)
    storage = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return storage.view(dtype).reshape(shape)


@dataclass(frozen=True)
class ExternalEncoderResult:
    """Versioned packed linear embeddings sent to aggregated vLLM."""

    features: Mapping[str, Any]
    row_splits: Sequence[int]
    image_token_id: int
    embedding_format: str = LINEAR_EMBEDDINGS_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.features, Mapping) or not self.features:
            raise ValueError("external encoder features must be a non-empty object")
        if self.embedding_format != LINEAR_EMBEDDINGS_FORMAT:
            raise ValueError(
                f"unsupported external encoder format {self.embedding_format!r}"
            )
        shape, _, _ = _validate_request_plane_tensor_payload(self.features)
        row_splits = tuple(self.row_splits)
        if len(row_splits) < 2 or row_splits[0] != 0:
            raise ValueError("external encoder row_splits must start at zero")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in row_splits
        ):
            raise ValueError(
                "external encoder row_splits must contain non-negative integers"
            )
        if any(left >= right for left, right in zip(row_splits, row_splits[1:])):
            raise ValueError("external encoder row_splits must be strictly increasing")
        if isinstance(self.image_token_id, bool) or not isinstance(
            self.image_token_id, int
        ):
            raise ValueError("external encoder image_token_id must be an integer")
        if self.image_token_id < 0:
            raise ValueError("external encoder image_token_id must be non-negative")
        if row_splits[-1] != shape[0]:
            raise ValueError(
                "external encoder row_splits do not cover the packed feature rows"
            )

        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
        object.__setattr__(self, "row_splits", row_splits)

    @classmethod
    def from_artifacts(
        cls,
        artifacts: Sequence[torch.Tensor],
        *,
        image_token_id: int,
    ) -> "ExternalEncoderResult":
        """Pack ordered per-image encoder artifacts into the wire contract."""

        if not artifacts:
            raise ValueError("external encoder returned no image artifacts")

        tensors: list[torch.Tensor] = []
        hidden_size: int | None = None
        dtype: torch.dtype | None = None
        row_splits = [0]
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, torch.Tensor):
                raise TypeError(
                    f"external encoder artifact {index} must be a torch.Tensor"
                )
            if artifact.dim() != 2 or any(size <= 0 for size in artifact.shape):
                raise ValueError(
                    f"external encoder artifact {index} must be a non-empty 2D tensor"
                )
            if artifact.device.type != "cpu":
                raise ValueError(f"external encoder artifact {index} must be on CPU")
            if hidden_size is None:
                hidden_size = artifact.shape[1]
                dtype = artifact.dtype
            elif artifact.shape[1] != hidden_size or artifact.dtype != dtype:
                raise ValueError(
                    "external encoder artifacts must have one hidden size and dtype"
                )
            tensors.append(artifact)
            row_splits.append(row_splits[-1] + artifact.shape[0])

        packed = torch.cat(tensors, dim=0).contiguous()
        return cls(
            features=encode_request_plane_tensor(packed),
            row_splits=row_splits,
            image_token_id=image_token_id,
        )

    def to_artifacts(self) -> list[torch.Tensor]:
        """Decode the wire payload into ordered per-image embedding tensors."""

        packed = decode_request_plane_tensor(self.features)
        return [
            packed[start:end]
            for start, end in zip(self.row_splits, self.row_splits[1:])
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned, MsgPack-compatible wire object."""

        return {
            "schema": EXTERNAL_ENCODER_RESULT_SCHEMA,
            "version": EXTERNAL_ENCODER_RESULT_VERSION,
            "format": self.embedding_format,
            "features": dict(self.features),
            "row_splits": list(self.row_splits),
            "image_token_id": self.image_token_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExternalEncoderResult":
        """Validate and parse the versioned wire object."""

        if not isinstance(data, Mapping):
            raise ValueError("external encoder result must be an object")
        _check_keys(
            data,
            {
                "schema",
                "version",
                "format",
                "features",
                "row_splits",
                "image_token_id",
            },
            "external encoder result",
        )
        if data["schema"] != EXTERNAL_ENCODER_RESULT_SCHEMA:
            raise ValueError(f"unsupported external encoder schema {data['schema']!r}")
        version = data["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != EXTERNAL_ENCODER_RESULT_VERSION
        ):
            raise ValueError(f"unsupported external encoder version {version!r}")
        row_splits = data["row_splits"]
        if not isinstance(row_splits, list):
            raise ValueError("external encoder row_splits must be an array")
        return cls(
            features=data["features"],
            row_splits=row_splits,
            image_token_id=data["image_token_id"],
            embedding_format=data["format"],
        )


def external_encoder_request_conflicts(request: Mapping[str, Any]) -> list[str]:
    """Return source-side multimodal fields that conflict with encoder_result."""

    conflicts = [
        key for key in EXTERNAL_ENCODER_REQUEST_FIELDS if request.get(key) is not None
    ]
    extra_args = request.get("extra_args")
    if isinstance(extra_args, Mapping):
        conflicts.extend(
            f"extra_args.{key}"
            for key in EXTERNAL_ENCODER_EXTRA_ARG_FIELDS
            if extra_args.get(key) is not None
        )
    return conflicts


def _extract_image_urls(request: Mapping[str, Any]) -> list[str]:
    multimodal = request.get("multi_modal_data") or {}
    if not isinstance(multimodal, Mapping):
        raise InvalidArgument("multi_modal_data must be an object")

    unsupported = sorted(
        key for key, value in multimodal.items() if key != _IMAGE_URL and value
    )
    if unsupported:
        raise InvalidArgument(
            "external encoder supports image inputs only; "
            f"got unsupported multimodal data: {unsupported}"
        )

    image_items = multimodal.get(_IMAGE_URL) or []
    if not isinstance(image_items, list) or not image_items:
        raise InvalidArgument("external encoder requires at least one image")

    image_urls: list[str] = []
    for index, item in enumerate(image_items):
        if not isinstance(item, Mapping):
            raise InvalidArgument(f"image_url item {index} must be an object")
        image_url = item.get(_URL)
        if not isinstance(image_url, str) or not image_url:
            raise InvalidArgument(
                f"image_url item {index} must contain a non-empty 'Url' string"
            )
        image_urls.append(image_url)
    return image_urls


def _build_downstream_request(
    request_value: Mapping[str, Any],
    encoder_result: ExternalEncoderResult,
    *,
    target_model: str,
) -> GenerateRequest:
    if not target_model:
        raise ValueError("target_model must not be empty")
    if request_value.get("encoder_result") is not None:
        raise InvalidArgument("request already contains encoder_result")
    if request_value.get("prompt_embeds") is not None:
        raise InvalidArgument(
            "external encoder result cannot be combined with prompt_embeds"
        )

    request = dict(request_value)
    request["encoder_result"] = encoder_result.to_dict()
    request["model"] = target_model
    for field_name in EXTERNAL_ENCODER_REQUEST_FIELDS:
        request.pop(field_name, None)

    extra_args = request.get("extra_args")
    if isinstance(extra_args, Mapping):
        copied_extra_args = dict(extra_args)
        for field_name in EXTERNAL_ENCODER_EXTRA_ARG_FIELDS:
            copied_extra_args.pop(field_name, None)
        request["extra_args"] = copied_extra_args

    return cast(GenerateRequest, request)


class ExternalEncoderHandoff:
    """Drive a ``VisionEncoderBackend`` and prepare a remote vLLM request."""

    def __init__(
        self,
        backend: VisionEncoderBackend[str, Any, torch.Tensor],
        *,
        preprocess_concurrency: int | None = None,
        name: str = "external-encoder",
    ) -> None:
        image_token_id = getattr(backend, "image_token_id", None)
        if (
            not isinstance(image_token_id, int)
            or isinstance(image_token_id, bool)
            or image_token_id < 0
        ):
            raise ValueError(
                "external linear encoder requires a non-negative integer "
                "image_token_id"
            )
        self._image_token_id = image_token_id
        self._encoder = AsyncVisionEncoder(
            backend,
            preprocess_concurrency=preprocess_concurrency,
            name=name,
        )

    def load(self, model_id: str) -> None:
        """Load the backend through Dynamo's maintained encoder driver."""

        self._encoder.load(model_id)

    async def prepare_request(
        self,
        request: Mapping[str, Any],
        *,
        target_model: str,
    ) -> GenerateRequest:
        """Encode request images and return a sanitized downstream request."""

        if not target_model:
            raise ValueError("target_model must not be empty")
        if request.get("encoder_result") is not None:
            raise InvalidArgument("request already contains encoder_result")
        if request.get("prompt_embeds") is not None:
            raise InvalidArgument(
                "external encoder result cannot be combined with prompt_embeds"
            )

        image_urls = _extract_image_urls(request)
        artifacts = await self._encoder.encode(image_urls)
        result = ExternalEncoderResult.from_artifacts(
            artifacts,
            image_token_id=self._image_token_id,
        )
        return _build_downstream_request(
            request,
            result,
            target_model=target_model,
        )

    def shutdown(self) -> None:
        """Release the encoder driver and backend resources."""

        self._encoder.shutdown()
