# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve an inline custom encoder backed by a remote stock vLLM worker."""

from __future__ import annotations

import asyncio
import importlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from dynamo.common.backend import GenerateRequest
from dynamo.common.external_encoder import (
    ExternalEncoderResult,
    encode_request_plane_tensor,
)
from dynamo.experimental.endpoint import serve_unary_endpoint
from dynamo.experimental.llm import LLMUnaryClient
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.llm.exceptions import InvalidArgument
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.vllm.multimodal_utils.custom_encoder import (
    AsyncVisionEncoder,
    VisionEncoderBackend,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ENCODER_CLASS = (
    "examples.custom_encoder.hitchhikers_vision_encoder.HitchhikersVisionEncoder"
)

MODEL = os.environ.get("DYN_MODEL", DEFAULT_MODEL)
ENCODER_CLASS = os.environ.get("DYN_ENCODER_CLASS", DEFAULT_ENCODER_CLASS)
PUBLIC_MODEL_NAME = os.environ.get("DYN_SERVED_MODEL_NAME", "remote-custom-encoder")
DECODER_MODEL_NAME = os.environ.get(
    "DYN_DECODER_MODEL_NAME", "remote-custom-encoder-decoder"
)
NAMESPACE = os.environ.get("DYN_NAMESPACE", "remote-custom-encoder")
GENERATOR_ENDPOINT = f"{NAMESPACE}.generator.generate"
ORCHESTRATOR_ENDPOINT = f"{NAMESPACE}.orchestrator.generate"
CHAT_TEMPLATE = Path(__file__).resolve().parents[1] / "templates/qwen_vl.jinja"

_IMAGE_URL = "image_url"
_URL = "Url"
_RAW_MULTIMODAL_FIELDS = (
    "multi_modal_data",
    "multi_modal_uuids",
    "mm_processor_kwargs",
    "mm_routing_info",
)


class _EncoderDriver(Protocol):
    async def encode(self, raws: list[str]) -> list[Any]:
        ...

    def shutdown(self) -> None:
        ...


class InlineEncoder:
    """Drive a custom encoder and package its ordered linear embeddings."""

    def __init__(
        self,
        encoder: _EncoderDriver,
        image_token_id: int,
    ) -> None:
        if (
            isinstance(image_token_id, bool)
            or not isinstance(image_token_id, int)
            or image_token_id < 0
        ):
            raise ValueError("encoder backend requires a non-negative image_token_id")
        self._encoder = encoder
        self._image_token_id = image_token_id

    @classmethod
    def from_backend(
        cls,
        backend: VisionEncoderBackend[Any, Any, torch.Tensor],
        *,
        model: str,
    ) -> "InlineEncoder":
        """Load an author-provided backend through Dynamo's encoder driver."""

        image_token_id = getattr(backend, "image_token_id", None)
        encoder: AsyncVisionEncoder[Any, Any, torch.Tensor] = AsyncVisionEncoder(
            backend,
            name="remote-custom-encoder",
        )
        try:
            encoder.load(model)
            return cls(encoder, image_token_id)
        except BaseException:
            encoder.shutdown()
            raise

    async def encode(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Return the versioned request-plane result for one Generate request."""

        artifacts = await self._encoder.encode(_image_urls(request))
        tensors = _validate_artifacts(artifacts)
        row_splits = [0]
        for tensor in tensors:
            row_splits.append(row_splits[-1] + tensor.shape[0])

        packed = torch.cat(tensors, dim=0).contiguous()
        return ExternalEncoderResult(
            features=encode_request_plane_tensor(packed),
            row_splits=tuple(row_splits),
            image_token_id=self._image_token_id,
        ).to_dict()

    def close(self) -> None:
        """Release the encoder driver and backend resources."""

        self._encoder.shutdown()


class ExternalEncoderOrchestrator:
    """Encode locally, then invoke a remote aggregated vLLM endpoint."""

    def __init__(
        self,
        encoder: InlineEncoder,
        decoder: LLMUnaryClient,
        decoder_model_name: str,
    ) -> None:
        if not decoder_model_name:
            raise ValueError("decoder_model_name must not be empty")
        self._encoder = encoder
        self._decoder = decoder
        self._decoder_model_name = decoder_model_name

    async def __call__(
        self,
        request: Mapping[str, Any],
        *,
        context: Any,
    ) -> dict[str, Any]:
        encoder_result = await self._encoder.encode(request)
        decoder_request = _with_encoder_result(
            request,
            encoder_result,
            decoder_model_name=self._decoder_model_name,
        )
        return await self._decoder.complete(decoder_request, context=context)


def _image_urls(request: Mapping[str, Any]) -> list[str]:
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

    image_urls = []
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


def _validate_artifacts(artifacts: list[Any]) -> list[torch.Tensor]:
    if not artifacts:
        raise InvalidArgument("external encoder returned no image artifacts")

    tensors: list[torch.Tensor] = []
    hidden_size: int | None = None
    dtype: torch.dtype | None = None
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, torch.Tensor):
            raise InvalidArgument(
                f"external encoder artifact {index} must be a torch.Tensor"
            )
        if artifact.dim() != 2 or any(size <= 0 for size in artifact.shape):
            raise InvalidArgument(
                f"external encoder artifact {index} must be a non-empty 2D tensor"
            )
        if artifact.device.type != "cpu":
            raise InvalidArgument(f"external encoder artifact {index} must be on CPU")
        if hidden_size is None:
            hidden_size = artifact.shape[1]
            dtype = artifact.dtype
        elif artifact.shape[1] != hidden_size or artifact.dtype != dtype:
            raise InvalidArgument(
                "external encoder artifacts must have one hidden size and dtype"
            )
        tensors.append(artifact)
    return tensors


def _with_encoder_result(
    request_value: Mapping[str, Any],
    encoder_result: Mapping[str, Any],
    *,
    decoder_model_name: str,
) -> GenerateRequest:
    request = dict(request_value)
    if request.get("encoder_result") is not None:
        raise InvalidArgument("request already contains encoder_result")
    if request.get("prompt_embeds") is not None:
        raise InvalidArgument(
            "external encoder result cannot be combined with prompt_embeds"
        )

    request["encoder_result"] = dict(encoder_result)
    request["model"] = decoder_model_name
    for field_name in _RAW_MULTIMODAL_FIELDS:
        request.pop(field_name, None)

    extra_args = request.get("extra_args")
    if isinstance(extra_args, Mapping):
        copied_extra_args = dict(extra_args)
        copied_extra_args.pop("mm_kwargs_shm", None)
        copied_extra_args.pop("mm_kwargs_nixl", None)
        request["extra_args"] = copied_extra_args
    return cast(GenerateRequest, request)


def _resolve_backend_class(
    dotted_path: str,
) -> type[VisionEncoderBackend[Any, Any, torch.Tensor]]:
    module_path, separator, class_name = dotted_path.rpartition(".")
    if not separator:
        raise ValueError("DYN_ENCODER_CLASS must be a dotted module.ClassName path")
    backend_class = getattr(importlib.import_module(module_path), class_name)
    if not (
        isinstance(backend_class, type)
        and issubclass(backend_class, VisionEncoderBackend)
    ):
        raise TypeError(
            f"DYN_ENCODER_CLASS must resolve to VisionEncoderBackend; got {backend_class!r}"
        )
    return backend_class


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    decoder_client = await runtime.endpoint(GENERATOR_ENDPOINT).client()
    await decoder_client.wait_for_instances()

    backend_class = _resolve_backend_class(ENCODER_CLASS)
    encoder = InlineEncoder.from_backend(backend_class(), model=MODEL)
    try:
        endpoint = runtime.endpoint(ORCHESTRATOR_ENDPOINT)
        await register_model(
            ModelInput.Tokens,
            ModelType.Chat,
            endpoint,
            MODEL,
            model_name=PUBLIC_MODEL_NAME,
            custom_template_path=str(CHAT_TEMPLATE),
            worker_type=WorkerType.Aggregated,
            ignore_weights=True,
        )
        orchestrator = ExternalEncoderOrchestrator(
            encoder,
            LLMUnaryClient(decoder_client),
            DECODER_MODEL_NAME,
        )
        await serve_unary_endpoint(endpoint, orchestrator)
    finally:
        encoder.close()


def main() -> None:
    asyncio.run(worker())


if __name__ == "__main__":
    main()
