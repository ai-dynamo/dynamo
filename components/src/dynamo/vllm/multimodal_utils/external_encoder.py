# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a workflow-produced external encoder result into a vLLM prompt."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import torch
from vllm.inputs import EmbedsPrompt

from dynamo.common.constants import EmbeddingTransferMode
from dynamo.common.external_encoder import ExternalEncoderResult
from dynamo.common.multimodal.embedding_transfer import (
    NixlWriteEmbeddingReceiver,
    TransferRequest,
)
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.linear import (
    build_mixed_embeds,
)
from dynamo.vllm.multimodal_utils.model_config import _hidden_size, _is_multimodal_model
from dynamo.workflow.perf import WORKFLOW_PERF_TRACE

logger = logging.getLogger(__name__)


class TensorImporter(Protocol):
    """Consumer-side subset of the workflow NIXL carrier."""

    async def import_tensor(self, reference: Mapping[str, Any]) -> Any:
        ...


class EmbeddingReceiver(Protocol):
    """Consumer-side subset of the stock embedding receiver."""

    async def receive_embeddings(
        self, request: TransferRequest
    ) -> tuple[int, torch.Tensor]:
        ...

    def release_tensor(self, tensor_id: int) -> None:
        ...


def _make_cpu_nixl_importer() -> TensorImporter:
    # Keep the workflow/NIXL dependency off the ordinary vLLM startup path.
    # The package (and its native connector) is loaded only when the first
    # request actually carries an external encoder result.
    from dynamo.workflow.nixl import NixlTensorCarrier

    return NixlTensorCarrier(receive_device="cpu")


def _make_nixl_write_receiver() -> EmbeddingReceiver:
    return NixlWriteEmbeddingReceiver()


class ExternalEncoderPromptLoader:
    """Import packed visual features and build a mixed ``EmbedsPrompt``."""

    def __init__(
        self,
        model_config: Any,
        engine_args: Any,
        *,
        importer_factory: Callable[[], TensorImporter] = _make_cpu_nixl_importer,
        receiver_factory: Callable[[], EmbeddingReceiver] = _make_nixl_write_receiver,
    ) -> None:
        if model_config is None:
            raise ValueError(
                "external encoder results require the resolved vLLM ModelConfig"
            )
        if _is_multimodal_model(model_config):
            raise ValueError(
                "external linear embeddings currently require a text-only decoder"
            )
        if not getattr(engine_args, "enable_prompt_embeds", False):
            raise ValueError(
                "external linear embeddings require --enable-prompt-embeds"
            )

        self._hidden_size = _hidden_size(model_config)
        model_dtype = getattr(model_config, "dtype", None)
        self._dtype = model_dtype if isinstance(model_dtype, torch.dtype) else None
        transfer_mode = getattr(
            engine_args,
            "embedding_transfer_mode",
            EmbeddingTransferMode.NIXL_WRITE,
        )
        if isinstance(transfer_mode, str):
            transfer_mode = EmbeddingTransferMode(transfer_mode)
        if transfer_mode not in (
            EmbeddingTransferMode.NIXL_WRITE,
            EmbeddingTransferMode.NIXL_READ,
        ):
            raise ValueError(
                "external encoder results require nixl-write or nixl-read "
                "embedding transfer mode"
            )
        self._transfer_mode = transfer_mode
        self._importer_factory = importer_factory
        self._receiver_factory = receiver_factory
        self._importer: TensorImporter | None = None
        self._receiver: EmbeddingReceiver | None = None

    async def load(
        self,
        encoder_result: Mapping[str, Any],
        token_ids: list[int],
        *,
        trace_id: str | None = None,
    ) -> EmbedsPrompt:
        """Read one packed feature tensor and adapt it to vLLM's mixed mode."""

        started_ns = time.perf_counter_ns()
        parsed = ExternalEncoderResult.from_dict(encoder_result)
        tensor_id: int | None = None
        if self._transfer_mode == EmbeddingTransferMode.NIXL_WRITE:
            if self._receiver is None:
                self._receiver = self._receiver_factory()
            request = TransferRequest.model_validate(parsed.features)
            tensor_id, packed = await self._receiver.receive_embeddings(request)
        else:
            if self._importer is None:
                self._importer = self._importer_factory()
            packed = await self._importer.import_tensor(parsed.features)
        received_ns = time.perf_counter_ns()
        try:
            # Building the dense mixed prompt performs CPU allocation and copies.
            # At the target 50 req/s workload this takes roughly one full event-loop
            # core, starving the receiver progress task that must observe subsequent
            # NIXL completions. Keep the ring-buffer view leased while a worker thread
            # materializes the owned prompt, then release it in the existing finally.
            prompt = await asyncio.to_thread(
                self._build_prompt,
                parsed,
                packed,
                token_ids,
            )
            completed_ns = time.perf_counter_ns()
            if trace_id is not None:
                WORKFLOW_PERF_TRACE.emit(
                    logger,
                    "external_encoder.prompt",
                    trace_id,
                    build_ms=(completed_ns - received_ns) / 1_000_000,
                    bytes=packed.numel() * packed.element_size(),
                    receive_ms=(received_ns - started_ns) / 1_000_000,
                    rows=packed.shape[0],
                    total_ms=(completed_ns - started_ns) / 1_000_000,
                )
            return prompt
        finally:
            if tensor_id is not None:
                assert self._receiver is not None
                self._receiver.release_tensor(tensor_id)

    def _build_prompt(
        self,
        parsed: ExternalEncoderResult,
        packed: Any,
        token_ids: list[int],
    ) -> EmbedsPrompt:
        if not isinstance(packed, torch.Tensor):
            raise TypeError(
                "external encoder NIXL import must produce a torch.Tensor; "
                f"got {type(packed).__name__}"
            )
        if packed.device.type != "cpu":
            raise ValueError(
                f"external encoder tensor is on {packed.device}; expected CPU"
            )
        if packed.dim() != 2 or packed.shape[1] != self._hidden_size:
            raise ValueError(
                f"external encoder tensor has shape {tuple(packed.shape)}; "
                f"expected 2D with decoder hidden size {self._hidden_size}"
            )
        if self._dtype is not None and packed.dtype != self._dtype:
            raise ValueError(
                f"external encoder tensor has dtype {packed.dtype}; "
                f"expected decoder dtype {self._dtype}"
            )
        if parsed.row_splits[-1] != packed.shape[0]:
            raise ValueError(
                "external encoder row_splits do not cover the imported tensor rows"
            )

        rows = [
            packed[start:end]
            for start, end in zip(parsed.row_splits, parsed.row_splits[1:])
        ]
        prompt_embeds, prompt_token_ids, prompt_is_token_ids = build_mixed_embeds(
            token_ids,
            rows,
            parsed.image_token_id,
        )
        return EmbedsPrompt(
            prompt_embeds=prompt_embeds,
            prompt_token_ids=prompt_token_ids,
            prompt_is_token_ids=prompt_is_token_ids,
        )
