# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton worker handling for the ``/v1/embeddings`` API.

Registers with ``skip_model_assets=True`` and reads the Triton model config
directly from ``config.pbtxt``.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, AsyncGenerator, Final, Optional

import numpy as np
import tritonclient.grpc.model_config_pb2 as mc
from tritonserver import MemoryType as TritonMemoryType
from tritonserver import Model as TritonModel
from tritonserver import Server as TritonServer
from tritonserver import Tensor as TritonTensor
from tritonserver import TritonError

from dynamo.common.backend.health_check import is_probe
from dynamo.triton.pooling_common import (
    extract_text_input,
    normalize_fp32_by_batching_contract,
    reject_unsupported_controls,
    require_batched_or_single,
    resolve_fp32_output_name,
    resolve_string_input_name,
    validate_string_input_dims,
)

logger = logging.getLogger(__name__)


class EmbeddingWorkerHandler:
    """Serve OpenAI ``/v1/embeddings`` on top of one Triton model.

    Always emits base64 on the worker->frontend wire; the frontend decodes
    to float at the HTTP boundary when ``encoding_format`` is not ``base64``.
    """

    def __init__(
        self,
        server: TritonServer,
        model: TritonModel,
        triton_model_config: mc.ModelConfig,
        embed_input_name: Optional[str] = None,
        embed_output_name: Optional[str] = None,
    ) -> None:
        self._server = server
        self._model = model
        self._config = triton_model_config
        self._input_name = resolve_string_input_name(
            self._config,
            model.name,
            embed_input_name,
            task_label="embedding",
            cli_flag="--embed-input-name",
        )
        validate_string_input_dims(
            self._config,
            self._input_name,
            model.name,
            task_label="embedding",
            endpoint="/v1/embeddings",
        )
        self._output_name = resolve_fp32_output_name(
            self._config,
            model.name,
            embed_output_name,
            task_label="embedding",
            cli_flag="--embed-output-name",
        )
        # Read batching from the parsed proto (not model.config()) so the
        # disk-fallback path in main.py._read_model_config still routes
        # batchable embeddings through the [N, 1] BYTES shape when the
        # runtime config is unavailable.
        self._batched = self._config.max_batch_size > 0
        logger.info(
            "Embedding worker for model '%s' initialized: input=%s, output=%s, batched=%s",
            model.name,
            self._input_name,
            self._output_name,
            self._batched,
        )

    async def generate(
        self, request: dict, context: Any = None
    ) -> AsyncGenerator[dict, None]:
        logger.debug("Received embeddings request for model %s", self._model.name)

        if is_probe(request):
            yield self._probe()
            return

        async for response in self._generate_embedding(request):
            yield response

    async def _generate_embedding(self, request: dict) -> AsyncGenerator[dict, None]:
        model_name = request.get("model") or self._model.name
        prompts = extract_text_input(request.get("input"), task_label="embedding")
        require_batched_or_single(
            prompts,
            self._batched,
            self._model.name,
            self._config.max_batch_size,
            task_label="embedding",
        )
        reject_unsupported_controls(
            request, _UNSUPPORTED_EMBEDDING_CONTROLS, task_label="embedding"
        )
        _validate_encoding_format(request.get("encoding_format"))

        # Send the whole batch through Triton in one InferRequest so the
        # backend's dynamic batcher sees them together.
        inference_request = self._model.create_request()
        # Triton BYTES input: object array of bytes-strings with shape [N, 1]
        # when the model is batchable (max_batch_size > 0), else [N].
        arr = np.array([[s.encode()] for s in prompts], dtype=object)
        if not self._batched:
            arr = arr.reshape(-1)
        inference_request.inputs[self._input_name] = arr

        prompt_tokens = 0
        data: list[dict[str, Any]] = []

        inference_responses = self._model.async_infer(inference_request)
        async for inference_response in inference_responses:
            output_tensor = inference_response.outputs[self._output_name]

            if (
                isinstance(output_tensor, TritonTensor)
                and output_tensor.memory_type != TritonMemoryType.CPU
            ):
                output_tensor = output_tensor.to_host()
            # copy=False makes astype a no-op on FP32 (the resolved output
            # is FP32 by construction), avoiding a payload-sized copy.
            embeds_arr = np.from_dlpack(output_tensor).astype(np.float32, copy=False)

            embeds_arr = normalize_fp32_by_batching_contract(
                embeds_arr,
                self._batched,
                self._model.name,
                self._output_name,
                self._config.max_batch_size,
            )

            for idx in range(embeds_arr.shape[0]):
                data.append(
                    {
                        "object": "embedding",
                        "embedding": _encode_fp32_vector_to_base64(embeds_arr[idx]),
                        "index": len(data),
                    }
                )

        # A batch of N inputs must produce exactly N embedding rows. A
        # non-batch-aligned Triton response (misconfigured ensemble, unbatched
        # model, wrong output shape) would otherwise silently return an
        # incomplete response with the wrong indices.
        if len(data) != len(prompts):
            raise RuntimeError(
                f"Triton model '{self._model.name}' returned {len(data)} "
                f"embedding row(s) for {len(prompts)} input(s); expected "
                "one row per input. Check the model's batching config or "
                "ensemble output shape."
            )

        yield {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                # Triton does not surface per-request prompt-token counts
                # from an ensemble; fields are kept for wire parity with
                # vLLM's response and default to 0.
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }

    def _probe(self) -> dict:
        try:
            if not self._server.ready():
                raise RuntimeError("server not ready")
            if not self._model.ready():
                raise RuntimeError(f"model {self._model.name} not ready")
        except TritonError as exc:
            raise RuntimeError(f"triton not ready: {exc}") from exc
        return {
            "object": "list",
            "model": self._model.name,
            "data": [],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }


# ``dimensions`` (Matryoshka truncation) and the tokenization kwargs are
# baked into the Triton model plan; clients relying on them get a 400 so a
# silent wrong-dim or wrong-tokenization vector is never possible.
_UNSUPPORTED_EMBEDDING_CONTROLS: Final[tuple[str, ...]] = (
    "dimensions",
    "add_special_tokens",
    "truncate_prompt_tokens",
)


_VALID_ENCODING_FORMATS: Final[tuple[str, ...]] = ("float", "base64")


def _validate_encoding_format(value: Any) -> None:
    if value is None or value in _VALID_ENCODING_FORMATS:
        return
    raise ValueError(
        f"the Triton embedding worker only supports encoding_format values "
        f"{list(_VALID_ENCODING_FORMATS)}; got {value!r}."
    )


def _encode_fp32_vector_to_base64(vec: np.ndarray) -> str:
    """Encode as OpenAI's base64 format: concatenated little-endian IEEE 754
    f32 bytes, standard-alphabet base64. Zero-copy on little-endian hosts."""
    return base64.b64encode(vec.astype("<f4", copy=False).tobytes()).decode("ascii")
