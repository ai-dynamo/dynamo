# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton worker handling for the ``/v1/classify`` API.

Registers with ``skip_model_assets=True`` and reads the Triton model config
directly from ``config.pbtxt``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncGenerator, Final, Optional

import numpy as np
import tritonclient.grpc.model_config_pb2 as mc
from tritonserver import MemoryType as TritonMemoryType
from tritonserver import Model as TritonModel
from tritonserver import Server as TritonServer
from tritonserver import Tensor as TritonTensor
from tritonserver import TritonError

from dynamo.common.backend.health_check import is_probe
from dynamo.triton.classification import classification_label
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


class ClassifyWorkerHandler:
    """Serve OpenAI ``/v1/classify`` on top of one Triton model.

    Mirrors the entry-point shape of ``dynamo.vllm.pooling_handlers.ClassifyWorkerHandler``:
    ``generate(request, context)`` receives an ``NvCreateClassifyRequest``
    already deserialized to a dict, and yields ``NvCreateClassifyResponse``
    as a dict. The frontend's aggregator folds the stream into the HTTP
    response.
    """

    def __init__(
        self,
        server: TritonServer,
        model: TritonModel,
        triton_model_config: mc.ModelConfig,
        classify_input_name: Optional[str] = None,
        classify_output_name: Optional[str] = None,
    ) -> None:
        self._server = server
        self._model = model
        self._config = triton_model_config
        self._input_name = resolve_string_input_name(
            self._config,
            model.name,
            classify_input_name,
            task_label="classify",
            cli_flag="--classify-input-name",
        )
        validate_string_input_dims(
            self._config,
            self._input_name,
            model.name,
            task_label="classify",
            endpoint="/v1/classify",
        )
        self._output_name = resolve_fp32_output_name(
            self._config,
            model.name,
            classify_output_name,
            task_label="classify",
            cli_flag="--classify-output-name",
        )
        # Read batching from the parsed proto (not model.config()) so the
        # disk-fallback path in main.py._read_model_config still routes
        # batchable classifiers through the [N, 1] BYTES shape when the
        # runtime config is unavailable.
        self._batched = self._config.max_batch_size > 0
        logger.info(
            "Classify worker for model '%s' initialized: input=%s, output=%s, batched=%s",
            model.name,
            self._input_name,
            self._output_name,
            self._batched,
        )

    async def generate(
        self, request: dict, context: Any = None
    ) -> AsyncGenerator[dict, None]:
        logger.debug("Received classify request for model %s", self._model.name)

        if is_probe(request):
            yield self._probe()
            return

        # NvCreatePoolingRequest always carries ``encoding_format``; the
        # NvCreateClassifyRequest never does. vLLM's shared handler uses the
        # same dispatch key. Pooling support lands in a follow-up; reject
        # rather than silently misroute to classify.
        if "encoding_format" in request:
            raise ValueError(
                "the Triton worker does not yet serve /v1/pooling; register "
                "with a pooling-capable backend or wait for the follow-up "
                "that adds pooling support to this worker"
            )

        async for response in self._generate_classify(request, context):
            yield response

    async def _generate_classify(
        self, request: dict, context: Any = None
    ) -> AsyncGenerator[dict, None]:
        model_name = request.get("model") or self._model.name
        prompts = extract_text_input(request.get("input"), task_label="classify")
        require_batched_or_single(
            prompts,
            self._batched,
            self._model.name,
            self._config.max_batch_size,
            task_label="classify",
        )
        reject_unsupported_controls(
            request, _UNSUPPORTED_CLASSIFY_CONTROLS, task_label="classify"
        )

        # Mirror vLLM's fallback so concurrent classify responses stay
        # correlatable even when the client omits ``request_id``. Context
        # is None on unit-test call sites that bypass the worker runtime.
        response_request_id = request.get("request_id") or (
            context.id() if context is not None else ""
        )

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
            probs_arr = np.from_dlpack(output_tensor).astype(np.float32, copy=False)

            probs_arr = normalize_fp32_by_batching_contract(
                probs_arr,
                self._batched,
                self._model.name,
                self._output_name,
                self._config.max_batch_size,
            )
            batch_size, num_classes = probs_arr.shape

            for idx in range(batch_size):
                probs_row = probs_arr[idx].tolist()
                argmax = int(np.argmax(probs_arr[idx])) if num_classes else None
                label = (
                    classification_label(inference_response, self._output_name, argmax)
                    if argmax is not None
                    else None
                )
                data.append(
                    {
                        "index": len(data),
                        "label": label,
                        "probs": probs_row,
                        "num_classes": num_classes,
                    }
                )

        # A batch of N inputs must produce exactly N classification rows. A
        # non-batch-aligned Triton response (misconfigured ensemble, unbatched
        # model, wrong output shape) would otherwise silently return an
        # incomplete response with the wrong indices.
        if len(data) != len(prompts):
            raise RuntimeError(
                f"Triton model '{self._model.name}' returned {len(data)} "
                f"classification row(s) for {len(prompts)} input(s); expected "
                "one row per input. Check the model's batching config or "
                "ensemble output shape."
            )

        yield {
            "id": f"classify-{response_request_id}"
            if response_request_id
            else "classify",
            "object": "list",
            "created": int(time.time()),
            "model": model_name,
            "data": data,
            "usage": {
                # Triton does not surface per-request prompt-token counts
                # from an ensemble; fields are kept for wire parity with
                # vLLM's response and default to 0.
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
                "completion_tokens": 0,
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
            "id": "",
            "object": "list",
            "created": 0,
            "model": self._model.name,
            "data": [],
            "usage": {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0},
        }


# NvCreateClassifyRequest carries controls the vLLM adapter honors during
# encode. The Triton path cannot: tokenization, classify-head activation,
# HF processor kwargs, prefix caching, and scheduling priority are handled
# by the model plan or Triton internals, not the Python InferRequest.
# Reject the Optional-typed fields explicitly (priority is handled below
# because it is not Optional in the wire schema). This keeps the wire
# contract honest: clients relying on them get a 400 instead of a silently
# different classification result.
_UNSUPPORTED_CLASSIFY_CONTROLS: Final[tuple[str, ...]] = (
    "use_activation",
    "add_special_tokens",
    "truncate_prompt_tokens",
    "truncation_side",
    "mm_processor_kwargs",
    "cache_salt",
)
