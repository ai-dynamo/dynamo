# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton worker handling for the ``/v1/classify`` API.

Sits alongside the tensor ``RequestHandler`` in ``handlers.py`` and is selected
at worker startup by ``--task classify``. The Rust classify PushRouter ships a
serialized ``NvCreateClassifyRequest`` onto the endpoint as a dict; this module
turns it into a Triton ``InferRequest`` (single BYTES text tensor), invokes
the model, reads the FP32 probability output, and returns the OpenAI-style
classify response dict shape the frontend aggregator expects.

The wire matches vLLM's ``/classify`` (`vllm/entrypoints/pooling/classify/protocol.py`)
via `lib/llm/src/protocols/openai/classify.rs`. ``/v1/pooling`` support will
land alongside; today the worker registers ``ModelType.Classify`` only.

Scope: text input (``Single`` / ``Batch`` variants of
``ClassificationInput``). Token-ID input (``Tokens`` / ``TokenBatch``) is
rejected with a clear error. A Triton token-input ensemble uses per-model
tensor names (``input_ids`` / ``attention_mask`` / ``token_type_ids``), so
supporting it needs per-model wiring that is out of scope here.
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

logger = logging.getLogger(__name__)


# Triton protobuf ``DataType`` enum values. Cached at module import so the
# hot path doesn't re-hit the protobuf descriptor on every request.
_TYPE_STRING: Final[int] = mc.DataType.TYPE_STRING
_TYPE_FP32: Final[int] = mc.DataType.TYPE_FP32


# Client-input errors are raised as plain ``ValueError``; the PyO3 binding
# maps that to ``BackendError::InvalidArgument`` in ``engine.rs``, which the
# frontend's classify route surfaces as HTTP 400. See
# ``lib/bindings/python/rust/engine.rs::py_err_to_dynamo`` and
# ``lib/llm/src/http/service/openai.rs::test_backend_invalid_argument_surfaces_as_400``.


class ClassifyWorkerHandler:
    """Serve OpenAI ``/v1/classify`` on top of one Triton model.

    Mirrors the entry-point shape of ``dynamo.vllm.pooling_handlers.ClassifyWorkerHandler``:
    ``generate(request, context)`` receives an ``NvCreateClassifyRequest``
    already deserialized to a dict, and yields ``NvCreateClassifyResponse``
    as a dict — the frontend's aggregator folds the stream into the HTTP
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
        self._input_name = classify_input_name or self._resolve_input_name()
        self._output_name = classify_output_name or self._resolve_output_name()
        # Batching is fixed at load time; cache the batchable flag so we don't
        # re-read the model config per request.
        self._batched = model.config().get("max_batch_size", 0) > 0
        logger.info(
            "Classify worker for model '%s' initialized: input=%s, output=%s, batched=%s",
            model.name,
            self._input_name,
            self._output_name,
            self._batched,
        )

    # ------------------------------------------------------------------
    # Input / output tensor resolution
    # ------------------------------------------------------------------

    def _resolve_input_name(self) -> str:
        """Find the sole BYTES input in the model's ``config.pbtxt``.

        A classify-shaped Triton model — leaf or ensemble — accepts one
        BYTES tensor of text and produces one FP32 tensor of probabilities.
        Anything else needs the operator to disambiguate with
        ``--classify-input-name``.
        """
        names = [i.name for i in self._config.input if i.data_type == _TYPE_STRING]
        if len(names) != 1:
            raise ValueError(
                f"Triton classify model '{self._model.name}' has {len(names)} "
                "TYPE_STRING input tensor(s); expected exactly 1. Pass "
                "--classify-input-name to disambiguate."
            )
        return names[0]

    def _resolve_output_name(self) -> str:
        """Find the sole FP32 output in the model's ``config.pbtxt``."""
        names = [o.name for o in self._config.output if o.data_type == _TYPE_FP32]
        if len(names) != 1:
            raise ValueError(
                f"Triton classify model '{self._model.name}' has {len(names)} "
                "TYPE_FP32 output tensor(s); expected exactly 1. Pass "
                "--classify-output-name to disambiguate."
            )
        return names[0]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def generate(
        self, request: dict, context: Any = None
    ) -> AsyncGenerator[dict, None]:
        logger.debug(f"Received classify request: {request}")

        # Health probes short-circuit before inference to match RequestHandler.
        if is_probe(request):
            yield self._probe()
            return

        # NvCreatePoolingRequest always carries ``encoding_format``; the
        # NvCreateClassifyRequest never does. That's the same dispatch key
        # vLLM's shared handler uses. We reject pooling explicitly rather
        # than silently misroute — pooling support lands in a follow-up.
        if "encoding_format" in request:
            raise ValueError(
                "the Triton worker does not yet serve /v1/pooling; register "
                "with a pooling-capable backend or wait for the follow-up "
                "that adds pooling support to this worker"
            )

        async for response in self._generate_classify(request, context):
            yield response

    # ------------------------------------------------------------------
    # Classify path
    # ------------------------------------------------------------------

    async def _generate_classify(
        self, request: dict, context: Any = None
    ) -> AsyncGenerator[dict, None]:
        model_name = request.get("model") or self._model.name
        prompts = _extract_text_input(request.get("input"))
        _reject_unsupported_controls(request)

        # Mirror vLLM's fallback so concurrent classify responses stay
        # correlatable even when the client omits ``request_id``. Context
        # is None on unit-test call sites that bypass the worker runtime.
        response_request_id = request.get("request_id") or (
            context.id() if context is not None else ""
        )

        # Send the whole batch through Triton in one InferRequest so the
        # backend's dynamic batcher sees them together. Each response tensor
        # slot corresponds to one input string; the top-level classify
        # response's ``data`` array is one entry per input, ordered by index.
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

            # FP32 outputs are DLPack-compatible; move GPU tensors to host
            # first so numpy can consume them.
            if (
                isinstance(output_tensor, TritonTensor)
                and output_tensor.memory_type != TritonMemoryType.CPU
            ):
                output_tensor = output_tensor.to_host()
            probs_arr = np.from_dlpack(output_tensor).astype(np.float32)

            # Batched output shape is [N, num_classes]; unbatched is
            # [num_classes]. Normalize to a 2D view so per-batch slicing is
            # uniform.
            if probs_arr.ndim == 1:
                probs_arr = probs_arr.reshape(1, -1)
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

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def _probe(self) -> dict:
        """Report readiness without invoking Triton inference."""
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


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


# NvCreateClassifyRequest exposes tokenization and activation controls that the
# vLLM adapter honors during encode. The Triton path cannot: the tokenizer and
# classify-head activation live inside the model plan and are not reachable
# from a Python InferRequest. Rejecting the fields explicitly keeps the wire
# contract honest: clients that rely on them get a 400 instead of a silently
# different classification result.
_UNSUPPORTED_CLASSIFY_CONTROLS: Final[tuple[str, ...]] = (
    "use_activation",
    "add_special_tokens",
    "truncate_prompt_tokens",
    "truncation_side",
)


def _reject_unsupported_controls(request: dict) -> None:
    for field in _UNSUPPORTED_CLASSIFY_CONTROLS:
        if request.get(field) is not None:
            raise ValueError(
                f"the Triton classify worker does not honor '{field}'; the "
                "Triton model plan owns tokenization and the classify head's "
                "activation. Send the field unset, or run the model behind a "
                "backend that applies it (vLLM)."
            )


def _extract_text_input(input_field: Any) -> list[str]:
    """Turn ``NvCreateClassifyRequest.input`` into a list of text prompts.

    ``ClassificationInput`` in ``lib/llm/src/protocols/openai/classify.rs`` is
    an untagged enum of four variants: ``Single(str)``, ``Batch(list[str])``,
    ``Tokens(list[int])``, ``TokenBatch(list[list[int]])``. This handler
    supports only the text variants; token-ID variants are rejected with 400
    since Triton token-input needs per-model tensor names not carried on the
    OpenAI request.
    """
    if input_field is None:
        raise ValueError("classify request missing required 'input' field")

    if isinstance(input_field, str):
        if not input_field:
            raise ValueError("classify 'input' cannot be an empty string")
        return [input_field]

    if isinstance(input_field, list):
        if not input_field:
            raise ValueError("classify 'input' cannot be an empty list")
        if all(isinstance(item, str) for item in input_field):
            if any(not item for item in input_field):
                raise ValueError("classify 'input' list must not contain empty strings")
            return list(input_field)
        # Anything non-str at this point is a token-ID variant.
        raise ValueError(
            "the Triton classify worker does not yet accept token-ID input "
            "(only text 'input' strings); ask the client to send text or run "
            "the model behind a backend that owns tokenization"
        )

    raise ValueError(
        f"classify 'input' has unsupported type {type(input_field).__name__}; "
        "expected str or list[str]"
    )
