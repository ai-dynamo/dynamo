# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers and base class shared by the Triton ``/v1/classify`` and
``/v1/embeddings`` workers.

Both handlers read the Triton model config directly from ``config.pbtxt``
and reject token-ID input, because a Triton token-input ensemble uses
per-model tensor names (``input_ids`` / ``attention_mask`` / etc.) that are
not carried on the OpenAI request.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, ClassVar, Final, Optional

import numpy as np
import tritonclient.grpc.model_config_pb2 as TritonModelConfigPb
from tritonserver import MemoryType as TritonMemoryType
from tritonserver import Model as TritonModel
from tritonserver import Server as TritonServer
from tritonserver import Tensor as TritonTensor
from tritonserver import TritonError

from dynamo.common.backend.health_check import is_probe

logger = logging.getLogger(__name__)

TYPE_STRING: Final[int] = TritonModelConfigPb.DataType.TYPE_STRING
TYPE_FP32: Final[int] = TritonModelConfigPb.DataType.TYPE_FP32


# Client-input errors are raised as plain ``ValueError``; the PyO3 binding
# maps that to ``BackendError::InvalidArgument`` in ``engine.rs``, which the
# frontend's classify and embeddings routes surface as HTTP 400. See
# ``lib/bindings/python/rust/engine.rs::py_err_to_dynamo`` and
# ``lib/llm/src/http/service/openai.rs::test_backend_invalid_argument_surfaces_as_400``.


def resolve_string_input_name(
    config: TritonModelConfigPb.ModelConfig,
    model_name: str,
    override: Optional[str],
    task_label: str,
    cli_flag: str,
) -> str:
    string_inputs = [i.name for i in config.input if i.data_type == TYPE_STRING]
    if override is not None:
        if override not in string_inputs:
            raise ValueError(
                f"Triton {task_label} model '{model_name}' has no "
                f"TYPE_STRING input named '{override}'; TYPE_STRING "
                f"inputs: {string_inputs}."
            )
        return override
    if len(string_inputs) != 1:
        raise ValueError(
            f"Triton {task_label} model '{model_name}' has "
            f"{len(string_inputs)} TYPE_STRING input tensor(s); "
            f"expected exactly 1. Pass {cli_flag} to disambiguate."
        )
    return string_inputs[0]


def validate_string_input_dims(
    config: TritonModelConfigPb.ModelConfig,
    name: str,
    model_name: str,
    task_label: str,
    endpoint: str,
) -> None:
    # The handler builds [N, 1] (batched) or [1] (unbatched) request
    # tensors, so the selected STRING input must declare dims=[1] (one
    # string per request item) or dims=[-1] (variable, which still
    # accepts shape 1). Reject other layouts at startup rather than
    # letting the model pass readiness and fail every request with a
    # Triton shape-mismatch error.
    selected = next(i for i in config.input if i.name == name)
    dims = list(selected.dims)
    if not (len(dims) == 1 and dims[0] in (1, -1)):
        raise ValueError(
            f"Triton {task_label} model '{model_name}' STRING input "
            f"'{name}' has dims={dims}; the {endpoint} path supports "
            "only dims=[1] or dims=[-1] (one string per request item). "
            "Change the model's config.pbtxt to one of the supported "
            "layouts, or serve the model with --task tensor and "
            "address it over KServe gRPC."
        )


def resolve_fp32_output_name(
    config: TritonModelConfigPb.ModelConfig,
    model_name: str,
    override: Optional[str],
    task_label: str,
    cli_flag: str,
) -> str:
    fp32_outputs = [o.name for o in config.output if o.data_type == TYPE_FP32]
    if override is not None:
        if override not in fp32_outputs:
            raise ValueError(
                f"Triton {task_label} model '{model_name}' has no "
                f"TYPE_FP32 output named '{override}'; TYPE_FP32 "
                f"outputs: {fp32_outputs}."
            )
        return override
    if len(fp32_outputs) != 1:
        raise ValueError(
            f"Triton {task_label} model '{model_name}' has "
            f"{len(fp32_outputs)} TYPE_FP32 output tensor(s); expected "
            f"exactly 1. Pass {cli_flag} to disambiguate."
        )
    return fp32_outputs[0]


def normalize_fp32_by_batching_contract(
    arr: np.ndarray,
    batched: bool,
    model_name: str,
    output_name: str,
    max_batch_size: int,
) -> np.ndarray:
    """Reshape to ``(batch_size, flat_width)``.

    Branches on the configured batching contract, not on tensor rank: an
    unbatched model with ``dims=[a, b]`` returns shape ``(a, b)`` that is
    one row, not ``a`` rows.
    """
    if batched:
        if arr.ndim < 2:
            raise RuntimeError(
                f"Triton model '{model_name}' declares "
                f"batching (max_batch_size={max_batch_size}) "
                f"but output '{output_name}' arrived with "
                f"shape {arr.shape}; expected a leading batch axis."
            )
        return arr.reshape(arr.shape[0], -1)
    return arr.reshape(1, -1)


def require_batched_or_single(
    prompts: list[str],
    batched: bool,
    model_name: str,
    max_batch_size: int,
    task_label: str,
) -> None:
    if not batched and len(prompts) > 1:
        raise ValueError(
            f"Triton {task_label} model '{model_name}' is unbatched "
            f"(max_batch_size={max_batch_size}) and received {len(prompts)} prompts. "
            "Send one prompt per request, or raise max_batch_size in "
            "the model's config.pbtxt."
        )
    # Reject at the handler before Triton returns a server-side shape
    # error so the client sees HTTP 400 with the model's limit, not 500.
    if batched and len(prompts) > max_batch_size:
        raise ValueError(
            f"Triton {task_label} model '{model_name}' accepts at most "
            f"{max_batch_size} prompts per request (max_batch_size); "
            f"received {len(prompts)}. Split the batch or raise "
            "max_batch_size in the model's config.pbtxt."
        )


def reject_unsupported_controls(
    request: dict,
    unsupported: tuple[str, ...],
    task_label: str,
) -> None:
    for field in unsupported:
        if request.get(field) is not None:
            raise ValueError(
                f"the Triton {task_label} worker does not honor '{field}'. "
                "Send the field unset, or run the model behind a backend "
                "that honors it (vLLM)."
            )
    # priority is declared as `i64` with `#[serde(default)]` and no
    # `skip_serializing_if`, so serde always emits it (default 0). Reject
    # only non-default values so the always-on-wire default passes through.
    if request.get("priority", 0) != 0:
        raise ValueError(
            f"the Triton {task_label} worker does not honor 'priority'; Triton's "
            "scheduling queue is not exposed to the Python worker. Send "
            "priority unset (0), or run the model behind a backend that "
            "honors it (vLLM)."
        )


def extract_text_input(input_field: Any, task_label: str) -> list[str]:
    """Convert the request's ``input`` field to a list of text prompts.

    Both ``ClassificationInput`` (lib/llm/src/protocols/openai/classify.rs)
    and ``EmbeddingInput`` (dynamo_protocols::types) are untagged enums
    over single-str / list-of-str / single-token-list / list-of-token-lists.
    The handler accepts text variants only; token-ID variants need per-model
    Triton tensor names the OpenAI request does not carry.
    """
    if input_field is None:
        raise ValueError(f"{task_label} request missing required 'input' field")

    if isinstance(input_field, str):
        if not input_field:
            raise ValueError(f"{task_label} 'input' cannot be an empty string")
        return [input_field]

    if isinstance(input_field, list):
        if not input_field:
            raise ValueError(f"{task_label} 'input' cannot be an empty list")
        if all(isinstance(item, str) for item in input_field):
            if any(not item for item in input_field):
                raise ValueError(
                    f"{task_label} 'input' list must not contain empty strings"
                )
            return list(input_field)
        # Anything non-str at this point is a token-ID variant.
        raise ValueError(
            f"the Triton {task_label} worker does not yet accept token-ID input "
            "(only text 'input' strings); ask the client to send text or run "
            "the model behind a backend that owns tokenization"
        )

    raise ValueError(
        f"{task_label} 'input' has unsupported type {type(input_field).__name__}; "
        "expected str or list[str]"
    )


class PoolingWorkerHandlerBase(ABC):
    """Shared pipeline for the Triton ``/v1/classify`` and ``/v1/embeddings`` workers."""

    # Subclass identity: set as class attributes on each concrete subclass.
    # No defaults, so a missing override fails at import rather than at
    # request time with an empty string in the error message.
    task_label: ClassVar[str]
    cli_input_flag: ClassVar[str]
    cli_output_flag: ClassVar[str]
    endpoint: ClassVar[str]
    unsupported_controls: ClassVar[tuple[str, ...]]

    # Per-instance state (assigned in ``__init__`` below).
    _server: TritonServer
    _model: TritonModel
    _config: TritonModelConfigPb.ModelConfig
    _input_name: str
    _output_name: str
    _batched: bool

    def __init__(
        self,
        server: TritonServer,
        model: TritonModel,
        triton_model_config: TritonModelConfigPb.ModelConfig,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> None:
        self._server = server
        self._model = model
        self._config = triton_model_config
        self._input_name = resolve_string_input_name(
            self._config,
            model.name,
            input_name,
            task_label=self.task_label,
            cli_flag=self.cli_input_flag,
        )
        validate_string_input_dims(
            self._config,
            self._input_name,
            model.name,
            task_label=self.task_label,
            endpoint=self.endpoint,
        )
        self._output_name = resolve_fp32_output_name(
            self._config,
            model.name,
            output_name,
            task_label=self.task_label,
            cli_flag=self.cli_output_flag,
        )
        # Read batching from the parsed proto (not model.config()) so the
        # disk-fallback path in main.py._read_model_config still routes
        # batchable models through the [N, 1] BYTES shape when the
        # runtime config is unavailable.
        self._batched = self._config.max_batch_size > 0
        logger.info(
            "%s worker for model '%s' initialized: input=%s, output=%s, batched=%s",
            self.task_label.capitalize(),
            model.name,
            self._input_name,
            self._output_name,
            self._batched,
        )

    async def generate(
        self, request: dict, context: Any = None
    ) -> AsyncGenerator[dict, None]:
        logger.debug(
            "Received %s request for model %s", self.task_label, self._model.name
        )

        if is_probe(request):
            yield self._probe()
            return

        self._pre_dispatch(request)

        async for response in self._run(request, context):
            yield response

    @abstractmethod
    def _pre_dispatch(self, request: dict) -> None:
        ...

    async def _run(self, request: dict, context: Any) -> AsyncGenerator[dict, None]:
        model_name = request.get("model") or self._model.name
        prompts = extract_text_input(request.get("input"), task_label=self.task_label)
        require_batched_or_single(
            prompts,
            self._batched,
            self._model.name,
            self._config.max_batch_size,
            task_label=self.task_label,
        )
        reject_unsupported_controls(
            request, self.unsupported_controls, task_label=self.task_label
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
            arr_out = np.from_dlpack(output_tensor).astype(np.float32, copy=False)

            arr_out = normalize_fp32_by_batching_contract(
                arr_out,
                self._batched,
                self._model.name,
                self._output_name,
                self._config.max_batch_size,
            )

            for idx in range(arr_out.shape[0]):
                row = self._row_from_output(arr_out[idx], inference_response)
                row["index"] = len(data)
                data.append(row)

        # A batch of N inputs must produce exactly N rows. A non-batch-aligned
        # Triton response (misconfigured ensemble, unbatched model, wrong
        # output shape) would otherwise silently return an incomplete response
        # with the wrong indices.
        if len(data) != len(prompts):
            raise RuntimeError(
                f"Triton model '{self._model.name}' returned {len(data)} "
                f"{self.task_label} row(s) for {len(prompts)} input(s); expected "
                "one row per input. Check the model's batching config or "
                "ensemble output shape."
            )

        yield self._response_envelope(data, model_name, request, context)

    @abstractmethod
    def _row_from_output(self, arr_row: np.ndarray, inference_response: Any) -> dict:
        ...

    @abstractmethod
    def _response_envelope(
        self, data: list[dict], model_name: str, request: dict, context: Any
    ) -> dict:
        ...

    def _probe(self) -> dict:
        try:
            if not self._server.ready():
                raise RuntimeError("server not ready")
            if not self._model.ready():
                raise RuntimeError(f"model {self._model.name} not ready")
        except TritonError as exc:
            raise RuntimeError(f"triton not ready: {exc}") from exc
        return self._probe_envelope()

    @abstractmethod
    def _probe_envelope(self) -> dict:
        ...
