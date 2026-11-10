# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton worker handling for the ``/v1/classify`` API.

Registers with ``skip_model_assets=True`` and reads the Triton model config
directly from ``config.pbtxt``.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import tritonclient.grpc.model_config_pb2 as TritonModelConfigPb
from tritonserver import Model as TritonModel
from tritonserver import Server as TritonServer

from dynamo.triton.classification import classification_label
from dynamo.triton.pooling_common import PoolingWorkerHandlerBase


class ClassifyWorkerHandler(PoolingWorkerHandlerBase):
    """Serve OpenAI ``/v1/classify`` on top of one Triton model.

    Mirrors the entry-point shape of ``dynamo.vllm.pooling_handlers.ClassifyWorkerHandler``:
    ``generate(request, context)`` receives an ``NvCreateClassifyRequest``
    already deserialized to a dict, and yields ``NvCreateClassifyResponse``
    as a dict. The frontend's aggregator folds the stream into the HTTP
    response.
    """

    task_label = "classify"
    cli_input_flag = "--classify-input-name"
    cli_output_flag = "--classify-output-name"
    endpoint = "/v1/classify"
    # NvCreateClassifyRequest carries controls the vLLM adapter honors during
    # encode. The Triton path cannot: tokenization, classify-head activation,
    # HF processor kwargs, prefix caching, and scheduling priority are handled
    # by the model plan or Triton internals, not the Python InferRequest.
    # Reject the Optional-typed fields explicitly (priority is handled by the
    # base ``reject_unsupported_controls`` because it is not Optional in the
    # wire schema). This keeps the wire contract honest: clients relying on
    # them get a 400 instead of a silently different classification result.
    unsupported_controls = (
        "use_activation",
        "add_special_tokens",
        "truncate_prompt_tokens",
        "truncation_side",
        "mm_processor_kwargs",
        "cache_salt",
    )

    def __init__(
        self,
        server: TritonServer,
        model: TritonModel,
        triton_model_config: TritonModelConfigPb.ModelConfig,
        classify_input_name: Optional[str] = None,
        classify_output_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            server,
            model,
            triton_model_config,
            input_name=classify_input_name,
            output_name=classify_output_name,
        )

    def _pre_dispatch(self, request: dict) -> None:
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

    def _row_from_output(self, arr_row: np.ndarray, inference_response: Any) -> dict:
        num_classes = int(arr_row.shape[0])
        probs_row = arr_row.tolist()
        argmax = int(np.argmax(arr_row)) if num_classes else None
        label = (
            classification_label(inference_response, self._output_name, argmax)
            if argmax is not None
            else None
        )
        return {
            "label": label,
            "probs": probs_row,
            "num_classes": num_classes,
        }

    def _response_envelope(
        self, data: list[dict], model_name: str, request: dict, context: Any
    ) -> dict:
        # Mirror vLLM's fallback so concurrent classify responses stay
        # correlatable even when the client omits ``request_id``. Context
        # is None on unit-test call sites that bypass the worker runtime.
        response_request_id = request.get("request_id") or (
            context.id() if context is not None else ""
        )
        return {
            "id": (
                f"classify-{response_request_id}" if response_request_id else "classify"
            ),
            "object": "list",
            "created": int(time.time()),
            "model": model_name,
            "data": data,
            "usage": {
                # Triton does not surface per-request prompt-token counts
                # from an ensemble; fields are kept for wire parity with
                # vLLM's response and default to 0.
                "prompt_tokens": 0,
                "total_tokens": 0,
                "completion_tokens": 0,
            },
        }

    def _probe_envelope(self) -> dict:
        return {
            "id": "",
            "object": "list",
            "created": 0,
            "model": self._model.name,
            "data": [],
            "usage": {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0},
        }
