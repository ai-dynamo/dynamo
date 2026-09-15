# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import AsyncGenerator

import numpy as np
from tritonserver import MemoryType as TritonMemoryType
from tritonserver import Model as TritonModel
from tritonserver import Server as TritonServer
from tritonserver import Tensor as TritonTensor
from tritonserver import TritonError

from dynamo.common.backend.health_check import is_probe
from dynamo.triton.classification import class_count, top_k_classifications
from dynamo.triton.util import (
    TRITON_TO_DYNAMO_DTYPE,
    dynamo_tensor_to_numpy,
    numpy_to_dynamo_values,
)

logger = logging.getLogger(__name__)

# Dynamo's tensor protocol currently lacks Float16 / BFloat16 variants.
# Remove this guard once upstream support lands.
_UNSUPPORTED_INPUT_DTYPES = frozenset({"Float16", "BFloat16"})


class RequestHandler:
    def __init__(self, server: TritonServer, model: TritonModel):
        self._server = server
        self._model = model
        # Output schema and batching are fixed at load time; cache to avoid a
        # per-response lookup.
        self._output_dtypes: dict[str, str] = {
            out["name"]: out["datatype"] for out in model.metadata()["outputs"]
        }
        self._batched = model.config().get("max_batch_size", 0) > 0

    async def generate(self, request: dict) -> AsyncGenerator[dict, None]:
        logger.debug(f"Received request: {request}")

        # Short-circuit health probes before inference to avoid poisoning stateful models.
        if is_probe(request):
            yield self._probe()
            return

        self._validate_request(request)
        requested_outputs = {
            output["name"]: output for output in request.get("outputs") or []
        }

        inference_request = self._model.create_request()
        for tensor in request["tensors"]:
            logger.debug(f"Tensor: {tensor}")
            arr = dynamo_tensor_to_numpy(tensor)
            inference_request.inputs[tensor["metadata"]["name"]] = arr

        inference_responses = self._model.async_infer(inference_request)
        async for inference_response in inference_responses:
            response_tensors = []
            for output_name, output_tensor in inference_response.outputs.items():
                requested_output = requested_outputs.get(output_name)
                if requested_outputs and requested_output is None:
                    continue

                triton_dtype = self._output_dtypes[output_name]
                req_class_count = (
                    class_count(requested_output.get("parameters") or {})
                    if requested_output is not None
                    else None
                )

                if triton_dtype == "BYTES":
                    # String/BYTES tensors are not DLPack-compatible; pull them as
                    # an object array of bytes via the Triton Tensor API.
                    response_arr = output_tensor.to_bytes_array()
                else:
                    # Handle GPU memory tensors by moving them to host memory
                    # before sending them to the worker protocol as a response.
                    if (
                        isinstance(output_tensor, TritonTensor)
                        and output_tensor.memory_type != TritonMemoryType.CPU
                    ):
                        output_tensor = output_tensor.to_host()

                    response_arr = np.from_dlpack(output_tensor)

                if req_class_count is not None:
                    response_arr = top_k_classifications(
                        response_arr,
                        req_class_count,
                        triton_dtype,
                        self._batched,
                        inference_response,
                        output_name,
                    )
                    triton_dtype = "BYTES"
                elif triton_dtype not in TRITON_TO_DYNAMO_DTYPE.keys():
                    raise TypeError(
                        f"Requested dtype '{triton_dtype}', for output '{output_name}', is not supported."
                    )

                dtype_str = TRITON_TO_DYNAMO_DTYPE.get(triton_dtype, triton_dtype)
                response_tensors.append(
                    {
                        "metadata": {
                            "name": output_name,
                            "shape": list(response_arr.shape),
                            "data_type": dtype_str,
                        },
                        "data": {
                            "data_type": dtype_str,
                            "values": numpy_to_dynamo_values(response_arr, dtype_str),
                        },
                    }
                )

            response = {
                "id": inference_response.request_id,
                "model": inference_response.model.name,
                "tensors": response_tensors,
            }

            yield response

    def _validate_request(self, request: dict) -> None:
        """Reject non-tensor requests, unsupported input dtypes, requested
        outputs the model does not produce, and an ill-formed classification
        parameter."""
        if "tensors" not in request:
            raise ValueError(
                "dynamo.triton only accepts tensor requests; missing 'tensors' "
                f"key. Received keys: {sorted(request.keys())}"
            )
        bad_dtypes = sorted(
            {
                tensor["metadata"]["data_type"]
                for tensor in request["tensors"]
                if tensor["metadata"]["data_type"] in _UNSUPPORTED_INPUT_DTYPES
            }
        )
        if bad_dtypes:
            raise ValueError(
                f"dynamo does not support {' / '.join(bad_dtypes)} input tensors."
            )
        unknown = sorted(
            output["name"]
            for output in request.get("outputs") or []
            if output["name"] not in self._output_dtypes
        )
        if unknown:
            names = " / ".join(f"'{name}'" for name in unknown)
            raise ValueError(
                f"unexpected inference output {names} for model "
                f"'{self._model.name}'"
            )
        for output in request.get("outputs") or []:
            class_count(output.get("parameters") or {})

    def _probe(self) -> dict:
        """Report readiness without invoking Triton inference.

        Server.ready() raises TritonError after stop(); collapse it into
        RuntimeError so the framework sees a single failure shape.
        """
        try:
            if not self._server.ready():
                raise RuntimeError("server not ready")
            if not self._model.ready():
                raise RuntimeError(f"model {self._model.name} not ready")
        except TritonError as exc:
            raise RuntimeError(f"triton not ready: {exc}") from exc
        return {"id": "", "model": self._model.name, "tensors": []}
