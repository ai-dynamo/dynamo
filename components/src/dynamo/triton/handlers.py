# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
from tritonserver import Model as TritonModel
from tritonserver import Server as TritonServer
from tritonserver import TritonError

from dynamo.common.backend.health_check import is_probe
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
        # Output schema is fixed at load time; cache to avoid a per-response lookup.
        self._output_dtypes: dict[str, str] = {
            out["name"]: out["datatype"] for out in model.metadata()["outputs"]
        }

    async def generate(self, request: dict) -> dict:
        logger.debug(f"Received request: {request}")

        # Short-circuit health probes before inference to avoid poisoning stateful models.
        if is_probe(request):
            yield self._probe()
            return

        self._validate_request(request)

        inference_request = self._model.create_request()
        for tensor in request["tensors"]:
            logger.debug(f"Tensor: {tensor}")
            arr = dynamo_tensor_to_numpy(tensor)
            inference_request.inputs[tensor["metadata"]["name"]] = arr

        inference_responses = self._model.async_infer(inference_request)
        async for inference_response in inference_responses:
            response_tensors = []
            for output_name, triton_dtype in self._output_dtypes.items():
                out_tensor = inference_response.outputs[output_name]
                if triton_dtype == "BYTES":
                    # String/BYTES tensors are not DLPack-compatible; pull them as
                    # an object array of bytes via the Triton Tensor API.
                    response_arr = out_tensor.to_bytes_array()
                else:
                    response_arr = np.from_dlpack(out_tensor)
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
        """Reject non-tensor requests and unsupported input dtypes early."""
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
