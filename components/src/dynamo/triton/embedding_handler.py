# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton worker handling for the ``/v1/embeddings`` API.

Registers with ``skip_model_assets=True`` and reads the Triton model config
directly from ``config.pbtxt``.
"""

from __future__ import annotations

import base64
from typing import Any, Final, Optional

import numpy as np
import tritonclient.grpc.model_config_pb2 as TritonModelConfigPb
from tritonserver import Model as TritonModel
from tritonserver import Server as TritonServer

from dynamo.triton.pooling_common import PoolingWorkerHandlerBase


class EmbeddingWorkerHandler(PoolingWorkerHandlerBase):
    """Serve OpenAI ``/v1/embeddings`` on top of one Triton model.

    Always emits base64 on the worker->frontend wire; the frontend decodes
    to float at the HTTP boundary when ``encoding_format`` is not ``base64``.
    """

    task_label = "embedding"
    cli_input_flag = "--embeddings-input-name"
    cli_output_flag = "--embeddings-output-name"
    endpoint = "/v1/embeddings"
    # ``dimensions`` (Matryoshka truncation) and the tokenization kwargs are
    # baked into the Triton model plan; clients relying on them get a 400 so a
    # silent wrong-dim or wrong-tokenization vector is never possible.
    unsupported_controls = (
        "dimensions",
        "add_special_tokens",
        "truncate_prompt_tokens",
    )

    def __init__(
        self,
        server: TritonServer,
        model: TritonModel,
        triton_model_config: TritonModelConfigPb.ModelConfig,
        embeddings_input_name: Optional[str] = None,
        embeddings_output_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            server,
            model,
            triton_model_config,
            input_name=embeddings_input_name,
            output_name=embeddings_output_name,
        )

    def _pre_dispatch(self, request: dict) -> None:
        _validate_encoding_format(request.get("encoding_format"))

    def _row_from_output(self, arr_row: np.ndarray, inference_response: Any) -> dict:
        return {
            "object": "embedding",
            "embedding": _encode_fp32_vector_to_base64(arr_row),
        }

    def _response_envelope(
        self, data: list[dict], model_name: str, request: dict, context: Any
    ) -> dict:
        return {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                # Triton does not surface per-request prompt-token counts
                # from an ensemble; fields are kept for wire parity with
                # vLLM's response and default to 0.
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }

    def _probe_envelope(self) -> dict:
        return {
            "object": "list",
            "model": self._model.name,
            "data": [],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }


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
