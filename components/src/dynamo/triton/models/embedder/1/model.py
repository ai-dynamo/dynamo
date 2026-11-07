# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic output per prompt so tests can assert the exact vector."""

import hashlib

import numpy as np
import triton_python_backend_utils as pb_utils

_EMBEDDING_DIM = 8


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            batch = text_tensor.as_numpy()
            n = batch.shape[0]
            vectors = np.zeros((n, _EMBEDDING_DIM), dtype=np.float32)
            for i in range(n):
                raw = batch[i, 0]
                text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                vectors[i] = (
                    np.frombuffer(digest[:_EMBEDDING_DIM], dtype=np.uint8) / 255.0
                )
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("embedding", vectors)]
                )
            )
        return responses
