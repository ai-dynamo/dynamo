# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic output per prompt so tests can assert exact probabilities."""

import numpy as np
import triton_python_backend_utils as pb_utils

_NUM_CLASSES = 3


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            batch = text_tensor.as_numpy()
            n = batch.shape[0]
            probs = np.zeros((n, _NUM_CLASSES), dtype=np.float32)
            for i in range(n):
                raw = batch[i, 0]
                text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                probs[i, sum(ord(c) for c in text) % _NUM_CLASSES] = 1.0
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("probs", probs)]
                )
            )
        return responses
