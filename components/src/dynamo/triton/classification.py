# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KServe classification (top-K) for Triton model outputs.

A requested output carrying the KServe ``classification`` parameter asks for the
output's top-K classes as ``"<score>:<index>[:<label>]"`` strings instead of raw
values. Triton implements this in its own HTTP / gRPC frontends rather than in
the core, so the worker reproduces it here: the same descending-score ordering,
score formatting, and BYTES output shape.
"""

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

CLASSIFICATION_PARAMETER = "classification"

# Bound on the elements sorted per batch entry, matching Triton's own limit. A
# larger output would make the full sort below pathologically expensive.
MAX_CLASSIFICATION_ELEMENTS = 1_000_000


def class_count(parameters: Mapping[str, Any]) -> Optional[int]:
    """Number of classes a requested output asked for.

    Returns None when the output carries no ``classification`` parameter, which
    means it asked for raw values.
    """
    value = parameters.get(CLASSIFICATION_PARAMETER)
    if value is None:
        return None

    # Dynamo serializes a tensor parameter as a single-entry {tag: value} map.
    # For example, {"classification": {"int64": 10}} becomes 10.
    if isinstance(value, Mapping):
        value = value.get("int64")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"invalid value type for '{CLASSIFICATION_PARAMETER}' parameter, "
            "expected int64_param"
        )
    if value < 1:
        raise ValueError(
            f"invalid value for '{CLASSIFICATION_PARAMETER}' parameter, expected >= 1"
        )
    return value


def classification_label(
    response: Any, output_name: str, class_index: int
) -> Optional[str]:
    """Label for ``output_name`` at ``class_index``, or None if unlabelled.

    Labels sit on the C ``TRITONSERVER_InferenceResponse``, exposed as
    each tensor's ``memory_buffer.owner``. That binding takes an output
    index, in the order ``outputs`` was filled — the same
    ``(response, output_idx, class_index)`` lookup ``TopkClassifications``
    makes. The high-level Python ``InferenceResponse`` does not wrap this
    yet; if the owner has no lookup, class strings stay unlabeled.
    """
    try:
        output_idx = list(response.outputs).index(output_name)
        owner = response.outputs[output_name].memory_buffer.owner
    except (AttributeError, KeyError, ValueError):
        return None
    lookup = getattr(owner, "output_classification_label", None)
    if lookup is None:
        return None
    label = lookup(output_idx, class_index)
    return label or None


def top_k_classifications(
    array: np.ndarray,
    req_class_count: int,
    triton_dtype: str,
    batched: bool,
    response: Any = None,
    output_name: str = "",
) -> np.ndarray:
    """Top-K classes of one output as an object array of BYTES class strings.

    ``batched`` marks the leading dimension of ``array`` as a batch dimension,
    in which case the top-K is taken per batch entry and the result keeps that
    dimension. Class indices are relative to a batch entry, as Triton reports
    them. Labels are read from ``response`` when it is provided.
    ``triton_dtype`` is the model's output datatype (``FP32``, ``INT32``, …),
    the same string Triton's frontends pass into ``TopkClassifications``.
    """
    # FP32/FP64 and INT/UINT, matching TopkClassifications. FP16 is rejected.
    is_float = triton_dtype in ("FP32", "FP64")
    if not is_float and not triton_dtype.startswith(("UINT", "INT")):
        raise ValueError(
            "class result not available for output due to unsupported type "
            f"'{triton_dtype}'"
        )

    flat = array.reshape(-1)
    batch_size = int(array.shape[0]) if batched and array.ndim > 0 else 0
    element_cnt = flat.size // batch_size if batch_size else flat.size
    if element_cnt > MAX_CLASSIFICATION_ELEMENTS:
        raise ValueError("classification output tensor too large")

    class_cnt = min(req_class_count, element_cnt)
    class_strs = []
    for bs in range(max(1, batch_size)):
        probs = flat[bs * element_cnt : (bs + 1) * element_cnt]
        idx = np.argsort(probs)[::-1][:class_cnt]
        for k in range(class_cnt):
            kth_class_index = idx[k]
            # Format the score as a string, matching Triton's std::to_string.
            score = (
                f"{probs[kth_class_index]:.6f}"
                if is_float
                else str(int(probs[kth_class_index]))
            )
            class_str = f"{score}:{kth_class_index}"
            if response is not None:
                label = classification_label(
                    response, output_name, int(kth_class_index)
                )
                if label:
                    class_str += f":{label}"
            class_strs.append(class_str.encode())

    shape = (batch_size, class_cnt) if batch_size else (class_cnt,)
    return np.array(class_strs, dtype=object).reshape(shape)
