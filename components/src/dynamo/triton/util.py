# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helper utilities for the Dynamo Triton worker: dtype conversion, endpoint
name slugification, and tensor (de)serialization."""

import hashlib
import logging
import re

import numpy as np
from tritonclient.utils import triton_to_np_dtype

from dynamo.runtime.logging import log_level_mapping

# Mapping from Triton dtype (uppercase) to Dynamo dtype (camelCase)
TRITON_TO_DYNAMO_DTYPE = {
    "BOOL": "Bool",
    "UINT8": "Uint8",
    "UINT16": "Uint16",
    "UINT32": "Uint32",
    "UINT64": "Uint64",
    "INT8": "Int8",
    "INT16": "Int16",
    "INT32": "Int32",
    "INT64": "Int64",
    "FP16": "Float16",
    "FP32": "Float32",
    "FP64": "Float64",
    "BYTES": "Bytes",
}

# Inverse map: Dynamo dtype (camelCase) -> Triton dtype (uppercase). Used to feed
# tritonclient.triton_to_np_dtype, which keys on the Triton spelling ("FP32"),
# not the Dynamo one ("Float32").
DYNAMO_TO_TRITON_DTYPE = {v: k for k, v in TRITON_TO_DYNAMO_DTYPE.items()}


def create_triton_log_callback(logger_name: str = "triton"):
    """Build a callback (matching triton_runtime.Options.log_callback) that sends
    each Triton log record through the worker's Python logging pipeline.
    The worker already uses configure_dynamo_logging(), which adds Dynamo's
    LogHandler to the root logger, so Triton logs follow the same format as
    worker logs.
    """
    log = logging.getLogger(logger_name)

    def _forward(level, filename, line, _, message):
        name = getattr(level, "name", str(level)).lower()
        py_level = logging.DEBUG if name == "verbose" else log_level_mapping(name)
        if not log.isEnabledFor(py_level):
            return

        record = log.makeRecord(
            log.name,
            py_level,
            filename or "triton",
            int(line or 0),
            message,
            (),
            None,
            func="<module>",
        )
        log.handle(record)

    return _forward


def endpoint_slug(model_name: str) -> str:
    """Derive a unique, Dynamo-safe endpoint name from a Triton model name.

    The endpoint name is embedded into NATS subjects (dot-delimited, with '*'
    and '>' as wildcards) and into '/'-delimited discovery/TCP routing keys, so
    Dynamo restricts it to [a-z0-9-_]. Triton model names may contain commas,
    uppercase, and other characters (e.g. the L0_infer name 'nop_TYPE_BOOL_-1,-1'),
    which are folded to '_' here.

    A short content hash is appended for uniqueness. The discovery store finds an
    endpoint's models with a boundary-less prefix scan (the key
    '{ns}/{comp}/{endpoint}/{instance}' is matched with starts_with), so when one
    endpoint name is a textual prefix of another (e.g. 'nop_..._-1_-1' vs
    'nop_..._-1_-1_-1') the shorter name's query sweeps up the longer name's
    registration and surfaces a false "different model already registered"
    conflict. The hash makes the names diverge so neither is a prefix of the
    other (mirrors the runtime's Slug::slugify_unique).
    """
    safe = re.sub(r"[^a-z0-9_-]", "_", model_name.lower())
    digest = hashlib.sha256(model_name.encode()).hexdigest()[:8]
    return f"{safe}_{digest}"


def dynamo_tensor_to_numpy(tensor: dict) -> np.ndarray:
    """Convert a Dynamo tensor dict to a NumPy array for Triton inference."""
    shape = tensor["metadata"]["shape"]
    data_type = tensor["metadata"]["data_type"]
    values = tensor["data"]["values"]

    if data_type == "Bytes":
        # Dynamo sends BYTES as {"values": [[b0, b1, ...], ...]} — one byte
        # list per string element. Triton expects an object-dtype array of bytes.
        byte_strings = [
            item if isinstance(item, (bytes, str)) else bytes(item) for item in values
        ]
        return np.array(byte_strings, dtype=object).reshape(shape)

    triton_dtype = DYNAMO_TO_TRITON_DTYPE.get(data_type, data_type.upper())
    np_dtype = triton_to_np_dtype(triton_dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported tensor data type '{data_type}'")

    return np.array(values, dtype=np_dtype).reshape(shape)


def numpy_to_dynamo_values(arr: np.ndarray, data_type: str) -> list:
    """Convert a NumPy array to Dynamo tensor values."""
    if data_type == "Bytes":
        flat = arr.reshape(-1)
        return [
            list(item) if isinstance(item, (bytes, bytearray)) else item
            for item in flat
        ]

    return arr.flatten().tolist()
