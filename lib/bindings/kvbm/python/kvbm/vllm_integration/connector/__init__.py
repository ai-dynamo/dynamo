# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Back-compat shim for the legacy path ``kvbm.vllm_integration.connector``.

vLLM resolves ``kv_connector_module_path`` via
``importlib.import_module(path)`` + ``getattr(mod, "DynamoConnector")``.
We redirect attribute access to the canonical v1 module so
``DynamoConnector.__module__`` stays
``kvbm.v1.vllm_integration.connector.dynamo_connector``.

The explicit v2 path is ``kvbm.v2.vllm.schedulers.connector``.

Lazy by design: simply importing this module must not force ``import vllm``.
v1's connector transitively pulls vLLM, so we defer the import until vLLM
asks for ``DynamoConnector`` (at which point vLLM is definitionally present).
"""

_V1_EXPORTS = {
    "DynamoConnector": (
        "kvbm.v1.vllm_integration.connector.dynamo_connector",
        "DynamoConnector",
    ),
    "DynamoConnectorMetadata": (
        "kvbm.v1.vllm_integration.connector.dynamo_connector",
        "DynamoConnectorMetadata",
    ),
    "PdConnector": (
        "kvbm.v1.vllm_integration.connector.pd_connector",
        "PdConnector",
    ),
    "PdConnectorMetadata": (
        "kvbm.v1.vllm_integration.connector.pd_connector",
        "PdConnectorMetadata",
    ),
}

__all__ = list(_V1_EXPORTS)


def __getattr__(name):
    try:
        mod_path, attr = _V1_EXPORTS[name]
    except KeyError as e:
        raise AttributeError(
            f"module 'kvbm.vllm_integration.connector' has no attribute {name!r}"
        ) from e
    import importlib

    return getattr(importlib.import_module(mod_path), attr)


def __dir__():
    return sorted(set(list(globals().keys()) + list(_V1_EXPORTS)))
