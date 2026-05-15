# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical v1 connector façade — lazy re-export of the v1 impl.

vLLM resolves ``kv_connector_module_path`` via
``importlib.import_module(path)`` + ``getattr(mod, "DynamoConnector")``.
This shim defers the redirect to attribute access so simply importing
the module does not force ``import vllm`` on hosts where the kvbm v1
substrate transitively pulls vLLM.

The mirror at ``kvbm.v2.vllm.connector`` differs by exactly one
character (the version segment), per the phase-4 design contract.
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
            f"module 'kvbm.v1.vllm.connector' has no attribute {name!r}"
        ) from e
    import importlib

    return getattr(importlib.import_module(mod_path), attr)


def __dir__():
    return sorted(set(list(globals().keys()) + list(_V1_EXPORTS)))
