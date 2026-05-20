# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Back-compat shim for the legacy path ``kvbm.vllm_integration.connector``.

vLLM resolves ``kv_connector_module_path`` via
``importlib.import_module(path)`` and ``getattr(module, connector_name)``.
Tracked examples and Dynamo vLLM's compatibility helper still point at this
legacy module path for ``PdConnector(DynamoConnector, NixlConnector)``. The
canonical Dynamo connector implementation now lives under v2, so this package
lazy-exports the v2 connector plus the KVBM-owned ``PdConnector`` wrapper.

Lazy by design: importing this module must not force ``import vllm``. The
connector implementations transitively import vLLM, so the redirect happens
only when vLLM asks for one of the exported connector classes.
"""

_EXPORTS = {
    "DynamoConnector": (
        "kvbm.v2.vllm.schedulers.connector",
        "DynamoConnector",
    ),
    "DynamoConnectorMetadata": (
        "kvbm.v2.vllm.schedulers.connector",
        "DynamoSchedulerConnectorMetadata",
    ),
    "DynamoSchedulerConnectorMetadata": (
        "kvbm.v2.vllm.schedulers.connector",
        "DynamoSchedulerConnectorMetadata",
    ),
    "PdConnector": (
        "kvbm.vllm_integration.connector.pd_connector",
        "PdConnector",
    ),
    "PdConnectorMetadata": (
        "kvbm.vllm_integration.connector.pd_connector",
        "PdConnectorMetadata",
    ),
    "PdHandshakeMetadata": (
        "kvbm.vllm_integration.connector.pd_connector",
        "PdHandshakeMetadata",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        mod_path, attr = _EXPORTS[name]
    except KeyError as e:
        raise AttributeError(
            f"module 'kvbm.vllm_integration.connector' has no attribute {name!r}"
        ) from e
    import importlib

    return getattr(importlib.import_module(mod_path), attr)


def __dir__():
    return sorted(set(list(globals().keys()) + list(_EXPORTS)))
