# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical v2 connector façade — lazy re-export of the v2 impl.

The v2 production connector lives at ``kvbm.v2.vllm.schedulers.connector``
today. A future cleanup will move it to ``kvbm.v2.vllm.connectors``;
when that lands, this façade is the only edit needed and external
callers (tests, vLLM kv_transfer config) keep working unchanged.

vLLM resolves ``kv_connector_module_path`` via
``importlib.import_module(path)`` + ``getattr(mod, "DynamoConnector")``.
The lazy ``__getattr__`` shim defers the redirect to attribute access so
simply importing this module does not force ``import vllm`` on hosts
where the schedulers chain transitively pulls vLLM.

Note: ``kvbm.v2.vllm.connector`` (singular) is intentionally distinct
from the placeholder ``kvbm.v2.vllm.connectors`` (plural) sibling
package.

The mirror at ``kvbm.v1.vllm.connector`` differs by exactly one
character (the version segment), per the phase-4 design contract.
"""

_V2_EXPORTS = {
    "DynamoConnector": (
        "kvbm.v2.vllm.schedulers.connector",
        "DynamoConnector",
    ),
}

__all__ = list(_V2_EXPORTS)


def __getattr__(name):
    try:
        mod_path, attr = _V2_EXPORTS[name]
    except KeyError as e:
        raise AttributeError(
            f"module 'kvbm.v2.vllm.connector' has no attribute {name!r}"
        ) from e
    import importlib

    return getattr(importlib.import_module(mod_path), attr)


def __dir__():
    return sorted(set(list(globals().keys()) + list(_V2_EXPORTS)))
