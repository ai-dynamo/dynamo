# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-Omni integration for Dynamo.

Handlers are exposed lazily so importing this package (or the vLLM-free
``RealtimeOmniHandler``) does not eagerly pull ``OmniHandler``'s ``vllm_omni``
dependency. ``from dynamo.vllm.omni import OmniHandler`` still works.
"""

from typing import TYPE_CHECKING

__all__ = ["BaseOmniHandler", "OmniHandler", "RealtimeOmniHandler"]

# Map each lazily-exported name to the submodule that defines it.
_LAZY_EXPORTS = {
    "BaseOmniHandler": ".base_handler",
    "OmniHandler": ".omni_handler",
    "RealtimeOmniHandler": ".realtime_handler",
}

if TYPE_CHECKING:
    from .base_handler import BaseOmniHandler
    from .omni_handler import OmniHandler
    from .realtime_handler import RealtimeOmniHandler


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
