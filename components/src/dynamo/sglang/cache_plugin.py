# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early loading for SGLang radix-cache backend plugins."""

import importlib
import sys
from collections.abc import Sequence

_RADIX_CACHE_PLUGINS = {
    "dynamo_kvbm": "kvbm.sglang_integration",
}


def _radix_cache_backend(argv: Sequence[str]) -> str | None:
    """Return the last explicitly selected radix-cache backend."""
    flag = "--radix-cache-backend"
    prefix = f"{flag}="
    backend = None
    for index, argument in enumerate(argv):
        if argument.startswith(prefix):
            backend = argument[len(prefix) :]
        elif argument == flag and index + 1 < len(argv):
            backend = argv[index + 1]
    return backend


def load_sglang_cache_plugin(argv: Sequence[str]) -> bool:
    """Import the selected cache plugin before SGLang resolves ``ServerArgs``."""
    module_name = _RADIX_CACHE_PLUGINS.get(_radix_cache_backend(argv))
    if module_name is None:
        return False
    importlib.import_module(module_name)
    return True


# ``args`` imports this module immediately before ServerArgs. The normal CLI
# therefore registers its selected backend as part of this module import.
load_sglang_cache_plugin(sys.argv[1:])
