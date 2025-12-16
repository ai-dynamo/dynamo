# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
import logging

import nixl
from kvbm._core import __version__ as __version__
from kvbm._feature_stubs import _make_feature_stub, _make_module_stub

# nixl needs to be loaded before any other imports to ensure that the nixl shared object is available for the KVBM core.
logger = logging.getLogger(__name__)
logger.info(f"Loaded nixl API module: {nixl._api}")

# v1 feature: BlockManager, KvbmLeader, KvbmWorker
try:
    from kvbm._core import BlockManager as BlockManager
    from kvbm._core import KvbmLeader as KvbmLeader
    from kvbm._core import KvbmWorker as KvbmWorker

    _V1_AVAILABLE = True
except ImportError:
    BlockManager = _make_feature_stub("BlockManager", "v1")
    KvbmLeader = _make_feature_stub("KvbmLeader", "v1")
    KvbmWorker = _make_feature_stub("KvbmWorker", "v1")
    _V1_AVAILABLE = False

# v2 feature: v2 submodule
try:
    from kvbm._core import v2 as v2

    _V2_AVAILABLE = True
except ImportError:
    v2 = _make_module_stub("kvbm.v2", "v2")
    _V2_AVAILABLE = False

# kernels feature: kernels submodule (optional)
try:
    from kvbm._core import kernels as kernels

    _KERNELS_AVAILABLE = True
except ImportError:
    kernels = _make_module_stub("kvbm.kernels", "kernels")
    _KERNELS_AVAILABLE = False

__all__ = [
    "__version__",
    "v2",
    "kernels",
    "BlockManager",
    "KvbmLeader",
    "KvbmWorker",
    "_V1_AVAILABLE",
    "_V2_AVAILABLE",
    "_KERNELS_AVAILABLE",
]
