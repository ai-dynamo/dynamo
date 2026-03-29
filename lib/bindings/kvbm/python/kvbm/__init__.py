# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
import logging

logger = logging.getLogger(__name__)

try:
    # Preload the optional Python NIXL package when it is installed so its
    # shared libraries are available before the native extension is imported.
    import nixl
except ModuleNotFoundError:
    nixl = None
else:
    logger.info("Loaded nixl API module: %s", nixl._api)

from kvbm._core import BlockManager as BlockManager
from kvbm._core import KvbmLeader as KvbmLeader
from kvbm._core import KvbmWorker as KvbmWorker
