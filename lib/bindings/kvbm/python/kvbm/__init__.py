# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
import logging
logger = logging.getLogger(__name__)


import nixl
logger.info(f"Loaded nixl API module: {nixl._api}")

from kvbm._core import BlockManager as BlockManager
from kvbm._core import KvbmLeader as KvbmLeader
from kvbm._core import KvbmWorker as KvbmWorker
from kvbm._core import kernels as kernels
