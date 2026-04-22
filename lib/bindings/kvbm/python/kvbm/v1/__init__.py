# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kvbm._feature_stubs import _make_feature_stub

try:
    from kvbm._core import BlockManager as BlockManager
    from kvbm._core import KvbmLeader as KvbmLeader
    from kvbm._core import KvbmWorker as KvbmWorker
except ImportError:
    BlockManager = _make_feature_stub("BlockManager", "v1")
    KvbmLeader = _make_feature_stub("KvbmLeader", "v1")
    KvbmWorker = _make_feature_stub("KvbmWorker", "v1")
