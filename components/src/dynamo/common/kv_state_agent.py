# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attachment helpers shared by KV state-agent workers."""

import hashlib
from collections.abc import Iterable

# Fixed width, so no worker's topic is a prefix of another's: SUB subscriptions
# are prefix filters, and the state agent routes pooled raw streams by topic.
_RAW_TOPIC_HEX_DIGITS = 16


def worker_raw_topic(cache_owner_ids: Iterable[str]) -> str:
    """Raw KV event topic of one worker, published by every DP rank it owns.

    Derived from the worker's CacheOwner IDs, so it is distinct per worker and
    stable across restarts.
    """
    digest = hashlib.sha256("\n".join(sorted(cache_owner_ids)).encode()).hexdigest()
    return digest[:_RAW_TOPIC_HEX_DIGITS]
