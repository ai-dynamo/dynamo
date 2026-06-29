# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang adapter for the gms_kv_ring handle.

`install_for_sglang(...)` monkey-patches SGLang's RadixCache so that:

  • on eviction, evicted blocks are pushed to the evict ring
  • on cache hit, restored block_id pairs are pushed to the restore
    ring and the compute stream waits on the per-slot counter

The monkey-patch is intentionally small. Engines pass us their
existing block-table machinery; we only intercept eviction and
restore call-sites.

NOTE: this file imports SGLang lazily (only inside `install_for_sglang`)
so unit tests for the handle don't require SGLang to be importable.
"""

from gms_kv_ring.engines.sglang.install_kv_ring import install_for_sglang

__all__ = ["install_for_sglang"]
