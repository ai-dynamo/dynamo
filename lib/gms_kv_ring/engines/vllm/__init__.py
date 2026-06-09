# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for the gms_kv_ring handle.

Same shape as the SGLang adapter: callers pass layer descriptors,
get back a `GMSKvRing` handle, and wire `record_evict` / `record_restore`
into the engine's eviction (KVCacheConnector.evict_kv_blocks) and
cache-hit (KVCacheConnector.start_load_kv) call-sites.
"""

from gms_kv_ring.engines.vllm.install_kv_ring import install_for_vllm

__all__ = ["install_for_vllm"]
