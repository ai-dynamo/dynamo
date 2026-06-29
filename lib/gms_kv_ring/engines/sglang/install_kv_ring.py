# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire a GMSKvRing handle into SGLang's RadixCache.

Public entry point: `install_for_sglang(engine_id, daemon_socket,
layer_descs, *, on_evict=None, on_hit=None)`.

The function returns a `GMSKvRing` handle. The caller (engine
bootstrap) is responsible for closing it on engine shutdown.

The on_evict / on_hit hooks the caller can pass in are optional —
if omitted, we install a default monkey-patch on
`sglang.srt.mem_cache.radix_cache.RadixCache.evict` and on the
`match_prefix` path that publishes the obvious record sets.

Why callback-injection instead of always monkey-patching: SGLang's
internals shift between releases (we've seen #199 hangs from
cross-stream sync). Giving the engine the option to call
`handle.record_evict(...)` from its own well-tested hook is safer
than auto-patching across SGLang versions.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from gms_kv_ring.engines.handle import GMSKvRing

logger = logging.getLogger(__name__)


def install_for_sglang(
    *,
    engine_id: str,
    daemon_socket: str,
    layers: list[dict],
    on_evict: Optional[Callable[[GMSKvRing], None]] = None,
    on_hit: Optional[Callable[[GMSKvRing], None]] = None,
    evict_ring_capacity: int = 4096,
    restore_ring_capacity: int = 4096,
    num_counters: int = 512,
) -> GMSKvRing:
    """Build the handle and (optionally) wire SGLang hooks to it.

    Parameters
    ----------
    engine_id : str
        Stable per-engine identifier. Use the same value across
        restarts so the daemon can reuse host-tier slots.
    daemon_socket : str
        Unix-socket path the daemon is listening on.
    layers : list[dict]
        Per-layer descriptors. Each entry: {layer_idx, va, size, stride}.
        `va` is the engine-process virtual address of the layer base
        (i.e. the engine has already CUDA-mapped the pool). The
        daemon's pool attachment will rely on these.
    on_evict, on_hit : callable | None
        Optional callbacks invoked exactly once with the handle, so
        the caller can wire its own engine hooks. If None, this is a
        plain-handle install — useful for tests and for callers that
        want to wire their hooks explicitly.
    """
    handle = GMSKvRing(
        engine_id=engine_id,
        daemon_socket=daemon_socket,
        layers=layers,
        evict_ring_capacity=evict_ring_capacity,
        restore_ring_capacity=restore_ring_capacity,
        num_counters=num_counters,
    )
    if on_evict is not None:
        on_evict(handle)
    if on_hit is not None:
        on_hit(handle)
    logger.info("gms_kv_ring installed for SGLang engine %r", engine_id)
    return handle
