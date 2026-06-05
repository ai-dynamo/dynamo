// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One [`NodeSlab`] per host-CPU NUMA node. Each slab owns a pinned
//! [`MmappedPinnedStorage`]; in production it is also registered with its
//! own dedicated [`NixlAgent`] so remote workers can RDMA into it.
//!
//! When the pool is built with `allow_no_nixl_backends = true` (test /
//! local-dev mode, or when `nixl_sys` is in stub mode and an agent cannot
//! be constructed at all), the slab carries the raw storage with no NIXL
//! wrapping — the slab still serves as a pinned host-memory region for
//! local use, just unreachable from remote NIXL peers.

use std::fmt;

use dynamo_memory::{
    HugepageTier, MmappedPinnedStorage, NumaNode,
    nixl::{NixlAgent, NixlRegistered},
};
use serde::Serialize;

/// Page-pinned host-memory slab on one NUMA node, optionally registered
/// with NIXL.
///
/// # Drop order
///
/// Field declaration order is load-bearing:
///
/// 1. `storage: SlabStorage` drops first. In the [`SlabStorage::Registered`]
///    variant, [`NixlRegistered::Drop`] drops the registration handle while
///    the agent below is still alive; the underlying
///    [`MmappedPinnedStorage`] then drops, calling `cuMemHostUnregister`
///    and `munmap` in that order (also field-order-enforced inside the
///    storage type). In the [`SlabStorage::Local`] variant it's just the
///    raw mmap cleanup.
/// 2. `agent: Option<NixlAgent>` drops last when present, so the NIXL
///    transport that backed the registration outlives the deregister call.
pub struct NodeSlab {
    storage: SlabStorage,
    /// Held when present to keep the NIXL agent (and its backends /
    /// transport) alive at least as long as the registered storage above.
    #[allow(dead_code)]
    agent: Option<NixlAgent>,

    // Plain metadata below; ordering after the drop-sensitive fields above
    // is irrelevant.
    numa_node: NumaNode,
    size_bytes: usize,
    hugepage_tier: HugepageTier,
    /// `None` when this slab was built without a NIXL agent — see
    /// [`SlabStorage::Local`].
    agent_name: Option<String>,
}

/// Either a NIXL-registered slab (production / normal path) or a
/// local-only slab (test / no-NIXL path).
pub enum SlabStorage {
    /// Storage wrapped in [`NixlRegistered`]; remote NIXL peers can pull
    /// from this region via the owning agent.
    Registered(NixlRegistered<MmappedPinnedStorage>),
    /// Raw pinned mmap storage with no NIXL registration. The region is
    /// still pinned and CUDA-host-registered, just unreachable from
    /// remote NIXL workers.
    Local(MmappedPinnedStorage),
}

impl fmt::Debug for SlabStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registered(_) => f.write_str("Registered(..)"),
            Self::Local(_) => f.write_str("Local(..)"),
        }
    }
}

impl NodeSlab {
    /// Construct a slab whose storage is registered with `agent`. The
    /// caller must have already called `storage.register(&agent, opt)` so
    /// `registered` is the live wrapper.
    pub(crate) fn new_registered(
        registered: NixlRegistered<MmappedPinnedStorage>,
        agent: NixlAgent,
        numa_node: NumaNode,
        size_bytes: usize,
        hugepage_tier: HugepageTier,
        agent_name: String,
    ) -> Self {
        Self {
            storage: SlabStorage::Registered(registered),
            agent: Some(agent),
            numa_node,
            size_bytes,
            hugepage_tier,
            agent_name: Some(agent_name),
        }
    }

    /// Construct a slab with no NIXL agent. Used when
    /// [`crate::pool::PoolConfig::allow_no_nixl_backends`] is `true` or
    /// when `nixl_sys` is in stub mode (no real NIXL libs linked).
    pub(crate) fn new_local(
        storage: MmappedPinnedStorage,
        numa_node: NumaNode,
        size_bytes: usize,
        hugepage_tier: HugepageTier,
    ) -> Self {
        Self {
            storage: SlabStorage::Local(storage),
            agent: None,
            numa_node,
            size_bytes,
            hugepage_tier,
            agent_name: None,
        }
    }

    /// NUMA node this slab is bound to.
    pub fn numa_node(&self) -> NumaNode {
        self.numa_node
    }

    /// Allocation size in bytes (rounded up to the effective page size by
    /// the underlying mmap).
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Page-backing tier the underlying mmap actually landed on.
    pub fn hugepage_tier(&self) -> HugepageTier {
        self.hugepage_tier
    }

    /// Name of the NIXL agent that owns the registration for this slab,
    /// or `None` if the slab is local-only ([`SlabStorage::Local`]).
    pub fn agent_name(&self) -> Option<&str> {
        self.agent_name.as_deref()
    }

    /// Returns `true` when this slab is registered with NIXL and remote
    /// peers can pull from it.
    pub fn is_registered(&self) -> bool {
        matches!(self.storage, SlabStorage::Registered(_))
    }

    /// Access the storage variant.
    pub fn storage(&self) -> &SlabStorage {
        &self.storage
    }
}

impl fmt::Debug for NodeSlab {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeSlab")
            .field("numa_node", &self.numa_node)
            .field("size_bytes", &self.size_bytes)
            .field("hugepage_tier", &self.hugepage_tier)
            .field("agent_name", &self.agent_name)
            .field("registered", &self.is_registered())
            .finish()
    }
}

/// Per-slab view for `/v1/pool` snapshots and metrics labels.
#[derive(Debug, Clone, Serialize)]
pub struct NodeSlabSnapshot {
    pub numa_node: u32,
    pub size_bytes: u64,
    pub hugepage_tier: HugepageTier,
    /// `None` for local-only slabs (built without a NIXL agent).
    pub agent_name: Option<String>,
    /// Whether the slab is reachable from remote NIXL peers.
    pub registered: bool,
}

impl From<&NodeSlab> for NodeSlabSnapshot {
    fn from(s: &NodeSlab) -> Self {
        Self {
            numa_node: s.numa_node.0,
            size_bytes: s.size_bytes as u64,
            hugepage_tier: s.hugepage_tier,
            agent_name: s.agent_name.clone(),
            registered: s.is_registered(),
        }
    }
}
