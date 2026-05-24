// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One [`NodeSlab`] per host-CPU NUMA node. Each slab owns a pinned
//! [`MmappedPinnedStorage`] registered with its own dedicated
//! [`NixlAgent`].

use std::fmt;

use dynamo_memory::{
    HugepageTier, MmappedPinnedStorage, NumaNode,
    nixl::{NixlAgent, NixlRegistered},
};
use serde::Serialize;

/// A pinned host-memory slab on one NUMA node, registered with its own
/// dedicated NIXL agent.
///
/// # Drop order
///
/// Field declaration order is load-bearing:
///
/// 1. `storage: NixlRegistered<_>` drops first — its `Drop` impl drops the
///    NIXL registration handle while the agent is still alive.
/// 2. The underlying [`MmappedPinnedStorage`] inside `storage` then drops,
///    calling `cuMemHostUnregister` and `munmap` in that order (also
///    field-order-enforced inside the storage type).
/// 3. `agent` drops last, so the NIXL transport that backed the
///    registration outlives the deregister call.
pub struct NodeSlab {
    storage: NixlRegistered<MmappedPinnedStorage>,
    /// Held to keep the NIXL agent (and its backends/transport) alive at
    /// least as long as the registered storage above.
    #[allow(dead_code)]
    agent: NixlAgent,

    // Plain metadata below; ordering after the drop-sensitive fields above
    // is irrelevant.
    numa_node: NumaNode,
    size_bytes: usize,
    hugepage_tier: HugepageTier,
    agent_name: String,
}

impl NodeSlab {
    /// Construct from the freshly-built storage + agent pair. Caller is
    /// responsible for having already called [`NixlAgent::register_memory`]
    /// via the storage's `.register(&agent, opt)` path so that `storage`
    /// is the live wrapper.
    pub(crate) fn new(
        storage: NixlRegistered<MmappedPinnedStorage>,
        agent: NixlAgent,
        numa_node: NumaNode,
        size_bytes: usize,
        hugepage_tier: HugepageTier,
        agent_name: String,
    ) -> Self {
        Self {
            storage,
            agent,
            numa_node,
            size_bytes,
            hugepage_tier,
            agent_name,
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

    /// Name of the NIXL agent that owns the registration for this slab.
    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }

    /// Access the registered storage for read-only inspection (address,
    /// size, etc.).
    pub fn storage(&self) -> &NixlRegistered<MmappedPinnedStorage> {
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
            .finish()
    }
}

/// Per-slab view for `/v1/pool` snapshots and metrics labels.
#[derive(Debug, Clone, Serialize)]
pub struct NodeSlabSnapshot {
    pub numa_node: u32,
    pub size_bytes: u64,
    pub hugepage_tier: HugepageTier,
    pub agent_name: String,
}

impl From<&NodeSlab> for NodeSlabSnapshot {
    fn from(s: &NodeSlab) -> Self {
        Self {
            numa_node: s.numa_node.0,
            size_bytes: s.size_bytes as u64,
            hugepage_tier: s.hugepage_tier,
            agent_name: s.agent_name.clone(),
        }
    }
}
