// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for [`crate::pool::HostMemoryPool`]. Folded into
//! [`crate::ServiceConfig::pool`].

use std::collections::HashMap;

use dynamo_memory::HugepageMode;
use serde::{Deserialize, Serialize};

/// Sizing policy for the host-memory pool.
///
/// All variants size **per host-CPU NUMA node** —
/// [`crate::pool::HostMemoryPool`] iterates
/// [`dynamo_memory::Resources::host_memory_nodes`] and creates one slab per
/// such node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolSizing {
    /// Fraction (0.0–1.0) of each host-memory node's `MemTotal`. The
    /// pool-wide bytes are therefore `sum(ratio * node.total_bytes)`. This
    /// is the default — operators think in "% of host memory" and the
    /// split scales naturally across boxes.
    Ratio(f64),
    /// Explicit total bytes, split across host-memory nodes **proportional
    /// to each node's `MemTotal`** so heterogeneous boxes (Grace + x86) get
    /// a sensible share per node.
    Total {
        /// Total bytes the pool should allocate across all host-memory
        /// nodes.
        bytes: u64,
    },
    /// Fixed bytes per host-memory node. Every node gets the same size
    /// regardless of capacity.
    PerNode {
        /// Bytes allocated on each host-memory NUMA node.
        bytes_per_node: u64,
    },
    /// Per-node override, keyed by NUMA node id.
    Explicit(HashMap<u32, u64>),
}

impl Default for PoolSizing {
    fn default() -> Self {
        Self::Ratio(DEFAULT_POOL_RATIO)
    }
}

/// Default fraction of host memory the pool claims when `sizing =
/// Ratio(_)` is left at its default.
pub const DEFAULT_POOL_RATIO: f64 = 0.85;

/// Settings for [`crate::pool::HostMemoryPool::new`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Per-node sizing policy.
    #[serde(default)]
    pub sizing: PoolSizing,
    /// Hugepage allocation strategy. Defaults to
    /// [`HugepageMode::BestEffort`].
    #[serde(default = "default_hugepage_mode")]
    pub hugepage_mode: HugepageMode,
    /// Page size to request from the explicit hugetlb pool. `None` uses
    /// the system default (`/proc/meminfo Hugepagesize:`, typically 2 MiB).
    #[serde(default)]
    pub hugepage_size_bytes: Option<usize>,
    /// CUDA device ordinal whose context is used for `cuMemHostRegister`.
    /// Any visible GPU works; defaults to `0`.
    #[serde(default)]
    pub ctx_device_id: u32,
    /// Sample-check page placement via `move_pages(2)` after allocation
    /// (slow, debug only).
    #[serde(default)]
    pub validate_placement: bool,
    /// Allow building the pool with no NIXL DRAM backend configured.
    ///
    /// In production, slabs without UCX (or POSIX) registration accept
    /// `register_memory` calls that the nixl_sys C++ layer logs as
    /// "no available backends for mem type 'DRAM_SEG'" — the Rust handle
    /// looks valid but the slab is unreachable from remote workers.
    /// The pool refuses to start in that state by default. Set this to
    /// `true` only for tests and local development where no remote
    /// transfer is expected.
    #[serde(default)]
    pub allow_no_nixl_backends: bool,
}

fn default_hugepage_mode() -> HugepageMode {
    HugepageMode::BestEffort
}

impl Default for PoolConfig {
    fn default() -> Self {
        // The service's default differs from the primitive type's default
        // (`HugepageMode::Disabled`): operators benefit from best-effort
        // hugepages with the tier surfaced in metrics. Disabled is reserved
        // for explicit opt-out (tests, debugging).
        Self {
            sizing: PoolSizing::default(),
            hugepage_mode: default_hugepage_mode(),
            hugepage_size_bytes: None,
            ctx_device_id: 0,
            validate_placement: false,
            allow_no_nixl_backends: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_sizing_is_ratio_85() {
        let sizing = PoolSizing::default();
        match sizing {
            PoolSizing::Ratio(r) => assert!((r - 0.85).abs() < f64::EPSILON),
            other => panic!("unexpected default: {other:?}"),
        }
    }

    #[test]
    fn default_pool_config_serializes_round_trip() {
        let cfg = PoolConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: PoolConfig = serde_json::from_str(&json).unwrap();
        match back.sizing {
            PoolSizing::Ratio(r) => assert!((r - 0.85).abs() < f64::EPSILON),
            other => panic!("unexpected sizing after round trip: {other:?}"),
        }
        assert_eq!(back.hugepage_mode, HugepageMode::BestEffort);
        assert!(back.hugepage_size_bytes.is_none());
        assert_eq!(back.ctx_device_id, 0);
        assert!(!back.validate_placement);
    }

    #[test]
    fn per_node_sizing_round_trip() {
        let cfg = PoolConfig {
            sizing: PoolSizing::PerNode {
                bytes_per_node: 16 * 1024 * 1024,
            },
            hugepage_mode: HugepageMode::Disabled,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: PoolConfig = serde_json::from_str(&json).unwrap();
        match back.sizing {
            PoolSizing::PerNode { bytes_per_node } => {
                assert_eq!(bytes_per_node, 16 * 1024 * 1024);
            }
            other => panic!("unexpected sizing: {other:?}"),
        }
        assert_eq!(back.hugepage_mode, HugepageMode::Disabled);
    }
}
