// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for [`crate::pool::HostMemoryPool`]. Folded into
//! [`crate::ServiceConfig::pool`]. Construct via [`PoolConfig::builder`]
//! for ergonomic tests / programmatic setup; deserialize from TOML / env
//! for operator-driven config.

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
    /// NIXL backends to initialize on every slab's agent. The default is
    /// `["UCX"]` — UCX is required for remote workers to RDMA into the
    /// pool. If this list is non-empty it takes precedence; if empty,
    /// the pool falls back to reading `DYN_KVBM_NIXL_BACKEND_*` env vars
    /// via [`dynamo_memory::nixl::NixlBackendConfig::from_env`]. Clear
    /// to `vec![]` together with `allow_no_nixl_backends = true` only
    /// for local-only deployments.
    #[serde(default = "default_nixl_backends")]
    pub backends: Vec<String>,
}

fn default_hugepage_mode() -> HugepageMode {
    HugepageMode::BestEffort
}

fn default_nixl_backends() -> Vec<String> {
    vec!["UCX".to_string()]
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
            backends: default_nixl_backends(),
        }
    }
}

impl PoolConfig {
    /// Start a new [`PoolConfigBuilder`] with the same defaults as
    /// [`Self::default`].
    pub fn builder() -> PoolConfigBuilder {
        PoolConfigBuilder::new()
    }
}

/// Fluent builder for [`PoolConfig`]. Preferred over struct-literal
/// construction in tests + programmatic call sites because the field set
/// will grow.
#[derive(Debug, Clone)]
pub struct PoolConfigBuilder {
    cfg: PoolConfig,
}

impl Default for PoolConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PoolConfigBuilder {
    /// Build seeded with [`PoolConfig::default`] (sizing = `Ratio(0.85)`,
    /// hugepages best-effort, backends = `["UCX"]`).
    pub fn new() -> Self {
        Self {
            cfg: PoolConfig::default(),
        }
    }

    /// Override the sizing policy.
    pub fn sizing(mut self, s: PoolSizing) -> Self {
        self.cfg.sizing = s;
        self
    }

    /// Shortcut for `.sizing(PoolSizing::PerNode { bytes_per_node })`.
    pub fn per_node_bytes(self, bytes_per_node: u64) -> Self {
        self.sizing(PoolSizing::PerNode { bytes_per_node })
    }

    /// Shortcut for `.sizing(PoolSizing::Ratio(r))`.
    pub fn ratio(self, r: f64) -> Self {
        self.sizing(PoolSizing::Ratio(r))
    }

    /// Override the hugepage allocation mode.
    pub fn hugepage_mode(mut self, m: HugepageMode) -> Self {
        self.cfg.hugepage_mode = m;
        self
    }

    /// Override the requested hugepage size (`None` => system default).
    pub fn hugepage_size_bytes(mut self, size: Option<usize>) -> Self {
        self.cfg.hugepage_size_bytes = size;
        self
    }

    /// CUDA device ordinal used for `cuMemHostRegister`.
    pub fn ctx_device_id(mut self, id: u32) -> Self {
        self.cfg.ctx_device_id = id;
        self
    }

    /// Append a NIXL backend by name (`"UCX"`, `"POSIX"`, …). Calls are
    /// idempotent at the [`dynamo_memory::nixl::NixlAgent`] layer.
    pub fn nixl_backend(mut self, name: impl Into<String>) -> Self {
        self.cfg.backends.push(name.into());
        self
    }

    /// Replace the backend list entirely (clearing the `["UCX"]` default).
    pub fn nixl_backends<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.cfg.backends = names.into_iter().map(Into::into).collect();
        self
    }

    /// Convenience for `.nixl_backend("UCX")`. UCX is already on the
    /// default list — use this when you want it explicit in a test.
    pub fn with_ucx(self) -> Self {
        self.nixl_backend("UCX")
    }

    /// Clear the backend list **and** set `allow_no_nixl_backends = true`
    /// — the only safe way to ask for a local-only pool with no NIXL
    /// reachability.
    pub fn local_only(mut self) -> Self {
        self.cfg.backends.clear();
        self.cfg.allow_no_nixl_backends = true;
        self
    }

    /// Override the `allow_no_nixl_backends` escape hatch directly.
    pub fn allow_no_nixl_backends(mut self, b: bool) -> Self {
        self.cfg.allow_no_nixl_backends = b;
        self
    }

    /// Enable per-page `move_pages(2)` placement sampling (slow; debug).
    pub fn validate_placement(mut self, b: bool) -> Self {
        self.cfg.validate_placement = b;
        self
    }

    /// Finalize the configuration.
    pub fn build(self) -> PoolConfig {
        self.cfg
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
        assert_eq!(back.backends, vec!["UCX".to_string()]);
    }

    #[test]
    fn builder_defaults_match_struct_default() {
        let from_builder = PoolConfig::builder().build();
        let from_default = PoolConfig::default();
        // Both should produce identical JSON.
        assert_eq!(
            serde_json::to_value(&from_builder).unwrap(),
            serde_json::to_value(&from_default).unwrap()
        );
    }

    #[test]
    fn builder_per_node_with_disabled_hugepages() {
        let cfg = PoolConfig::builder()
            .per_node_bytes(16 * 1024 * 1024)
            .hugepage_mode(HugepageMode::Disabled)
            .build();
        match cfg.sizing {
            PoolSizing::PerNode { bytes_per_node } => {
                assert_eq!(bytes_per_node, 16 * 1024 * 1024)
            }
            other => panic!("unexpected sizing: {other:?}"),
        }
        assert_eq!(cfg.hugepage_mode, HugepageMode::Disabled);
        // UCX still on the default backend list — builder doesn't drop it.
        assert_eq!(cfg.backends, vec!["UCX".to_string()]);
    }

    #[test]
    fn builder_with_ucx_appends_to_default() {
        // .with_ucx() on top of the default ["UCX"] is harmless: NixlAgent
        // dedups at registration time.
        let cfg = PoolConfig::builder().with_ucx().build();
        assert!(cfg.backends.iter().any(|b| b == "UCX"));
    }

    #[test]
    fn builder_nixl_backends_replaces_default() {
        let cfg = PoolConfig::builder()
            .nixl_backends(["UCX", "POSIX"])
            .build();
        assert_eq!(cfg.backends, vec!["UCX".to_string(), "POSIX".to_string()]);
    }

    #[test]
    fn builder_local_only_clears_backends_and_opts_in() {
        let cfg = PoolConfig::builder().local_only().build();
        assert!(cfg.backends.is_empty());
        assert!(cfg.allow_no_nixl_backends);
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
