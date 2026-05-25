// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-memory pool.
//!
//! Allocates one [`NodeSlab`] per host-CPU NUMA node at process startup,
//! sizes per [`PoolConfig::sizing`], and exposes the slabs to the eventual
//! kvbm-engine container via [`HostMemoryPool::lease_all`].
//!
//! Single-tenant for the MVP: only one outstanding [`PoolLease`] at a time.
//! Moving to multi-tenant is a state-machine widening on
//! [`HostMemoryPool::inner`], not an API change.

pub mod config;
pub mod slab;

pub use config::{DEFAULT_POOL_RATIO, PoolConfig, PoolSizing};
pub use slab::{NodeSlab, NodeSlabSnapshot};

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_memory::{
    HugepageInfo, MmappedPinnedOptions, NumaNode, NumaNodeView, Resources,
    nixl::{NixlAgent, NixlBackendConfig, NixlRegisterExt, is_stub as nixl_is_stub},
    numa::worker_pool::NumaWorkerPool,
    resources::CgroupInfo,
};
use parking_lot::Mutex;
use serde::Serialize;

use crate::error::{ServiceError, ServiceResult};

/// Live host-memory pool. Owns one [`NodeSlab`] per host-CPU NUMA node.
#[derive(Debug)]
pub struct HostMemoryPool {
    slabs: Vec<Arc<NodeSlab>>,
    resources: Resources,
    hugepage_info: HugepageInfo,
    instance_id: String,
    inner: Mutex<PoolState>,
}

#[derive(Debug, Default)]
struct PoolState {
    leased: bool,
}

/// Single-tenant lease handle. Dropping releases the lease back to the
/// pool. The held [`Arc<HostMemoryPool>`] keeps the pool alive at least as
/// long as any outstanding lease.
#[derive(Debug)]
pub struct PoolLease {
    pool: Arc<HostMemoryPool>,
}

impl PoolLease {
    /// Slabs accessible under this lease.
    pub fn slabs(&self) -> &[Arc<NodeSlab>] {
        self.pool.slabs()
    }

    /// Reference back to the owning pool (e.g. for snapshot reads).
    pub fn pool(&self) -> &Arc<HostMemoryPool> {
        &self.pool
    }
}

impl Drop for PoolLease {
    fn drop(&mut self) {
        self.pool.inner.lock().leased = false;
    }
}

/// Read-only snapshot of pool state for the `/v1/pool` HTTP endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct PoolSnapshot {
    pub instance_id: String,
    pub leased: bool,
    pub total_bytes: u64,
    pub slabs: Vec<NodeSlabSnapshot>,
    pub hugepage_default_size_bytes: usize,
    pub thp_enabled: String,
}

impl HostMemoryPool {
    /// Discover host topology + hugepage state, allocate one slab per
    /// host-memory NUMA node, register each with its own NIXL agent.
    /// Returns the populated pool wrapped in [`Arc`] so the caller can
    /// take leases.
    ///
    /// The `instance_id` is incorporated into each slab's NIXL agent name
    /// (`"kvbm-svc:{instance_id}:n{node_id}"`) so multiple service
    /// processes on the same host produce non-colliding agent identities.
    pub fn new(config: &PoolConfig, instance_id: &str) -> ServiceResult<Arc<Self>> {
        let resources = Resources::discover();
        let hugepage_info = HugepageInfo::discover();
        Self::new_with_resources(config, instance_id, resources, hugepage_info)
    }

    /// Same as [`Self::new`] but with caller-supplied snapshots — used by
    /// tests so they can drive deterministic topologies without depending
    /// on the host they run on.
    pub fn new_with_resources(
        config: &PoolConfig,
        instance_id: &str,
        resources: Resources,
        hugepage_info: HugepageInfo,
    ) -> ServiceResult<Arc<Self>> {
        let all_host_nodes: Vec<&NumaNodeView> = resources.host_memory_nodes().collect();
        if all_host_nodes.is_empty() {
            return Err(ServiceError::Internal(
                "no host-memory NUMA nodes discovered; cannot build pool".into(),
            ));
        }

        // Apply cgroup cpuset mems mask (effective wins on v2 so we honor
        // inherited restrictions). `mbind(MPOL_BIND)` against a node
        // outside the mask fails with `EPERM`; refuse early and tell the
        // operator what they're missing.
        let total_before_cgroup = all_host_nodes.len();
        let host_nodes = nodes_allowed_by_cgroup(all_host_nodes, &resources.cgroup);
        let effective_mems_log = effective_mems(&resources.cgroup);
        if host_nodes.is_empty() {
            return Err(ServiceError::InvalidArgument(format!(
                "cgroup cpuset mems={:?} excludes every CPU-bearing NUMA node; \
                 cannot build pool",
                effective_mems_log,
            )));
        }
        if host_nodes.len() < total_before_cgroup {
            tracing::warn!(
                "cgroup cpuset mems={:?} restricted pool to {}/{} host-memory nodes",
                effective_mems_log,
                host_nodes.len(),
                total_before_cgroup,
            );
        }

        let total_host_memory: u64 = host_nodes.iter().filter_map(|n| n.total_bytes).sum();
        let raw_per_node = compute_per_node_sizes(&config.sizing, &host_nodes)?;
        let (per_node_bytes, capped) =
            cap_by_cgroup_memory(raw_per_node, resources.cgroup.memory_max);
        if capped {
            tracing::warn!(
                "cgroup memory.max={} bytes scaled the pool down; per-node sizes adjusted",
                resources.cgroup.memory_max.unwrap_or(0),
            );
        }
        let pool_total: u64 = per_node_bytes.values().sum();

        // Sanity warnings.
        if let PoolSizing::Ratio(r) = config.sizing
            && r > 0.95
        {
            tracing::warn!(
                "PoolSizing::Ratio({:.3}) > 0.95: high ratio leaves little headroom \
                 for frameworks and OS — set lower if the box hosts other tenants",
                r,
            );
        }
        for node in &host_nodes {
            if let (Some(req), Some(total)) = (per_node_bytes.get(&node.node.0), node.total_bytes)
                && *req as f64 > 0.9 * total as f64
            {
                tracing::warn!(
                    "node {}: requested {:.1} GiB exceeds 90% of node MemTotal \
                     ({:.1} GiB) — may evict caches",
                    node.node.0,
                    *req as f64 / (1024.0 * 1024.0 * 1024.0),
                    total as f64 / (1024.0 * 1024.0 * 1024.0),
                );
            }
        }
        if !matches!(config.sizing, PoolSizing::Ratio(_))
            && total_host_memory > 0
            && (pool_total as f64) < 0.5 * (total_host_memory as f64)
        {
            tracing::info!(
                "pool covers {:.1}% of {:.1} GiB host memory; {:.1} GiB unused",
                (pool_total as f64) / (total_host_memory as f64) * 100.0,
                total_host_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                (total_host_memory - pool_total) as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }

        // Resolve which NIXL backends to actually use. `from_env` is
        // wrapped in a closure so it only runs on the legacy fallback
        // path; a malformed `DYN_KVBM_NIXL_BACKEND_*` env var would
        // otherwise abort startup for configurations that never consult
        // env at all (explicit list / `.local_only()`).
        let backend_config =
            resolve_backend_config(&config.backends, config.allow_no_nixl_backends, || {
                NixlBackendConfig::from_env()
                    .map_err(|e| ServiceError::Internal(format!("parse NIXL backend env: {e}")))
            })?;
        // Production trap: a NIXL agent with no DRAM-capable backend will
        // silently accept register_memory and produce slabs whose handle
        // looks valid but cannot satisfy remote pulls. Refuse to start in
        // that configuration unless the operator explicitly opts in.
        //
        // Stub mode (nixl_sys linked without real libs) is another route
        // to "no NIXL functionality" — also requires the explicit opt-in,
        // and additionally forces slabs to local-only because we can't
        // even build an agent.
        let stub = nixl_is_stub();
        let has_dram_backend =
            backend_config.has_backend("UCX") || backend_config.has_backend("POSIX");
        let can_build_agent = !stub && has_dram_backend;
        if !can_build_agent && !config.allow_no_nixl_backends {
            return Err(ServiceError::InvalidArgument(format!(
                "no usable NIXL DRAM transport ({}). Slabs without a backend would \
                 be unreachable from remote workers. Set \
                 PoolConfig::allow_no_nixl_backends = true to override for local-only use.",
                if stub {
                    "nixl_sys is in stub mode (no real NIXL libs linked)".to_string()
                } else {
                    "no DRAM backend configured (set DYN_KVBM_NIXL_BACKEND_UCX=true \
                     or DYN_KVBM_NIXL_BACKEND_POSIX=true)"
                        .to_string()
                },
            )));
        }
        let build_agents = can_build_agent;
        if !build_agents {
            tracing::warn!(
                "HostMemoryPool: building slabs without NIXL agents \
                 (stub={}, has_dram_backend={}); slabs are local-only and \
                 not reachable from remote workers",
                stub,
                has_dram_backend,
            );
        }
        let mut slabs: Vec<Arc<NodeSlab>> = Vec::with_capacity(host_nodes.len());
        for node in &host_nodes {
            let size = match per_node_bytes.get(&node.node.0).copied() {
                Some(s) if s > 0 => s as usize,
                _ => {
                    return Err(ServiceError::Internal(format!(
                        "computed zero-byte allocation for node {}",
                        node.node.0
                    )));
                }
            };
            let slab = allocate_slab(
                &backend_config,
                config,
                instance_id,
                node.node,
                size,
                build_agents,
            )?;
            slabs.push(Arc::new(slab));
        }

        let total_human = pool_total as f64 / (1024.0 * 1024.0 * 1024.0);
        let pct = if total_host_memory > 0 {
            format!(
                "{:.1}%",
                (pool_total as f64) / (total_host_memory as f64) * 100.0
            )
        } else {
            "?%".to_string()
        };
        tracing::info!(
            "HostMemoryPool: allocated {:.1} GiB across {} host-memory node(s) ({} of host)",
            total_human,
            slabs.len(),
            pct,
        );
        for slab in &slabs {
            tracing::info!(
                "  node {}  size={:.1} GiB  tier={:?}  agent={}",
                slab.numa_node().0,
                slab.size_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                slab.hugepage_tier(),
                slab.agent_name().unwrap_or("(local-only)"),
            );
        }

        Ok(Arc::new(Self {
            slabs,
            resources,
            hugepage_info,
            instance_id: instance_id.to_string(),
            inner: Mutex::new(PoolState::default()),
        }))
    }

    /// All slabs (one per host-memory node).
    pub fn slabs(&self) -> &[Arc<NodeSlab>] {
        &self.slabs
    }

    /// Topology snapshot captured at construction.
    pub fn resources(&self) -> &Resources {
        &self.resources
    }

    /// Hugepage snapshot captured at construction.
    pub fn hugepage_info(&self) -> &HugepageInfo {
        &self.hugepage_info
    }

    /// Take an exclusive lease over the entire pool. Fails if a lease is
    /// already outstanding. The lease must be held by the
    /// [`crate::container::ServiceContainer`] for the duration it owns the
    /// slabs.
    pub fn lease_all(self: &Arc<Self>) -> ServiceResult<PoolLease> {
        let mut state = self.inner.lock();
        if state.leased {
            return Err(ServiceError::Internal(
                "host-memory pool is already leased".into(),
            ));
        }
        state.leased = true;
        Ok(PoolLease {
            pool: Arc::clone(self),
        })
    }

    /// Build an empty pool for tests that exercise lease state without
    /// touching CUDA / NIXL. Production code always uses [`Self::new`].
    #[cfg(test)]
    pub(crate) fn empty_for_tests(instance_id: &str) -> Arc<Self> {
        Arc::new(Self {
            slabs: Vec::new(),
            resources: Resources::discover(),
            hugepage_info: HugepageInfo::default(),
            instance_id: instance_id.to_string(),
            inner: Mutex::new(PoolState::default()),
        })
    }

    /// Snapshot suitable for the `/v1/pool` HTTP endpoint.
    pub fn snapshot(&self) -> PoolSnapshot {
        let total_bytes: u64 = self.slabs.iter().map(|s| s.size_bytes() as u64).sum();
        PoolSnapshot {
            instance_id: self.instance_id.clone(),
            leased: self.inner.lock().leased,
            total_bytes,
            slabs: self
                .slabs
                .iter()
                .map(|s| NodeSlabSnapshot::from(&**s))
                .collect(),
            hugepage_default_size_bytes: self.hugepage_info.default_size_bytes,
            thp_enabled: self.hugepage_info.thp_enabled.to_string(),
        }
    }
}

/// Build one slab end-to-end: allocate the mmap'd pinned storage on the
/// target NUMA node, and — if `build_agent` is true — build a dedicated
/// NIXL agent and register the storage with it.
fn allocate_slab(
    backend_config: &NixlBackendConfig,
    config: &PoolConfig,
    instance_id: &str,
    node: NumaNode,
    size: usize,
    build_agent: bool,
) -> ServiceResult<NodeSlab> {
    let opt = MmappedPinnedOptions {
        size,
        numa_node: node,
        hugepage_mode: config.hugepage_mode,
        hugepage_size: config.hugepage_size_bytes,
        ctx_device_id: config.ctx_device_id,
    };
    let storage = NumaWorkerPool::global()
        .allocate_mmap_pinned_on_node(opt)
        .map_err(|e| ServiceError::Internal(format!("allocate slab on node {}: {}", node.0, e)))?;

    let tier = storage.hugepage_tier();
    let mapped_len = storage.mapped_len();

    if !build_agent {
        return Ok(NodeSlab::new_local(storage, node, mapped_len, tier));
    }

    let agent_name = format!("kvbm-svc:{instance_id}:n{}", node.0);
    let agent = NixlAgent::from_nixl_backend_config(&agent_name, backend_config.clone())
        .map_err(|e| ServiceError::Internal(format!("build NixlAgent {agent_name}: {e:?}")))?;

    let registered = storage.register(&agent, None).map_err(|_storage| {
        ServiceError::Internal(format!("register slab with NIXL agent {agent_name} failed"))
    })?;

    Ok(NodeSlab::new_registered(
        registered, agent, node, mapped_len, tier, agent_name,
    ))
}

/// Resolve the NIXL backend set the pool will hand to each slab's agent.
///
/// Precedence:
/// 1. **Explicit list** (`requested_backends` non-empty) — wins
///    unconditionally. Operator opted in via TOML / builder.
/// 2. **Explicit opt-out** (`allow_no_nixl_backends = true` with empty
///    list) — return an empty config; **the env fallback is suppressed**
///    so an inherited `DYN_KVBM_NIXL_BACKEND_UCX` cannot silently
///    re-enable NIXL behind a `.local_only()` builder call.
/// 3. **Legacy env fallback** — read `DYN_KVBM_NIXL_BACKEND_*` via the
///    `env_backends` closure. Preserved for operators who set backends
///    purely via environment.
///
/// `env_backends` is a closure so the (potentially failing) env read only
/// runs when the legacy fallback branch is actually taken. A malformed
/// env var must not abort startup for configs that never look at env.
fn resolve_backend_config<F>(
    requested_backends: &[String],
    allow_no_nixl_backends: bool,
    env_backends: F,
) -> ServiceResult<NixlBackendConfig>
where
    F: FnOnce() -> ServiceResult<NixlBackendConfig>,
{
    if !requested_backends.is_empty() {
        let mut bc = NixlBackendConfig::default();
        for name in requested_backends {
            bc = bc.with_backend(name);
        }
        return Ok(bc);
    }
    if allow_no_nixl_backends {
        return Ok(NixlBackendConfig::default());
    }
    env_backends()
}

/// Restrict the host-memory nodes to those allowed by the cgroup's
/// effective mems mask. Outside the mask, `mbind(MPOL_BIND)` fails with
/// `EPERM` (or silently misplaces pages), so allocating on those nodes is
/// unsafe.
///
/// Prefers `cpuset.mems.effective` (v2; what the kernel actually enforces
/// after ancestor + online-node intersection) and falls back to
/// `cpuset.mems` (v1, or v2 hosts where the effective file is missing).
/// `None` for both means "no cgroup constraint" — return input unchanged.
fn nodes_allowed_by_cgroup<'a>(
    host_nodes: Vec<&'a NumaNodeView>,
    cgroup: &CgroupInfo,
) -> Vec<&'a NumaNodeView> {
    let mems = effective_mems(cgroup);
    let mems = match mems {
        Some(m) if !m.is_empty() => m,
        _ => return host_nodes,
    };
    let allowed: std::collections::HashSet<u32> = mems.iter().map(|&n| n as u32).collect();
    host_nodes
        .into_iter()
        .filter(|n| allowed.contains(&n.node.0))
        .collect()
}

/// Pick the cpuset mems mask the kernel actually enforces. Prefer v2's
/// `cpuset.mems.effective`; fall back to the configured `cpuset.mems` only
/// when the effective field is absent (v1, or unreadable).
fn effective_mems(cgroup: &CgroupInfo) -> Option<&Vec<usize>> {
    cgroup
        .cpuset_mems_effective
        .as_ref()
        .or(cgroup.cpuset_mems.as_ref())
}

/// If the cgroup imposes a `memory.max` and the pool would exceed it,
/// scale every per-node allocation down proportionally so the sum fits.
/// Returns `(adjusted_map, was_capped)`.
fn cap_by_cgroup_memory(
    mut per_node: HashMap<u32, u64>,
    cgroup_memory_max: Option<u64>,
) -> (HashMap<u32, u64>, bool) {
    let limit = match cgroup_memory_max {
        Some(l) => l,
        None => return (per_node, false),
    };
    let total: u64 = per_node.values().sum();
    if total <= limit || total == 0 {
        return (per_node, false);
    }
    let scale = (limit as f64) / (total as f64);
    for v in per_node.values_mut() {
        *v = ((*v as f64) * scale).floor() as u64;
    }
    (per_node, true)
}

/// Resolve the user's [`PoolSizing`] into a concrete bytes-per-node map,
/// keyed by NUMA node id.
fn compute_per_node_sizes(
    sizing: &PoolSizing,
    host_nodes: &[&NumaNodeView],
) -> ServiceResult<HashMap<u32, u64>> {
    let mut out: HashMap<u32, u64> = HashMap::with_capacity(host_nodes.len());
    match sizing {
        PoolSizing::Ratio(r) => {
            if !(0.0..=1.0).contains(r) {
                return Err(ServiceError::InvalidArgument(format!(
                    "PoolSizing::Ratio({r}) must be in [0.0, 1.0]"
                )));
            }
            for n in host_nodes {
                let total = n.total_bytes.ok_or_else(|| {
                    ServiceError::Internal(format!(
                        "node {} has no MemTotal reading; Ratio sizing needs per-node \
                         capacity. Set PoolSizing::PerNode or PoolSizing::Explicit instead.",
                        n.node.0
                    ))
                })?;
                out.insert(n.node.0, ((total as f64) * r).floor() as u64);
            }
        }
        PoolSizing::Total { bytes } => {
            // Proportional split by node MemTotal so heterogeneous boxes
            // get a fair per-node share.
            let totals: Vec<(u32, u64)> = host_nodes
                .iter()
                .map(|n| {
                    n.total_bytes.map(|t| (n.node.0, t)).ok_or_else(|| {
                        ServiceError::Internal(format!(
                            "node {} has no MemTotal reading; Total sizing needs \
                                 per-node capacity. Set PoolSizing::PerNode or Explicit \
                                 instead.",
                            n.node.0
                        ))
                    })
                })
                .collect::<ServiceResult<_>>()?;
            let total_sum: u64 = totals.iter().map(|(_, t)| *t).sum();
            if total_sum == 0 {
                return Err(ServiceError::Internal(
                    "host-memory nodes report zero total MemTotal; cannot proportionally split"
                        .into(),
                ));
            }
            let mut assigned: u64 = 0;
            let last_idx = totals.len() - 1;
            for (i, (id, t)) in totals.iter().enumerate() {
                let share = if i == last_idx {
                    bytes.saturating_sub(assigned)
                } else {
                    let v = ((*bytes as f64) * (*t as f64) / (total_sum as f64)).floor() as u64;
                    assigned = assigned.saturating_add(v);
                    v
                };
                out.insert(*id, share);
            }
        }
        PoolSizing::PerNode { bytes_per_node } => {
            for n in host_nodes {
                out.insert(n.node.0, *bytes_per_node);
            }
        }
        PoolSizing::Explicit(map) => {
            for n in host_nodes {
                match map.get(&n.node.0).copied() {
                    Some(b) => {
                        out.insert(n.node.0, b);
                    }
                    None => {
                        return Err(ServiceError::InvalidArgument(format!(
                            "PoolSizing::Explicit missing entry for host-memory node {}",
                            n.node.0
                        )));
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u32, cpus: Vec<usize>, total_bytes: Option<u64>) -> NumaNodeView {
        NumaNodeView {
            node: NumaNode(id),
            cpus,
            gpu_indices: vec![],
            role: dynamo_memory::NumaNodeRole::HostCpu,
            total_bytes,
        }
    }

    #[test]
    fn ratio_sizing_scales_per_node_total() {
        let n0 = node(0, vec![0, 1], Some(100));
        let n1 = node(1, vec![2, 3], Some(200));
        let nodes = vec![&n0, &n1];
        let out = compute_per_node_sizes(&PoolSizing::Ratio(0.5), &nodes).unwrap();
        assert_eq!(out.get(&0).copied(), Some(50));
        assert_eq!(out.get(&1).copied(), Some(100));
    }

    #[test]
    fn ratio_sizing_rejects_out_of_range() {
        let n0 = node(0, vec![0], Some(100));
        let nodes = vec![&n0];
        assert!(compute_per_node_sizes(&PoolSizing::Ratio(-0.1), &nodes).is_err());
        assert!(compute_per_node_sizes(&PoolSizing::Ratio(1.1), &nodes).is_err());
    }

    #[test]
    fn per_node_sizing_gives_every_node_same() {
        let n0 = node(0, vec![0], Some(100));
        let n1 = node(1, vec![1], Some(50));
        let nodes = vec![&n0, &n1];
        let out =
            compute_per_node_sizes(&PoolSizing::PerNode { bytes_per_node: 64 }, &nodes).unwrap();
        assert_eq!(out.get(&0).copied(), Some(64));
        assert_eq!(out.get(&1).copied(), Some(64));
    }

    #[test]
    fn total_sizing_proportional_to_capacity() {
        let n0 = node(0, vec![0], Some(100));
        let n1 = node(1, vec![1], Some(300));
        let nodes = vec![&n0, &n1];
        // 400 total bytes split proportionally to 100/300 -> 100 + 300
        let out = compute_per_node_sizes(&PoolSizing::Total { bytes: 400 }, &nodes).unwrap();
        assert_eq!(out.get(&0).copied(), Some(100));
        assert_eq!(out.get(&1).copied(), Some(300));
        let total: u64 = out.values().sum();
        assert_eq!(total, 400);
    }

    #[test]
    fn cgroup_cpuset_mems_filters_nodes() {
        let n0 = node(0, vec![0, 1], Some(100));
        let n1 = node(1, vec![2, 3], Some(100));
        let all = vec![&n0, &n1];
        let cg = CgroupInfo {
            cpuset_mems: Some(vec![0]),
            ..Default::default()
        };
        let kept: Vec<u32> = nodes_allowed_by_cgroup(all, &cg)
            .into_iter()
            .map(|n| n.node.0)
            .collect();
        assert_eq!(kept, vec![0]);
    }

    fn env_ucx_loader() -> impl FnOnce() -> ServiceResult<NixlBackendConfig> {
        || Ok(NixlBackendConfig::default().with_backend("UCX"))
    }

    /// Records whether the env-loading closure ran. Used to prove that
    /// opt-out paths never read env vars (so a malformed
    /// `DYN_KVBM_NIXL_BACKEND_*` can't abort startup).
    struct EnvProbe {
        called: std::cell::Cell<bool>,
    }
    impl EnvProbe {
        fn new() -> Self {
            Self {
                called: std::cell::Cell::new(false),
            }
        }
        fn loader(&self) -> impl FnOnce() -> ServiceResult<NixlBackendConfig> + '_ {
            || {
                self.called.set(true);
                Ok(NixlBackendConfig::default().with_backend("UCX"))
            }
        }
        fn was_called(&self) -> bool {
            self.called.get()
        }
    }

    #[test]
    fn resolve_backends_explicit_list_wins() {
        let resolved =
            resolve_backend_config(&["POSIX".to_string()], false, env_ucx_loader()).unwrap();
        assert!(resolved.has_backend("POSIX"));
        assert!(!resolved.has_backend("UCX"));
    }

    #[test]
    fn resolve_backends_empty_list_falls_back_to_env() {
        let resolved = resolve_backend_config(&[], false, env_ucx_loader()).unwrap();
        assert!(
            resolved.has_backend("UCX"),
            "empty backends + allow_no_nixl_backends=false must consult env"
        );
    }

    /// The bug fix: an operator who calls `.local_only()` on the builder
    /// gets `backends = []` and `allow_no_nixl_backends = true`. If a
    /// `DYN_KVBM_NIXL_BACKEND_UCX=true` env var happens to be set, the
    /// old code would silently re-enable NIXL, defeating the intent.
    #[test]
    fn resolve_backends_local_only_ignores_env_var() {
        let resolved = resolve_backend_config(&[], true, env_ucx_loader()).unwrap();
        assert!(
            !resolved.has_backend("UCX"),
            "allow_no_nixl_backends=true must suppress env fallback"
        );
        assert!(resolved.backends().is_empty());
    }

    #[test]
    fn resolve_backends_explicit_list_overrides_allow_no_nixl() {
        // Edge case: user set a non-empty list AND allow_no_nixl_backends
        // (e.g. via TOML); explicit list still wins. The flag is a fail-
        // safe for "no backends OK", not a global mute.
        let resolved =
            resolve_backend_config(&["UCX".to_string()], true, env_ucx_loader()).unwrap();
        assert!(resolved.has_backend("UCX"));
    }

    /// Opt-out and explicit-list paths must NOT call the env loader.
    /// Regression for the "env parsing still runs on opt-out paths" issue:
    /// a malformed env var must not affect configs that never look at env.
    #[test]
    fn resolve_backends_explicit_list_skips_env_loader() {
        let probe = EnvProbe::new();
        let _ = resolve_backend_config(&["UCX".to_string()], false, probe.loader()).unwrap();
        assert!(
            !probe.was_called(),
            "explicit backend list must not trigger env parsing"
        );
    }

    #[test]
    fn resolve_backends_local_only_skips_env_loader() {
        let probe = EnvProbe::new();
        let _ = resolve_backend_config(&[], true, probe.loader()).unwrap();
        assert!(
            !probe.was_called(),
            "allow_no_nixl_backends=true must not trigger env parsing"
        );
    }

    #[test]
    fn resolve_backends_env_loader_error_only_fails_on_fallback_path() {
        // If the env loader fails, only the legacy-fallback branch
        // surfaces the error. Explicit list / local-only succeed without
        // touching the loader.
        let failing = || Err(ServiceError::Internal("simulated bad env".into()));
        assert!(resolve_backend_config(&["UCX".to_string()], false, failing).is_ok());
        let failing = || Err(ServiceError::Internal("simulated bad env".into()));
        assert!(resolve_backend_config(&[], true, failing).is_ok());
        let failing = || Err(ServiceError::Internal("simulated bad env".into()));
        assert!(resolve_backend_config(&[], false, failing).is_err());
    }

    #[test]
    fn cgroup_effective_mems_preferred_over_configured() {
        // Configured mems says [0,1] but the kernel-effective mask is [0] —
        // honor the latter because that's what mbind() will enforce.
        let n0 = node(0, vec![0], Some(100));
        let n1 = node(1, vec![1], Some(100));
        let all = vec![&n0, &n1];
        let cg = CgroupInfo {
            cpuset_mems: Some(vec![0, 1]),
            cpuset_mems_effective: Some(vec![0]),
            ..Default::default()
        };
        let kept: Vec<u32> = nodes_allowed_by_cgroup(all, &cg)
            .into_iter()
            .map(|n| n.node.0)
            .collect();
        assert_eq!(kept, vec![0]);
    }

    #[test]
    fn cgroup_falls_back_to_configured_when_effective_missing() {
        // v1 path (or v2 file unreadable): effective is None, fall back to
        // configured.
        let n0 = node(0, vec![0], Some(100));
        let n1 = node(1, vec![1], Some(100));
        let all = vec![&n0, &n1];
        let cg = CgroupInfo {
            cpuset_mems: Some(vec![1]),
            cpuset_mems_effective: None,
            ..Default::default()
        };
        let kept: Vec<u32> = nodes_allowed_by_cgroup(all, &cg)
            .into_iter()
            .map(|n| n.node.0)
            .collect();
        assert_eq!(kept, vec![1]);
    }

    #[test]
    fn cgroup_cpuset_mems_none_passes_through() {
        let n0 = node(0, vec![0], Some(100));
        let n1 = node(1, vec![1], Some(100));
        let all = vec![&n0, &n1];
        let cg = CgroupInfo::default();
        let kept = nodes_allowed_by_cgroup(all, &cg);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn cgroup_memory_max_scales_pool_down() {
        let mut per_node = HashMap::new();
        per_node.insert(0u32, 100u64);
        per_node.insert(1u32, 300u64);
        // Limit = 200 → total of 400 must shrink to ≤ 200.
        let (capped, was_capped) = cap_by_cgroup_memory(per_node, Some(200));
        assert!(was_capped);
        let total: u64 = capped.values().sum();
        assert!(total <= 200, "total {total} should be <= 200");
        // Proportionality: node 1 had 3x more before, still has ~3x more.
        assert!(capped[&1] >= 2 * capped[&0]);
    }

    #[test]
    fn cgroup_memory_max_noop_when_under_limit() {
        let mut per_node = HashMap::new();
        per_node.insert(0u32, 50u64);
        per_node.insert(1u32, 50u64);
        let (out, was_capped) = cap_by_cgroup_memory(per_node.clone(), Some(1000));
        assert!(!was_capped);
        assert_eq!(out, per_node);
    }

    #[test]
    fn cgroup_memory_max_none_is_noop() {
        let mut per_node = HashMap::new();
        per_node.insert(0u32, 1_000_000u64);
        let (out, was_capped) = cap_by_cgroup_memory(per_node.clone(), None);
        assert!(!was_capped);
        assert_eq!(out, per_node);
    }

    #[test]
    fn lease_is_exclusive_and_returns_on_drop() {
        let pool = HostMemoryPool::empty_for_tests("test");
        let lease = pool.lease_all().expect("first lease");
        assert!(pool.lease_all().is_err(), "second lease should fail");
        drop(lease);
        // Drop releases the lease back to the pool.
        let _again = pool.lease_all().expect("lease after drop");
    }

    #[test]
    fn snapshot_reflects_lease_state() {
        let pool = HostMemoryPool::empty_for_tests("test");
        assert!(!pool.snapshot().leased);
        let lease = pool.lease_all().unwrap();
        assert!(pool.snapshot().leased);
        drop(lease);
        assert!(!pool.snapshot().leased);
    }

    #[test]
    fn explicit_sizing_requires_entry_per_node() {
        let n0 = node(0, vec![0], Some(100));
        let n1 = node(1, vec![1], Some(200));
        let nodes = vec![&n0, &n1];
        let mut map = HashMap::new();
        map.insert(0u32, 32);
        // missing node 1 → error
        assert!(compute_per_node_sizes(&PoolSizing::Explicit(map.clone()), &nodes).is_err());
        map.insert(1u32, 64);
        let out = compute_per_node_sizes(&PoolSizing::Explicit(map), &nodes).unwrap();
        assert_eq!(out.get(&0).copied(), Some(32));
        assert_eq!(out.get(&1).copied(), Some(64));
    }
}
