// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host GPU / NUMA / CPU resource discovery.
//!
//! [`Resources::discover`] produces an immutable snapshot of every NVIDIA GPU on
//! the host, the NUMA topology, and a deterministic per-GPU slice of the
//! relevant CPU set so that multiple GPUs sharing a NUMA node do not contend
//! for the same cores. Allocation is intentionally out of scope — this module
//! describes the system, it does not change it.
//!
//! ## Container safety
//!
//! GPU enumeration is driven by sysfs (`/sys/bus/pci/devices`), which is not
//! network-namespaced and reflects host topology even when NVML's view is
//! restricted by the container runtime. NVML is only consulted to fill in
//! details sysfs cannot provide.
//!
//! ## Slicing policy
//!
//! By default ([`SlicingMode::AssumeAllBusy`]), each NUMA node's CPU list is
//! divided evenly across **every** host GPU on that node, even GPUs the
//! current process cannot address. This prevents a container holding 2 of 8
//! host GPUs from claiming all of a node's CPUs and fighting siblings on the
//! same box. [`SlicingMode::VisibleOnly`] is available for callers that own
//! every GPU on the host.
//!
//! ## Fallbacks
//!
//! Funky topologies (no NUMA, GPUs without affinity info, memory-only NUMA
//! nodes, missing sysfs) never produce errors. Each GPU's [`SliceSource`]
//! tags how its slice was derived; see the variants for details.

use std::collections::{HashMap, HashSet};

use crate::numa::{
    self, GpuInfo, NumaNode, get_pci_bus_address_from_cuda, is_numa_enabled,
    topology::{NumaTopology, parse_cpulist},
};

/// How a [`GpuView`]'s [`GpuView::cpu_slice`] was derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceSource {
    /// Happy path: the GPU's NUMA node had a non-empty cpulist and the slice
    /// is its deterministic share of that cpulist.
    Numa(NumaNode),
    /// The GPU's NUMA node was `-1` or unreadable. Slice was carved from the
    /// host cpuset against the bucket of no-affinity GPUs.
    NoAffinityBucket,
    /// The GPU's NUMA node existed in topology but had an empty cpulist
    /// (e.g. memory-only NUMA nodes on Grace/GB200). Slice is the full host
    /// cpuset; not subdivided further because the nearest CPU-bearing node is
    /// not derivable without distance info.
    EmptyNumaNodeFallback,
    /// NUMA was disabled, the system reported a single node, or topology was
    /// otherwise degenerate. All GPUs share a host-wide bucket sliced evenly.
    HostCpuset,
    /// NUMA topology could not be read at all. `cpu_slice` is empty; the
    /// caller should not pin and should let the OS scheduler decide.
    NoTopology,
}

/// How the slicing across siblings should be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlicingMode {
    /// Assume every host GPU may be running concurrently — divide each NUMA
    /// node's CPUs across all host GPUs on that node. Safe default for
    /// shared hosts and multi-tenant containers.
    AssumeAllBusy,
    /// Only the CUDA-visible GPUs count as siblings — divide each NUMA
    /// node's CPUs across the visible GPUs on that node. Use when the
    /// caller knows it owns every active GPU on the host.
    VisibleOnly,
}

impl std::fmt::Display for SlicingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AssumeAllBusy => f.write_str("assume-all-busy"),
            Self::VisibleOnly => f.write_str("visible-only"),
        }
    }
}

/// View of a single NUMA node.
#[derive(Debug, Clone)]
pub struct NumaNodeView {
    /// The node ID.
    pub node: NumaNode,
    /// CPUs belonging to this node (sorted, deduplicated). Empty for
    /// memory-only nodes.
    pub cpus: Vec<usize>,
    /// Indices into [`Resources::gpus`] for GPUs attached to this node.
    pub gpu_indices: Vec<usize>,
}

/// View of a single host GPU.
#[derive(Debug, Clone)]
pub struct GpuView {
    /// Normalized PCI bus address, e.g. `"0000:3b:00.0"`.
    pub pci_address: String,
    /// CUDA ordinal for this process, if the device is visible. `None`
    /// indicates the GPU exists on the host but is hidden from this process
    /// (typically by `CUDA_VISIBLE_DEVICES` or container GPU allotment).
    pub cuda_ordinal: Option<u32>,
    /// NUMA node from `/sys/bus/pci/devices/<pci>/numa_node`. `None` means
    /// `-1` (no affinity info available).
    pub numa_node: Option<NumaNode>,
    /// Deterministic CPU slice for this GPU. May equal the full host cpuset
    /// in certain fallback paths; empty only when [`Self::slice_source`] is
    /// [`SliceSource::NoTopology`].
    pub cpu_slice: Vec<usize>,
    /// How `cpu_slice` was derived. See [`SliceSource`] for the meanings.
    pub slice_source: SliceSource,
}

/// Immutable snapshot of host GPU / NUMA / CPU topology with per-GPU CPU slices.
///
/// Built via [`Resources::discover`] (or [`Resources::discover_with`] to pick a
/// non-default [`SlicingMode`]). Sync, never panics, never errors — instead
/// degraded topologies are surfaced through [`GpuView::slice_source`].
#[derive(Debug, Clone)]
pub struct Resources {
    /// One entry per NUMA node that has at least one CPU or at least one GPU.
    pub nodes: Vec<NumaNodeView>,
    /// Every NVIDIA GPU discovered on the host. The list is canonical
    /// (sysfs-derived in the normal path) and stable across `CUDA_VISIBLE_DEVICES`.
    pub gpus: Vec<GpuView>,
    /// Whether NUMA optimizations are enabled (mirror of
    /// `dynamo_memory::is_numa_enabled()` at discovery time).
    pub numa_enabled: bool,
    /// Union of every NUMA node's cpulist, or
    /// `std::thread::available_parallelism()` as a last-ditch fallback.
    pub host_cpus: Vec<usize>,
    /// CPUs this process is allowed to schedule on, parsed from
    /// `/proc/self/status` `Cpus_allowed_list`. Diagnostic only — not folded
    /// into [`GpuView::cpu_slice`].
    pub process_allowed_cpus: Vec<usize>,
    /// Process launch context: cgroup version, cgroup-imposed cpuset and
    /// memory/CPU limits, and a container hint. Diagnostic only.
    pub cgroup: CgroupInfo,
    /// The slicing mode that produced this snapshot.
    pub mode: SlicingMode,
}

/// Which cgroup hierarchy the current process is under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CgroupVersion {
    /// Legacy cgroup v1 (per-controller mounts under `/sys/fs/cgroup/<ctrl>`).
    V1,
    /// Unified cgroup v2 (single hierarchy under `/sys/fs/cgroup`).
    V2,
    /// Could not determine — neither v1 controller dirs nor the v2 marker
    /// file were found.
    #[default]
    Unknown,
}

impl std::fmt::Display for CgroupVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V1 => f.write_str("v1"),
            Self::V2 => f.write_str("v2"),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

/// CPU bandwidth limit imposed by the cgroup (`cpu.max` in v2 or
/// `cpu.cfs_{quota,period}_us` in v1).
#[derive(Debug, Clone, Copy)]
pub struct CgroupCpuMax {
    /// Quota in microseconds per period. `None` means unlimited (`"max"`
    /// in v2, `-1` in v1).
    pub quota_us: Option<u64>,
    /// Period length in microseconds (typically 100000 = 100 ms).
    pub period_us: u64,
}

impl CgroupCpuMax {
    /// Effective CPU-core equivalent, e.g. `quota=200000 period=100000`
    /// returns `Some(2.0)`. `None` if quota is unlimited.
    pub fn cores(&self) -> Option<f64> {
        let q = self.quota_us?;
        if self.period_us == 0 {
            return None;
        }
        Some(q as f64 / self.period_us as f64)
    }
}

/// cgroup-derived diagnostic info for the current process.
#[derive(Debug, Clone, Default)]
pub struct CgroupInfo {
    /// Detected hierarchy version.
    pub version: CgroupVersion,
    /// Cgroup path within the hierarchy (e.g. `/` for the root, or
    /// `/docker/<id>` in many container runtimes). Parsed from
    /// `/proc/self/cgroup`. `None` if the file could not be read.
    pub path: Option<String>,
    /// `cpuset.cpus` (configured) for this cgroup, parsed as CPU IDs.
    pub cpuset_cpus: Option<Vec<usize>>,
    /// `cpuset.cpus.effective` (after intersecting with ancestor and online
    /// CPUs) for this cgroup. v2 only; v1 callers see `None`.
    pub cpuset_cpus_effective: Option<Vec<usize>>,
    /// `cpuset.mems` — which NUMA memory nodes the cgroup is restricted to.
    pub cpuset_mems: Option<Vec<usize>>,
    /// CPU bandwidth limit. `None` if `cpu.max` / `cpu.cfs_*` could not be read.
    pub cpu_max: Option<CgroupCpuMax>,
    /// `memory.max` (v2) or `memory.limit_in_bytes` (v1). `None` means
    /// unlimited (`"max"` or sentinel). `Some(0)` would be a kernel-imposed
    /// disable; we treat any unparseable value as `None`.
    pub memory_max: Option<u64>,
    /// Short string describing why we think the process is in a container,
    /// or `None` if no signal was found. Heuristic only.
    pub container_hint: Option<String>,
}


impl Resources {
    /// Discover the host resources using [`SlicingMode::AssumeAllBusy`].
    pub fn discover() -> Self {
        Self::discover_with(SlicingMode::AssumeAllBusy)
    }

    /// Discover the host resources using the given slicing mode.
    pub fn discover_with(mode: SlicingMode) -> Self {
        let numa_enabled = is_numa_enabled();
        let topology = numa::topology::get_numa_topology().ok();

        let all_gpus = numa::enumerate_all_gpus();

        // Best-effort CUDA init so a fresh process (e.g. the inspection
        // binary) can still see ordinals. Errors are ignored: a CUDA-less
        // host should still produce a useful sysfs-derived snapshot.
        let _ = cudarc::driver::result::init();

        let mut cuda_ordinals_by_pci: HashMap<String, u32> = HashMap::new();
        if let Ok(count) = cudarc::driver::result::device::get_count() {
            for i in 0..count as u32 {
                if let Some(pci) = get_pci_bus_address_from_cuda(i) {
                    cuda_ordinals_by_pci.insert(pci, i);
                }
            }
        }

        let process_allowed_cpus = read_process_allowed_cpus();
        let host_cpus_fallback = available_parallelism_range();
        let cgroup = read_cgroup_info();

        compute_resources_from_inputs(
            topology,
            all_gpus,
            cuda_ordinals_by_pci,
            numa_enabled,
            host_cpus_fallback,
            process_allowed_cpus,
            cgroup,
            mode,
        )
    }

    /// Find a GPU by PCI bus address (e.g. `"0000:3b:00.0"`).
    pub fn by_pci(&self, pci: &str) -> Option<&GpuView> {
        self.gpus.iter().find(|g| g.pci_address == pci)
    }

    /// Find a GPU by CUDA ordinal (as seen by the current process).
    pub fn by_cuda_ordinal(&self, ordinal: u32) -> Option<&GpuView> {
        self.gpus.iter().find(|g| g.cuda_ordinal == Some(ordinal))
    }

    /// Iterate GPUs attached to the given NUMA node.
    pub fn gpus_on_node(&self, node: NumaNode) -> impl Iterator<Item = &GpuView> {
        self.gpus
            .iter()
            .filter(move |g| g.numa_node == Some(node))
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_resources_from_inputs(
    topology: Option<&'static NumaTopology>,
    all_gpus: Vec<GpuInfo>,
    cuda_ordinals_by_pci: HashMap<String, u32>,
    numa_enabled: bool,
    host_cpus_fallback: Vec<usize>,
    process_allowed_cpus: Vec<usize>,
    cgroup: CgroupInfo,
    mode: SlicingMode,
) -> Resources {
    let visible_pcis: HashSet<&String> = cuda_ordinals_by_pci.keys().collect();

    // Seed GpuView rows from the canonical host GPU list.
    let mut gpus: Vec<GpuView> = all_gpus
        .iter()
        .map(|g| GpuView {
            pci_address: g.pci_address.clone(),
            cuda_ordinal: cuda_ordinals_by_pci.get(&g.pci_address).copied(),
            numa_node: g.numa_node.map(NumaNode),
            cpu_slice: Vec::new(),
            slice_source: SliceSource::NoTopology,
        })
        .collect();

    // CUDA-visible PCIs not in sysfs (only possible if /sys is missing or
    // we fell back to CUDA-driver enumeration that disagrees) — append as
    // synthetic entries so the visibility view is faithful.
    let known: HashSet<&String> = gpus.iter().map(|g| &g.pci_address).collect();
    let extras: Vec<(String, u32)> = cuda_ordinals_by_pci
        .iter()
        .filter(|(pci, _)| !known.contains(pci))
        .map(|(pci, ord)| (pci.clone(), *ord))
        .collect();
    drop(known);
    for (pci, ord) in extras {
        gpus.push(GpuView {
            pci_address: pci,
            cuda_ordinal: Some(ord),
            numa_node: None,
            cpu_slice: Vec::new(),
            slice_source: SliceSource::NoTopology,
        });
    }

    // Compute host_cpus: union of every node's cpulist, or fallback.
    let mut host_cpus: Vec<usize> = match topology {
        Some(t) => {
            let mut set: HashSet<usize> = HashSet::new();
            // NumaTopology does not expose node IDs publicly; sweep a wide
            // range and collect anything present.
            for node_id in 0..1024u32 {
                if let Some(cpus) = t.cpus_for_node(node_id) {
                    set.extend(cpus.iter().copied());
                }
            }
            let mut v: Vec<usize> = set.into_iter().collect();
            v.sort_unstable();
            v
        }
        None => host_cpus_fallback.clone(),
    };
    if host_cpus.is_empty() {
        host_cpus = host_cpus_fallback;
    }

    // Decide the slicing path.
    let topology = match topology {
        Some(t) => t,
        None => {
            eprintln!(
                "dynamo-memory::resources: NUMA topology unavailable, all GPUs share host cpuset (or empty)"
            );
            let slice = host_cpus.clone();
            let source = if slice.is_empty() {
                SliceSource::NoTopology
            } else {
                SliceSource::HostCpuset
            };
            for g in &mut gpus {
                g.cpu_slice = slice.clone();
                g.slice_source = source;
            }
            return Resources {
                nodes: Vec::new(),
                gpus,
                numa_enabled,
                host_cpus,
                process_allowed_cpus,
                cgroup,
                mode,
            };
        }
    };

    if !numa_enabled || topology.is_single_node() {
        // One host-wide bucket sliced evenly.
        let pcis: Vec<String> = gpus.iter().map(|g| g.pci_address.clone()).collect();
        let slices = slice_evenly(&host_cpus, &pcis);
        for g in &mut gpus {
            g.cpu_slice = slices.get(&g.pci_address).cloned().unwrap_or_default();
            g.slice_source = SliceSource::HostCpuset;
        }
    } else {
        // Bucket by NUMA node.
        let mut by_node: HashMap<u32, Vec<String>> = HashMap::new();
        let mut no_affinity_bucket: Vec<String> = Vec::new();
        for g in &gpus {
            match g.numa_node {
                Some(n) => by_node.entry(n.0).or_default().push(g.pci_address.clone()),
                None => no_affinity_bucket.push(g.pci_address.clone()),
            }
        }

        // Build a quick map for setting slice/source on each GpuView.
        let mut slice_map: HashMap<String, (Vec<usize>, SliceSource)> = HashMap::new();

        for (node, members) in &by_node {
            let cpus_opt = topology.cpus_for_node(*node);
            match cpus_opt {
                Some(cpus) if !cpus.is_empty() => {
                    let bucket: Vec<String> = match mode {
                        SlicingMode::AssumeAllBusy => members.clone(),
                        SlicingMode::VisibleOnly => members
                            .iter()
                            .filter(|p| visible_pcis.contains(p))
                            .cloned()
                            .collect(),
                    };
                    let slices_for_bucket = if !bucket.is_empty() {
                        slice_evenly(cpus, &bucket)
                    } else {
                        HashMap::new()
                    };
                    // Visible-only members get bucket slices; non-bucket members
                    // get the AssumeAllBusy slice (computed against all members).
                    let all_slices = slice_evenly(cpus, members);
                    for pci in members {
                        let slice = slices_for_bucket
                            .get(pci)
                            .cloned()
                            .unwrap_or_else(|| all_slices.get(pci).cloned().unwrap_or_default());
                        slice_map.insert(
                            pci.clone(),
                            (slice, SliceSource::Numa(NumaNode(*node))),
                        );
                    }
                }
                _ => {
                    // Bucket C: NUMA node has no cpulist (memory-only) → host cpuset.
                    eprintln!(
                        "dynamo-memory::resources: NUMA node {} has empty cpulist; GPUs on this node fall back to host cpuset",
                        node
                    );
                    for pci in members {
                        slice_map.insert(
                            pci.clone(),
                            (host_cpus.clone(), SliceSource::EmptyNumaNodeFallback),
                        );
                    }
                }
            }
        }

        if !no_affinity_bucket.is_empty() {
            eprintln!(
                "dynamo-memory::resources: {} GPU(s) without NUMA affinity; slicing host cpuset across that bucket",
                no_affinity_bucket.len()
            );
            let slices = slice_evenly(&host_cpus, &no_affinity_bucket);
            for pci in &no_affinity_bucket {
                let slice = slices.get(pci).cloned().unwrap_or_default();
                slice_map.insert(pci.clone(), (slice, SliceSource::NoAffinityBucket));
            }
        }

        for g in &mut gpus {
            if let Some((slice, source)) = slice_map.remove(&g.pci_address) {
                g.cpu_slice = slice;
                g.slice_source = source;
            }
        }
    }

    // Build node views (only for nodes with cpus OR attached GPUs).
    let mut nodes: Vec<NumaNodeView> = Vec::new();
    {
        let mut node_ids: HashSet<u32> = HashSet::new();
        for node_id in 0..1024u32 {
            if topology.cpus_for_node(node_id).is_some() {
                node_ids.insert(node_id);
            }
        }
        for g in &gpus {
            if let Some(n) = g.numa_node {
                node_ids.insert(n.0);
            }
        }
        let mut sorted: Vec<u32> = node_ids.into_iter().collect();
        sorted.sort_unstable();
        for node_id in sorted {
            let cpus = topology
                .cpus_for_node(node_id)
                .map(|c| c.to_vec())
                .unwrap_or_default();
            let gpu_indices: Vec<usize> = gpus
                .iter()
                .enumerate()
                .filter_map(|(idx, g)| (g.numa_node == Some(NumaNode(node_id))).then_some(idx))
                .collect();
            if cpus.is_empty() && gpu_indices.is_empty() {
                continue;
            }
            nodes.push(NumaNodeView {
                node: NumaNode(node_id),
                cpus,
                gpu_indices,
            });
        }
    }

    Resources {
        nodes,
        gpus,
        numa_enabled,
        host_cpus,
        process_allowed_cpus,
        cgroup,
        mode,
    }
}

/// Slice a CPU set evenly across the given (sorted) PCI list.
///
/// `pcis` is expected to be the deterministic ordering of the bucket; the
/// caller sorts before invoking. Returns one slice per input PCI. If the
/// bucket has more GPUs than CPUs, every GPU gets the full set.
fn slice_evenly(cpus: &[usize], pcis: &[String]) -> HashMap<String, Vec<usize>> {
    let mut out: HashMap<String, Vec<usize>> = HashMap::new();
    if pcis.is_empty() || cpus.is_empty() {
        return out;
    }
    let mut sorted_pcis: Vec<&String> = pcis.iter().collect();
    sorted_pcis.sort();
    let n = sorted_pcis.len();
    let chunk = cpus.len() / n;
    if chunk == 0 {
        for pci in sorted_pcis {
            out.insert(pci.clone(), cpus.to_vec());
        }
        return out;
    }
    for (i, pci) in sorted_pcis.iter().enumerate() {
        let start = i * chunk;
        let end = if i == n - 1 { cpus.len() } else { start + chunk };
        out.insert((*pci).clone(), cpus[start..end].to_vec());
    }
    out
}

fn read_cgroup_info() -> CgroupInfo {
    let version = detect_cgroup_version();
    let path = read_self_cgroup_path(version);
    let container_hint = detect_container_hint();

    let (cpuset_cpus, cpuset_cpus_effective, cpuset_mems, cpu_max, memory_max) = match version {
        CgroupVersion::V2 => {
            let rel = path.as_deref().unwrap_or("");
            (
                read_cgroup_v2(rel, "cpuset.cpus").and_then(|s| parse_cpulist(&s).ok()),
                read_cgroup_v2(rel, "cpuset.cpus.effective").and_then(|s| parse_cpulist(&s).ok()),
                read_cgroup_v2(rel, "cpuset.mems").and_then(|s| parse_cpulist(&s).ok()),
                read_cgroup_v2(rel, "cpu.max").and_then(|s| parse_cgroup_v2_cpu_max(&s)),
                read_cgroup_v2(rel, "memory.max").and_then(|s| parse_cgroup_v2_memory_max(&s)),
            )
        }
        CgroupVersion::V1 => {
            let rel = path.as_deref().unwrap_or("");
            let quota = read_cgroup_v1(rel, "cpu", "cpu.cfs_quota_us")
                .and_then(|s| s.trim().parse::<i64>().ok());
            let period = read_cgroup_v1(rel, "cpu", "cpu.cfs_period_us")
                .and_then(|s| s.trim().parse::<u64>().ok());
            let cpu_max = match (quota, period) {
                (Some(q), Some(p)) => Some(CgroupCpuMax {
                    quota_us: if q < 0 { None } else { Some(q as u64) },
                    period_us: p,
                }),
                _ => None,
            };
            (
                read_cgroup_v1(rel, "cpuset", "cpuset.cpus").and_then(|s| parse_cpulist(&s).ok()),
                None,
                read_cgroup_v1(rel, "cpuset", "cpuset.mems").and_then(|s| parse_cpulist(&s).ok()),
                cpu_max,
                read_cgroup_v1(rel, "memory", "memory.limit_in_bytes")
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .and_then(|v| {
                        // The v1 sentinel for "no limit" is a huge value. Treat
                        // anything >= 1 EiB as unlimited.
                        if v >= (1u64 << 60) { None } else { Some(v) }
                    }),
            )
        }
        CgroupVersion::Unknown => (None, None, None, None, None),
    };

    CgroupInfo {
        version,
        path,
        cpuset_cpus,
        cpuset_cpus_effective,
        cpuset_mems,
        cpu_max,
        memory_max,
        container_hint,
    }
}

fn detect_cgroup_version() -> CgroupVersion {
    if std::path::Path::new("/sys/fs/cgroup/cgroup.controllers").exists() {
        CgroupVersion::V2
    } else if std::path::Path::new("/sys/fs/cgroup/cpuset").is_dir()
        || std::path::Path::new("/sys/fs/cgroup/cpu").is_dir()
    {
        CgroupVersion::V1
    } else {
        CgroupVersion::Unknown
    }
}

fn read_self_cgroup_path(version: CgroupVersion) -> Option<String> {
    let content = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    match version {
        CgroupVersion::V2 => {
            // v2 format: "0::/path"
            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("0::") {
                    return Some(rest.to_string());
                }
            }
            None
        }
        CgroupVersion::V1 => {
            // v1: pick the cpuset path if present, otherwise any cpu line.
            let mut fallback: Option<String> = None;
            for line in content.lines() {
                let parts: Vec<&str> = line.splitn(3, ':').collect();
                if parts.len() != 3 {
                    continue;
                }
                let controllers = parts[1];
                let path = parts[2].to_string();
                if controllers.split(',').any(|c| c == "cpuset") {
                    return Some(path);
                }
                if controllers.split(',').any(|c| c == "cpu") {
                    fallback = Some(path);
                }
            }
            fallback
        }
        CgroupVersion::Unknown => None,
    }
}

fn read_cgroup_v2(rel_path: &str, filename: &str) -> Option<String> {
    let candidates = [
        // Namespaced containers: /sys/fs/cgroup is rooted at the container's
        // cgroup, so root-level filenames are the relevant ones.
        format!("/sys/fs/cgroup/{}", filename),
        // Otherwise, descend the path from /proc/self/cgroup.
        format!(
            "/sys/fs/cgroup{}/{}",
            rel_path.trim_end_matches('/'),
            filename
        ),
    ];
    for path in &candidates {
        if let Ok(s) = std::fs::read_to_string(path) {
            return Some(s);
        }
    }
    None
}

fn read_cgroup_v1(rel_path: &str, controller: &str, filename: &str) -> Option<String> {
    let candidates = [
        format!(
            "/sys/fs/cgroup/{}{}/{}",
            controller,
            rel_path.trim_end_matches('/'),
            filename
        ),
        format!("/sys/fs/cgroup/{}/{}", controller, filename),
    ];
    for path in &candidates {
        if let Ok(s) = std::fs::read_to_string(path) {
            return Some(s);
        }
    }
    None
}

fn parse_cgroup_v2_cpu_max(s: &str) -> Option<CgroupCpuMax> {
    // Format: "<quota|max> <period>"
    let mut it = s.split_whitespace();
    let q = it.next()?;
    let p = it.next()?;
    let period_us: u64 = p.parse().ok()?;
    let quota_us = if q == "max" {
        None
    } else {
        Some(q.parse::<u64>().ok()?)
    };
    Some(CgroupCpuMax {
        quota_us,
        period_us,
    })
}

fn parse_cgroup_v2_memory_max(s: &str) -> Option<u64> {
    let t = s.trim();
    if t == "max" {
        return None;
    }
    t.parse::<u64>().ok()
}

fn detect_container_hint() -> Option<String> {
    if std::path::Path::new("/.dockerenv").exists() {
        return Some("/.dockerenv present".to_string());
    }
    if let Ok(s) = std::fs::read_to_string("/proc/1/cgroup") {
        for marker in &["/docker/", "/kubepods", "/containerd", "/lxc/", "/podman/"] {
            if s.contains(marker) {
                return Some(format!("PID 1 cgroup matches '{}'", marker.trim_matches('/')));
            }
        }
    }
    None
}

fn read_process_allowed_cpus() -> Vec<usize> {
    let content = match std::fs::read_to_string("/proc/self/status") {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("Cpus_allowed_list:") {
            return parse_cpulist(rest.trim()).unwrap_or_default();
        }
    }
    Vec::new()
}

fn available_parallelism_range() -> Vec<usize> {
    match std::thread::available_parallelism() {
        Ok(n) => (0..n.get()).collect(),
        Err(_) => Vec::new(),
    }
}

impl std::fmt::Display for Resources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let visible_count = self.gpus.iter().filter(|g| g.cuda_ordinal.is_some()).count();
        writeln!(f, "Dynamo Resources Inspection")?;
        writeln!(f, "===========================")?;
        writeln!(f, "NUMA enabled:           {}", self.numa_enabled)?;
        writeln!(f, "NUMA nodes:             {}", self.nodes.len())?;
        writeln!(
            f,
            "Host GPUs:              {} ({} cuda-visible)",
            self.gpus.len(),
            visible_count
        )?;
        writeln!(f, "Host CPUs:              [{}]", compact_range(&self.host_cpus))?;
        writeln!(
            f,
            "Process-allowed CPUs:   [{}]",
            compact_range(&self.process_allowed_cpus)
        )?;
        writeln!(f, "Slicing mode:           {}", self.mode)?;

        writeln!(f)?;
        writeln!(f, "Process launch context")?;
        writeln!(f, "  cgroup version:       {}", self.cgroup.version)?;
        if let Some(p) = &self.cgroup.path {
            writeln!(f, "  cgroup path:          {}", p)?;
        }
        if let Some(c) = &self.cgroup.cpuset_cpus {
            writeln!(f, "  cpuset.cpus:          [{}]", compact_range(c))?;
        }
        if let Some(c) = &self.cgroup.cpuset_cpus_effective {
            writeln!(f, "  cpuset.cpus.effective: [{}]", compact_range(c))?;
        }
        if let Some(m) = &self.cgroup.cpuset_mems {
            writeln!(f, "  cpuset.mems:          [{}]", compact_range(m))?;
        }
        if let Some(c) = &self.cgroup.cpu_max {
            match c.quota_us {
                Some(q) => writeln!(
                    f,
                    "  cpu.max:              {} / {} us (= {:.2} cores)",
                    q,
                    c.period_us,
                    c.cores().unwrap_or(0.0)
                )?,
                None => writeln!(f, "  cpu.max:              max ({} us period)", c.period_us)?,
            }
        }
        match self.cgroup.memory_max {
            Some(m) => writeln!(
                f,
                "  memory.max:           {} bytes (= {})",
                m,
                format_bytes(m)
            )?,
            None if self.cgroup.version != CgroupVersion::Unknown => {
                writeln!(f, "  memory.max:           max (unlimited)")?
            }
            None => {}
        }
        if let Some(hint) = &self.cgroup.container_hint {
            writeln!(f, "  container hint:       {}", hint)?;
        }

        for node in &self.nodes {
            writeln!(f)?;
            writeln!(
                f,
                "NUMA node {}  cpus=[{}]",
                node.node.0,
                compact_range(&node.cpus)
            )?;
            for idx in &node.gpu_indices {
                let g = &self.gpus[*idx];
                writeln!(
                    f,
                    "  pci={}  cuda={}  cpus=[{}]  source={}",
                    g.pci_address,
                    match g.cuda_ordinal {
                        Some(o) => o.to_string(),
                        None => "-".to_string(),
                    },
                    compact_range(&g.cpu_slice),
                    format_slice_source(g.slice_source),
                )?;
            }
        }

        let orphans: Vec<&GpuView> = self
            .gpus
            .iter()
            .filter(|g| g.numa_node.is_none())
            .collect();
        if !orphans.is_empty() {
            writeln!(f)?;
            writeln!(f, "Unaffinitized GPUs (numa_node=-1)")?;
            for g in orphans {
                writeln!(
                    f,
                    "  pci={}  cuda={}  cpus=[{}]  source={}",
                    g.pci_address,
                    match g.cuda_ordinal {
                        Some(o) => o.to_string(),
                        None => "-".to_string(),
                    },
                    compact_range(&g.cpu_slice),
                    format_slice_source(g.slice_source),
                )?;
            }
        }
        Ok(())
    }
}

fn format_slice_source(s: SliceSource) -> String {
    match s {
        SliceSource::Numa(n) => format!("numa({})", n.0),
        SliceSource::NoAffinityBucket => "no-affinity-bucket".to_string(),
        SliceSource::EmptyNumaNodeFallback => "empty-numa-node-fallback".to_string(),
        SliceSource::HostCpuset => "host-cpuset".to_string(),
        SliceSource::NoTopology => "no-topology".to_string(),
    }
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    const TIB: f64 = GIB * 1024.0;
    let b = bytes as f64;
    if b >= TIB {
        format!("{:.1} TiB", b / TIB)
    } else if b >= GIB {
        format!("{:.1} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.1} KiB", b / KIB)
    } else {
        format!("{} B", bytes)
    }
}

fn compact_range(cpus: &[usize]) -> String {
    if cpus.is_empty() {
        return String::new();
    }
    let mut sorted = cpus.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut out = String::new();
    let mut i = 0;
    while i < sorted.len() {
        let start = sorted[i];
        let mut end = start;
        while i + 1 < sorted.len() && sorted[i + 1] == end + 1 {
            i += 1;
            end = sorted[i];
        }
        if !out.is_empty() {
            out.push(',');
        }
        if start == end {
            out.push_str(&start.to_string());
        } else {
            out.push_str(&format!("{}-{}", start, end));
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(pci: &str, node: Option<u32>) -> GpuInfo {
        GpuInfo {
            pci_address: pci.to_string(),
            numa_node: node,
        }
    }

    #[test]
    fn slicing_with_no_topology_uses_host_cpuset_fallback() {
        let gpus = vec![gpu("0000:01:00.0", Some(0)), gpu("0000:02:00.0", Some(0))];
        let r = compute_resources_from_inputs(
            None,
            gpus,
            HashMap::new(),
            true,
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3],
            CgroupInfo::default(),
            SlicingMode::AssumeAllBusy,
        );
        assert_eq!(r.gpus.len(), 2);
        for g in &r.gpus {
            assert_eq!(g.slice_source, SliceSource::HostCpuset);
            assert_eq!(g.cpu_slice, vec![0, 1, 2, 3]);
        }
    }

    #[test]
    fn slicing_with_no_topology_and_empty_fallback_yields_no_topology() {
        let gpus = vec![gpu("0000:01:00.0", None)];
        let r = compute_resources_from_inputs(
            None,
            gpus,
            HashMap::new(),
            true,
            Vec::new(),
            Vec::new(),
            CgroupInfo::default(),
            SlicingMode::AssumeAllBusy,
        );
        assert_eq!(r.gpus[0].slice_source, SliceSource::NoTopology);
        assert!(r.gpus[0].cpu_slice.is_empty());
    }

    #[test]
    fn deterministic_ordering_across_random_input_order() {
        // Same GPUs in different input orders → identical PCI → slice map.
        // This is the contract that makes us safe across CUDA_VISIBLE_DEVICES.
        let g_in_order = [
            gpu("0000:01:00.0", Some(0)),
            gpu("0000:02:00.0", Some(0)),
            gpu("0000:03:00.0", Some(0)),
            gpu("0000:04:00.0", Some(0)),
        ];
        let g_reversed: Vec<GpuInfo> = g_in_order.iter().rev().cloned().collect();

        let pcis: Vec<String> = g_in_order.iter().map(|g| g.pci_address.clone()).collect();
        let cpus = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let a = slice_evenly(&cpus, &pcis);
        let b = slice_evenly(
            &cpus,
            &g_reversed
                .iter()
                .map(|g| g.pci_address.clone())
                .collect::<Vec<_>>(),
        );
        assert_eq!(a, b);
        assert_eq!(a.get("0000:01:00.0").unwrap(), &vec![0, 1]);
        assert_eq!(a.get("0000:04:00.0").unwrap(), &vec![6, 7]);
    }

    #[test]
    fn more_gpus_than_cpus_gives_all_to_everyone() {
        let pcis: Vec<String> = (0..5)
            .map(|i| format!("0000:0{}:00.0", i))
            .collect();
        let cpus = vec![0, 1, 2];
        let slices = slice_evenly(&cpus, &pcis);
        for p in &pcis {
            assert_eq!(slices.get(p).unwrap(), &cpus);
        }
    }

    #[test]
    fn slice_evenly_last_takes_remainder() {
        let pcis = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let cpus = vec![0, 1, 2, 3, 4, 5, 6]; // 7 cpus / 3 = 2, last gets 3
        let s = slice_evenly(&cpus, &pcis);
        assert_eq!(s["a"], vec![0, 1]);
        assert_eq!(s["b"], vec![2, 3]);
        assert_eq!(s["c"], vec![4, 5, 6]);
    }

    #[test]
    fn compact_range_handles_runs_and_gaps() {
        assert_eq!(compact_range(&[]), "");
        assert_eq!(compact_range(&[5]), "5");
        assert_eq!(compact_range(&[0, 1, 2, 3]), "0-3");
        assert_eq!(compact_range(&[0, 1, 2, 8, 9, 10]), "0-2,8-10");
        assert_eq!(compact_range(&[2, 0, 1, 4]), "0-2,4");
    }

    #[test]
    fn parse_v2_cpu_max_handles_max_and_numbers() {
        assert!(parse_cgroup_v2_cpu_max("max 100000").unwrap().quota_us.is_none());
        let q = parse_cgroup_v2_cpu_max("200000 100000").unwrap();
        assert_eq!(q.quota_us, Some(200000));
        assert_eq!(q.period_us, 100000);
        assert!((q.cores().unwrap() - 2.0).abs() < f64::EPSILON);
        assert!(parse_cgroup_v2_cpu_max("garbage").is_none());
    }

    #[test]
    fn parse_v2_memory_max_handles_max_and_numbers() {
        assert!(parse_cgroup_v2_memory_max("max").is_none());
        assert_eq!(parse_cgroup_v2_memory_max("17179869184"), Some(17179869184));
        assert!(parse_cgroup_v2_memory_max("nope").is_none());
    }

    #[test]
    fn format_bytes_human_readable() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(2048), "2.0 KiB");
        assert_eq!(format_bytes(8 * 1024 * 1024 * 1024), "8.0 GiB");
    }

    #[test]
    fn slice_source_display_strings() {
        assert_eq!(format_slice_source(SliceSource::Numa(NumaNode(0))), "numa(0)");
        assert_eq!(format_slice_source(SliceSource::HostCpuset), "host-cpuset");
        assert_eq!(format_slice_source(SliceSource::NoAffinityBucket), "no-affinity-bucket");
        assert_eq!(
            format_slice_source(SliceSource::EmptyNumaNodeFallback),
            "empty-numa-node-fallback"
        );
        assert_eq!(format_slice_source(SliceSource::NoTopology), "no-topology");
    }
}
