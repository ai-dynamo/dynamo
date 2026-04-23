// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NUMA topology detection
//!
//! This module provides utilities to read the actual CPU-to-NUMA mapping from the system,
//! replacing heuristic assumptions with real topology data.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global cached topology
static TOPOLOGY: std::sync::OnceLock<Result<NumaTopology, String>> = std::sync::OnceLock::new();

/// Ensures the error-path probe dump in `get_numa_topology` runs at most once.
static PROBE_LOGGED: AtomicBool = AtomicBool::new(false);

/// Represents the CPU topology for NUMA nodes.
///
/// This struct provides bidirectional lookup between NUMA nodes and CPUs,
/// read from the Linux sysfs interface at `/sys/devices/system/node/`.
///
/// CPU-less memory-only nodes (e.g. GPU-attached HBM exposed as NUMA, CXL
/// expanders) are recorded in `node_to_cpus` with an empty `Vec` so callers
/// can distinguish "node exists but has no CPUs" from "node does not exist".
#[derive(Debug)]
pub struct NumaTopology {
    /// Maps NUMA node ID -> list of CPU IDs (empty for memory-only nodes)
    node_to_cpus: HashMap<u32, Vec<usize>>,
    /// Maps CPU ID -> NUMA node ID
    cpu_to_node: HashMap<usize, u32>,
    /// Sorted list of nodes that exist but have no CPUs (memory-only).
    memory_only_nodes: Vec<u32>,
}

impl NumaTopology {
    /// Read NUMA topology from the standard sysfs path `/sys/devices/system/node`.
    pub fn from_sysfs() -> Result<Self, String> {
        Self::from_dir(Path::new("/sys/devices/system/node"))
    }

    /// Read NUMA topology from an arbitrary directory that matches the sysfs
    /// layout (`nodeN/cpulist` files). Used by tests with a tempdir, and
    /// factored out for that reason.
    ///
    /// Failures on individual nodes are logged and skipped — one unusual or
    /// malformed node does not poison the rest of the topology. Empty
    /// `cpulist` files are accepted and recorded as memory-only nodes.
    pub fn from_dir(node_dir: &Path) -> Result<Self, String> {
        let mut node_to_cpus: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut cpu_to_node: HashMap<usize, u32> = HashMap::new();
        let mut memory_only_nodes: Vec<u32> = Vec::new();

        if !node_dir.exists() {
            return Err(format!(
                "Node directory not found: {}",
                node_dir.display()
            ));
        }
        let entries = fs::read_dir(node_dir)
            .map_err(|e| format!("Failed to read node directory {}: {}", node_dir.display(), e))?;

        // Collect and sort by node id so log output is deterministic.
        let mut node_paths: Vec<(u32, std::path::PathBuf)> = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            if !name.starts_with("node") {
                continue;
            }
            match name[4..].parse::<u32>() {
                Ok(id) => node_paths.push((id, path)),
                Err(_) => {
                    tracing::debug!("Skipping non-node directory entry: {}", name);
                }
            }
        }
        node_paths.sort_by_key(|(id, _)| *id);

        for (node_id, path) in node_paths {
            let cpulist_path = path.join("cpulist");
            if !cpulist_path.exists() {
                tracing::debug!(
                    "Node {} has no cpulist file at {}, skipping",
                    node_id,
                    cpulist_path.display()
                );
                continue;
            }

            let raw = match fs::read_to_string(&cpulist_path) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        "Failed to read {} for node {}: {} — skipping node, other nodes still processed",
                        cpulist_path.display(),
                        node_id,
                        e
                    );
                    continue;
                }
            };
            let trimmed = raw.trim();

            let cpus = match parse_cpulist(trimmed) {
                Ok(cpus) => cpus,
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse cpulist for node {} ({}): {} — raw={:?}, skipping node, other nodes still processed",
                        node_id,
                        cpulist_path.display(),
                        e,
                        trimmed
                    );
                    continue;
                }
            };

            if cpus.is_empty() {
                tracing::info!(
                    "NUMA node {} has no CPUs (memory-only node, e.g. GPU/CXL-attached memory); raw cpulist={:?}",
                    node_id,
                    trimmed
                );
                memory_only_nodes.push(node_id);
                node_to_cpus.insert(node_id, Vec::new());
                continue;
            }

            for cpu in &cpus {
                cpu_to_node.insert(*cpu, node_id);
            }
            node_to_cpus.insert(node_id, cpus);
        }

        if node_to_cpus.is_empty() {
            // Richer diagnostic: enumerate what was actually present so a
            // container/sandbox failure is debuggable from the log alone.
            let present: Vec<String> = fs::read_dir(node_dir)
                .ok()
                .map(|rd| {
                    rd.flatten()
                        .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            return Err(format!(
                "No NUMA nodes found under {}; directory entries={:?}",
                node_dir.display(),
                present
            ));
        }

        memory_only_nodes.sort_unstable();

        let total_cpus: usize = node_to_cpus.values().map(|v| v.len()).sum();
        let cpu_bearing: Vec<(u32, usize)> = {
            let mut v: Vec<(u32, usize)> = node_to_cpus
                .iter()
                .filter(|(_, c)| !c.is_empty())
                .map(|(n, c)| (*n, c.len()))
                .collect();
            v.sort_unstable_by_key(|(n, _)| *n);
            v
        };
        tracing::debug!(
            "NUMA topology loaded: {} nodes ({} CPU-bearing, {} memory-only), {} total CPUs; CPU-bearing={:?}, memory-only={:?}",
            node_to_cpus.len(),
            cpu_bearing.len(),
            memory_only_nodes.len(),
            total_cpus,
            cpu_bearing,
            memory_only_nodes
        );

        Ok(Self {
            node_to_cpus,
            cpu_to_node,
            memory_only_nodes,
        })
    }

    /// Returns all CPU IDs belonging to the given NUMA node.
    ///
    /// Returns `None` if the node ID is not in the topology. For memory-only
    /// nodes the returned slice is empty.
    pub fn cpus_for_node(&self, node_id: u32) -> Option<&[usize]> {
        self.node_to_cpus.get(&node_id).map(|v| v.as_slice())
    }

    /// Returns the NUMA node ID that contains the given CPU.
    ///
    /// Returns `None` if the CPU ID is not in the topology.
    pub fn node_for_cpu(&self, cpu_id: usize) -> Option<u32> {
        self.cpu_to_node.get(&cpu_id).copied()
    }

    /// Returns the number of NUMA nodes in the system (including memory-only).
    pub fn num_nodes(&self) -> usize {
        self.node_to_cpus.len()
    }

    /// Returns `true` if this is a single-node (non-NUMA) system.
    pub fn is_single_node(&self) -> bool {
        self.num_nodes() == 1
    }

    /// Sorted list of node IDs that exist but have no CPUs.
    pub fn memory_only_nodes(&self) -> &[u32] {
        &self.memory_only_nodes
    }

    /// Returns a log-friendly summary of what the topology knows:
    /// `"CPU-bearing=[0 (32 cpus), 1 (32 cpus)], memory-only=[2, 3]"`.
    pub fn summary_string(&self) -> String {
        let mut cpu_bearing: Vec<(u32, usize)> = self
            .node_to_cpus
            .iter()
            .filter(|(_, c)| !c.is_empty())
            .map(|(n, c)| (*n, c.len()))
            .collect();
        cpu_bearing.sort_unstable_by_key(|(n, _)| *n);
        let cpu_bearing_str: Vec<String> = cpu_bearing
            .iter()
            .map(|(n, c)| format!("{n} ({c} cpus)"))
            .collect();
        format!(
            "CPU-bearing=[{}], memory-only={:?}",
            cpu_bearing_str.join(", "),
            self.memory_only_nodes
        )
    }
}

/// Parse Linux cpulist format.
///
/// Empty tokens (e.g. from trailing/leading/double commas, or an entirely
/// empty file from a CPU-less memory node) are skipped rather than errored.
///
/// # Examples
/// - `""`           -> `[]` (empty — e.g. memory-only node)
/// - `"0-15"`       -> `[0,1,2,...,15]`
/// - `"0,4,8"`      -> `[0,4,8]`
/// - `"0-3,8-11"`   -> `[0,1,2,3,8,9,10,11]`
/// - `"0-15,"`      -> `[0,..,15]` (trailing comma tolerated)
fn parse_cpulist(cpulist: &str) -> Result<Vec<usize>, String> {
    let mut cpus = Vec::new();

    for raw_part in cpulist.split(',') {
        let part = raw_part.trim();
        if part.is_empty() {
            continue;
        }

        if let Some((a, b)) = part.split_once('-') {
            let a = a.trim();
            let b = b.trim();
            let start: usize = a.parse().map_err(|_| {
                format!(
                    "Invalid CPU ID {:?} in range {:?} (cpulist {:?})",
                    a, part, cpulist
                )
            })?;
            let end: usize = b.parse().map_err(|_| {
                format!(
                    "Invalid CPU ID {:?} in range {:?} (cpulist {:?})",
                    b, part, cpulist
                )
            })?;
            if end < start {
                return Err(format!(
                    "Reversed range {:?} (start {} > end {}) in cpulist {:?}",
                    part, start, end, cpulist
                ));
            }
            for cpu in start..=end {
                cpus.push(cpu);
            }
        } else {
            let cpu: usize = part
                .parse()
                .map_err(|_| format!("Invalid CPU ID {:?} in cpulist {:?}", part, cpulist))?;
            cpus.push(cpu);
        }
    }

    cpus.sort_unstable();
    cpus.dedup();

    Ok(cpus)
}

/// Get the global NUMA topology (cached after first call).
///
/// Returns an error if NUMA topology cannot be read from sysfs. This indicates either:
/// - System doesn't support NUMA
/// - `/sys` is not mounted (e.g., restricted container)
/// - Kernel NUMA support is disabled
///
/// On the first error observation, a one-shot `warn!` dumps live probe data
/// (directory contents and per-node `cpulist` snippets) so the failure is
/// diagnosable from the log alone, without shelling into the host.
///
/// Callers should handle errors gracefully by disabling NUMA optimizations.
pub fn get_numa_topology() -> Result<&'static NumaTopology, &'static str> {
    TOPOLOGY
        .get_or_init(NumaTopology::from_sysfs)
        .as_ref()
        .map_err(|e| {
            if !PROBE_LOGGED.swap(true, Ordering::AcqRel) {
                tracing::warn!(
                    "NUMA topology unavailable: {} — probe follows: {}",
                    e,
                    probe_sysfs_node_dir()
                );
            } else {
                tracing::trace!("NUMA topology unavailable (cached): {}", e);
            }
            "NUMA topology unavailable"
        })
}

/// Live probe of `/sys/devices/system/node` used when topology load fails.
/// Returns a single-line string safe to append to a log message. Each per-node
/// cpulist is truncated so the log line stays bounded.
fn probe_sysfs_node_dir() -> String {
    let node_dir = Path::new("/sys/devices/system/node");
    if !node_dir.exists() {
        return format!("{} does not exist", node_dir.display());
    }
    let rd = match fs::read_dir(node_dir) {
        Ok(rd) => rd,
        Err(e) => return format!("read_dir({}) failed: {}", node_dir.display(), e),
    };

    let mut entries: Vec<String> = Vec::new();
    let mut node_snippets: Vec<(u32, String)> = Vec::new();

    for entry in rd.flatten() {
        let name = match entry.file_name().into_string() {
            Ok(s) => s,
            Err(_) => continue,
        };
        entries.push(name.clone());

        if let Some(num) = name.strip_prefix("node") {
            if let Ok(id) = num.parse::<u32>() {
                let cpulist_path = entry.path().join("cpulist");
                let snippet = match fs::read_to_string(&cpulist_path) {
                    Ok(s) => {
                        let t = s.trim();
                        if t.len() > 128 {
                            format!("{}...", &t[..128])
                        } else {
                            t.to_string()
                        }
                    }
                    Err(e) => format!("<read failed: {e}>"),
                };
                node_snippets.push((id, snippet));
            }
        }
    }
    entries.sort();
    node_snippets.sort_by_key(|(id, _)| *id);

    format!(
        "entries={:?}; per-node cpulist: {:?}",
        entries, node_snippets
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cpulist_range() {
        let cpus = parse_cpulist("0-3").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_parse_cpulist_list() {
        let cpus = parse_cpulist("0,4,8").unwrap();
        assert_eq!(cpus, vec![0, 4, 8]);
    }

    #[test]
    fn test_parse_cpulist_mixed() {
        let cpus = parse_cpulist("0-2,8,16-17").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 8, 16, 17]);
    }

    #[test]
    fn test_parse_cpulist_ht() {
        let cpus = parse_cpulist("0-15,32-47").unwrap();
        assert_eq!(cpus.len(), 32);
        assert_eq!(cpus[0], 0);
        assert_eq!(cpus[15], 15);
        assert_eq!(cpus[16], 32);
        assert_eq!(cpus[31], 47);
    }

    #[test]
    fn test_parse_cpulist_real_numa_system() {
        let cpus = parse_cpulist("0-15,128-143").unwrap();
        assert_eq!(cpus.len(), 32);
        assert_eq!(cpus[0], 0);
        assert_eq!(cpus[15], 15);
        assert_eq!(cpus[16], 128);
        assert_eq!(cpus[31], 143);

        let cpus = parse_cpulist("16-31,144-159").unwrap();
        assert_eq!(cpus.len(), 32);
        assert_eq!(cpus[0], 16);
        assert_eq!(cpus[15], 31);
        assert_eq!(cpus[16], 144);
        assert_eq!(cpus[31], 159);
    }

    #[test]
    fn test_parse_cpulist_out_of_order() {
        let cpus = parse_cpulist("4,2,0,1,3").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_parse_cpulist_duplicates() {
        let cpus = parse_cpulist("0-2,1-3").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_parse_cpulist_empty() {
        // Empty cpulist is now valid (CPU-less memory node case).
        let cpus = parse_cpulist("").unwrap();
        assert!(cpus.is_empty());
    }

    #[test]
    fn test_parse_cpulist_whitespace_only() {
        let cpus = parse_cpulist("   ").unwrap();
        assert!(cpus.is_empty());
    }

    #[test]
    fn test_parse_cpulist_trailing_comma() {
        let cpus = parse_cpulist("0-15,").unwrap();
        assert_eq!(cpus.len(), 16);
    }

    #[test]
    fn test_parse_cpulist_leading_comma() {
        let cpus = parse_cpulist(",0-15").unwrap();
        assert_eq!(cpus.len(), 16);
    }

    #[test]
    fn test_parse_cpulist_double_comma() {
        let cpus = parse_cpulist("0-3,,8-11").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 3, 8, 9, 10, 11]);
    }

    #[test]
    fn test_parse_cpulist_reversed_range_errors() {
        let err = parse_cpulist("5-3").unwrap_err();
        assert!(err.contains("Reversed range"), "got: {err}");
        assert!(err.contains("5-3"), "got: {err}");
    }

    #[test]
    fn test_parse_cpulist_nonnumeric_errors_include_raw() {
        let err = parse_cpulist("abc").unwrap_err();
        assert!(err.contains("abc"), "err should echo bad token: {err}");
        assert!(err.contains("Invalid CPU ID"), "got: {err}");
    }

    #[test]
    fn test_parse_cpulist_single_cpu() {
        let cpus = parse_cpulist("5").unwrap();
        assert_eq!(cpus, vec![5]);
    }

    #[test]
    fn test_topology_bidirectional_lookup() {
        let mut node_to_cpus = std::collections::HashMap::new();
        let mut cpu_to_node = std::collections::HashMap::new();

        node_to_cpus.insert(0, vec![0, 1, 2, 3]);
        node_to_cpus.insert(1, vec![4, 5, 6, 7]);

        for (node, cpus) in &node_to_cpus {
            for cpu in cpus {
                cpu_to_node.insert(*cpu, *node);
            }
        }

        let topology = NumaTopology {
            node_to_cpus,
            cpu_to_node,
            memory_only_nodes: Vec::new(),
        };

        assert_eq!(topology.cpus_for_node(0), Some(&[0, 1, 2, 3][..]));
        assert_eq!(topology.cpus_for_node(1), Some(&[4, 5, 6, 7][..]));

        assert_eq!(topology.node_for_cpu(0), Some(0));
        assert_eq!(topology.node_for_cpu(3), Some(0));
        assert_eq!(topology.node_for_cpu(4), Some(1));
        assert_eq!(topology.node_for_cpu(7), Some(1));

        assert_eq!(topology.node_for_cpu(999), None);
    }

    /// Simulate the sysfs layout this incident exposed: two healthy CPU-bearing
    /// nodes plus one memory-only node with an empty `cpulist`. Prior to the
    /// fix, the empty file poisoned the entire topology; after the fix the two
    /// CPU-bearing nodes load cleanly and the third is recorded as memory-only.
    #[test]
    fn test_from_dir_tolerates_cpuless_memory_node() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        for (node_id, cpulist) in &[(0u32, "0-15"), (1u32, "16-31"), (2u32, "")] {
            let node_dir = root.join(format!("node{node_id}"));
            std::fs::create_dir(&node_dir).unwrap();
            std::fs::write(node_dir.join("cpulist"), cpulist).unwrap();
        }

        let topo = NumaTopology::from_dir(root).expect("topology should load");
        assert_eq!(topo.num_nodes(), 3);
        assert_eq!(topo.cpus_for_node(0).unwrap().len(), 16);
        assert_eq!(topo.cpus_for_node(1).unwrap().len(), 16);
        assert_eq!(topo.cpus_for_node(2), Some(&[][..]));
        assert_eq!(topo.memory_only_nodes(), &[2]);
        assert_eq!(topo.node_for_cpu(0), Some(0));
        assert_eq!(topo.node_for_cpu(16), Some(1));
        assert_eq!(topo.node_for_cpu(9999), None);
    }

    /// A single unparseable node should not poison the rest of the topology.
    #[test]
    fn test_from_dir_isolates_bad_node() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        std::fs::create_dir(root.join("node0")).unwrap();
        std::fs::write(root.join("node0").join("cpulist"), "0-15").unwrap();
        std::fs::create_dir(root.join("node1")).unwrap();
        std::fs::write(root.join("node1").join("cpulist"), "garbage-data").unwrap();

        let topo = NumaTopology::from_dir(root).expect("topology should still load");
        assert_eq!(topo.cpus_for_node(0).unwrap().len(), 16);
        assert_eq!(topo.cpus_for_node(1), None);
    }

    /// Empty directory produces a rich error listing what *was* present.
    #[test]
    fn test_from_dir_empty_errors_with_probe() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let err = NumaTopology::from_dir(tmp.path()).unwrap_err();
        assert!(err.contains("No NUMA nodes found"), "got: {err}");
    }

    #[test]
    fn test_from_dir_missing_errors() {
        let err = NumaTopology::from_dir(Path::new("/nonexistent/path/xyz")).unwrap_err();
        assert!(err.contains("Node directory not found"), "got: {err}");
    }
}
