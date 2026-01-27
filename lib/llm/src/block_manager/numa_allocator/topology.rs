// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NUMA topology detection
//!
//! This module provides utilities to read the actual CPU-to-NUMA mapping from the system,
//! replacing heuristic assumptions with real topology data.

use std::collections::HashMap;
use std::fs;

/// Global cached topology
static TOPOLOGY: std::sync::OnceLock<Result<NumaTopology, String>> = std::sync::OnceLock::new();

/// Represents the CPU topology for NUMA nodes
pub struct NumaTopology {
    /// Maps NUMA node ID -> list of CPU IDs
    node_to_cpus: HashMap<u32, Vec<usize>>,
    /// Maps CPU ID -> NUMA node ID
    cpu_to_node: HashMap<usize, u32>,
}

impl NumaTopology {
    /// Read NUMA topology from sysfs
    pub fn from_sysfs() -> Result<Self, String> {
        let mut node_to_cpus: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut cpu_to_node: HashMap<usize, u32> = HashMap::new();

        // TODO: Read /sys/devices/system/node directory
        let node_dir = std::path::Path::new("/sys/devices/system/node");
        if !node_dir.exists() {
            return Err("Node directory not found".to_string());
        }
        let entries =
            fs::read_dir(node_dir).map_err(|e| format!("Failed to read node directory: {}", e))?;

        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            // Only process "nodeN" directories
            if !name.starts_with("node") {
                continue;
            }
            // Extract node number
            let node_id: u32 = name[4..]
                .parse()
                .map_err(|_| format!("Invalid node dir: {}", name))?;

            // Read cpulist file
            let cpulist_path = path.join("cpulist");
            if !cpulist_path.exists() {
                continue;
            }

            let cpulist = fs::read_to_string(&cpulist_path)
                .map_err(|e| format!("Failed to read {}: {}", cpulist_path.display(), e))?;

            let cpus = parse_cpulist(cpulist.trim())?;

            // Populate both maps
            for cpu in &cpus {
                cpu_to_node.insert(*cpu, node_id);
            }
            node_to_cpus.insert(node_id, cpus);
        }

        if node_to_cpus.is_empty() {
            return Err("No NUMA nodes found".to_string());
        }

        Ok(Self {
            node_to_cpus,
            cpu_to_node,
        })
    }

    /// Get all CPUs for a NUMA node
    pub fn cpus_for_node(&self, node_id: u32) -> Option<&[usize]> {
        self.node_to_cpus.get(&node_id).map(|v| v.as_slice())
    }

    /// Get NUMA node for a CPU
    pub fn node_for_cpu(&self, cpu_id: usize) -> Option<u32> {
        self.cpu_to_node.get(&cpu_id).copied()
    }

    /// Get number of NUMA nodes
    pub fn num_nodes(&self) -> usize {
        self.node_to_cpus.len()
    }

    /// Check if single-node system
    pub fn is_single_node(&self) -> bool {
        self.num_nodes() == 1
    }
}

/// Parse Linux cpulist format
/// Examples:
///   "0-15"        -> [0,1,2,...,15]
///   "0,4,8"       -> [0,4,8]
///   "0-3,8-11"    -> [0,1,2,3,8,9,10,11]
fn parse_cpulist(cpulist: &str) -> Result<Vec<usize>, String> {
    let mut cpus = Vec::new();

    for part in cpulist.split(',') {
        if part.contains('-') {
            // Range: "0-15"
            let range: Vec<&str> = part.split('-').collect();
            if range.len() != 2 {
                return Err(format!("Invalid range: {}", part));
            }

            let start: usize = range[0]
                .parse()
                .map_err(|_| format!("Invalid CPU ID: {}", range[0]))?;
            let end: usize = range[1]
                .parse()
                .map_err(|_| format!("Invalid CPU ID: {}", range[1]))?;

            for cpu in start..=end {
                cpus.push(cpu);
            }
        } else {
            // Single CPU
            let cpu: usize = part
                .parse()
                .map_err(|_| format!("Invalid CPU ID: {}", part))?;
            cpus.push(cpu);
        }
    }

    cpus.sort_unstable();
    cpus.dedup();

    Ok(cpus)
}

/// Get the global NUMA topology (cached after first call)
///
/// Returns an error if NUMA topology cannot be read from sysfs. This indicates either:
/// - System doesn't support NUMA
/// - `/sys` is not mounted (e.g., restricted container)
/// - Kernel NUMA support is disabled
///
/// Callers should handle errors gracefully by disabling NUMA optimizations.
pub fn get_numa_topology() -> Result<&'static NumaTopology, &'static str> {
    TOPOLOGY
        .get_or_init(NumaTopology::from_sysfs)
        .as_ref()
        .map_err(|e| {
            tracing::warn!("NUMA topology unavailable: {}", e);
            "NUMA topology unavailable"
        })
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
        // Hyperthreading: 0-15,32-47 (physical cores 0-15, HT siblings 32-47)
        let cpus = parse_cpulist("0-15,32-47").unwrap();
        assert_eq!(cpus.len(), 32);
        assert_eq!(cpus[0], 0);
        assert_eq!(cpus[15], 15);
        assert_eq!(cpus[16], 32);
        assert_eq!(cpus[31], 47);
    }

    #[test]
    fn test_parse_cpulist_real_numa_system() {
        // Real dual-socket system with hyperthreading (discovered pattern)
        // Node 0: CPUs 0-15, 128-143
        let cpus = parse_cpulist("0-15,128-143").unwrap();
        assert_eq!(cpus.len(), 32);
        assert_eq!(cpus[0], 0);
        assert_eq!(cpus[15], 15);
        assert_eq!(cpus[16], 128);
        assert_eq!(cpus[31], 143);

        // Node 1: CPUs 16-31, 144-159
        let cpus = parse_cpulist("16-31,144-159").unwrap();
        assert_eq!(cpus.len(), 32);
        assert_eq!(cpus[0], 16);
        assert_eq!(cpus[15], 31);
        assert_eq!(cpus[16], 144);
        assert_eq!(cpus[31], 159);
    }

    #[test]
    fn test_parse_cpulist_out_of_order() {
        // Test that parser handles out-of-order input (seen in some systems)
        let cpus = parse_cpulist("4,2,0,1,3").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 3, 4]); // Should be sorted
    }

    #[test]
    fn test_parse_cpulist_duplicates() {
        // Test deduplication (in case kernel reports duplicates)
        let cpus = parse_cpulist("0-2,1-3").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 3]); // Should remove duplicates
    }

    #[test]
    fn test_parse_cpulist_empty() {
        // Edge case: empty cpulist
        let result = parse_cpulist("");
        assert!(result.is_err() || result.unwrap().is_empty());
    }

    #[test]
    fn test_parse_cpulist_single_cpu() {
        // Single CPU node (uncommon but valid)
        let cpus = parse_cpulist("5").unwrap();
        assert_eq!(cpus, vec![5]);
    }

    #[test]
    fn test_topology_bidirectional_lookup() {
        // Test that node->cpu and cpu->node mappings are consistent
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
        };

        // Verify forward lookup (node -> cpus)
        assert_eq!(topology.cpus_for_node(0), Some(&[0, 1, 2, 3][..]));
        assert_eq!(topology.cpus_for_node(1), Some(&[4, 5, 6, 7][..]));

        // Verify reverse lookup (cpu -> node)
        assert_eq!(topology.node_for_cpu(0), Some(0));
        assert_eq!(topology.node_for_cpu(3), Some(0));
        assert_eq!(topology.node_for_cpu(4), Some(1));
        assert_eq!(topology.node_for_cpu(7), Some(1));

        // Verify unknown CPU
        assert_eq!(topology.node_for_cpu(999), None);
    }
}
