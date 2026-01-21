// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NUMA-aware memory allocation utilities.
//!
//! This module provides utilities for NUMA-aware memory allocation, which is critical
//! for optimal performance on multi-socket systems with GPUs. Memory allocated on the
//! NUMA node closest to the target GPU has significantly lower access latency.
//!
//! ## Architecture
//!
//! - [`NumaNode`]: Represents a NUMA node ID
//! - [`topology`]: Reads CPU-to-NUMA mapping from `/sys/devices/system/node`
//! - [`worker_pool`]: Dedicated worker threads pinned to specific NUMA nodes
//!
//! ## Usage
//!
//! NUMA optimization is opt-in via environment variable:
//! ```bash
//! export DYN_KVBM_ENABLE_NUMA=1
//! ```
//!
//! When enabled, pinned memory allocations are routed through NUMA workers
//! that are pinned to the target GPU's NUMA node, ensuring first-touch policy
//! places pages on the correct node.

pub mod topology;
pub mod worker_pool;

use nix::libc;
use serde::{Deserialize, Serialize};
use std::{mem, process::Command};

/// Check if NUMA optimization is enabled via environment variable
///
/// Set `DYN_KVBM_ENABLE_NUMA=1` to enable NUMA-aware allocation.
/// Default: disabled (opt-in)
pub fn is_numa_enabled() -> bool {
    std::env::var("DYN_KVBM_ENABLE_NUMA")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

/// Represents a NUMA node identifier.
///
/// NUMA nodes are typically numbered 0, 1, 2, etc. corresponding to physical
/// CPU sockets. Use [`NumaNode::UNKNOWN`] when the node cannot be determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NumaNode(pub u32);

impl NumaNode {
    /// Sentinel value for unknown NUMA node.
    pub const UNKNOWN: NumaNode = NumaNode(u32::MAX);

    /// Returns true if this represents an unknown NUMA node.
    pub fn is_unknown(&self) -> bool {
        self.0 == u32::MAX
    }
}

impl std::fmt::Display for NumaNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_unknown() {
            write!(f, "UNKNOWN")
        } else {
            write!(f, "NumaNode({})", self.0)
        }
    }
}

/// Get the current CPU's NUMA node.
///
/// Uses the Linux `getcpu` syscall to determine which NUMA node the current CPU belongs to.
/// Returns [`NumaNode::UNKNOWN`] if the syscall fails.
pub fn get_current_cpu_numa_node() -> NumaNode {
    unsafe {
        let mut cpu: libc::c_uint = 0;
        let mut node: libc::c_uint = 0;

        // getcpu syscall: int getcpu(unsigned *cpu, unsigned *node, struct getcpu_cache *tcache);
        let result = libc::syscall(
            libc::SYS_getcpu,
            &mut cpu,
            &mut node,
            std::ptr::null_mut::<libc::c_void>(),
        );
        if result == 0 {
            NumaNode(node)
        } else {
            NumaNode::UNKNOWN
        }
    }
}

/// Get NUMA node for a GPU device.
///
/// For GPU memory, the NUMA affinity depends on which PCIe bus the GPU is attached to.
/// This is queried via nvidia-smi. Falls back to a heuristic (device_id % 2) if nvidia-smi
/// is unavailable.
///
/// # Arguments
/// * `device_id` - CUDA device index (0, 1, 2, ...)
///
/// # Returns
/// The NUMA node closest to the specified GPU, or a heuristic fallback.
pub fn get_device_numa_node(device_id: u32) -> NumaNode {
    // Use nvidia-smi topo to get NUMA ID of nearest CPU
    // This directly returns the NUMA node
    let output = match Command::new("nvidia-smi")
        .args([
            "topo",
            "--get-numa-id-of-nearby-cpu",
            "-i",
            &device_id.to_string(),
        ])
        .output()
    {
        Ok(out) if out.status.success() => out,
        _ => {
            tracing::warn!("nvidia-smi failed for GPU {}, using heuristic", device_id);
            return NumaNode(device_id % 2);
        }
    };

    if let Ok(stdout) = std::str::from_utf8(&output.stdout)
        && let Some(line) = stdout.lines().next()
        && let Some(numa_str) = line.split(':').nth(1)
        && let Ok(node) = numa_str.trim().parse::<u32>()
    {
        tracing::trace!("GPU {} on NUMA node {}", device_id, node);
        return NumaNode(node);
    }
    tracing::warn!("Failed to get NUMA node for GPU {}", device_id);
    NumaNode::UNKNOWN
}

/// Pin the current thread to a specific NUMA node's CPUs.
///
/// This sets the CPU affinity for the calling thread to only run on CPUs
/// belonging to the specified NUMA node. This is critical for ensuring
/// that memory allocations follow the first-touch policy on the correct node.
///
/// # Arguments
/// * `node` - The NUMA node to pin the thread to
///
/// # Errors
/// Returns an error if:
/// - NUMA topology cannot be read
/// - No CPUs are found for the specified node
/// - The `sched_setaffinity` syscall fails
pub fn pin_thread_to_numa_node(node: NumaNode) -> Result<(), String> {
    let topology =
        topology::get_numa_topology().map_err(|e| format!("Can not get NUMA topology: {}", e))?;

    let cpus = topology
        .cpus_for_node(node.0)
        .ok_or_else(|| format!("No CPUs found for NUMA node {}", node.0))?;

    if cpus.is_empty() {
        return Err(format!("No CPUs found for NUMA node {}", node.0));
    }

    unsafe {
        let mut cpu_set: libc::cpu_set_t = mem::zeroed();

        for cpu in cpus {
            libc::CPU_SET(*cpu, &mut cpu_set);
        }

        let result = libc::sched_setaffinity(
            0, // current thread
            mem::size_of::<libc::cpu_set_t>(),
            &cpu_set,
        );

        if result != 0 {
            let err = std::io::Error::last_os_error();
            return Err(format!("Failed to set CPU affinity: {}", err));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_node_equality() {
        let node0a = NumaNode(0);
        let node0b = NumaNode(0);
        let node1 = NumaNode(1);

        assert_eq!(node0a, node0b);
        assert_ne!(node0a, node1);
    }

    #[test]
    fn test_numa_node_unknown() {
        let unknown = NumaNode::UNKNOWN;
        assert!(unknown.is_unknown());
        assert_eq!(unknown.0, u32::MAX);

        let valid = NumaNode(0);
        assert!(!valid.is_unknown());
    }

    #[test]
    fn test_numa_node_display() {
        assert_eq!(format!("{}", NumaNode(0)), "NumaNode(0)");
        assert_eq!(format!("{}", NumaNode(7)), "NumaNode(7)");
        assert_eq!(format!("{}", NumaNode::UNKNOWN), "UNKNOWN");
    }

    #[test]
    fn test_numa_node_serialization() {
        // Verify NumaNode can be serialized (important for benchmarking)
        let node = NumaNode(1);
        let json = serde_json::to_string(&node).unwrap();
        let deserialized: NumaNode = serde_json::from_str(&json).unwrap();
        assert_eq!(node, deserialized);
    }

    #[test]
    fn test_get_current_cpu_numa_node() {
        // Should either return a valid node or UNKNOWN
        let node = get_current_cpu_numa_node();

        // If not unknown, should be a reasonable NUMA node number (< 8 on most systems)
        if !node.is_unknown() {
            assert!(node.0 < 8, "NUMA node {} seems unreasonably high", node.0);
        }
    }

    #[test]
    fn test_get_device_numa_node_valid_gpu() {
        // Test GPU 0 detection
        let node = get_device_numa_node(0);

        // Should return either a valid node (0-7) or use heuristic (gpu_id % 2)
        // On dual-socket systems, GPU 0 typically on node 0 or 1
        println!("GPU 0 detected on NUMA node: {}", node.0);
    }

    #[test]
    fn test_numa_node_hash() {
        // Verify NumaNode can be used as a HashMap key
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(NumaNode(0), "node0");
        map.insert(NumaNode(1), "node1");

        assert_eq!(map.get(&NumaNode(0)), Some(&"node0"));
        assert_eq!(map.get(&NumaNode(1)), Some(&"node1"));
        assert_eq!(map.get(&NumaNode(2)), None);
    }

    #[test]
    fn test_numa_node_copy_clone() {
        // Verify NumaNode is Copy and Clone
        let node1 = NumaNode(5);
        let node2 = node1; // Copy
        let node3 = node1; // Clone

        assert_eq!(node1, node2);
        assert_eq!(node1, node3);
        assert_eq!(node2, node3);
    }
}
