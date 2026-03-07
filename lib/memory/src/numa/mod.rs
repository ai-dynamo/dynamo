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

use cudarc::driver::sys::CUdevice_attribute_enum;
use nix::libc;
use serde::{Deserialize, Serialize};
use std::{fs, mem, process::Command};

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

/// Format a PCI bus address from domain, bus, and device IDs.
///
/// Returns a string in the format `"DDDD:BB:DD.0"` suitable for sysfs lookups.
fn format_pci_bus_address(domain: i32, bus: i32, device: i32) -> String {
    format!("{:04x}:{:02x}:{:02x}.0", domain, bus, device)
}

/// Query the PCI bus address for a CUDA device from the CUDA driver API.
///
/// Uses `CudaContext::attribute()` to read PCI domain, bus, and device IDs.
/// This transparently handles `CUDA_VISIBLE_DEVICES` remapping since
/// `CudaContext::new(ordinal)` operates on the process-local device index.
fn get_pci_bus_address_from_cuda(device_id: u32) -> Option<String> {
    let ctx = crate::device::cuda_context(device_id).ok()?;
    let domain = ctx
        .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)
        .ok()?;
    let bus = ctx
        .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
        .ok()?;
    let device = ctx
        .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
        .ok()?;
    Some(format_pci_bus_address(domain, bus, device))
}

/// Read the NUMA node for a PCI device from sysfs.
///
/// Reads `/sys/bus/pci/devices/<pci_address>/numa_node`. Returns `None` if the
/// file doesn't exist, can't be read, or contains `-1` (no NUMA affinity).
fn read_numa_node_from_sysfs(pci_address: &str) -> Option<NumaNode> {
    let path = format!("/sys/bus/pci/devices/{}/numa_node", pci_address);
    let content = fs::read_to_string(&path).ok()?;
    let node: i32 = content.trim().parse().ok()?;
    if node < 0 {
        // -1 means no NUMA affinity info available
        None
    } else {
        Some(NumaNode(node as u32))
    }
}

/// Fallback: query NUMA node from nvidia-smi using PCI bus address.
///
/// Uses the PCI BDF address (not env-var-based device index) so it is
/// correct regardless of `CUDA_VISIBLE_DEVICES` remapping.
fn get_numa_node_from_nvidia_smi(pci_address: &str) -> Option<NumaNode> {
    let output = Command::new("nvidia-smi")
        .args(["topo", "--get-numa-id-of-nearby-cpu", "-i", pci_address])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = std::str::from_utf8(&output.stdout).ok()?;
    let line = stdout.lines().next()?;
    let numa_str = line.split(':').nth(1)?;
    let node: u32 = numa_str.trim().parse().ok()?;
    Some(NumaNode(node))
}

/// Get NUMA node for a GPU device.
///
/// Queries the PCI bus address from the CUDA driver API, then reads the NUMA
/// node from sysfs. Falls back to nvidia-smi with the PCI address, then to
/// `NumaNode(0)` as a last resort. Never returns `NumaNode::UNKNOWN`.
///
/// `CUDA_VISIBLE_DEVICES` is handled transparently because `CudaContext::new(ordinal)`
/// operates on the process-local device index.
///
/// # Arguments
/// * `device_id` - CUDA device index (0, 1, 2, ...) as seen by the process
///
/// # Returns
/// The NUMA node closest to the specified GPU, or `NumaNode(0)` as a safe default.
pub fn get_device_numa_node(device_id: u32) -> NumaNode {
    // Step 1: Get PCI bus address from CUDA driver
    let pci_address = match get_pci_bus_address_from_cuda(device_id) {
        Some(addr) => addr,
        None => {
            tracing::warn!(
                "Failed to get PCI address from CUDA for device {}, defaulting to NUMA node 0",
                device_id
            );
            return NumaNode(0);
        }
    };

    // Step 2: Read NUMA node from sysfs
    if let Some(node) = read_numa_node_from_sysfs(&pci_address) {
        tracing::trace!(
            "GPU {} (PCI {}) on NUMA node {} (sysfs)",
            device_id,
            pci_address,
            node.0
        );
        return node;
    }

    // Step 3: Fallback to nvidia-smi with PCI address
    if let Some(node) = get_numa_node_from_nvidia_smi(&pci_address) {
        tracing::trace!(
            "GPU {} (PCI {}) on NUMA node {} (nvidia-smi)",
            device_id,
            pci_address,
            node.0
        );
        return node;
    }

    // Step 4: Last resort — node 0 always exists
    tracing::warn!(
        "Could not determine NUMA node for GPU {} (PCI {}), defaulting to NUMA node 0",
        device_id,
        pci_address
    );
    NumaNode(0)
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
        let node = NumaNode(1);
        let json = serde_json::to_string(&node).unwrap();
        let deserialized: NumaNode = serde_json::from_str(&json).unwrap();
        assert_eq!(node, deserialized);
    }

    #[test]
    fn test_get_current_cpu_numa_node() {
        let node = get_current_cpu_numa_node();
        if !node.is_unknown() {
            assert!(node.0 < 8, "NUMA node {} seems unreasonably high", node.0);
        }
    }

    #[test]
    fn test_numa_node_hash() {
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
        let node1 = NumaNode(5);
        let node2 = node1;
        let node3 = node1;

        assert_eq!(node1, node2);
        assert_eq!(node1, node3);
        assert_eq!(node2, node3);
    }

    #[test]
    fn test_format_pci_bus_address() {
        assert_eq!(format_pci_bus_address(0, 0, 0), "0000:00:00.0");
        assert_eq!(format_pci_bus_address(0, 0x3b, 0), "0000:3b:00.0");
        assert_eq!(format_pci_bus_address(0, 0xaf, 0), "0000:af:00.0");
        assert_eq!(format_pci_bus_address(0x10, 0x1a, 0x03), "0010:1a:03.0");
    }

    #[test]
    fn test_read_numa_node_from_sysfs_nonexistent() {
        assert!(read_numa_node_from_sysfs("ffff:ff:ff.0").is_none());
    }
}

#[cfg(all(test, feature = "testing-cuda"))]
mod cuda_tests {
    use super::*;

    #[test]
    fn test_get_pci_bus_address_from_cuda() {
        let addr = get_pci_bus_address_from_cuda(0).expect("should get PCI address for GPU 0");
        // Validate BDF format: DDDD:BB:DD.0
        let parts: Vec<&str> = addr.split(':').collect();
        assert_eq!(
            parts.len(),
            3,
            "PCI address should have 3 colon-separated parts: {}",
            addr
        );
        assert_eq!(parts[0].len(), 4, "domain should be 4 hex chars: {}", addr);
        assert!(parts[2].ends_with(".0"), "should end with .0: {}", addr);
        println!("GPU 0 PCI address: {}", addr);
    }

    #[test]
    fn test_read_numa_node_from_sysfs_real_gpu() {
        let addr = get_pci_bus_address_from_cuda(0).expect("should get PCI address for GPU 0");
        if let Some(node) = read_numa_node_from_sysfs(&addr) {
            assert!(node.0 < 16, "NUMA node {} seems unreasonably high", node.0);
            println!("GPU 0 (PCI {}) sysfs NUMA node: {}", addr, node.0);
        } else {
            println!(
                "GPU 0 (PCI {}) has no sysfs NUMA info (single-socket?)",
                addr
            );
        }
    }

    #[test]
    fn test_get_device_numa_node_never_returns_unknown() {
        let node = get_device_numa_node(0);
        assert!(
            !node.is_unknown(),
            "get_device_numa_node should never return UNKNOWN"
        );
    }

    #[test]
    fn test_get_device_numa_node_returns_valid_node() {
        let node = get_device_numa_node(0);
        assert!(node.0 < 16, "NUMA node {} seems unreasonably high", node.0);
        println!("GPU 0 detected on NUMA node: {}", node.0);
    }
}
