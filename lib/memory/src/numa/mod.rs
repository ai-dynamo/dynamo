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
//! NUMA optimization is enabled by default. To disable it:
//! ```bash
//! export DYN_MEMORY_DISABLE_NUMA=1
//! ```
//!
//! When enabled, pinned memory allocations are routed through NUMA workers
//! that are pinned to the target GPU's NUMA node, ensuring first-touch policy
//! places pages on the correct node. If the GPU's NUMA node cannot be
//! determined, allocation falls back to the non-NUMA path transparently.

pub(crate) mod nvml;
pub mod topology;
pub mod worker_pool;

use cudarc::driver::{result::device as cuda_device, sys as cuda_sys};
use nix::libc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::{fs, mem, process::Command};

/// Cache for GPU PCI address → NUMA node lookups.
/// The mapping never changes at runtime, so we cache results (including negative
/// lookups) to avoid repeated sysfs reads and nvidia-smi subprocesses.
static NUMA_NODE_CACHE: OnceLock<Mutex<HashMap<String, Option<NumaNode>>>> = OnceLock::new();

/// Check if NUMA optimization is disabled via environment variable.
///
/// NUMA-aware allocation is enabled by default. Set `DYN_MEMORY_DISABLE_NUMA=1`
/// (or any truthy value) to disable it.
pub fn is_numa_enabled() -> bool {
    !crate::env_is_truthy("DYN_MEMORY_DISABLE_NUMA")
}

/// Convenience inverse of [`is_numa_enabled`].
pub fn is_numa_disabled() -> bool {
    !is_numa_enabled()
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
/// node from sysfs. Falls back to nvidia-smi with the PCI address. Returns
/// `None` if the NUMA node cannot be determined, signaling the caller to skip
/// NUMA-aware allocation entirely rather than guessing wrong.
///
/// `CUDA_VISIBLE_DEVICES` is handled transparently because `CudaContext::new(ordinal)`
/// operates on the process-local device index.
///
/// # Arguments
/// * `device_id` - CUDA device index (0, 1, 2, ...) as seen by the process
///
/// # Returns
/// The NUMA node closest to the specified GPU, or `None` if it cannot be determined.
pub fn get_device_numa_node(device_id: u32) -> Option<NumaNode> {
    // Step 1: Get PCI bus address from CUDA driver
    let pci_address = match get_pci_bus_address_from_cuda(device_id) {
        Some(addr) => addr,
        None => {
            tracing::warn!(
                "Failed to get PCI address from CUDA for device {}, skipping NUMA optimization",
                device_id
            );
            return None;
        }
    };

    // Step 2: Check cache (includes negative lookups)
    let cache = NUMA_NODE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock().unwrap();
        if let Some(cached) = guard.get(&pci_address) {
            return *cached;
        }
    }

    // Step 3: Read NUMA node from sysfs
    let result = read_numa_node_from_sysfs(&pci_address)
        .or_else(|| get_numa_node_from_nvidia_smi(&pci_address));

    match result {
        Some(node) => {
            tracing::trace!(
                "GPU {} (PCI {}) on NUMA node {}",
                device_id,
                pci_address,
                node.0
            );
        }
        None => {
            tracing::warn!(
                "Could not determine NUMA node for GPU {} (PCI {}), skipping NUMA optimization",
                device_id,
                pci_address
            );
        }
    }

    // Cache result (including None for negative lookups)
    cache.lock().unwrap().insert(pci_address, result);
    result
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

/// Get PCI bus address for a CUDA device via the CUDA driver API.
///
/// Returns a normalized PCI address string like "0000:3b:00.0".
/// The device_id here is a CUDA ordinal (affected by CUDA_VISIBLE_DEVICES).
fn get_pci_bus_address_from_cuda(device_id: u32) -> Option<String> {
    // SAFETY: We're calling CUDA driver API functions with valid device ordinals.
    // cuDeviceGet and get_attribute are safe as long as CUDA is initialized
    // (which CudaContext::new handles).
    unsafe {
        let mut dev = std::mem::MaybeUninit::uninit();
        if cuda_sys::cuDeviceGet(dev.as_mut_ptr(), device_id as i32)
            .result()
            .is_err()
        {
            return None;
        }
        let dev = dev.assume_init();

        let domain = cuda_device::get_attribute(
            dev,
            cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
        )
        .ok()?;
        let bus = cuda_device::get_attribute(
            dev,
            cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
        )
        .ok()?;
        let device = cuda_device::get_attribute(
            dev,
            cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
        )
        .ok()?;

        Some(format!("{:04x}:{:02x}:{:02x}.0", domain, bus, device))
    }
}

/// GPU info with PCI address and NUMA node, used for CPU set subdivision.
#[derive(Debug, Clone)]
struct GpuTopoInfo {
    pci_address: String,
    numa_node: Option<u32>,
}

/// Enumerate all GPUs visible to CUDA with their PCI addresses and NUMA nodes.
fn enumerate_cuda_gpus() -> Vec<GpuTopoInfo> {
    let count = match cuda_device::get_count() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    (0..count as u32)
        .filter_map(|i| {
            let pci = get_pci_bus_address_from_cuda(i)?;
            let numa = read_numa_node_from_sysfs(&pci).map(|n| n.0);
            Some(GpuTopoInfo {
                pci_address: pci,
                numa_node: numa,
            })
        })
        .collect()
}

/// Enumerate all GPUs on the system, preferring NVML (sees all GPUs)
/// over CUDA driver (only sees CUDA_VISIBLE_DEVICES).
fn enumerate_all_gpus() -> Vec<GpuTopoInfo> {
    // Try NVML first — it sees all GPUs regardless of CUDA_VISIBLE_DEVICES
    if let Some(nvml) = nvml::try_nvml() {
        let nvml_gpus = nvml.enumerate_gpus();
        if !nvml_gpus.is_empty() {
            tracing::debug!(
                "NVML enumerated {} GPUs (ignoring CUDA_VISIBLE_DEVICES)",
                nvml_gpus.len()
            );
            return nvml_gpus
                .into_iter()
                .map(|g| {
                    let numa = read_numa_node_from_sysfs(&g.pci_address).map(|n| n.0);
                    GpuTopoInfo {
                        pci_address: g.pci_address,
                        numa_node: numa,
                    }
                })
                .collect();
        }
    }

    // Fallback: enumerate via CUDA driver (may miss hidden devices)
    tracing::debug!("Falling back to CUDA driver GPU enumeration");
    enumerate_cuda_gpus()
}

/// Cached CPU set results per CUDA device ordinal.
static DEVICE_CPU_SETS: OnceLock<HashMap<u32, Option<Vec<usize>>>> = OnceLock::new();

/// Get a deterministic CPU subset for a CUDA device, subdivided among ALL GPUs
/// sharing the same NUMA node (including those hidden by CUDA_VISIBLE_DEVICES).
///
/// # Algorithm
/// 1. Get PCI address + NUMA node for target device (CUDA driver API)
/// 2. Enumerate ALL GPUs on the system:
///    - Try NVML first (sees all GPUs, ignores CUDA_VISIBLE_DEVICES)
///    - Fall back to CUDA driver API (only sees visible devices)
/// 3. For each GPU, get its NUMA node via sysfs (PCI address → /sys/.../numa_node)
/// 4. Group GPUs by NUMA node
/// 5. Sort by PCI address within each group (deterministic)
/// 6. Get full CPU set for the node via topology
/// 7. Divide into N equal slices (N = GPUs on same node)
/// 8. Return the slice for the target device's position
///
/// # Example
/// System: 8 GPUs, 2 NUMA nodes, 4 GPUs per node.
/// CUDA_VISIBLE_DEVICES=0,1 (only 2 visible).
/// NVML sees all 8 → correctly subdivides into 4 slices per node.
///
/// Returns None if NUMA node can't be determined.
pub fn get_device_cpu_set(device_id: u32) -> Option<Vec<usize>> {
    DEVICE_CPU_SETS
        .get_or_init(compute_all_device_cpu_sets)
        .get(&device_id)
        .cloned()
        .flatten()
}

fn compute_all_device_cpu_sets() -> HashMap<u32, Option<Vec<usize>>> {
    let topology = match topology::get_numa_topology() {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("Cannot subdivide CPU sets: {e}");
            return HashMap::new();
        }
    };

    // Get the target device's PCI address and NUMA node
    let cuda_count = cuda_device::get_count().unwrap_or(0);
    if cuda_count == 0 {
        return HashMap::new();
    }

    // Build info for each visible CUDA device
    let mut cuda_devices: Vec<(u32, String, Option<u32>)> = Vec::new();
    for i in 0..cuda_count as u32 {
        if let Some(pci) = get_pci_bus_address_from_cuda(i) {
            let numa = read_numa_node_from_sysfs(&pci).map(|n| n.0);
            cuda_devices.push((i, pci, numa));
        }
    }

    // Enumerate ALL GPUs on the system (NVML preferred)
    let all_gpus = enumerate_all_gpus();

    // Group all GPUs by NUMA node
    let mut node_groups: HashMap<u32, Vec<String>> = HashMap::new();
    for gpu in &all_gpus {
        if let Some(node) = gpu.numa_node {
            node_groups
                .entry(node)
                .or_default()
                .push(gpu.pci_address.clone());
        }
    }

    // Sort each group by PCI address for deterministic ordering
    for group in node_groups.values_mut() {
        group.sort();
    }

    // For each CUDA device, find its position in its NUMA group and subdivide
    let mut results = HashMap::new();
    for (device_id, pci_addr, numa_node) in &cuda_devices {
        let cpu_set = numa_node.and_then(|node| {
            let group = node_groups.get(&node)?;
            let position = group.iter().position(|addr| addr == pci_addr)?;
            let all_cpus = topology.cpus_for_node(node)?;

            if all_cpus.is_empty() || group.is_empty() {
                return None;
            }

            // Divide CPUs into N equal slices
            let n = group.len();
            let chunk_size = all_cpus.len() / n;
            if chunk_size == 0 {
                // More GPUs than CPUs on this node — give all CPUs to everyone
                return Some(all_cpus.to_vec());
            }

            let start = position * chunk_size;
            let end = if position == n - 1 {
                all_cpus.len() // last slice gets remainder
            } else {
                start + chunk_size
            };

            Some(all_cpus[start..end].to_vec())
        });

        results.insert(*device_id, cpu_set);
    }

    results
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
    fn test_get_device_numa_node_returns_some_or_none() {
        let result = get_device_numa_node(0);
        match result {
            Some(node) => {
                assert!(node.0 < 16, "NUMA node {} seems unreasonably high", node.0);
                assert!(
                    !node.is_unknown(),
                    "should never return UNKNOWN inside Some"
                );
                println!("GPU 0 detected on NUMA node: {}", node.0);
            }
            None => {
                println!("GPU 0 has no determinable NUMA node (single-socket or no sysfs info)");
            }
        }
    }
}
