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

use cudarc::driver::{CudaContext, result::device as cuda_device, sys as cuda_sys};
use kvbm_common::DeviceBackend;
use nix::libc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Get or create a CUDA context for NUMA-aware operations.
pub(crate) fn cuda_context(device_id: u32) -> crate::Result<Arc<CudaContext>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<u32, Arc<CudaContext>>>> = OnceLock::new();
    let mut map = CONTEXTS.get_or_init(Default::default).lock().unwrap();
    if let Some(existing) = map.get(&device_id) {
        return Ok(existing.clone());
    }
    let ctx = CudaContext::new(device_id as usize).map_err(|e| {
        crate::StorageError::AllocationFailed(format!("CUDA context creation failed: {e}"))
    })?;
    map.insert(device_id, ctx.clone());
    Ok(ctx)
}
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

/// Fallback: query NUMA node from xpu-smi using PCI bus address.
///
/// Uses `xpu-smi discovery` to enumerate devices and match by PCI BDF,
/// then reads the NUMA affinity from device properties.
///
/// Falls back gracefully if xpu-smi is not installed.
fn get_numa_node_from_xpu_smi(pci_address: &str) -> Option<NumaNode> {
    // xpu-smi topology -d <device_id> shows NUMA affinity, but we need
    // to first map PCI address → device ID. Use `xpu-smi discovery` which
    // lists all devices with their PCI BDF.
    //
    // `xpu-smi discovery` output format (example):
    //   +-----------+----------------------+
    //   | Device ID | Device Information   |
    //   +-----------+----------------------+
    //   | 0         | ...                  |
    //   |           | PCI BDF Address: 0000:04:00.0 |
    //   +-----------+----------------------+
    //
    // We parse this to find the device ID matching the PCI address, then
    // query its NUMA node via `xpu-smi topology -d <id>`.

    let output = Command::new("xpu-smi")
        .args(["discovery"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = std::str::from_utf8(&output.stdout).ok()?;
    let pci_upper = pci_address.to_lowercase();

    // Find device ID whose PCI BDF matches
    let mut current_device_id: Option<&str> = None;
    for line in stdout.lines() {
        let trimmed = line.trim();
        // Lines like "| 0         |" or "| 1         |" indicate device IDs
        if trimmed.starts_with('|') && trimmed.ends_with('|') {
            let parts: Vec<&str> = trimmed.split('|').collect();
            if parts.len() >= 3 {
                let maybe_id = parts[1].trim();
                if maybe_id.parse::<u32>().is_ok() {
                    current_device_id = Some(maybe_id);
                }
            }
        }
        // Look for PCI BDF line
        if let Some(_dev_id) = current_device_id {
            if trimmed.to_lowercase().contains(&pci_upper) {
                // Found the device. Now query its NUMA node.
                let topo_output = Command::new("xpu-smi")
                    .args(["topology", "-d", _dev_id])
                    .output()
                    .ok()?;

                if !topo_output.status.success() {
                    return None;
                }

                let topo_stdout = std::str::from_utf8(&topo_output.stdout).ok()?;
                // Look for "CPU Affinity" or "NUMA" in topology output
                for topo_line in topo_stdout.lines() {
                    let tl = topo_line.to_lowercase();
                    if tl.contains("numa") || tl.contains("cpu affinity") {
                        // Try to extract a number from this line
                        for word in topo_line.split_whitespace() {
                            if let Ok(n) = word.trim_matches(|c: char| !c.is_ascii_digit()).parse::<u32>() {
                                return Some(NumaNode(n));
                            }
                        }
                    }
                }
                return None;
            }
        }
    }

    None
}

/// Get NUMA node for a device given its PCI BDF address (backend-agnostic).
///
/// Reads `/sys/bus/pci/devices/<pci_address>/numa_node`. Falls back to
/// nvidia-smi for NVIDIA GPUs. Returns `None` if the NUMA node cannot be
/// determined.
///
/// This is the preferred entry point for callers that already have a PCI
/// address (e.g. via `DeviceContext::pci_bdf_address()`).
///
/// # Arguments
/// * `pci_address` - PCI BDF address string, e.g. "0000:04:00.0"
///
/// # Returns
/// The NUMA node closest to the device, or `None` if it cannot be determined.
pub fn get_numa_node_for_pci_address(pci_address: &str) -> Option<NumaNode> {
    // Check cache
    let cache = NUMA_NODE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock().unwrap();
        if let Some(cached) = guard.get(pci_address) {
            return *cached;
        }
    }

    // Read from sysfs (backend-agnostic: works for any PCI device).
    // If sysfs has no NUMA info (e.g. -1 or missing), try vendor management
    // tools as fallback: nvidia-smi for NVIDIA GPUs, xpu-smi for Intel XPUs.
    let result = read_numa_node_from_sysfs(pci_address)
        .or_else(|| get_numa_node_from_nvidia_smi(pci_address))
        .or_else(|| get_numa_node_from_xpu_smi(pci_address));

    match result {
        Some(node) => {
            tracing::trace!("PCI {} on NUMA node {}", pci_address, node.0);
        }
        None => {
            tracing::debug!(
                "Could not determine NUMA node for PCI {}, skipping NUMA optimization",
                pci_address
            );
        }
    }

    // Cache (including None for negative lookups)
    cache
        .lock()
        .unwrap()
        .insert(pci_address.to_string(), result);
    result
}

/// Get NUMA node for a GPU device (CUDA-specific legacy API).
///
/// Queries the PCI bus address from the CUDA driver API, then reads the NUMA
/// node from sysfs. Falls back to nvidia-smi with the PCI address. Returns
/// `None` if the NUMA node cannot be determined, signaling the caller to skip
/// NUMA-aware allocation entirely rather than guessing wrong.
///
/// `CUDA_VISIBLE_DEVICES` is handled transparently because `CudaContext::new(ordinal)`
/// operates on the process-local device index.
///
/// For new code that works across backends, prefer
/// [`get_numa_node_for_pci_address`] with the PCI address from
/// `DeviceContext::pci_bdf_address()`.
///
/// # Arguments
/// * `device_id` - CUDA device index (0, 1, 2, ...) as seen by the process
///
/// # Returns
/// The NUMA node closest to the specified GPU, or `None` if it cannot be determined.
pub fn get_device_numa_node(device_id: u32) -> Option<NumaNode> {
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
    // Vendor-smi fallbacks are handled inside get_numa_node_for_pci_address
    get_numa_node_for_pci_address(&pci_address)
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
pub(crate) fn get_pci_bus_address_from_cuda(device_id: u32) -> Option<String> {
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

// ---------------------------------------------------------------------------
// Backend GPU enumerator trait + implementations
// ---------------------------------------------------------------------------

/// Internal trait for backend-specific GPU enumeration.
///
/// Each backend provides a way to discover ALL GPUs of that vendor on the
/// system (ignoring env-var filters like `CUDA_VISIBLE_DEVICES` or
/// `ONEAPI_DEVICE_SELECTOR`), so the CPU-set subdivision is fair.
trait BackendGpuEnumerator {
    /// Enumerate ALL GPUs for this backend, including those hidden by env vars.
    fn enumerate_all_gpus(&self) -> Vec<GpuTopoInfo>;
}

/// CUDA backend: NVML (all GPUs) → CUDA driver API fallback (visible only).
struct CudaGpuEnumerator;

impl BackendGpuEnumerator for CudaGpuEnumerator {
    fn enumerate_all_gpus(&self) -> Vec<GpuTopoInfo> {
        // NVML sees all NVIDIA GPUs regardless of CUDA_VISIBLE_DEVICES
        if let Some(nvml) = nvml::try_nvml() {
            let nvml_gpus = nvml.enumerate_gpus();
            if !nvml_gpus.is_empty() {
                tracing::debug!(
                    "NVML enumerated {} NVIDIA GPUs (ignoring CUDA_VISIBLE_DEVICES)",
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

        // Fallback: CUDA driver (only sees visible devices)
        tracing::debug!("Falling back to CUDA driver GPU enumeration");
        enumerate_cuda_gpus()
    }
}

/// SYCL/XPU backend: sysfs PCI scan (all Intel GPUs) → SYCL runtime fallback.
#[cfg(feature = "xpu-sycl")]
struct SyclGpuEnumerator;

#[cfg(feature = "xpu-sycl")]
impl BackendGpuEnumerator for SyclGpuEnumerator {
    fn enumerate_all_gpus(&self) -> Vec<GpuTopoInfo> {
        // sysfs scan sees all Intel GPUs regardless of ONEAPI_DEVICE_SELECTOR / ZE_AFFINITY_MASK
        let sysfs_gpus = enumerate_intel_gpus_sysfs();
        if !sysfs_gpus.is_empty() {
            tracing::debug!(
                "sysfs enumerated {} Intel GPUs (ignoring ONEAPI_DEVICE_SELECTOR)",
                sysfs_gpus.len()
            );
            return sysfs_gpus;
        }

        // Fallback: SYCL runtime (only sees filtered devices)
        tracing::debug!("Falling back to SYCL runtime GPU enumeration");
        enumerate_sycl_gpus()
    }
}

/// Enumerate GPUs visible to the SYCL runtime with their PCI addresses.
#[cfg(feature = "xpu-sycl")]
fn enumerate_sycl_gpus() -> Vec<GpuTopoInfo> {
    use oneapi_rs::safe::SyclDevice;

    let count = match SyclDevice::count() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    (0..count)
        .filter_map(|i| {
            let dev = SyclDevice::by_ordinal(i).ok()?;
            let pci = dev.info().ok()?.pci_address?;
            let numa = read_numa_node_from_sysfs(&pci).map(|n| n.0);
            Some(GpuTopoInfo {
                pci_address: pci,
                numa_node: numa,
            })
        })
        .collect()
}

/// Scan sysfs for ALL Intel GPU PCI devices (vendor 0x8086, class 0x03xxxx).
///
/// This is the SYCL/XPU equivalent of NVML for CUDA: it sees every Intel GPU
/// on the system regardless of `ONEAPI_DEVICE_SELECTOR` or `ZE_AFFINITY_MASK`.
#[cfg(feature = "xpu-sycl")]
fn enumerate_intel_gpus_sysfs() -> Vec<GpuTopoInfo> {
    let pci_dir = std::path::Path::new("/sys/bus/pci/devices");
    let entries = match fs::read_dir(pci_dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut gpus = Vec::new();
    for entry in entries.flatten() {
        let dev_path = entry.path();

        // GPU / display controller: PCI class 0x03xxxx
        let class = match fs::read_to_string(dev_path.join("class")) {
            Ok(c) => c.trim().to_string(),
            Err(_) => continue,
        };
        if !class.starts_with("0x03") {
            continue;
        }

        // Intel vendor ID: 0x8086
        let vendor = match fs::read_to_string(dev_path.join("vendor")) {
            Ok(v) => v.trim().to_string(),
            Err(_) => continue,
        };
        if vendor != "0x8086" {
            continue;
        }

        let pci_address = entry.file_name().to_string_lossy().to_string();
        let numa = read_numa_node_from_sysfs(&pci_address).map(|n| n.0);
        gpus.push(GpuTopoInfo {
            pci_address,
            numa_node: numa,
        });
    }

    // Deterministic ordering by PCI address
    gpus.sort_by(|a, b| a.pci_address.cmp(&b.pci_address));
    gpus
}

// ---------------------------------------------------------------------------
// Shared CPU-set subdivision logic (backend-agnostic)
// ---------------------------------------------------------------------------

/// Subdivide a NUMA node's CPUs among all GPUs sharing that node.
///
/// Given a list of GPUs (from one backend's enumeration) and a target PCI
/// address, returns the CPU subset assigned to that particular device.
fn subdivide_cpu_set_for_device(
    all_gpus: &[GpuTopoInfo],
    target_pci: &str,
    topology: &topology::NumaTopology,
) -> Option<Vec<usize>> {
    // Find the target GPU and its NUMA node
    let target = all_gpus.iter().find(|g| g.pci_address == target_pci)?;
    let target_node = target.numa_node?;

    // Collect all GPUs on the same NUMA node, sorted by PCI address
    let mut siblings: Vec<&str> = all_gpus
        .iter()
        .filter(|g| g.numa_node == Some(target_node))
        .map(|g| g.pci_address.as_str())
        .collect();
    siblings.sort();

    let position = siblings.iter().position(|&addr| addr == target_pci)?;
    let all_cpus = topology.cpus_for_node(target_node)?;

    if all_cpus.is_empty() || siblings.is_empty() {
        return None;
    }

    // Divide CPUs into N equal slices
    let n = siblings.len();
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
}

// ---------------------------------------------------------------------------
// Public API: backend-scoped CPU-set lookup
// ---------------------------------------------------------------------------

/// Cache: (backend, pci_address) → CPU set.
static BACKEND_CPU_SETS: OnceLock<Mutex<HashMap<(DeviceBackend, String), Option<Vec<usize>>>>> =
    OnceLock::new();

/// Get a deterministic CPU subset for a device, subdivided among ALL GPUs
/// of the **same backend** sharing the same NUMA node.
///
/// This is the primary entry point for NUMA-aware CPU pinning. The caller
/// provides the backend kind and the PCI BDF address (obtainable from
/// `DeviceContext::pci_bdf_address()` in `kvbm-physical`).
///
/// # Algorithm
/// 1. Enumerate ALL GPUs for the given backend:
///    - CUDA: NVML first (sees all GPUs, ignores `CUDA_VISIBLE_DEVICES`),
///      falling back to CUDA driver API.
///    - SYCL: sysfs PCI scan for Intel GPUs (ignores `ONEAPI_DEVICE_SELECTOR`),
///      falling back to SYCL runtime.
/// 2. Group enumerated GPUs by NUMA node.
/// 3. Sort by PCI address within each group (deterministic).
/// 4. Get full CPU set for the target device's NUMA node.
/// 5. Divide into N equal slices (N = GPUs on same node).
/// 6. Return the slice for the target device's position.
///
/// # Example
/// System with 8 NVIDIA GPUs, 2 NUMA nodes, 4 GPUs per node.
/// `CUDA_VISIBLE_DEVICES=0,1` (only 2 visible to CUDA runtime).
/// NVML sees all 8 → correctly subdivides into 4 CPU slices per node.
///
/// Same logic for Intel XPUs: 4 XPUs across 2 NUMA nodes,
/// `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` (only 2 visible to SYCL runtime).
/// sysfs scan sees all 4 → correctly subdivides into 2 CPU slices per node.
///
/// Returns `None` if the NUMA node cannot be determined.
pub fn get_device_cpu_set(backend: DeviceBackend, pci_address: &str) -> Option<Vec<usize>> {
    let cache = BACKEND_CPU_SETS.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (backend, pci_address.to_string());

    {
        let guard = cache.lock().unwrap();
        if let Some(cached) = guard.get(&key) {
            return cached.clone();
        }
    }

    let result = compute_cpu_set_for_device(backend, pci_address);
    cache.lock().unwrap().insert(key, result.clone());
    result
}

/// Compute the CPU set for a single device by enumerating its backend's GPUs.
fn compute_cpu_set_for_device(
    backend: DeviceBackend,
    pci_address: &str,
) -> Option<Vec<usize>> {
    let topology = match topology::get_numa_topology() {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("Cannot subdivide CPU sets: {e}");
            return None;
        }
    };

    let all_gpus = match backend {
        DeviceBackend::Cuda => CudaGpuEnumerator.enumerate_all_gpus(),
        DeviceBackend::Sycl => {
            #[cfg(feature = "xpu-sycl")]
            {
                SyclGpuEnumerator.enumerate_all_gpus()
            }
            #[cfg(not(feature = "xpu-sycl"))]
            {
                tracing::warn!("SYCL feature not enabled, cannot enumerate SYCL GPUs");
                return None;
            }
        }
    };

    subdivide_cpu_set_for_device(&all_gpus, pci_address, &topology)
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


#[cfg(all(test, feature = "testing-xpu-sycl"))]
mod sycl_tests {
    use super::*;
    use oneapi_rs::safe::SyclDevice;

    fn xpu_pci_address() -> Option<String> {
        let dev = SyclDevice::by_ordinal(0).ok()?;
        dev.info().ok()?.pci_address
    }

    #[test]
    fn test_xpu_pci_address_format() {
        let pci = match xpu_pci_address() {
            Some(p) => p,
            None => { println!("No PCI address for XPU 0, skipping"); return; }
        };
        // Expected format: DDDD:BB:DD.F (e.g. "0000:04:00.0")
        assert!(pci.len() >= 10, "PCI address too short: {}", pci);
        assert!(pci.contains(':'), "PCI address should contain ':'");
        assert!(pci.contains('.'), "PCI address should contain '.'");
        println!("XPU 0 PCI address: {}", pci);
    }

    #[test]
    fn test_get_numa_node_for_xpu() {
        let pci = match xpu_pci_address() {
            Some(p) => p,
            None => { println!("No PCI address, skipping"); return; }
        };

        let result = get_numa_node_for_pci_address(&pci);
        match result {
            Some(node) => {
                assert!(!node.is_unknown());
                assert!(node.0 < 16, "NUMA node {} seems unreasonably high", node.0);
                println!("XPU PCI {} → NUMA node {}", pci, node.0);
            }
            None => {
                println!("XPU PCI {} → no NUMA affinity (single-socket expected)", pci);
            }
        }
    }


    #[test]
    fn test_sysfs_numa_node_for_xpu() {
        let pci = match xpu_pci_address() {
            Some(p) => p,
            None => { println!("No PCI address, skipping"); return; }
        };

        let result = read_numa_node_from_sysfs(&pci);
        match result {
            Some(node) => {
                println!("sysfs reports NUMA node {} for PCI {}", node.0, pci);
            }
            None => {
                println!("sysfs has no NUMA info for PCI {} (returns -1 or missing)", pci);
            }
        }
    }

    #[test]
    fn test_numa_cache_consistency_xpu() {
        let pci = match xpu_pci_address() {
            Some(p) => p,
            None => { println!("No PCI address, skipping"); return; }
        };

        // Call twice — second should be cached
        let r1 = get_numa_node_for_pci_address(&pci);
        let r2 = get_numa_node_for_pci_address(&pci);
        assert_eq!(r1, r2, "Cached result should match first lookup");
    }

    #[test]
    fn test_bogus_pci_address_returns_none() {
        let result = get_numa_node_for_pci_address("ffff:ff:ff.f");
        assert!(result.is_none(), "Bogus PCI address should return None");
    }
}

