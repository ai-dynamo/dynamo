// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-aware GPU topology discovery and CPU-set subdivision.
//!
//! The functions here dispatch on [`DeviceBackend`] to enumerate every
//! GPU of a given vendor on the system (using NVML for CUDA, sysfs PCI
//! scan for SYCL/Intel), then subdivide each NUMA node's CPUs fairly
//! across the GPUs that share it.
//!
//! The backend-agnostic primitives this module relies on
//! (`get_numa_node_for_pci_address`, `subdivide_cpu_set_for_device`,
//! `read_numa_node_from_sysfs`, `GpuTopoInfo`) live in `dynamo-memory`'s
//! `numa` module so this crate can stay focused on device-tier dispatch.

use crate::DeviceBackend;
use dynamo_memory::numa::{GpuTopoInfo, subdivide_cpu_set_for_device, topology};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[cfg(any(feature = "cuda", feature = "xpu-sycl"))]
use dynamo_memory::numa::read_numa_node_from_sysfs;
#[cfg(feature = "cuda")]
use cudarc::driver::result::device as cuda_device;
#[cfg(feature = "cuda")]
use dynamo_memory::numa::{enumerate_nvml_gpus, get_pci_bus_address_from_cuda};

// ---------------------------------------------------------------------------
// Backend GPU enumerator trait + implementations
// ---------------------------------------------------------------------------

/// Internal trait for backend-specific GPU enumeration.
///
/// Each backend provides a way to discover ALL GPUs of that vendor on
/// the system (ignoring env-var filters like `CUDA_VISIBLE_DEVICES` or
/// `ONEAPI_DEVICE_SELECTOR`), so the CPU-set subdivision is fair.
trait BackendGpuEnumerator {
    fn enumerate_all_gpus(&self) -> Vec<GpuTopoInfo>;
}

/// CUDA backend: NVML (all GPUs) → CUDA driver API fallback (visible only).
#[cfg(feature = "cuda")]
struct CudaGpuEnumerator;

#[cfg(feature = "cuda")]
impl BackendGpuEnumerator for CudaGpuEnumerator {
    fn enumerate_all_gpus(&self) -> Vec<GpuTopoInfo> {
        // NVML sees all NVIDIA GPUs regardless of CUDA_VISIBLE_DEVICES.
        if let Some(nvml_gpus) = enumerate_nvml_gpus() {
            tracing::debug!(
                "NVML enumerated {} NVIDIA GPUs (ignoring CUDA_VISIBLE_DEVICES)",
                nvml_gpus.len()
            );
            return nvml_gpus;
        }
        // Fallback: CUDA driver (only sees visible devices).
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
        let sysfs_gpus = enumerate_intel_gpus_sysfs();
        if !sysfs_gpus.is_empty() {
            tracing::debug!(
                "sysfs enumerated {} Intel GPUs (ignoring ONEAPI_DEVICE_SELECTOR)",
                sysfs_gpus.len()
            );
            return sysfs_gpus;
        }
        tracing::debug!("Falling back to SYCL runtime GPU enumeration");
        enumerate_sycl_gpus()
    }
}

// ---------------------------------------------------------------------------
// Per-backend GPU enumeration helpers
// ---------------------------------------------------------------------------

/// Enumerate GPUs visible to the CUDA driver.
#[cfg(feature = "cuda")]
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
    use std::fs;

    let pci_dir = std::path::Path::new("/sys/bus/pci/devices");
    let entries = match fs::read_dir(pci_dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut gpus = Vec::new();
    for entry in entries.flatten() {
        let dev_path = entry.path();

        // GPU / display controller: PCI class 0x03xxxx.
        let class = match fs::read_to_string(dev_path.join("class")) {
            Ok(c) => c.trim().to_string(),
            Err(_) => continue,
        };
        if !class.starts_with("0x03") {
            continue;
        }

        // Intel vendor ID: 0x8086.
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

    gpus.sort_by(|a, b| a.pci_address.cmp(&b.pci_address));
    gpus
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
/// `DeviceContext::pci_bdf_address()`).
///
/// # Algorithm
/// 1. Enumerate ALL GPUs for the given backend:
///    - CUDA: NVML first (sees all GPUs, ignores `CUDA_VISIBLE_DEVICES`),
///      falling back to CUDA driver API.
///    - SYCL: sysfs PCI scan for Intel GPUs (ignores
///      `ONEAPI_DEVICE_SELECTOR`), falling back to SYCL runtime.
/// 2. Group enumerated GPUs by NUMA node.
/// 3. Sort by PCI address within each group (deterministic).
/// 4. Get full CPU set for the target device's NUMA node.
/// 5. Divide into N equal slices (N = GPUs on same node).
/// 6. Return the slice for the target device's position.
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

    let all_gpus: Vec<GpuTopoInfo> = match backend {
        DeviceBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                CudaGpuEnumerator.enumerate_all_gpus()
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!("CUDA feature not enabled, cannot enumerate CUDA GPUs");
                return None;
            }
        }
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
