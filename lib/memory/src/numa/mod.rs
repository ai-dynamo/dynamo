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

use nix::libc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::{fs, mem, process::Command};

#[cfg(all(test, feature = "testing-cuda"))]
use cudarc::driver::CudaContext;
#[cfg(all(test, feature = "testing-cuda"))]
use std::sync::Arc;

/// Get or create a CUDA context for NUMA-aware operations.
///
/// Used only by the in-tree CUDA test fixtures. Real device-context
/// management lives in `dynamo-device`.
#[cfg(all(test, feature = "testing-cuda"))]
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
/// The input may be in any of the PCI BDF string forms (4- or 8-char domain,
/// any case). The actual sysfs directory name on the running kernel is
/// resolved via [`pci_sysfs_dirnames`] and used to construct the path, so
/// callers do not have to know the kernel's preferred width. Returns `None`
/// if the entry can't be found, can't be read, or contains `-1`.
fn read_numa_node_from_sysfs(pci_address: &str) -> Option<NumaNode> {
    let parsed = PciAddress::parse(pci_address);
    let dirname = match parsed {
        Some(addr) => pci_sysfs_dirnames()
            .get(&addr)
            .cloned()
            .unwrap_or_else(|| addr.to_string()),
        None => pci_address.to_lowercase(),
    };
    let path = format!("/sys/bus/pci/devices/{}/numa_node", dirname);
    let content = fs::read_to_string(&path).ok()?;
    let node: i32 = content.trim().parse().ok()?;
    if node < 0 {
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

/// Resolve a CUDA device ordinal to its PCI BDF via the CUDA driver API.
///
/// Returns a normalized address string like `"0000:3b:00.0"`. The
/// `device_id` is a CUDA ordinal (affected by `CUDA_VISIBLE_DEVICES`).
///
/// Lives in `dynamo-memory` because `cudarc` is an unconditional dep of
/// this crate already, and the function is a leaf primitive — no
/// backend-aware dispatch. Both `dynamo-device::topology` and the
/// `testing-cuda` tests in this crate consume it as the canonical
/// CUDA-BDF helper.
pub fn get_pci_bus_address_from_cuda(device_id: u32) -> Option<String> {
    // cudarc panics (rather than returning Err) when libcuda.so is absent.
    std::panic::catch_unwind(|| {
        use cudarc::driver::{result::device as cuda_device, sys as cuda_sys};
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
    })
    .unwrap_or(None)
}

/// Get NUMA node for a CUDA device by ordinal — CUDA-only legacy helper.
///
/// Queries the PCI bus address via the CUDA driver API, then delegates
/// to [`get_numa_node_for_pci_address`]. Production code should obtain
/// the PCI BDF from `dynamo_device::DeviceContextOps::pci_bdf_address()`
/// and call [`get_numa_node_for_pci_address`] directly — that path is
/// backend-agnostic.
pub fn get_device_numa_node(device_id: u32) -> Option<NumaNode> {
    let pci_address = match get_pci_bus_address_from_cuda(device_id) {
        Some(a) => a,
        None => {
            tracing::warn!(
                "Failed to get PCI address from CUDA for device {}, skipping NUMA optimization",
                device_id
            );
            return None;
        }
    };
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

/// Numeric components of a PCI BDF address.
///
/// The PCI sysfs ABI (`Documentation/PCI/sysfs-pci.rst`) defines the
/// serialization as `DOMAIN:BUS:DEVICE.FUNCTION`, each field a lowercase
/// hex number. The serialization width varies (4 hex chars for `domain`
/// on most x86 systems, 8 on platforms with 32-bit PCI domains such as
/// Grace / GB200; bus/device are typically 2 chars; function is 1).
/// Width and capitalization differences across kernels and reporting
/// tools (CUDA, NVML, lspci) are normalized away by parsing into this
/// tuple — equality on a `PciAddress` is the only safe way to match
/// addresses from different sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PciAddress {
    /// PCI domain (segment) ID. Up to 32 bits per Linux kernel.
    pub domain: u32,
    /// PCI bus number, 0–255.
    pub bus: u8,
    /// Device (slot) number within the bus, 0–31.
    pub device: u8,
    /// Function number, 0–7.
    pub function: u8,
}

impl PciAddress {
    /// Parse a PCI BDF address string. Accepts any domain width and either
    /// case; rejects anything that cannot be decomposed into the four
    /// hex components.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        let parts: Vec<&str> = s.splitn(3, ':').collect();
        if parts.len() != 3 {
            return None;
        }
        let domain = u32::from_str_radix(parts[0], 16).ok()?;
        let bus = u8::from_str_radix(parts[1], 16).ok()?;
        let (dev_str, fn_str) = parts[2].split_once('.')?;
        let device = u8::from_str_radix(dev_str, 16).ok()?;
        let function = u8::from_str_radix(fn_str, 16).ok()?;
        Some(Self {
            domain,
            bus,
            device,
            function,
        })
    }
}

impl std::fmt::Display for PciAddress {
    /// Render with the kernel's native sysfs domain width on this host,
    /// so the result is a valid `/sys/bus/pci/devices/<this>/...` path
    /// component on the running system.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let w = sysfs_pci_domain_width();
        if w >= 8 {
            write!(
                f,
                "{:08x}:{:02x}:{:02x}.{:x}",
                self.domain, self.bus, self.device, self.function
            )
        } else {
            write!(
                f,
                "{:04x}:{:02x}:{:02x}.{:x}",
                self.domain, self.bus, self.device, self.function
            )
        }
    }
}

/// Detect the hex-character width the kernel uses for PCI domain components
/// in sysfs directory names (typically 4 on x86, 8 on platforms with 32-bit
/// PCI domains such as Grace / GB200). Falls back to 4 if `/sys/bus/pci/devices`
/// cannot be enumerated.
fn sysfs_pci_domain_width() -> usize {
    static WIDTH: OnceLock<usize> = OnceLock::new();
    *WIDTH.get_or_init(|| {
        if let Ok(entries) = fs::read_dir("/sys/bus/pci/devices") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if let Some(colon) = name.find(':') {
                    return colon;
                }
            }
        }
        4
    })
}

/// Map from a parsed PCI address to the exact directory name the kernel
/// uses under `/sys/bus/pci/devices/`. Built once by walking sysfs. This is
/// the safe way to construct a sysfs path from an address obtained from
/// CUDA/NVML, since we never reconstruct the directory name from format
/// assumptions.
fn pci_sysfs_dirnames() -> &'static HashMap<PciAddress, String> {
    static MAP: OnceLock<HashMap<PciAddress, String>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m: HashMap<PciAddress, String> = HashMap::new();
        if let Ok(entries) = fs::read_dir("/sys/bus/pci/devices") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if let Some(addr) = PciAddress::parse(&name) {
                    m.insert(addr, name);
                }
            }
        }
        m
    })
}

/// Parse `s` and re-emit it in the canonical form for this kernel.
/// Returns `None` if `s` is not a parseable PCI BDF address.
fn normalize_pci_address(s: &str) -> Option<String> {
    Some(PciAddress::parse(s)?.to_string())
}

/// GPU topology info: PCI address and (optional) NUMA node.
///
/// Returned by [`enumerate_all_gpus`] and [`enumerate_host_gpus_from_sysfs`].
/// The PCI address is normalized to lowercase `"DDDD:BB:DD.0"` form.
/// `numa_node` is `None` when sysfs reports `-1` (no affinity) or the entry
/// cannot be read.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Normalized PCI bus address, e.g. `"0000:3b:00.0"`.
    pub pci_address: String,
    /// NUMA node ID from `/sys/bus/pci/devices/<pci>/numa_node`, or `None`
    /// when the device has no affinity info.
    pub numa_node: Option<u32>,
}

/// Enumerate GPUs visible to the CUDA driver in this process (subject to
/// `CUDA_VISIBLE_DEVICES`). Used as a last-ditch fallback when neither
/// sysfs nor NVML succeed.
fn enumerate_cuda_gpus() -> Vec<GpuInfo> {
    // cudarc panics (rather than returning Err) when libcuda.so is absent.
    std::panic::catch_unwind(|| {
        use cudarc::driver::result::device as cuda_device;
        let count = match cuda_device::get_count() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        (0..count as u32)
            .filter_map(|i| {
                let pci = get_pci_bus_address_from_cuda(i)?;
                let numa = read_numa_node_from_sysfs(&pci).map(|n| n.0);
                Some(GpuInfo {
                    pci_address: pci,
                    numa_node: numa,
                })
            })
            .collect()
    })
    .unwrap_or_default()
}

/// Enumerate all NVIDIA GPUs on the host by walking sysfs.
///
/// Scans `/sys/bus/pci/devices/*`, keeping entries whose `vendor == 0x10de`
/// (NVIDIA) and whose `class` starts with `0x0300` (display / 3D / VGA
/// controllers). Reads each device's `numa_node`. The result is sorted by
/// PCI address for determinism.
///
/// This path is the source of truth for "what GPUs are on this host" because
/// `/sys` is not network-namespaced and remains accurate inside containers
/// even when NVML's view collapses to the container's allotment.
/// Returns an empty `Vec` if `/sys/bus/pci/devices` cannot be read.
pub fn enumerate_host_gpus_from_sysfs() -> Vec<GpuInfo> {
    let root = std::path::Path::new("/sys/bus/pci/devices");
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut gpus: Vec<GpuInfo> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        let vendor = match fs::read_to_string(path.join("vendor")) {
            Ok(s) => s.trim().to_lowercase(),
            Err(_) => continue,
        };
        // 0x10de = NVIDIA, 0x8086 = Intel. Both backends share the same cache.
        if vendor != "0x10de" && vendor != "0x8086" {
            continue;
        }

        let class = match fs::read_to_string(path.join("class")) {
            Ok(s) => s.trim().to_lowercase(),
            Err(_) => continue,
        };
        // 0x0300xx = display controller / VGA / 3D controller.
        if !class.starts_with("0x0300") {
            continue;
        }

        let numa = read_numa_node_from_sysfs(&name).map(|n| n.0);
        let normalized = normalize_pci_address(&name).unwrap_or_else(|| name.to_lowercase());
        gpus.push(GpuInfo {
            pci_address: normalized,
            numa_node: numa,
        });
    }

    gpus.sort_by(|a, b| a.pci_address.cmp(&b.pci_address));
    gpus
}

/// Enumerate every NVIDIA GPU on the host.
///
/// Prefers the sysfs walk (container-safe, sees host-wide topology), falls
/// back to NVML (loses container-hidden GPUs), and finally to the CUDA
/// driver (subject to `CUDA_VISIBLE_DEVICES`). Returns an empty `Vec` if all
/// three paths fail.
pub fn enumerate_all_gpus() -> Vec<GpuInfo> {
    let sysfs_gpus = enumerate_host_gpus_from_sysfs();
    if !sysfs_gpus.is_empty() {
        tracing::debug!("sysfs enumerated {} GPUs", sysfs_gpus.len());
        return sysfs_gpus;
    }

    if let Some(nvml) = nvml::try_nvml() {
        let nvml_gpus = nvml.enumerate_gpus();
        if !nvml_gpus.is_empty() {
            tracing::debug!(
                "NVML enumerated {} GPUs (sysfs unavailable)",
                nvml_gpus.len()
            );
            return nvml_gpus
                .into_iter()
                .map(|g| {
                    let pci = normalize_pci_address(&g.pci_address)
                        .unwrap_or_else(|| g.pci_address.to_lowercase());
                    let numa = read_numa_node_from_sysfs(&pci).map(|n| n.0);
                    GpuInfo {
                        pci_address: pci,
                        numa_node: numa,
                    }
                })
                .collect();
        }
    }

    tracing::debug!("Falling back to CUDA driver GPU enumeration");
    enumerate_cuda_gpus()
}

/// Cached map of PCI bus address → deterministic CPU slice.
///
/// Built once from the sysfs-derived host GPU list and the system NUMA
/// topology. Populated lazily on first call to [`cpu_slices_by_pci`].
static CPU_SLICES_BY_PCI: OnceLock<HashMap<String, Vec<usize>>> = OnceLock::new();

/// Return the global PCI-keyed map of per-GPU CPU slices.
///
/// Built from [`enumerate_all_gpus`] (sysfs-first) and the system NUMA
/// topology. For each NUMA node with non-empty cpulist, all sibling GPUs on
/// that node are sorted by PCI address and the node's CPUs are divided into
/// equal slices — last slice absorbs the remainder. GPUs whose NUMA node is
/// unknown, or whose node's cpulist is empty, are absent from the map.
///
/// Computed once and cached for the process lifetime.
pub fn cpu_slices_by_pci() -> &'static HashMap<String, Vec<usize>> {
    CPU_SLICES_BY_PCI.get_or_init(compute_cpu_slices_by_pci)
}

fn compute_cpu_slices_by_pci() -> HashMap<String, Vec<usize>> {
    let topology = match topology::get_numa_topology() {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("Cannot subdivide CPU sets: {e}");
            return HashMap::new();
        }
    };

    let all_gpus = enumerate_all_gpus();
    if all_gpus.is_empty() {
        return HashMap::new();
    }

    let mut node_groups: HashMap<u32, Vec<String>> = HashMap::new();
    for gpu in &all_gpus {
        if let Some(node) = gpu.numa_node {
            node_groups
                .entry(node)
                .or_default()
                .push(gpu.pci_address.clone());
        }
    }
    for group in node_groups.values_mut() {
        group.sort();
    }

    let mut results = HashMap::new();
    for (node, group) in &node_groups {
        let all_cpus = match topology.cpus_for_node(*node) {
            Some(cpus) if !cpus.is_empty() => cpus,
            _ => continue,
        };

        let n = group.len();
        let chunk_size = all_cpus.len() / n;
        if chunk_size == 0 {
            // More GPUs than CPUs on this node — every GPU gets all of them.
            for pci in group {
                results.insert(pci.clone(), all_cpus.to_vec());
            }
            continue;
        }

        for (position, pci) in group.iter().enumerate() {
            let start = position * chunk_size;
            let end = if position == n - 1 {
                all_cpus.len()
            } else {
                start + chunk_size
            };
            results.insert(pci.clone(), all_cpus[start..end].to_vec());
        }
    }

    results
}

// Backend-aware CPU-set lookup (`get_device_cpu_set(backend, pci)`)
// lives in `dynamo-device::topology`, which dispatches the
// per-backend GPU enumeration (CUDA/NVML, SYCL/sysfs) and calls back
// into `cpu_slices_by_pci` above for the backend-agnostic subdivision.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pci_address_parse_canonical_forms() {
        let want = PciAddress {
            domain: 0,
            bus: 0x89,
            device: 0,
            function: 0,
        };
        // 4-char domain
        assert_eq!(PciAddress::parse("0000:89:00.0"), Some(want));
        // 8-char domain (Grace/GB200 sysfs form)
        assert_eq!(PciAddress::parse("00000000:89:00.0"), Some(want));
        // Uppercase
        assert_eq!(PciAddress::parse("0000:89:00.0"), Some(want));
        assert_eq!(PciAddress::parse("0000:89:0.0"), Some(want));
        // Leading/trailing whitespace
        assert_eq!(PciAddress::parse("  0000:89:00.0  "), Some(want));
    }

    #[test]
    fn test_pci_address_parse_uppercase_and_short_function() {
        let want = PciAddress {
            domain: 0x0000_0001,
            bus: 0xAB,
            device: 0x1F,
            function: 7,
        };
        assert_eq!(PciAddress::parse("0001:AB:1F.7"), Some(want));
        assert_eq!(PciAddress::parse("00000001:ab:1f.7"), Some(want));
    }

    #[test]
    fn test_pci_address_parse_rejects_garbage() {
        assert_eq!(PciAddress::parse(""), None);
        assert_eq!(PciAddress::parse("not-a-pci-addr"), None);
        assert_eq!(PciAddress::parse("0000:89"), None);
        assert_eq!(PciAddress::parse("0000:89:00"), None);
        assert_eq!(PciAddress::parse("0000:89:zz.0"), None);
    }

    #[test]
    fn test_pci_address_round_trip_identity() {
        // Whatever Display produces must parse back to the same tuple, so
        // callers that stringify-then-reparse don't drift.
        let cases = [
            PciAddress {
                domain: 0,
                bus: 0,
                device: 0,
                function: 0,
            },
            PciAddress {
                domain: 0,
                bus: 0xc2,
                device: 0,
                function: 0,
            },
            PciAddress {
                domain: 0xdead_beef,
                bus: 0xff,
                device: 0x1f,
                function: 7,
            },
        ];
        for a in &cases {
            let s = a.to_string();
            assert_eq!(PciAddress::parse(&s), Some(*a), "round trip for {}", s);
        }
    }

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
    use oneapi_rs::sycl::safe::SyclDevice;

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

