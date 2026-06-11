// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal NVML FFI via dlopen.
//!
//! Dynamically loads `libnvidia-ml.so.1` to enumerate ALL GPUs on the system,
//! regardless of `CUDA_VISIBLE_DEVICES`. This is critical for CPU set subdivision:
//! when multiple GPUs share a NUMA node, we need to know about ALL siblings to
//! divide CPU cores fairly.
//!
//! If NVML is unavailable (e.g., in containers without the management library),
//! callers fall back to CUDA driver enumeration (which only sees visible devices).

use libloading::{Library, Symbol};
use std::sync::OnceLock;

/// PCI info struct matching NVML's `nvmlPciInfo_t`.
#[repr(C)]
struct NvmlPciInfo {
    bus_id_legacy: [u8; 16], // "DDDD:BB:DD.F\0" (legacy, 16 chars)
    domain: u32,
    bus: u32,
    device: u32,
    pci_device_id: u32,
    pci_subsystem_id: u32,
    bus_id: [u8; 32], // "DDDD:BB:DD.F\0" (full, 32 chars)
}

/// GPU info from NVML enumeration.
#[derive(Debug, Clone)]
pub(crate) struct NvmlGpuInfo {
    /// PCI bus address, e.g. "0000:3b:00.0"
    pub pci_address: String,
}

// NVML return codes
const NVML_SUCCESS: u32 = 0;

/// Minimal NVML handle. Sees ALL GPUs regardless of CUDA_VISIBLE_DEVICES.
pub(crate) struct NvmlHandle {
    _lib: Library,
    device_get_count: unsafe extern "C" fn(*mut u32) -> u32,
    device_get_handle_by_index: unsafe extern "C" fn(u32, *mut u64) -> u32,
    device_get_pci_info: unsafe extern "C" fn(u64, *mut NvmlPciInfo) -> u32,
    shutdown: unsafe extern "C" fn() -> u32,
}

// SAFETY: NVML functions are thread-safe per NVML documentation
unsafe impl Send for NvmlHandle {}
unsafe impl Sync for NvmlHandle {}

impl NvmlHandle {
    /// Try to load NVML. Returns None if libnvidia-ml.so.1 is not available.
    pub fn try_load() -> Option<Self> {
        // SAFETY: We are loading a well-known system library and resolving documented
        // NVML API symbols. The library is kept alive for the lifetime of NvmlHandle.
        unsafe {
            let lib = Library::new("libnvidia-ml.so.1").ok()?;

            // Initialize NVML
            let init: Symbol<unsafe extern "C" fn() -> u32> = lib.get(b"nvmlInit_v2\0").ok()?;
            if init() != NVML_SUCCESS {
                tracing::warn!("nvmlInit_v2 failed");
                return None;
            }

            let device_get_count: Symbol<unsafe extern "C" fn(*mut u32) -> u32> =
                lib.get(b"nvmlDeviceGetCount_v2\0").ok()?;
            let device_get_handle_by_index: Symbol<unsafe extern "C" fn(u32, *mut u64) -> u32> =
                lib.get(b"nvmlDeviceGetHandleByIndex_v2\0").ok()?;
            let device_get_pci_info: Symbol<unsafe extern "C" fn(u64, *mut NvmlPciInfo) -> u32> =
                lib.get(b"nvmlDeviceGetPciInfo_v3\0").ok()?;
            let shutdown: Symbol<unsafe extern "C" fn() -> u32> =
                lib.get(b"nvmlShutdown\0").ok()?;

            Some(Self {
                device_get_count: *device_get_count,
                device_get_handle_by_index: *device_get_handle_by_index,
                device_get_pci_info: *device_get_pci_info,
                shutdown: *shutdown,
                _lib: lib,
            })
        }
    }

    /// Enumerate ALL GPUs on the system with PCI addresses.
    pub fn enumerate_gpus(&self) -> Vec<NvmlGpuInfo> {
        let mut count: u32 = 0;
        // SAFETY: NVML API call with valid pointer
        unsafe {
            if (self.device_get_count)(&mut count) != NVML_SUCCESS {
                tracing::warn!("nvmlDeviceGetCount_v2 failed");
                return Vec::new();
            }
        }

        let mut gpus = Vec::with_capacity(count as usize);
        for i in 0..count {
            let mut handle: u64 = 0;
            // SAFETY: NVML API call with valid index and pointer
            unsafe {
                if (self.device_get_handle_by_index)(i, &mut handle) != NVML_SUCCESS {
                    tracing::warn!("nvmlDeviceGetHandleByIndex_v2 failed for index {i}");
                    continue;
                }
            }

            let mut pci_info = std::mem::MaybeUninit::<NvmlPciInfo>::zeroed();
            // SAFETY: NVML API call with valid handle and pointer to zeroed struct
            unsafe {
                if (self.device_get_pci_info)(handle, pci_info.as_mut_ptr()) != NVML_SUCCESS {
                    tracing::warn!("nvmlDeviceGetPciInfo_v3 failed for index {i}");
                    continue;
                }
                let pci_info = pci_info.assume_init();

                // Parse bus_id field: "DDDD:BB:DD.F\0" padded with zeros
                let bus_id = &pci_info.bus_id;
                let len = bus_id.iter().position(|&b| b == 0).unwrap_or(bus_id.len());
                let pci_address = std::str::from_utf8(&bus_id[..len])
                    .unwrap_or("")
                    .to_lowercase();

                if !pci_address.is_empty() {
                    gpus.push(NvmlGpuInfo { pci_address });
                }
            }
        }

        gpus
    }
}

impl Drop for NvmlHandle {
    fn drop(&mut self) {
        // SAFETY: Matching nvmlInit_v2 call from try_load
        unsafe {
            (self.shutdown)();
        }
    }
}

/// Cached NVML load attempt (None = tried and failed).
static NVML: OnceLock<Option<NvmlHandle>> = OnceLock::new();

/// Get a reference to the global NVML handle, if available.
pub(crate) fn try_nvml() -> Option<&'static NvmlHandle> {
    NVML.get_or_init(|| {
        let handle = NvmlHandle::try_load();
        if handle.is_some() {
            tracing::debug!("NVML loaded successfully");
        } else {
            tracing::debug!("NVML not available, will fall back to CUDA driver enumeration");
        }
        handle
    })
    .as_ref()
}
