// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-aware GPU topology lookup.
//!
//! All host-wide GPU enumeration and CPU-set subdivision lives in
//! `dynamo-memory::numa` (the sysfs walk covers both NVIDIA `0x10de` and
//! Intel `0x8086` GPUs in one pass and feeds `cpu_slices_by_pci()`).
//! This module is a thin device-tier wrapper that exposes the result
//! keyed by `(DeviceBackend, pci_bdf)` so callers in `kvbm-*` crates
//! can stay device-agnostic.

use crate::DeviceBackend;
use dynamo_memory::numa::cpu_slices_by_pci;

/// Get a deterministic CPU subset for a device, subdivided among ALL GPUs
/// sharing the same NUMA node.
///
/// The CPU slices come from `dynamo-memory::numa::cpu_slices_by_pci()`,
/// which builds the map once from a single sysfs walk that includes both
/// NVIDIA and Intel GPUs. The `backend` argument is retained for caller
/// observability (tracing, future per-backend policy); the lookup itself
/// is keyed solely by PCI BDF.
///
/// Returns `None` when the device's NUMA node cannot be determined or
/// the slicing map is empty.
pub fn get_device_cpu_set(backend: DeviceBackend, pci_address: &str) -> Option<Vec<usize>> {
    let result = cpu_slices_by_pci().get(pci_address).cloned();
    if result.is_none() {
        tracing::debug!(
            backend = ?backend,
            pci = %pci_address,
            "no CPU slice for device (NUMA unknown or empty topology)"
        );
    }
    result
}
