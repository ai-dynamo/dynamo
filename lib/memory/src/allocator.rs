// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic device memory allocation trait.
//!
//! [`DeviceAllocator`] provides the minimal interface needed by [`super::DeviceStorage`]
//! and [`super::PinnedStorage`] to allocate and free memory on any hardware backend.
//! Concrete implementations (CUDA, Level-Zero, …) live in downstream crates
//! (e.g. `kvbm-physical`) where the backend-specific dependencies are available.

use crate::Result;
use std::fmt;

/// Trait for backend-agnostic device memory allocation.
///
/// Implementations manage the lifecycle of device and pinned host memory.
/// Both [`super::DeviceStorage`] and [`super::PinnedStorage`] take an
/// `Arc<dyn DeviceAllocator>` and delegate allocation/deallocation to it.
pub trait DeviceAllocator: Send + Sync + fmt::Debug {
    /// Allocate device memory of the given size. Returns a device pointer.
    fn allocate_device(&self, size: usize) -> Result<u64>;

    /// Free device memory at the given pointer.
    fn free_device(&self, ptr: u64) -> Result<()>;

    /// Allocate pinned (page-locked) host memory. Returns a host pointer.
    fn allocate_pinned(&self, size: usize) -> Result<u64>;

    /// Free pinned host memory at the given pointer.
    fn free_pinned(&self, ptr: u64) -> Result<()>;

    /// Device ordinal this allocator is bound to.
    fn device_id(&self) -> u32;
}
