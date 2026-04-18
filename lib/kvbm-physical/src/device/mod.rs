// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Device abstraction layer for multi-backend support.
//!
//! This module provides a unified interface for different hardware backends
//! (CUDA, Level-Zero/XPU) using the Static Enum + Trait Objects pattern.

pub mod traits;
pub mod detection;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "xpu-ze")]
pub mod ze;
#[cfg(feature = "xpu-sycl")]
pub mod sycl;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub use traits::{DeviceContextOps, DeviceStreamOps, DeviceEventOps, DeviceMemPoolOps, EngineHint};

/// Device backend type selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceBackend {
    Cuda,
    Ze,
    /// SYCL backend — pure SYCL implementation (no Level Zero Rust crate needed).
    Sycl,
}

impl DeviceBackend {
    /// Get human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Ze => "Level-Zero (XPU)",
            Self::Sycl => "SYCL (XPU)",
        }
    }

    /// Check if backend is available on current system.
    pub fn is_available(&self) -> bool {
        match self {
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                { cuda::is_available() }
                #[cfg(not(feature = "cuda"))]
                { false }
            }
            Self::Ze => {
                #[cfg(feature = "xpu-ze")]
                { ze::is_available() }
                #[cfg(not(feature = "xpu-ze"))]
                { false }
            }
            Self::Sycl => {
                #[cfg(feature = "xpu-sycl")]
                { sycl::is_available() }
                #[cfg(not(feature = "xpu-sycl"))]
                { false }
            }
        }
    }
}

impl FromStr for DeviceBackend {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cuda" | "gpu" | "nvidia" => Ok(Self::Cuda),
            "ze" | "level-zero" => Ok(Self::Ze),
            "sycl" | "xpu" | "intel" => Ok(Self::Sycl),
            _ => bail!("Unknown device backend: {}", s),
        }
    }
}

/// Unified device context holding polymorphic implementation.
pub struct DeviceContext {
    backend: DeviceBackend,
    device_id: u32,
    pub(crate) ops: Box<dyn DeviceContextOps>,
}

impl DeviceContext {
    /// Create a new device context for the specified backend and device.
    pub fn new(backend: DeviceBackend, device_id: u32) -> Result<Self> {
        let ops: Box<dyn DeviceContextOps> = match backend {
            DeviceBackend::Cuda => {
                #[cfg(feature = "cuda")]
                { Box::new(cuda::CudaContext::new(device_id)?) }
                #[cfg(not(feature = "cuda"))]
                { bail!("CUDA backend not compiled (enable 'cuda' feature)") }
            }
            DeviceBackend::Ze => {
                #[cfg(feature = "xpu-ze")]
                { Box::new(ze::ZeContext::new(device_id)?) }
                #[cfg(not(feature = "xpu-ze"))]
                { bail!("Level-Zero backend not compiled (enable 'xpu-ze' feature)") }
            }
            DeviceBackend::Sycl => {
                #[cfg(feature = "xpu-sycl")]
                { Box::new(sycl::SyclContext::new(device_id)?) }
                #[cfg(not(feature = "xpu-sycl"))]
                { bail!("SYCL backend not compiled (enable 'xpu-sycl' feature)") }
            }
        };

        Ok(Self { backend, device_id, ops })
    }

    pub fn backend(&self) -> DeviceBackend {
        self.backend
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn create_stream(&self, hint: EngineHint) -> Result<DeviceStream> {
        let stream_ops = self.ops.create_stream(hint)?;
        Ok(DeviceStream {
            backend: self.backend,
            ops: stream_ops,
        })
    }

    pub fn allocate_device(&self, size: usize) -> Result<u64> {
        self.ops.allocate_device(size)
    }

    pub fn free_device(&self, ptr: u64) -> Result<()> {
        self.ops.free_device(ptr)
    }

    pub fn allocate_pinned(&self, size: usize) -> Result<u64> {
        self.ops.allocate_pinned(size)
    }

    pub fn free_pinned(&self, ptr: u64) -> Result<()> {
        self.ops.free_pinned(ptr)
    }

    /// Create a memory pool for stream-ordered async allocations.
    pub fn create_memory_pool(
        &self,
        reserve_size: usize,
        release_threshold: Option<u64>,
    ) -> Result<DeviceMemPool> {
        let ops = self.ops.create_memory_pool(reserve_size, release_threshold)?;
        Ok(DeviceMemPool {
            backend: self.backend,
            ops,
        })
    }
}

unsafe impl Send for DeviceContext {}
unsafe impl Sync for DeviceContext {}

impl std::fmt::Debug for DeviceContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceContext")
            .field("backend", &self.backend)
            .field("device_id", &self.device_id)
            .finish()
    }
}

/// Device stream wrapper.
pub struct DeviceStream {
    backend: DeviceBackend,
    pub ops: Box<dyn DeviceStreamOps>,
}

impl DeviceStream {
    pub fn backend(&self) -> DeviceBackend {
        self.backend
    }

    pub fn batch_copy(&self, src_ptrs: &[u64], dst_ptrs: &[u64], size: usize) -> Result<()> {
        self.ops.batch_copy(src_ptrs, dst_ptrs, size)
    }

    pub fn memcpy_htod(&self, dst_device: u64, src_host: &[u8]) -> Result<()> {
        self.ops.memcpy_htod(dst_device, src_host)
    }

    pub fn memcpy_dtoh(&self, src_device: u64, dst_host: &mut [u8]) -> Result<()> {
        self.ops.memcpy_dtoh(src_device, dst_host)
    }

    pub fn vectorized_copy(
        &self,
        src_ptrs_device: u64,
        dst_ptrs_device: u64,
        chunk_size: usize,
        count: usize,
    ) -> Result<()> {
        self.ops.vectorized_copy(src_ptrs_device, dst_ptrs_device, chunk_size, count)
    }

    pub fn record_event(&self) -> Result<DeviceEvent> {
        let event_ops = self.ops.record_event()?;
        Ok(DeviceEvent {
            backend: self.backend,
            ops: event_ops,
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.ops.synchronize()
    }
}

unsafe impl Send for DeviceStream {}
unsafe impl Sync for DeviceStream {}


/// Synchronous device-to-host memcpy using the device abstraction.
///
/// Creates a temporary stream, enqueues the copy, and synchronizes.
/// Used by test helpers (checksum, fill) that need backend-agnostic device access.
pub fn sync_memcpy_dtoh(device_id: u32, src_device: u64, dst_host: &mut [u8]) -> Result<()> {
    let backend = DeviceBackend::detect_backend()?;
    let ctx = DeviceContext::new(backend, device_id)?;
    let stream = ctx.create_stream(EngineHint::Copy)?;
    stream.memcpy_dtoh(src_device, dst_host)?;
    stream.synchronize()?;
    Ok(())
}

/// Synchronous host-to-device memcpy using the device abstraction.
///
/// Creates a temporary stream, enqueues the copy, and synchronizes.
/// Used by test helpers (checksum, fill) that need backend-agnostic device access.
pub fn sync_memcpy_htod(device_id: u32, dst_device: u64, src_host: &[u8]) -> Result<()> {
    let backend = DeviceBackend::detect_backend()?;
    let ctx = DeviceContext::new(backend, device_id)?;
    let stream = ctx.create_stream(EngineHint::Copy)?;
    stream.memcpy_htod(dst_device, src_host)?;
    stream.synchronize()?;
    Ok(())
}

impl std::fmt::Debug for DeviceStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceStream")
            .field("backend", &self.backend)
            .finish()
    }
}

/// Device event wrapper.
pub struct DeviceEvent {
    backend: DeviceBackend,
    pub ops: Box<dyn DeviceEventOps>,
}

impl DeviceEvent {
    pub fn backend(&self) -> DeviceBackend {
        self.backend
    }

    pub fn is_complete(&self) -> Result<bool> {
        self.ops.is_complete()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.ops.synchronize()
    }
}

unsafe impl Send for DeviceEvent {}
unsafe impl Sync for DeviceEvent {}

impl std::fmt::Debug for DeviceEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceEvent")
            .field("backend", &self.backend)
            .finish()
    }
}


// ======================================================================
// DeviceAllocator — bridge to dynamo-memory's trait
// ======================================================================

impl dynamo_memory::DeviceAllocator for DeviceContext {
    fn allocate_device(&self, size: usize) -> dynamo_memory::Result<u64> {
        self.ops
            .allocate_device(size)
            .map_err(|e| dynamo_memory::StorageError::AllocationFailed(e.to_string()))
    }

    fn free_device(&self, ptr: u64) -> dynamo_memory::Result<()> {
        self.ops
            .free_device(ptr)
            .map_err(|e| dynamo_memory::StorageError::OperationFailed(e.to_string()))
    }

    fn allocate_pinned(&self, size: usize) -> dynamo_memory::Result<u64> {
        self.ops
            .allocate_pinned(size)
            .map_err(|e| dynamo_memory::StorageError::AllocationFailed(e.to_string()))
    }

    fn free_pinned(&self, ptr: u64) -> dynamo_memory::Result<()> {
        self.ops
            .free_pinned(ptr)
            .map_err(|e| dynamo_memory::StorageError::OperationFailed(e.to_string()))
    }

    fn device_id(&self) -> u32 {
        self.device_id
    }
}

// ======================================================================
// DeviceMemPool — unified memory pool wrapper
// ======================================================================

/// Unified device memory pool for stream-ordered async allocation.
pub struct DeviceMemPool {
    backend: DeviceBackend,
    ops: Box<dyn DeviceMemPoolOps>,
}

impl DeviceMemPool {
    pub fn backend(&self) -> DeviceBackend {
        self.backend
    }

    /// Allocate memory from the pool, ordered on the given stream.
    pub fn alloc_async(&self, size: usize, stream: &DeviceStream) -> Result<u64> {
        self.ops.alloc_async(size, &*stream.ops)
    }

    /// Free memory back to the pool, ordered on the given stream.
    pub fn free_async(&self, ptr: u64, stream: &DeviceStream) -> Result<()> {
        self.ops.free_async(ptr, &*stream.ops)
    }
}

unsafe impl Send for DeviceMemPool {}
unsafe impl Sync for DeviceMemPool {}

impl std::fmt::Debug for DeviceMemPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMemPool")
            .field("backend", &self.backend)
            .finish()
    }
}
