//! Device abstraction layer for multi-backend support
//!
//! This module provides a unified interface for different hardware backends
//! (CUDA, Level-Zero, Synapse) using the Static Enum + Trait Objects pattern.

pub mod traits;
pub mod detection;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "xpu")]
pub mod ze;
#[cfg(feature = "hpu")]
pub mod hpu;

#[cfg(all(test, feature = "hpu"))]
mod test_hpu_minimal;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub use traits::{DeviceContextOps, DeviceStreamOps, DeviceEventOps};

/// Device backend type selector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceBackend {
    Cuda,
    Ze,
    Hpu,
}

impl DeviceBackend {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Ze => "Level-Zero (XPU)",
            Self::Hpu => "Synapse (HPU)",
        }
    }

    /// Check if backend is available on current system
    pub fn is_available(&self) -> bool {
        match self {
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                { cuda::is_available() }
                #[cfg(not(feature = "cuda"))]
                { false }
            }
            Self::Ze => {
                #[cfg(feature = "xpu")]
                { ze::is_available() }
                #[cfg(not(feature = "xpu"))]
                { false }
            }
            Self::Hpu => {
                #[cfg(feature = "hpu")]
                { hpu::is_available() }
                #[cfg(not(feature = "hpu"))]
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
            "ze" | "xpu" | "intel" | "level-zero" => Ok(Self::Ze),
            "hpu" | "habana" | "gaudi" | "synapse" => Ok(Self::Hpu),
            _ => bail!("Unknown device backend: {}", s),
        }
    }
}

/// Unified device context holding polymorphic implementation
pub struct DeviceContext {
    backend: DeviceBackend,
    device_id: u32,
    pub ops: Box<dyn DeviceContextOps>,
}

impl DeviceContext {
    /// Create a new device context for the specified backend and device
    pub fn new(backend: DeviceBackend, device_id: u32) -> Result<Self> {
        let ops: Box<dyn DeviceContextOps> = match backend {
            DeviceBackend::Cuda => {
                #[cfg(feature = "cuda")]
                { Box::new(cuda::CudaContext::new(device_id)?) }
                #[cfg(not(feature = "cuda"))]
                { bail!("CUDA backend not compiled (enable 'cuda' feature)") }
            }
            DeviceBackend::Ze => {
                #[cfg(feature = "xpu")]
                { Box::new(ze::ZeContext::new(device_id)?) }
                #[cfg(not(feature = "xpu"))]
                { bail!("Level-Zero backend not compiled (enable 'xpu' feature)") }
            }
            DeviceBackend::Hpu => {
                #[cfg(feature = "hpu")]
                { Box::new(hpu::HpuContext::new(device_id)?) }
                #[cfg(not(feature = "hpu"))]
                { bail!("Synapse backend not compiled (enable 'hpu' feature)") }
            }
        };

        Ok(Self {
            backend,
            device_id,
            ops,
        })
    }

    pub fn backend(&self) -> DeviceBackend {
        self.backend
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    // Delegate to trait object
    pub fn create_stream(&self) -> Result<DeviceStream> {
        let stream_ops = self.ops.create_stream()?;
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

/// Device stream wrapper
pub struct DeviceStream {
    backend: DeviceBackend,
    pub ops: Box<dyn DeviceStreamOps>,
}

impl DeviceStream {
    pub fn backend(&self) -> DeviceBackend {
        self.backend
    }

    pub fn copy_h2d(&self, dst: u64, src: &[u8]) -> Result<()> {
        self.ops.copy_h2d(dst, src)
    }

    pub fn copy_d2h(&self, dst: &mut [u8], src: u64) -> Result<()> {
        self.ops.copy_d2h(dst, src)
    }

    pub fn copy_d2d(&self, dst: u64, src: u64, size: usize) -> Result<()> {
        self.ops.copy_d2d(dst, src, size)
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

impl std::fmt::Debug for DeviceStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceStream")
            .field("backend", &self.backend)
            .finish()
    }
}

/// Device event wrapper
pub struct DeviceEvent {
    pub backend: DeviceBackend,
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
