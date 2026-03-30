//! CUDA backend implementation
//!
//! Wraps cudarc types with the device abstraction traits.

use crate::block_manager::v2::device::traits::*;
use anyhow::{Result, Context as _};
use cudarc::driver::result as cuda_result;
use cudarc::driver::sys::{CUresult, CU_MEMHOSTALLOC_PORTABLE};
use cudarc::driver::DriverError;
use std::sync::Arc;

/// CUDA device context wrapping cudarc::CudaContext
#[derive(Debug)]
pub struct CudaContext {
    context: Arc<cudarc::driver::CudaContext>,
    device_id: u32,
}

impl CudaContext {
    pub fn new(device_id: u32) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(device_id as usize)
            .with_context(|| format!("Failed to create CUDA context for device {}", device_id))?;

        Ok(Self {
            context,
            device_id,
        })
    }

    /// Create from an existing CudaContext (for compatibility with existing code)
    pub fn from_context(context: Arc<cudarc::driver::CudaContext>, device_id: u32) -> Self {
        Self { context, device_id }
    }

    /// Get the underlying cudarc context
    pub fn inner(&self) -> &Arc<cudarc::driver::CudaContext> {
        &self.context
    }
}

impl DeviceContextOps for CudaContext {
    fn device_id(&self) -> u32 {
        self.device_id
    }

    fn create_stream(&self) -> Result<Box<dyn DeviceStreamOps>> {
        let stream = self.context.new_stream()
            .context("Failed to create CUDA stream")?;

        Ok(Box::new(CudaStreamWrapper {
            stream,
        }))
    }

    fn allocate_device(&self, size: usize) -> Result<u64> {
        self.context.bind_to_thread()?;
        let ptr = unsafe {
            cuda_result::malloc_sync(size)
                .context("Failed to allocate device memory")?
        };
        Ok(ptr)
    }

    fn free_device(&self, ptr: u64) -> Result<()> {
        // cudarc manages memory via RAII, but for explicit free we need to drop
        // In practice, this is handled by the allocator dropping the buffer
        // For now, we can't explicitly free without the buffer handle
        // This will be addressed when we integrate with the actual allocation system
        tracing::warn!("CUDA free_device called with raw pointer {} - memory managed by allocator", ptr);
        Ok(())
    }

    fn allocate_pinned(&self, size: usize) -> Result<u64> {
        // Use cudarc's pinned allocation
        unsafe {
            let ptr = cuda_result::malloc_host(size, CU_MEMHOSTALLOC_PORTABLE)
                .context("Failed to allocate pinned host memory")?;
            Ok(ptr as u64)
        }
    }

    fn free_pinned(&self, ptr: u64) -> Result<()> {
        unsafe {
            cuda_result::free_host(ptr as *mut std::ffi::c_void)
                .context("Failed to free pinned host memory")?;
        }
        Ok(())
    }

    fn bind_to_thread(&self) -> Result<()> {
        // CUDA contexts are automatically bound to threads by cudarc
        Ok(())
    }

    unsafe fn disable_event_tracking(&self) -> Result<()> {
        // Disable cudarc's automatic event tracking for manual event management
        self.context.disable_event_tracking();
        Ok(())
    }

    fn raw_handle(&self) -> Option<u64> {
        // Return the raw CUDA device ID
        Some(self.context.cu_device() as u64)
    }
}

/// CUDA stream wrapper
#[derive(Debug)]
pub struct CudaStreamWrapper {
    stream: Arc<cudarc::driver::CudaStream>,
}

impl CudaStreamWrapper {
    /// Get the underlying CUDA stream (temporary for Phase 4)
    /// TODO: Remove in Phase 5 when executor is generalized
    pub fn inner(&self) -> &Arc<cudarc::driver::CudaStream> {
        &self.stream
    }
}

impl DeviceStreamOps for CudaStreamWrapper {
    fn copy_h2d(&self, dst_device_ptr: u64, src_host_data: &[u8]) -> Result<()> {
        unsafe {
            cuda_result::memcpy_htod_async(
                dst_device_ptr,
                src_host_data,
                self.stream.cu_stream(),
            )
            .context("CUDA H2D copy failed")?;
        }
        Ok(())
    }

    fn copy_d2h(&self, dst_host_data: &mut [u8], src_device_ptr: u64) -> Result<()> {
        unsafe {
            cuda_result::memcpy_dtoh_async(
                dst_host_data,
                src_device_ptr,
                self.stream.cu_stream(),
            )
            .context("CUDA D2H copy failed")?;
        }
        Ok(())
    }

    fn copy_d2d(&self, dst_device_ptr: u64, src_device_ptr: u64, size: usize) -> Result<()> {
        unsafe {
            cuda_result::memcpy_dtod_async(
                dst_device_ptr,
                src_device_ptr,
                size,
                self.stream.cu_stream(),
            )
            .context("CUDA D2D copy failed")?;
        }
        Ok(())
    }

    fn record_event(&self) -> Result<Box<dyn DeviceEventOps>> {
        let event = self.stream.record_event(None)
            .context("Failed to record CUDA event")?;

        Ok(Box::new(CudaEventWrapper { event }))
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
            .context("CUDA stream synchronization failed")?;
        Ok(())
    }

    fn raw_handle(&self) -> Option<u64> {
        Some(self.stream.cu_stream() as u64)
    }
}

/// CUDA event wrapper
#[derive(Debug)]
pub struct CudaEventWrapper {
    pub event: cudarc::driver::CudaEvent,
}

impl DeviceEventOps for CudaEventWrapper {
    fn is_complete(&self) -> Result<bool> {
        unsafe {
            match cuda_result::event::query(self.event.cu_event()) {
                Ok(()) => Ok(true),
                Err(DriverError(CUresult::CUDA_ERROR_NOT_READY)) => Ok(false),
                Err(e) => Err(anyhow::anyhow!("CUDA event query failed: {:?}", e)),
            }
        }
    }

    fn synchronize(&self) -> Result<()> {
        self.event.synchronize()
            .context("CUDA event synchronization failed")?;
        Ok(())
    }

    fn raw_handle(&self) -> Option<u64> {
        Some(self.event.cu_event() as u64)
    }
}

/// Check if CUDA backend is available on this system
pub fn is_available() -> bool {
    // Check if we can enumerate CUDA devices
    match cuda_result::init() {
        Ok(()) => {
            match cuda_result::device::get_count() {
                Ok(count) => count > 0,
                Err(_) => false,
            }
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // Just check that the function doesn't panic
        let available = is_available();
        println!("CUDA available: {}", available);
    }

    #[test]
    #[cfg(feature = "testing-cuda")]
    fn test_cuda_context_creation() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        assert_eq!(ctx.device_id(), 0);
    }

    #[test]
    #[cfg(feature = "testing-cuda")]
    fn test_cuda_stream_creation() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.create_stream().expect("Failed to create stream");

        // Test synchronize
        stream.synchronize().expect("Failed to synchronize stream");
    }

    #[test]
    #[cfg(feature = "testing-cuda")]
    fn test_cuda_event_creation() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.create_stream().expect("Failed to create stream");

        let event = stream.record_event().expect("Failed to record event");

        // Event should complete quickly for an empty stream
        event.synchronize().expect("Failed to synchronize event");
        assert!(event.is_complete().expect("Failed to query event"));
    }

    #[test]
    #[cfg(feature = "testing-cuda")]
    fn test_cuda_h2d_d2h() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.create_stream().expect("Failed to create stream");

        // Allocate device memory
        let size = 1024;
        let dev_ptr = ctx.allocate_device(size).expect("Failed to allocate device memory");

        // Prepare host data
        let host_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let mut host_result = vec![0u8; size];

        // Copy H2D
        stream.copy_h2d(dev_ptr, &host_data).expect("H2D copy failed");

        // Copy D2H
        stream.copy_d2h(&mut host_result, dev_ptr).expect("D2H copy failed");

        // Synchronize
        stream.synchronize().expect("Failed to synchronize");

        // Verify data
        assert_eq!(host_data, host_result, "Data mismatch after H2D->D2H roundtrip");
    }

    #[test]
    #[cfg(feature = "testing-cuda")]
    fn test_cuda_d2d() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.create_stream().expect("Failed to create stream");

        let size = 1024;
        let dev_ptr1 = ctx.allocate_device(size).expect("Failed to allocate device memory 1");
        let dev_ptr2 = ctx.allocate_device(size).expect("Failed to allocate device memory 2");

        // Prepare host data
        let host_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let mut host_result = vec![0u8; size];

        // Copy H2D to first buffer
        stream.copy_h2d(dev_ptr1, &host_data).expect("H2D copy failed");

        // Copy D2D
        stream.copy_d2d(dev_ptr2, dev_ptr1, size).expect("D2D copy failed");

        // Copy D2H from second buffer
        stream.copy_d2h(&mut host_result, dev_ptr2).expect("D2H copy failed");

        // Synchronize
        stream.synchronize().expect("Failed to synchronize");

        // Verify data
        assert_eq!(host_data, host_result, "Data mismatch after H2D->D2D->D2H");
    }

    #[test]
    #[cfg(feature = "testing-cuda")]
    fn test_pinned_memory() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let size = 4096;
        let pinned_ptr = ctx.allocate_pinned(size).expect("Failed to allocate pinned memory");

        // Verify we can write to it
        unsafe {
            let slice = std::slice::from_raw_parts_mut(pinned_ptr as *mut u8, size);
            slice.fill(42);
            assert_eq!(slice[0], 42);
        }

        ctx.free_pinned(pinned_ptr).expect("Failed to free pinned memory");
    }
}
