// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the storage-next module.

use super::*;

// ---------------------------------------------------------------------------
// Test-only DeviceAllocator — auto-selects Ze (XPU) or CUDA backend
// ---------------------------------------------------------------------------

/// Create a test DeviceAllocator: prefers Ze if available, falls back to CUDA.
#[cfg(any(feature = "testing-cuda", feature = "testing-ze"))]
fn test_device_ctx() -> std::sync::Arc<dyn DeviceAllocator> {
    #[cfg(feature = "testing-ze")]
    {
        if let Ok(alloc) = TestZeAllocator::new(0) {
            return std::sync::Arc::new(alloc);
        }
    }
    #[cfg(feature = "testing-cuda")]
    {
        return std::sync::Arc::new(
            TestCudaAllocator::new(0).expect("CUDA device 0 required for tests"),
        );
    }
    #[cfg(not(feature = "testing-cuda"))]
    {
        panic!("No device backend available for tests (need testing-cuda or testing-ze)");
    }
}

// ---- CUDA test allocator ----

#[cfg(feature = "testing-cuda")]
#[derive(Debug)]
struct TestCudaAllocator {
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    device_id: u32,
}

#[cfg(feature = "testing-cuda")]
impl TestCudaAllocator {
    fn new(device_id: u32) -> anyhow::Result<Self> {
        let ctx = cudarc::driver::CudaContext::new(device_id as usize)?;
        Ok(Self { ctx, device_id })
    }
}

#[cfg(feature = "testing-cuda")]
impl DeviceAllocator for TestCudaAllocator {
    fn allocate_device(&self, size: usize) -> Result<u64> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| StorageError::AllocationFailed(e.to_string()))?;
        unsafe { cudarc::driver::result::malloc_sync(size) }
            .map_err(|e| StorageError::AllocationFailed(e.to_string()))
    }

    fn free_device(&self, ptr: u64) -> Result<()> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;
        unsafe { cudarc::driver::result::free_sync(ptr) }
            .map_err(|e| StorageError::OperationFailed(e.to_string()))
    }

    fn allocate_pinned(&self, size: usize) -> Result<u64> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| StorageError::AllocationFailed(e.to_string()))?;
        unsafe { cudarc::driver::result::malloc_host(size, 0) }
            .map(|ptr| ptr as u64)
            .map_err(|e| StorageError::AllocationFailed(e.to_string()))
    }

    fn free_pinned(&self, ptr: u64) -> Result<()> {
        unsafe { cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void) }
            .map_err(|e| StorageError::OperationFailed(e.to_string()))
    }

    fn device_id(&self) -> u32 {
        self.device_id
    }
}

// ---- Level-Zero (XPU) test allocator ----

#[cfg(feature = "testing-ze")]
struct TestZeAllocator {
    context: std::sync::Arc<level_zero::Context>,
    device: level_zero::Device,
    device_id: u32,
    /// Keep host buffers alive (Drop frees via zeMemFree).
    host_buffers: std::sync::Mutex<std::collections::HashMap<u64, level_zero::HostBuffer>>,
    /// Keep device buffers alive.
    device_buffers: std::sync::Mutex<std::collections::HashMap<u64, level_zero::DeviceBuffer>>,
}

#[cfg(feature = "testing-ze")]
impl std::fmt::Debug for TestZeAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestZeAllocator")
            .field("device_id", &self.device_id)
            .finish()
    }
}

// SAFETY: Device wraps a ze_device_handle_t which is process-global and immutable
// after discovery. All mutable state is behind Mutex.
#[cfg(feature = "testing-ze")]
unsafe impl Send for TestZeAllocator {}
#[cfg(feature = "testing-ze")]
unsafe impl Sync for TestZeAllocator {}

#[cfg(feature = "testing-ze")]
impl TestZeAllocator {
    fn new(device_id: u32) -> anyhow::Result<Self> {
        let drivers = level_zero::drivers()?;
        let driver = drivers.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("No Level-Zero drivers found"))?;
        let devices = driver.devices()?;
        let device = devices.into_iter().nth(device_id as usize)
            .ok_or_else(|| anyhow::anyhow!("Level-Zero device {} not found", device_id))?;
        let context = std::sync::Arc::new(level_zero::Context::create(&driver)?);
        Ok(Self {
            context,
            device,
            device_id,
            host_buffers: std::sync::Mutex::new(std::collections::HashMap::new()),
            device_buffers: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }
}

#[cfg(feature = "testing-ze")]
impl DeviceAllocator for TestZeAllocator {
    fn allocate_device(&self, size: usize) -> Result<u64> {
        let buffer = self.context.alloc_device(&self.device, size, 1)
            .map_err(|e| StorageError::AllocationFailed(format!("Ze device alloc: {:?}", e)))?;
        let ptr = buffer.as_mut_ptr() as u64;
        self.device_buffers.lock().unwrap().insert(ptr, buffer);
        Ok(ptr)
    }

    fn free_device(&self, ptr: u64) -> Result<()> {
        self.device_buffers.lock().unwrap().remove(&ptr);
        Ok(())
    }

    fn allocate_pinned(&self, size: usize) -> Result<u64> {
        let buffer = self.context.alloc_host(size, 1)
            .map_err(|e| StorageError::AllocationFailed(format!("Ze host alloc: {:?}", e)))?;
        let ptr = buffer.as_mut_ptr() as u64;
        self.host_buffers.lock().unwrap().insert(ptr, buffer);
        Ok(ptr)
    }

    fn free_pinned(&self, ptr: u64) -> Result<()> {
        self.host_buffers.lock().unwrap().remove(&ptr);
        Ok(())
    }

    fn device_id(&self) -> u32 {
        self.device_id
    }
}

/// Helper function to validate NIXL descriptor consistency.
///
/// For any MemoryDescriptor that returns Some from nixl_descriptor(),
/// this validates that the descriptor's addr and size match the memory region's addr and size.
///
/// # Panics
/// Panics if descriptor values don't match memory region values.
#[allow(dead_code)]
fn validate_nixl_descriptor<M: MemoryDescriptor>(memory: &M) {
    if let Some(desc) = memory.nixl_descriptor() {
        assert_eq!(
            desc.addr as usize,
            memory.addr(),
            "NIXL descriptor addr ({}) does not match memory region addr ({})",
            desc.addr,
            memory.addr()
        );
        assert_eq!(
            desc.size,
            memory.size(),
            "NIXL descriptor size ({}) does not match memory region size ({})",
            desc.size,
            memory.size()
        );
    }
}

// ========== StorageKind tests ==========

#[test]
fn test_storage_kind_cuda_device_index_device() {
    let kind = StorageKind::Device(3);
    assert_eq!(kind.cuda_device_index(), Some(3));
}

#[test]
fn test_storage_kind_cuda_device_index_system() {
    let kind = StorageKind::System;
    assert_eq!(kind.cuda_device_index(), None);
}

#[test]
fn test_storage_kind_cuda_device_index_pinned() {
    let kind = StorageKind::Pinned;
    assert_eq!(kind.cuda_device_index(), None);
}

#[test]
fn test_storage_kind_cuda_device_index_disk() {
    let kind = StorageKind::Disk(123);
    assert_eq!(kind.cuda_device_index(), None);
}

#[test]
fn test_storage_kind_is_cuda() {
    assert!(StorageKind::Device(0).is_cuda());
    assert!(!StorageKind::System.is_cuda());
    assert!(!StorageKind::Pinned.is_cuda());
    assert!(!StorageKind::Disk(1).is_cuda());
}

#[test]
fn test_storage_kind_is_system() {
    assert!(StorageKind::System.is_system());
    assert!(!StorageKind::Device(0).is_system());
    assert!(!StorageKind::Pinned.is_system());
    assert!(!StorageKind::Disk(1).is_system());
}

#[test]
fn test_storage_kind_is_pinned() {
    assert!(StorageKind::Pinned.is_pinned());
    assert!(!StorageKind::System.is_pinned());
    assert!(!StorageKind::Device(0).is_pinned());
    assert!(!StorageKind::Disk(1).is_pinned());
}

#[test]
fn test_storage_kind_is_disk() {
    assert!(StorageKind::Disk(1).is_disk());
    assert!(!StorageKind::System.is_disk());
    assert!(!StorageKind::Pinned.is_disk());
    assert!(!StorageKind::Device(0).is_disk());
}

// ========== Buffer tests ==========

#[test]
fn test_buffer_new() {
    let storage = SystemStorage::new(1024).unwrap();
    let buffer = Buffer::new(storage);
    assert_eq!(buffer.size(), 1024);
    assert_eq!(buffer.storage_kind(), StorageKind::System);
}

#[test]
fn test_buffer_from_arc() {
    use std::sync::Arc;
    let storage = SystemStorage::new(2048).unwrap();
    let arc: Arc<dyn MemoryDescriptor> = Arc::new(storage);
    let buffer = Buffer::from_arc(arc);
    assert_eq!(buffer.size(), 2048);
}

#[test]
fn test_buffer_from_impl() {
    use std::sync::Arc;
    let storage = SystemStorage::new(512).unwrap();
    let arc: Arc<dyn MemoryDescriptor> = Arc::new(storage);
    let buffer: Buffer = arc.into();
    assert_eq!(buffer.size(), 512);
}

#[test]
fn test_buffer_deref() {
    let storage = SystemStorage::new(1024).unwrap();
    let buffer = Buffer::new(storage);
    // Deref allows calling MemoryDescriptor methods directly
    let size = buffer.size();
    assert_eq!(size, 1024);
}

#[test]
fn test_buffer_debug() {
    let storage = SystemStorage::new(1024).unwrap();
    let buffer = Buffer::new(storage);
    let debug_str = format!("{:?}", buffer);
    assert!(debug_str.contains("Buffer"));
    assert!(debug_str.contains("size"));
    assert!(debug_str.contains("addr"));
}

#[test]
fn test_buffer_clone() {
    let storage = SystemStorage::new(1024).unwrap();
    let buffer = Buffer::new(storage);
    let cloned = buffer.clone();
    assert_eq!(buffer.addr(), cloned.addr());
    assert_eq!(buffer.size(), cloned.size());
}

// ========== MemoryRegion tests ==========

#[test]
fn test_memory_region_new() {
    let region = MemoryRegion::new(0x1000, 4096);
    assert_eq!(region.addr, 0x1000);
    assert_eq!(region.size, 4096);
}

#[test]
fn test_memory_region_accessors() {
    let region = MemoryRegion::new(0x2000, 8192);
    assert_eq!(region.addr(), 0x2000);
    assert_eq!(region.size(), 8192);
}

#[test]
fn test_memory_region_zero_address() {
    let region = MemoryRegion::new(0, 1024);
    assert_eq!(region.addr(), 0);
    assert_eq!(region.size(), 1024);
}

#[test]
fn test_memory_region_zero_size() {
    let region = MemoryRegion::new(0x1000, 0);
    assert_eq!(region.addr(), 0x1000);
    assert_eq!(region.size(), 0);
}

#[test]
fn test_memory_region_clone() {
    let region = MemoryRegion::new(0x3000, 2048);
    let cloned = region;
    assert_eq!(region.addr(), cloned.addr());
    assert_eq!(region.size(), cloned.size());
}

#[test]
fn test_memory_region_eq() {
    let region1 = MemoryRegion::new(0x1000, 4096);
    let region2 = MemoryRegion::new(0x1000, 4096);
    let region3 = MemoryRegion::new(0x2000, 4096);
    assert_eq!(region1, region2);
    assert_ne!(region1, region3);
}

#[test]
fn test_memory_region_debug() {
    let region = MemoryRegion::new(0x1000, 4096);
    let debug_str = format!("{:?}", region);
    assert!(debug_str.contains("MemoryRegion"));
}

// ========== create_buffer helper tests ==========

#[test]
fn test_create_buffer_helper() {
    let storage = SystemStorage::new(1024).unwrap();
    let buffer = create_buffer(storage);
    assert_eq!(buffer.size(), 1024);
    assert_eq!(buffer.storage_kind(), StorageKind::System);
}

// ========== Original tests ==========

#[test]
fn test_system_storage() {
    let storage = SystemStorage::new(1024).unwrap();
    assert_eq!(storage.size(), 1024);
    assert_eq!(storage.storage_kind(), StorageKind::System);
    assert!(storage.addr() != 0);

    // Test that we can create multiple allocations
    let storage2 = SystemStorage::new(2048).unwrap();
    assert_eq!(storage2.size(), 2048);
    assert_ne!(storage.addr(), storage2.addr());
}

#[test]
fn test_system_storage_zero_size() {
    let result = SystemStorage::new(0);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        StorageError::AllocationFailed(_)
    ));
}

#[cfg(target_os = "linux")]
#[test]
fn test_disk_storage_temp() {
    let storage = DiskStorage::new(4096).unwrap();
    assert_eq!(storage.size(), 4096);
    assert!(matches!(storage.storage_kind(), StorageKind::Disk(_)));
    // Disk storage is file-backed, so addr() returns 0 (no memory address)
    assert_eq!(storage.addr(), 0);
    assert!(storage.path().exists());
}

#[cfg(target_os = "linux")]
#[test]
fn test_disk_storage_at_path() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("test.bin");

    let storage = DiskStorage::new_at(&path, 8192).unwrap();
    assert_eq!(storage.size(), 8192);
    assert!(matches!(storage.storage_kind(), StorageKind::Disk(_)));
    assert!(path.exists());
}

#[test]
fn test_type_erasure() {
    let storage = SystemStorage::new(1024).unwrap();
    let buffer = create_buffer(storage);

    assert_eq!(buffer.size(), 1024);
    assert_eq!(buffer.storage_kind(), StorageKind::System);
}

#[test]
fn test_memory_descriptor() {
    let desc = MemoryRegion::new(0x1000, 4096);
    assert_eq!(desc.addr, 0x1000);
    assert_eq!(desc.size, 4096);
}

#[test]
fn test_system_storage_unregistered_no_nixl_descriptor() {
    let storage = SystemStorage::new(1024).unwrap();
    assert!(storage.nixl_descriptor().is_none());
}

#[cfg(target_os = "linux")]
#[test]
fn test_disk_storage_unregistered_no_nixl_descriptor() {
    let storage = DiskStorage::new(4096).unwrap();
    assert!(storage.nixl_descriptor().is_none());
}

#[cfg(any(feature = "testing-cuda", feature = "testing-ze"))]
mod device_tests {
    use super::*;

    #[test]
    fn test_pinned_storage() {
        let storage = PinnedStorage::new(2048, test_device_ctx()).unwrap();
        assert_eq!(storage.size(), 2048);
        assert_eq!(storage.storage_kind(), StorageKind::Pinned);
        assert!(storage.addr() != 0);
    }

    #[test]
    fn test_pinned_storage_zero_size() {
        let storage = PinnedStorage::new(0, test_device_ctx());
        assert!(storage.is_err());
        assert!(matches!(
            storage.unwrap_err(),
            StorageError::AllocationFailed(_)
        ));
    }

    #[test]
    fn test_device_storage() {
        let storage = DeviceStorage::new(4096, test_device_ctx()).unwrap();
        assert_eq!(storage.size(), 4096);
        assert_eq!(storage.storage_kind(), StorageKind::Device(0));
        assert!(storage.addr() != 0);
        assert_eq!(storage.device_id(), 0);
    }

    #[test]
    fn test_device_storage_zero_size() {
        let result = DeviceStorage::new(0, test_device_ctx());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StorageError::AllocationFailed(_)
        ));
    }

    #[test]
    fn test_pinned_storage_unregistered_no_nixl_descriptor() {
        let storage = PinnedStorage::new(1024, test_device_ctx()).unwrap();
        assert!(storage.nixl_descriptor().is_none());
    }

    #[test]
    fn test_device_storage_unregistered_no_nixl_descriptor() {
        let storage = DeviceStorage::new(4096, test_device_ctx()).unwrap();
        assert!(storage.nixl_descriptor().is_none());
    }
}

#[cfg(feature = "testing-nixl")]
mod nixl_tests {
    use super::super::nixl::{NixlAgent, RegisteredView, register_with_nixl};
    use super::*;

    // System Storage Tests
    #[test]
    fn test_system_storage_registration() {
        let storage = SystemStorage::new(2048).unwrap();
        let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
        let registered = register_with_nixl(storage, &agent, None).unwrap();

        assert_eq!(registered.agent_name(), "test_agent");
        assert_eq!(registered.size(), 2048);
        assert_eq!(registered.storage_kind(), StorageKind::System);
        assert!(registered.is_registered());
    }

    #[test]
    fn test_system_storage_descriptor_consistency() {
        let storage = SystemStorage::new(1024).unwrap();
        let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
        let registered = register_with_nixl(storage, &agent, None).unwrap();

        // Validate descriptor consistency
        validate_nixl_descriptor(&registered);

        // Get descriptor and validate fields
        let desc = registered.descriptor();
        assert_eq!(desc.addr as usize, registered.addr());
        assert_eq!(desc.size, registered.size());
        assert_eq!(desc.mem_type, nixl_sys::MemType::Dram);
        assert_eq!(desc.device_id, 0);
    }

    // Note: into_storage() test removed due to implementation issue
    // The current implementation uses mem::zeroed() which is invalid for types with NonNull
    // TODO: Fix NixlRegistered::into_storage() implementation

    // Disk Storage Tests (Linux only)
    #[cfg(target_os = "linux")]
    #[test]
    fn test_disk_storage_registration() {
        let storage = DiskStorage::new(4096).unwrap();
        let agent = NixlAgent::with_backends("test_agent", &["POSIX"]).unwrap();
        let registered = register_with_nixl(storage, &agent, None).unwrap();

        assert_eq!(registered.agent_name(), "test_agent");
        assert_eq!(registered.size(), 4096);
        assert!(matches!(registered.storage_kind(), StorageKind::Disk(_)));
        assert!(registered.is_registered());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_disk_storage_descriptor_consistency() {
        let storage = DiskStorage::new(8192).unwrap();
        let agent = NixlAgent::with_backends("test_agent", &["POSIX"]).unwrap();
        let registered = register_with_nixl(storage, &agent, None).unwrap();

        // Validate descriptor consistency
        validate_nixl_descriptor(&registered);

        // Get descriptor and validate fields
        let desc = registered.descriptor();
        assert_eq!(desc.size, registered.size());
        assert_eq!(desc.mem_type, nixl_sys::MemType::File);
    }

    // CUDA tests (when both testing-nixl and testing-cuda are enabled)
    #[cfg(any(feature = "testing-all", all(feature = "testing-nixl", feature = "testing-ze")))]
    mod device_nixl_tests {
        use super::*;

        #[test]
        fn test_pinned_storage_registration() {
            let storage = PinnedStorage::new(2048, test_device_ctx()).unwrap();
            let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
            let registered = register_with_nixl(storage, &agent, None).unwrap();

            assert_eq!(registered.agent_name(), "test_agent");
            assert_eq!(registered.size(), 2048);
            assert_eq!(registered.storage_kind(), StorageKind::Pinned);
            assert!(registered.is_registered());
        }

        #[test]
        fn test_pinned_storage_descriptor_consistency() {
            let storage = PinnedStorage::new(1024, test_device_ctx()).unwrap();
            let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
            let registered = register_with_nixl(storage, &agent, None).unwrap();

            // Validate descriptor consistency
            validate_nixl_descriptor(&registered);

            // Get descriptor and validate fields
            let desc = registered.descriptor();
            assert_eq!(desc.addr as usize, registered.addr());
            assert_eq!(desc.size, registered.size());
            assert_eq!(desc.mem_type, nixl_sys::MemType::Dram);
        }

        #[test]
        fn test_device_storage_registration() {
            let storage = DeviceStorage::new(4096, test_device_ctx()).unwrap();
            let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
            let registered = register_with_nixl(storage, &agent, None).unwrap();

            assert_eq!(registered.agent_name(), "test_agent");
            assert_eq!(registered.size(), 4096);
            assert_eq!(registered.storage_kind(), StorageKind::Device(0));
            assert!(registered.is_registered());
        }

        #[test]
        fn test_device_storage_descriptor_consistency() {
            let storage = DeviceStorage::new(2048, test_device_ctx()).unwrap();
            let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
            let registered = register_with_nixl(storage, &agent, None).unwrap();

            // Validate descriptor consistency
            validate_nixl_descriptor(&registered);

            // Get descriptor and validate fields
            let desc = registered.descriptor();
            assert_eq!(desc.addr as usize, registered.addr());
            assert_eq!(desc.size, registered.size());
            assert_eq!(desc.mem_type, nixl_sys::MemType::Vram);
            assert_eq!(desc.device_id, 0);
        }
    }

    // Type Erasure Tests
    #[test]
    fn test_type_erasure_preserves_nixl_descriptor() {
        let storage = SystemStorage::new(1024).unwrap();
        let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
        let registered = register_with_nixl(storage, &agent, None).unwrap();

        let buffer = create_buffer(registered);

        // Validate descriptor through type erasure
        validate_nixl_descriptor(&buffer);

        // Verify descriptor is Some and has correct values
        let desc = buffer.nixl_descriptor().unwrap();
        assert_eq!(desc.addr as usize, buffer.addr());
        assert_eq!(desc.size, buffer.size());
    }

    #[cfg(any(feature = "testing-cuda", feature = "testing-ze"))]
    #[test]
    fn test_type_erasure_pinned_storage() {
        let storage = PinnedStorage::new(2048, test_device_ctx()).unwrap();
        let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
        let registered = register_with_nixl(storage, &agent, None).unwrap();

        let buffer = create_buffer(registered);

        validate_nixl_descriptor(&buffer);
        assert_eq!(buffer.storage_kind(), StorageKind::Pinned);
    }

    #[cfg(any(feature = "testing-cuda", feature = "testing-ze"))]
    #[test]
    fn test_type_erasure_device_storage() {
        let storage = DeviceStorage::new(4096, test_device_ctx()).unwrap();
        let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
        let registered = register_with_nixl(storage, &agent, None).unwrap();

        let buffer = create_buffer(registered);

        validate_nixl_descriptor(&buffer);
        assert_eq!(buffer.storage_kind(), StorageKind::Device(0));
    }
}

// Arena allocator tests with NIXL registration
#[cfg(feature = "testing-nixl")]
mod arena_nixl_tests {
    use super::super::arena::ArenaAllocator;
    use super::super::nixl::{NixlAgent, register_with_nixl};
    use super::*;

    const PAGE_SIZE: usize = 4096;
    const PAGE_COUNT: usize = 10;
    const TOTAL_SIZE: usize = PAGE_SIZE * PAGE_COUNT;

    #[test]
    fn test_arena_with_registered_storage_single_allocation() {
        let storage = SystemStorage::new(TOTAL_SIZE).unwrap();
        let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
        let registered = register_with_nixl(storage, &agent, None).unwrap();
        let base_addr = registered.addr();

        let allocator = ArenaAllocator::new(registered, PAGE_SIZE).unwrap();
        let buffer = allocator.allocate(PAGE_SIZE * 2).unwrap();

        // Validate buffer properties
        assert_eq!(buffer.size(), PAGE_SIZE * 2);
        assert_eq!(buffer.addr(), base_addr); // First allocation starts at base
        assert_eq!(buffer.agent_name(), "test_agent");

        // Validate descriptor
        let desc = buffer.registered_descriptor();
        assert_eq!(desc.addr as usize, buffer.addr());
        assert_eq!(desc.size, buffer.size());
    }

    #[test]
    fn test_arena_with_registered_storage_multiple_allocations() {
        let storage = SystemStorage::new(TOTAL_SIZE).unwrap();
        let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
        let registered = register_with_nixl(storage, &agent, None).unwrap();
        let base_addr = registered.addr();

        let allocator = ArenaAllocator::new(registered, PAGE_SIZE).unwrap();

        // Allocate three buffers
        let buffer1 = allocator.allocate(PAGE_SIZE).unwrap();
        let buffer2 = allocator.allocate(PAGE_SIZE * 2).unwrap();
        let buffer3 = allocator.allocate(PAGE_SIZE).unwrap();

        // Validate first buffer (starts at base, uses 1 page)
        assert_eq!(buffer1.size(), PAGE_SIZE);
        assert_eq!(buffer1.addr(), base_addr);

        // Validate second buffer (starts after buffer1, uses 2 pages)
        assert_eq!(buffer2.size(), PAGE_SIZE * 2);
        assert_eq!(buffer2.addr(), base_addr + PAGE_SIZE);

        // Validate third buffer (starts after buffer2, uses 1 page)
        assert_eq!(buffer3.size(), PAGE_SIZE);
        assert_eq!(buffer3.addr(), base_addr + PAGE_SIZE * 3);

        // Validate descriptors for all buffers
        let desc1 = buffer1.registered_descriptor();
        assert_eq!(desc1.addr as usize, buffer1.addr());
        assert_eq!(desc1.size, PAGE_SIZE);

        let desc2 = buffer2.registered_descriptor();
        assert_eq!(desc2.addr as usize, buffer2.addr());
        assert_eq!(desc2.size, PAGE_SIZE * 2);

        let desc3 = buffer3.registered_descriptor();
        assert_eq!(desc3.addr as usize, buffer3.addr());
        assert_eq!(desc3.size, PAGE_SIZE);
    }

    #[test]
    fn test_arena_buffer_agent_name_preservation() {
        let storage = SystemStorage::new(TOTAL_SIZE).unwrap();
        let agent = NixlAgent::with_backends("my_special_agent", &["UCX"]).unwrap();
        let registered = register_with_nixl(storage, &agent, None).unwrap();

        let allocator = ArenaAllocator::new(registered, PAGE_SIZE).unwrap();
        let buffer = allocator.allocate(PAGE_SIZE).unwrap();

        assert_eq!(buffer.agent_name(), "my_special_agent");
    }

    #[test]
    fn test_arena_multiple_buffers_stress_test() {
        let storage = SystemStorage::new(TOTAL_SIZE).unwrap();
        let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
        let registered = register_with_nixl(storage, &agent, None).unwrap();
        let base_addr = registered.addr();

        let allocator = ArenaAllocator::new(registered, PAGE_SIZE).unwrap();

        // Allocate 10 single-page buffers
        let mut buffers = Vec::new();
        for i in 0..10 {
            let buffer = allocator.allocate(PAGE_SIZE).unwrap();
            assert_eq!(buffer.size(), PAGE_SIZE);
            assert_eq!(buffer.addr(), base_addr + i * PAGE_SIZE);

            // Validate descriptor
            let desc = buffer.registered_descriptor();
            assert_eq!(desc.addr as usize, buffer.addr());
            assert_eq!(desc.size, PAGE_SIZE);

            buffers.push(buffer);
        }
    }

    #[test]
    fn test_arena_reallocation_after_drop() {
        let storage = SystemStorage::new(TOTAL_SIZE).unwrap();
        let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
        let registered = register_with_nixl(storage, &agent, None).unwrap();
        let base_addr = registered.addr();

        let allocator = ArenaAllocator::new(registered, PAGE_SIZE).unwrap();

        // Allocate and drop
        {
            let buffer = allocator.allocate(PAGE_SIZE * 5).unwrap();
            assert_eq!(buffer.addr(), base_addr);

            let desc = buffer.registered_descriptor();
            assert_eq!(desc.addr as usize, base_addr);
            assert_eq!(desc.size, PAGE_SIZE * 5);
        } // buffer dropped here

        // Reallocate same size - should reuse the space
        let buffer2 = allocator.allocate(PAGE_SIZE * 5).unwrap();
        assert_eq!(buffer2.addr(), base_addr);

        // Validate new descriptor
        let desc2 = buffer2.registered_descriptor();
        assert_eq!(desc2.addr as usize, base_addr);
        assert_eq!(desc2.size, PAGE_SIZE * 5);
    }

    #[cfg(any(feature = "testing-cuda", feature = "testing-ze"))]
    mod device_arena_tests {
        use super::*;

        #[test]
        fn test_arena_with_pinned_storage() {
            let storage = PinnedStorage::new(TOTAL_SIZE, test_device_ctx()).unwrap();
            let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
            let registered = register_with_nixl(storage, &agent, None).unwrap();

            let allocator = ArenaAllocator::new(registered, PAGE_SIZE).unwrap();
            let buffer = allocator.allocate(PAGE_SIZE * 2).unwrap();

            assert_eq!(buffer.size(), PAGE_SIZE * 2);
            assert_eq!(buffer.agent_name(), "test_agent");

            let desc = buffer.registered_descriptor();
            assert_eq!(desc.addr as usize, buffer.addr());
            assert_eq!(desc.size, PAGE_SIZE * 2);
            assert_eq!(desc.mem_type, nixl_sys::MemType::Dram);
        }

        #[test]
        fn test_arena_with_device_storage() {
            let storage = DeviceStorage::new(TOTAL_SIZE, test_device_ctx()).unwrap();
            let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
            let registered = register_with_nixl(storage, &agent, None).unwrap();

            let allocator = ArenaAllocator::new(registered, PAGE_SIZE).unwrap();
            let buffer = allocator.allocate(PAGE_SIZE * 3).unwrap();

            assert_eq!(buffer.size(), PAGE_SIZE * 3);
            assert_eq!(buffer.agent_name(), "test_agent");

            let desc = buffer.registered_descriptor();
            assert_eq!(desc.addr as usize, buffer.addr());
            assert_eq!(desc.size, PAGE_SIZE * 3);
            assert_eq!(desc.mem_type, nixl_sys::MemType::Vram);
            assert_eq!(desc.device_id, 0);
        }
    }
}
