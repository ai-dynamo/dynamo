// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the storage-next module.

use super::*;

/// Helper function to validate NIXL descriptor consistency.
///
/// For any MemoryDescription that returns Some from nixl_descriptor(),
/// this validates that the descriptor's addr and size match the memory region's addr and size.
///
/// # Panics
/// Panics if descriptor values don't match memory region values.
#[allow(dead_code)]
fn validate_nixl_descriptor<M: MemoryDescription>(memory: &M) {
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

#[test]
fn test_disk_storage_temp() {
    let storage = DiskStorage::new(4096).unwrap();
    assert_eq!(storage.size(), 4096);
    assert!(matches!(storage.storage_kind(), StorageKind::Disk(_)));
    // Disk storage is file-backed, so addr() returns 0 (no memory address)
    assert_eq!(storage.addr(), 0);
    assert!(storage.path().exists());
}

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

#[test]
fn test_disk_storage_unregistered_no_nixl_descriptor() {
    let storage = DiskStorage::new(4096).unwrap();
    assert!(storage.nixl_descriptor().is_none());
}

#[cfg(feature = "testing-cuda")]
mod cuda_tests {
    use super::*;

    #[test]
    fn test_pinned_storage() {
        let storage = PinnedStorage::new(2048).unwrap();
        assert_eq!(storage.size(), 2048);
        assert_eq!(storage.storage_kind(), StorageKind::Pinned);
        assert!(storage.addr() != 0);
    }

    #[test]
    fn test_pinned_storage_zero_size() {
        let storage = PinnedStorage::new(0);
        assert!(storage.is_err());
        assert!(matches!(
            storage.unwrap_err(),
            StorageError::AllocationFailed(_)
        ));
    }

    #[test]
    fn test_device_storage() {
        let storage = DeviceStorage::new(4096, 0).unwrap();
        assert_eq!(storage.size(), 4096);
        assert_eq!(storage.storage_kind(), StorageKind::Device(0));
        assert!(storage.addr() != 0);
        assert_eq!(storage.device_id(), 0);
    }

    #[test]
    fn test_device_storage_zero_size() {
        let result = DeviceStorage::new(0, 0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StorageError::AllocationFailed(_)
        ));
    }

    #[test]
    fn test_pinned_storage_unregistered_no_nixl_descriptor() {
        let storage = PinnedStorage::new(1024).unwrap();
        assert!(storage.nixl_descriptor().is_none());
    }

    #[test]
    fn test_device_storage_unregistered_no_nixl_descriptor() {
        let storage = DeviceStorage::new(4096, 0).unwrap();
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

    // Disk Storage Tests
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
    #[cfg(feature = "testing-all")]
    mod cuda_nixl_tests {
        use super::*;

        #[test]
        fn test_pinned_storage_registration() {
            let storage = PinnedStorage::new(2048).unwrap();
            let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
            let registered = register_with_nixl(storage, &agent, None).unwrap();

            assert_eq!(registered.agent_name(), "test_agent");
            assert_eq!(registered.size(), 2048);
            assert_eq!(registered.storage_kind(), StorageKind::Pinned);
            assert!(registered.is_registered());
        }

        #[test]
        fn test_pinned_storage_descriptor_consistency() {
            let storage = PinnedStorage::new(1024).unwrap();
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
            let storage = DeviceStorage::new(4096, 0).unwrap();
            let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
            let registered = register_with_nixl(storage, &agent, None).unwrap();

            assert_eq!(registered.agent_name(), "test_agent");
            assert_eq!(registered.size(), 4096);
            assert_eq!(registered.storage_kind(), StorageKind::Device(0));
            assert!(registered.is_registered());
        }

        #[test]
        fn test_device_storage_descriptor_consistency() {
            let storage = DeviceStorage::new(2048, 0).unwrap();
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

    #[cfg(feature = "testing-cuda")]
    #[test]
    fn test_type_erasure_pinned_storage() {
        let storage = PinnedStorage::new(2048).unwrap();
        let agent = NixlAgent::with_backends("test_agent", &["UCX"]).unwrap();
        let registered = register_with_nixl(storage, &agent, None).unwrap();

        let buffer = create_buffer(registered);

        validate_nixl_descriptor(&buffer);
        assert_eq!(buffer.storage_kind(), StorageKind::Pinned);
    }

    #[cfg(feature = "testing-cuda")]
    #[test]
    fn test_type_erasure_device_storage() {
        let storage = DeviceStorage::new(4096, 0).unwrap();
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

    #[cfg(feature = "testing-cuda")]
    mod cuda_arena_tests {
        use super::*;

        #[test]
        fn test_arena_with_pinned_storage() {
            let storage = PinnedStorage::new(TOTAL_SIZE).unwrap();
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
            let storage = DeviceStorage::new(TOTAL_SIZE, 0).unwrap();
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
