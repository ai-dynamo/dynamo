// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the storage-next module.

use super::*;

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
    let erased: OwnedMemoryRegion = erase_storage(storage);

    assert_eq!(erased.size(), 1024);
    assert_eq!(erased.storage_kind(), StorageKind::System);
}

#[test]
fn test_memory_descriptor() {
    let desc = MemoryDescriptor::new(0x1000, 4096);
    assert_eq!(desc.addr, 0x1000);
    assert_eq!(desc.size, 4096);
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
}

// Tests for NIXL registration would require a real NIXL agent,
// so we'll skip those for now. In practice, you'd mock the agent
// or use integration tests.
#[cfg(all(feature = "testing-nixl", feature = "testing-cuda"))]
mod nixl_tests {
    use super::super::registered::register_with_nixl;
    use super::*;
    use nixl_sys::Agent as NixlAgent;

    // These tests would require a mock NIXL agent or real NIXL setup
    // Placeholder for now

    #[test]
    fn test_nixl_registration() {
        let pinned = PinnedStorage::new(2048).unwrap();
        let agent = NixlAgent::new("test_agent").unwrap();
        let registered = register_with_nixl(pinned, &agent, None).unwrap();
        assert_eq!(registered.agent_name(), "test_agent");
    }
}
