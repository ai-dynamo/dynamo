// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the storage module (re-exported from dynamo-memory).

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
    use crate::block_manager::v2::device::{DeviceBackend, DeviceContext};
    use std::sync::Arc;

    #[test]
    fn test_pinned_storage() {
        let ctx = Arc::new(DeviceContext::new(DeviceBackend::Cuda, 0).unwrap());
        let storage = PinnedStorage::new(2048, ctx).unwrap();
        assert_eq!(storage.size(), 2048);
        assert_eq!(storage.storage_kind(), StorageKind::Pinned);
        assert!(storage.host_ptr() != 0);
    }

    #[test]
    fn test_pinned_storage_zero_size() {
        let ctx = Arc::new(DeviceContext::new(DeviceBackend::Cuda, 0).unwrap());
        let storage = PinnedStorage::new(0, ctx);
        assert!(storage.is_err());
        assert!(matches!(
            storage.unwrap_err(),
            StorageError::AllocationFailed(_)
        ));
    }

    #[test]
    fn test_device_storage() {
        let ctx = Arc::new(DeviceContext::new(DeviceBackend::Cuda, 0).unwrap());
        let storage = DeviceStorage::new(4096, ctx).unwrap();
        assert_eq!(storage.size(), 4096);
        assert_eq!(storage.storage_kind(), StorageKind::Device(0));
        assert!(storage.device_ptr() != 0);
        assert_eq!(storage.device_id(), 0);
    }

    #[test]
    fn test_device_storage_zero_size() {
        let ctx = Arc::new(DeviceContext::new(DeviceBackend::Cuda, 0).unwrap());
        let result = DeviceStorage::new(0, ctx);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StorageError::AllocationFailed(_)
        ));
    }
}

#[cfg(all(feature = "testing-nixl", feature = "testing-cuda"))]
mod nixl_tests {
    use super::*;
    use crate::block_manager::v2::device::{DeviceBackend, DeviceContext};
    use nixl_sys::Agent as NixlAgent;
    use std::sync::Arc;

    #[test]
    fn test_nixl_registration() {
        let ctx = Arc::new(DeviceContext::new(DeviceBackend::Cuda, 0).unwrap());
        let pinned = PinnedStorage::new(2048, ctx).unwrap();
        let agent = NixlAgent::new("test_agent").unwrap();
        let registered = register_with_nixl(pinned, &agent, None).unwrap();
        assert_eq!(registered.agent_name(), "test_agent");
    }
}

#[cfg(feature = "hpu")]
mod hpu_tests {
    use super::*;
    use crate::block_manager::v2::device::{DeviceBackend, DeviceContext};
    use std::sync::Arc;

    fn is_hpu_available() -> bool {
        DeviceBackend::Hpu.is_available()
    }

    #[test]
    fn test_hpu_pinned_storage() {
        if !is_hpu_available() {
            println!("Skipping test: HPU not available");
            return;
        }

        let ctx = Arc::new(DeviceContext::new(DeviceBackend::Hpu, 0).unwrap());
        let storage = PinnedStorage::new(2048, ctx).unwrap();
        assert_eq!(storage.size(), 2048);
        assert_eq!(storage.storage_kind(), StorageKind::Pinned);
        assert!(storage.host_ptr() != 0);
    }

    #[test]
    fn test_hpu_device_storage() {
        if !is_hpu_available() {
            println!("Skipping test: HPU not available");
            return;
        }

        let ctx = Arc::new(DeviceContext::new(DeviceBackend::Hpu, 0).unwrap());
        let storage = DeviceStorage::new(4096, ctx).unwrap();
        assert_eq!(storage.size(), 4096);
        assert_eq!(storage.storage_kind(), StorageKind::Device(0));
        assert!(storage.device_ptr() != 0);
        assert_eq!(storage.device_id(), 0);
        assert_eq!(storage.backend(), DeviceBackend::Hpu);
    }

    #[test]
    fn test_hpu_device_storage_zero_size() {
        if !is_hpu_available() {
            println!("Skipping test: HPU not available");
            return;
        }

        let ctx = Arc::new(DeviceContext::new(DeviceBackend::Hpu, 0).unwrap());
        let result = DeviceStorage::new(0, ctx);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StorageError::AllocationFailed(_)
        ));
    }
}
