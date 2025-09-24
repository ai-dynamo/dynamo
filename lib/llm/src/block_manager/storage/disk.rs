// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use core::ffi::c_char;
use nix::fcntl::{FallocateFlags, fallocate};
use nix::unistd::unlink;
use std::ffi::CStr;
use std::ffi::CString;
use std::path::Path;

const DISK_CACHE_KEY: &str = "DYN_KVBM_DISK_CACHE_DIR";
const DEFAULT_DISK_CACHE_DIR: &str = "/tmp/";

#[derive(Debug)]
pub struct DiskStorage {
    fd: u64,
    file_name: String,
    size: usize,
    handles: RegistrationHandles,
    unlinked: bool,
}

impl Local for DiskStorage {}
impl SystemAccessible for DiskStorage {}

impl DiskStorage {
    pub fn new(size: usize) -> Result<Self, StorageError> {
        // We need to open our file with some special flags that aren't supported by the tempfile crate.
        // Instead, we'll use the mkostemp function to create a temporary file with the correct flags.

        let specified_dir =
            std::env::var(DISK_CACHE_KEY).unwrap_or_else(|_| DEFAULT_DISK_CACHE_DIR.to_string());
        let file_path = Path::new(&specified_dir).join("dynamo-kvbm-disk-cache-XXXXXX");

        if !file_path.exists() {
            std::fs::create_dir_all(file_path.parent().unwrap()).unwrap();
        }

        tracing::debug!("Allocating disk cache file at {}", file_path.display());

        let template = CString::new(file_path.to_str().unwrap()).unwrap();
        let mut template_bytes = template.into_bytes_with_nul();

        let raw_fd = unsafe {
            nix::libc::mkostemp(
                template_bytes.as_mut_ptr() as *mut c_char,
                // For maximum performance, GPU DirectStorage requires O_DIRECT.
                // This allows transfers to bypass the kernel page cache.
                // It also introduces the restriction that all accesses must be page-aligned.
                nix::libc::O_RDWR | nix::libc::O_DIRECT,
            )
        };

        let file_name = CStr::from_bytes_with_nul(template_bytes.as_slice())
            .unwrap()
            .to_str()
            .map_err(|e| {
                StorageError::AllocationFailed(format!("Failed to read temp file name: {}", e))
            })?
            .to_string();

        // We need to use fallocate to actually allocate the storage and create the blocks on disk.
        fallocate(raw_fd, FallocateFlags::empty(), 0, size as i64).map_err(|e| {
            StorageError::AllocationFailed(format!("Failed to allocate temp file: {}", e))
        })?;

        Ok(Self {
            fd: raw_fd as u64,
            file_name,
            size,
            handles: RegistrationHandles::new(),
            unlinked: false,
        })
    }

    pub fn fd(&self) -> u64 {
        self.fd
    }

    /// Unlink our temp file.
    /// This means that when this process terminates, the file will be automatically deleted by the OS.
    /// Unfortunately, GDS requires that files we try to register must be linked.
    /// To get around this, we unlink the file only after we've registered it with NIXL.
    pub fn unlink(&mut self) -> Result<(), StorageError> {
        if self.unlinked {
            return Ok(());
        }

        self.unlinked = true;

        unlink(self.file_name.as_str()).map_err(|e| {
            StorageError::AllocationFailed(format!("Failed to unlink temp file: {}", e))
        })
    }

    pub fn unlinked(&self) -> bool {
        self.unlinked
    }
}

impl Drop for DiskStorage {
    fn drop(&mut self) {
        self.handles.release();
        let _ = self.unlink();
    }
}

impl Storage for DiskStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Disk(self.fd())
    }

    fn addr(&self) -> u64 {
        0
    }

    fn size(&self) -> usize {
        self.size
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        std::ptr::null()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        std::ptr::null_mut()
    }
}

impl RegisterableStorage for DiskStorage {
    fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        self.handles.register(key, handle)
    }

    fn is_registered(&self, key: &str) -> bool {
        self.handles.is_registered(key)
    }

    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.registration_handle(key)
    }
}

#[derive(Default)]
pub struct DiskAllocator;

impl StorageAllocator<DiskStorage> for DiskAllocator {
    fn allocate(&self, size: usize) -> Result<DiskStorage, StorageError> {
        DiskStorage::new(size)
    }
}
