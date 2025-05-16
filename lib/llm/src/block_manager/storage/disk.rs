// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

use std::fs::File;
use std::os::unix::io::{AsRawFd, FromRawFd};
use nix::fcntl::{fallocate, FallocateFlags};
use std::ffi::CString;

#[derive(Debug)]
pub struct DiskStorage {
    file: File,
    file_name: String,
    size: usize,
    handles: RegistrationHandles,
}

impl Local for DiskStorage {}
impl SystemAccessible for DiskStorage {}

impl DiskStorage {
    pub fn new(size: usize) -> Result<Self, StorageError> {
        // We need to open our file with some special flags that aren't supported by the tempfile crate.
        // Instead, we'll use the mkostemp function to create a temporary file with the correct flags.

        let template = CString::new("/tmp/dynamo-kvbm-disk-cache-XXXXXX").unwrap();
        let mut template_bytes = template.into_bytes_with_nul();

        let raw_fd = unsafe {
            nix::libc::mkostemp(template_bytes.as_mut_ptr() as *mut i8, nix::libc::O_RDWR | nix::libc::O_DIRECT)
        };

        let file = unsafe { File::from_raw_fd(raw_fd) };

        // Expand the file to the desired size, and allocate all blocks of the file.
        file.set_len(size as u64).map_err(|_| {
            StorageError::AllocationFailed("Failed to set temp file size".to_string())
        })?;

        fallocate(file.as_raw_fd(), FallocateFlags::empty(), 0, size as i64).map_err(|_| {
            StorageError::AllocationFailed("Failed to allocate temp file".to_string())
        })?;

        Ok(Self {
            file,
            file_name: String::from_utf8(template_bytes).unwrap(),
            size,
            handles: RegistrationHandles::new(),
        })
    }

    pub fn fd(&self) -> u64 {
        self.file.as_raw_fd() as u64
    }
}

impl Drop for DiskStorage {
    fn drop(&mut self) {
        std::fs::remove_file(self.file_name.clone()).unwrap();
    }
}

impl Storage for DiskStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Disk
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

