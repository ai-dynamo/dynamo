// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Disk-backed memory storage using memory-mapped files.

use super::{MemoryDescription, Result, StorageError, StorageKind, nixl::NixlDescriptor};
use std::any::Any;
use std::path::{Path, PathBuf};

use core::ffi::c_char;
use nix::fcntl::{FallocateFlags, fallocate};
use nix::unistd::unlink;
use std::ffi::CString;
use std::os::fd::BorrowedFd;

const DISK_CACHE_KEY: &str = "DYN_KVBM_DISK_CACHE_DIR";
const DEFAULT_DISK_CACHE_DIR: &str = "/tmp/";

#[derive(Debug)]
pub struct DiskStorage {
    fd: u64,
    path: PathBuf,
    size: usize,
    unlinked: bool,
}

impl DiskStorage {
    pub fn new(size: usize) -> Result<Self> {
        // We need to open our file with some special flags that aren't supported by the tempfile crate.
        // Instead, we'll use the mkostemp function to create a temporary file with the correct flags.

        let specified_dir =
            std::env::var(DISK_CACHE_KEY).unwrap_or_else(|_| DEFAULT_DISK_CACHE_DIR.to_string());
        let file_path = Path::new(&specified_dir).join("dynamo-kvbm-disk-cache-XXXXXX");

        Self::new_at(file_path, size)
    }

    pub fn new_at(path: impl AsRef<Path>, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }

        let file_path = path.as_ref().to_path_buf();

        if !file_path.exists() {
            let parent = file_path.parent().ok_or_else(|| {
                StorageError::AllocationFailed(format!(
                    "disk cache path {} has no parent directory",
                    file_path.display()
                ))
            })?;
            std::fs::create_dir_all(parent).map_err(|e| {
                StorageError::AllocationFailed(format!(
                    "failed to create disk cache directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        tracing::debug!("Allocating disk cache file at {}", file_path.display());

        let path_str = file_path.to_str().ok_or_else(|| {
            StorageError::AllocationFailed(format!(
                "disk cache path {} is not valid UTF-8",
                file_path.display()
            ))
        })?;
        let is_template = path_str.contains("XXXXXX");

        let (raw_fd, actual_path) = if is_template {
            // Template path - use mkostemp to generate unique filename
            let template = CString::new(path_str).unwrap();
            let mut template_bytes = template.into_bytes_with_nul();

            let fd = unsafe {
                nix::libc::mkostemp(
                    template_bytes.as_mut_ptr() as *mut c_char,
                    nix::libc::O_RDWR | nix::libc::O_DIRECT,
                )
            };

            if fd == -1 {
                return Err(StorageError::AllocationFailed(format!(
                    "mkostemp failed: {}",
                    std::io::Error::last_os_error()
                )));
            }

            // Extract the actual path created by mkostemp
            let actual = PathBuf::from(
                CString::from_vec_with_nul(template_bytes)
                    .unwrap()
                    .to_str()
                    .unwrap(),
            );

            (fd, actual)
        } else {
            // Specific path - use open with O_CREAT
            let path_cstr = CString::new(path_str).unwrap();
            let fd = unsafe {
                nix::libc::open(
                    path_cstr.as_ptr(),
                    nix::libc::O_CREAT | nix::libc::O_RDWR | nix::libc::O_DIRECT,
                    0o644,
                )
            };

            if fd == -1 {
                return Err(StorageError::AllocationFailed(format!(
                    "open failed: {}",
                    std::io::Error::last_os_error()
                )));
            }

            (fd, file_path)
        };

        // We need to use fallocate to actually allocate the storage and create the blocks on disk.
        unsafe {
            fallocate(
                BorrowedFd::borrow_raw(raw_fd),
                FallocateFlags::empty(),
                0,
                len as i64,
            )
            .map_err(|e| {
                StorageError::AllocationFailed(format!("Failed to allocate temp file: {}", e))
            })?
        };

        Ok(Self {
            fd: raw_fd as u64,
            path: actual_path,
            size: len,
            unlinked: false,
        })
    }

    pub fn fd(&self) -> u64 {
        self.fd
    }

    pub fn path(&self) -> &Path {
        self.path.as_path()
    }

    /// Unlink our temp file.
    /// This means that when this process terminates, the file will be automatically deleted by the OS.
    /// Unfortunately, GDS requires that files we try to register must be linked.
    /// To get around this, we unlink the file only after we've registered it with NIXL.
    pub fn unlink(&mut self) -> Result<()> {
        if self.unlinked {
            return Ok(());
        }

        unlink(self.path.as_path())
            .map_err(|e| StorageError::AllocationFailed(format!("Failed to unlink file: {}", e)))?;
        self.unlinked = true;
        Ok(())
    }

    pub fn unlinked(&self) -> bool {
        self.unlinked
    }
}

impl Drop for DiskStorage {
    fn drop(&mut self) {
        let _ = self.unlink();
        if let Err(e) = nix::unistd::close(self.fd as std::os::fd::RawFd) {
            tracing::debug!("failed to close disk cache fd {}: {e}", self.fd);
        }
    }
}

impl MemoryDescription for DiskStorage {
    fn addr(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        self.size
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::Disk(self.fd)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

// Support for NIXL registration
impl super::nixl::NixlCompatible for DiskStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        #[cfg(unix)]
        {
            // Use file descriptor as device_id for MemType::File
            (
                std::ptr::null(),
                self.size,
                nixl_sys::MemType::File,
                self.fd,
            )
        }

        #[cfg(not(unix))]
        {
            // On non-Unix systems, we can't get the file descriptor easily
            // Return device_id as 0 - registration will fail on these systems
            (
                self.mmap.as_ptr(),
                self.mmap.len(),
                nixl_sys::MemType::File,
                0,
            )
        }
    }
}

// mod mmap {
//     use super::*;

//     #[cfg(unix)]
//     use std::os::unix::io::AsRawFd;

//     use memmap2::{MmapMut, MmapOptions};
//     use std::fs::{File, OpenOptions};
//     use tempfile::NamedTempFile;

//     /// Disk-backed storage using memory-mapped files.
//     #[derive(Debug)]
//     pub struct MemMappedFileStorage {
//         _file: File, // Keep file alive for the lifetime of the mmap
//         mmap: MmapMut,
//         path: PathBuf,
//         #[cfg(unix)]
//         fd: i32,
//     }

//     unsafe impl Send for MemMappedFileStorage {}
//     unsafe impl Sync for MemMappedFileStorage {}

//     impl MemMappedFileStorage {
//         /// Create new disk storage with a temporary file.
//         pub fn new_temp(len: usize) -> Result<Self> {
//             if len == 0 {
//                 return Err(StorageError::AllocationFailed(
//                     "zero-sized allocations are not supported".into(),
//                 ));
//             }

//             // Create temporary file
//             let temp_file = NamedTempFile::new()?;
//             let path = temp_file.path().to_path_buf();
//             let file = temp_file.into_file();

//             // Set file size
//             file.set_len(len as u64)?;

//             #[cfg(unix)]
//             let fd = file.as_raw_fd();

//             // Memory map the file
//             let mmap = unsafe { MmapOptions::new().len(len).map_mut(&file)? };

//             Ok(Self {
//                 _file: file,
//                 mmap,
//                 path,
//                 #[cfg(unix)]
//                 fd,
//             })
//         }

//         /// Create new disk storage with a specific file path.
//         pub fn new_at(path: impl AsRef<Path>, len: usize) -> Result<Self> {
//             if len == 0 {
//                 return Err(StorageError::AllocationFailed(
//                     "zero-sized allocations are not supported".into(),
//                 ));
//             }

//             let path = path.as_ref().to_path_buf();

//             // Create or open file
//             let file = OpenOptions::new()
//                 .read(true)
//                 .write(true)
//                 .create(true)
//                 .open(&path)?;

//             // Set file size
//             file.set_len(len as u64)?;

//             #[cfg(unix)]
//             let fd = file.as_raw_fd();

//             // Memory map the file
//             let mmap = unsafe { MmapOptions::new().len(len).map_mut(&file)? };

//             Ok(Self {
//                 _file: file,
//                 mmap,
//                 path,
//                 #[cfg(unix)]
//                 fd,
//             })
//         }

//         /// Get the path to the backing file.
//         pub fn path(&self) -> &Path {
//             &self.path
//         }

//         /// Get the file descriptor (Unix only).
//         #[cfg(unix)]
//         pub fn fd(&self) -> i32 {
//             self.fd
//         }

//         /// Get a pointer to the memory-mapped region.
//         ///
//         /// # Safety
//         /// The caller must ensure the pointer is not used after this storage is dropped.
//         pub unsafe fn as_ptr(&self) -> *const u8 {
//             self.mmap.as_ptr()
//         }

//         /// Get a mutable pointer to the memory-mapped region.
//         ///
//         /// # Safety
//         /// The caller must ensure the pointer is not used after this storage is dropped
//         /// and that there are no other references to this memory.
//         pub unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
//             self.mmap.as_mut_ptr()
//         }
//     }

//     impl MemoryDescription for MemMappedFileStorage {
//         fn addr(&self) -> usize {
//             self.mmap.as_ptr() as usize
//         }

//         fn size(&self) -> usize {
//             self.mmap.len()
//         }

//         fn storage_kind(&self) -> StorageKind {
//             StorageKind::Disk
//         }

//         fn as_any(&self) -> &dyn Any {
//             self
//         }
//     }

//     // Support for NIXL registration
//     impl super::super::registered::NixlCompatible for MemMappedFileStorage {
//         fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
//             #[cfg(unix)]
//             {
//                 // Use file descriptor as device_id for MemType::File
//                 (
//                     self.mmap.as_ptr(),
//                     self.mmap.len(),
//                     nixl_sys::MemType::File,
//                     self.fd as u64,
//                 )
//             }

//             #[cfg(not(unix))]
//             {
//                 // On non-Unix systems, we can't get the file descriptor easily
//                 // Return device_id as 0 - registration will fail on these systems
//                 (
//                     self.mmap.as_ptr(),
//                     self.mmap.len(),
//                     nixl_sys::MemType::File,
//                     0,
//                 )
//             }
//         }
//     }
// }
