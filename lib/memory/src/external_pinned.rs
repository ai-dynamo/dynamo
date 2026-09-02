// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA registration for externally owned, file-backed host memory.

use std::{
    any::Any,
    fs::{File, OpenOptions},
    os::unix::{fs::MetadataExt, fs::OpenOptionsExt},
    path::PathBuf,
    sync::Arc,
};

use cudarc::driver::{CudaContext, sys};
use memmap2::{MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};

use crate::{
    MemoryDescriptor, Result, StorageError, StorageKind, actions,
    nixl::{MemType, NixlCompatible, NixlDescriptor},
};

/// The file range and identity expected for an external shared-memory mapping.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalFileMappingDescriptor {
    /// Owner-controlled backing path.
    pub path: PathBuf,
    /// Device ID from `stat(2)` when the owner created the file.
    pub device: u64,
    /// Inode from `stat(2)` when the owner created the file.
    pub inode: u64,
    /// Byte offset at which the opaque data area begins.
    pub offset: u64,
    /// Length of the opaque data area.
    pub len: u64,
    /// Required alignment of the mapped data address.
    pub alignment: u64,
}

/// A checked writable `MAP_SHARED` file range.
pub struct ExternalFileMapping {
    descriptor: ExternalFileMappingDescriptor,
    mmap: MmapMut,
    _file: File,
}

impl std::fmt::Debug for ExternalFileMapping {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExternalFileMapping")
            .field("descriptor", &self.descriptor)
            .field("addr", &(self.mmap.as_ptr() as usize))
            .finish_non_exhaustive()
    }
}

impl ExternalFileMapping {
    /// Securely open and map the exact file range described by `descriptor`.
    pub fn open(descriptor: ExternalFileMappingDescriptor) -> Result<Self> {
        validate_descriptor(&descriptor)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&descriptor.path)?;
        let metadata = file.metadata()?;
        if metadata.dev() != descriptor.device || metadata.ino() != descriptor.inode {
            return Err(StorageError::OperationFailed(
                "external mapping file identity does not match the owner descriptor".into(),
            ));
        }
        let end = descriptor
            .offset
            .checked_add(descriptor.len)
            .ok_or_else(|| {
                StorageError::OperationFailed("external mapping range overflows".into())
            })?;
        if metadata.len() < end {
            return Err(StorageError::OperationFailed(format!(
                "external mapping ends at {end}, but the backing file is only {} bytes",
                metadata.len()
            )));
        }
        let len = usize::try_from(descriptor.len).map_err(|_| {
            StorageError::OperationFailed("external mapping length does not fit usize".into())
        })?;
        let mmap = unsafe {
            MmapOptions::new()
                .offset(descriptor.offset)
                .len(len)
                .map_mut(&file)
        }?;
        if !(mmap.as_ptr() as usize).is_multiple_of(descriptor.alignment as usize) {
            return Err(StorageError::OperationFailed(format!(
                "external mapping address does not satisfy {}-byte alignment",
                descriptor.alignment
            )));
        }
        Ok(Self {
            descriptor,
            mmap,
            _file: file,
        })
    }

    /// Base host address of the mapped data range.
    pub fn addr(&self) -> usize {
        self.mmap.as_ptr() as usize
    }

    /// Length of the mapped data range.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Whether the mapped data range is empty.
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Mapping descriptor validated by [`Self::open`].
    pub fn descriptor(&self) -> &ExternalFileMappingDescriptor {
        &self.descriptor
    }

    /// Immutable view of the mapped bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    /// Mutable view of the mapped bytes.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.mmap
    }
}

/// An owner-provided shared-memory mapping registered as CUDA pinned host memory.
///
/// Unlike [`crate::PinnedStorage`], this type does not call `cuMemFreeHost`.
/// Drop unregisters the host range from CUDA before the mapping and file are released.
pub struct ExternalPinnedStorage {
    mapping: Option<ExternalFileMapping>,
    ctx: Arc<CudaContext>,
    registered: bool,
}

unsafe impl Send for ExternalPinnedStorage {}
unsafe impl Sync for ExternalPinnedStorage {}

impl std::fmt::Debug for ExternalPinnedStorage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExternalPinnedStorage")
            .field("mapping", &self.mapping)
            .field("cuda_device", &self.ctx.cu_device())
            .field("registered", &self.registered)
            .finish()
    }
}

impl ExternalPinnedStorage {
    /// Map an owner-provided file range and register it in `device_id`'s CUDA context.
    pub fn new(descriptor: ExternalFileMappingDescriptor, device_id: u32) -> Result<Self> {
        let mapping = ExternalFileMapping::open(descriptor)?;
        let ctx = crate::device::cuda_context(device_id)?;
        ctx.bind_to_thread().map_err(StorageError::Cuda)?;
        let result = unsafe {
            sys::cuMemHostRegister_v2(
                mapping.addr() as *mut std::ffi::c_void,
                mapping.len(),
                sys::CU_MEMHOSTREGISTER_DEVICEMAP | sys::CU_MEMHOSTREGISTER_PORTABLE,
            )
        };
        result.result().map_err(StorageError::Cuda)?;
        Ok(Self {
            mapping: Some(mapping),
            ctx,
            registered: true,
        })
    }

    /// Descriptor of the backing file range.
    pub fn mapping_descriptor(&self) -> &ExternalFileMappingDescriptor {
        self.mapping().descriptor()
    }

    /// CUDA context used to register this mapping.
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Raw host pointer. It remains valid until this storage is dropped.
    pub fn as_ptr(&self) -> *const u8 {
        self.mapping().addr() as *const u8
    }

    /// Raw mutable host pointer.
    ///
    /// # Safety
    /// The caller must prevent concurrent aliases that access this memory.
    pub unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.mapping().addr() as *mut u8
    }

    fn mapping(&self) -> &ExternalFileMapping {
        self.mapping
            .as_ref()
            .expect("external mapping remains present until storage drop")
    }
}

impl Drop for ExternalPinnedStorage {
    fn drop(&mut self) {
        if !self.registered {
            return;
        }
        if let Err(error) = self.ctx.bind_to_thread() {
            tracing::error!(%error, "failed to bind CUDA context before unregistering external host memory");
        }
        let address = self.mapping().addr();
        let result = unsafe { sys::cuMemHostUnregister(address as *mut std::ffi::c_void) };
        if let Err(error) = result.result() {
            tracing::error!(%error, "failed to unregister external host memory from CUDA");
            // Unmapping memory that CUDA still considers registered is unsafe. Leak this
            // exceptional mapping (and its FD) instead of invalidating the CUDA registration.
            if let Some(mapping) = self.mapping.take() {
                std::mem::forget(mapping);
            }
        } else {
            self.registered = false;
        }
    }
}

impl MemoryDescriptor for ExternalPinnedStorage {
    fn addr(&self) -> usize {
        self.mapping().addr()
    }

    fn size(&self) -> usize {
        self.mapping().len()
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::Pinned
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

impl NixlCompatible for ExternalPinnedStorage {
    fn nixl_params(&self) -> (*const u8, usize, MemType, u64) {
        (self.as_ptr(), self.mapping().len(), MemType::Dram, 0)
    }
}

impl actions::Memset for ExternalPinnedStorage {
    fn memset(&mut self, value: u8, offset: usize, size: usize) -> Result<()> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| StorageError::OperationFailed("memset range overflows".into()))?;
        if end > self.mapping().len() {
            return Err(StorageError::OperationFailed(
                "memset range exceeds external storage".into(),
            ));
        }
        self.mapping
            .as_mut()
            .expect("external mapping remains present until storage drop")
            .as_mut_slice()[offset..end]
            .fill(value);
        Ok(())
    }
}

fn validate_descriptor(descriptor: &ExternalFileMappingDescriptor) -> Result<()> {
    if descriptor.len == 0 {
        return Err(StorageError::OperationFailed(
            "external mapping length must be positive".into(),
        ));
    }
    if descriptor.alignment == 0 || !descriptor.alignment.is_power_of_two() {
        return Err(StorageError::OperationFailed(
            "external mapping alignment must be a non-zero power of two".into(),
        ));
    }
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(StorageError::Io(std::io::Error::last_os_error()));
    }
    let page_size = page_size as u64;
    if !descriptor.offset.is_multiple_of(page_size) {
        return Err(StorageError::OperationFailed(format!(
            "external mapping offset must be aligned to the system page size ({page_size})"
        )));
    }
    if descriptor.alignment > page_size || !page_size.is_multiple_of(descriptor.alignment) {
        return Err(StorageError::OperationFailed(format!(
            "external mapping alignment must not exceed the system page size ({page_size})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::MetadataExt;

    use super::*;

    #[test]
    fn checked_mapping_is_shared_and_rejects_identity_mismatch() {
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(8192).unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let descriptor = ExternalFileMappingDescriptor {
            path: file.path().to_path_buf(),
            device: metadata.dev(),
            inode: metadata.ino(),
            offset: 4096,
            len: 4096,
            alignment: 4096,
        };
        let mut first = ExternalFileMapping::open(descriptor.clone()).unwrap();
        let second = ExternalFileMapping::open(descriptor.clone()).unwrap();
        first.as_mut_slice()[9..13].copy_from_slice(b"kvbm");
        assert_eq!(&second.as_slice()[9..13], b"kvbm");

        let mut forged = descriptor;
        forged.inode = forged.inode.wrapping_add(1);
        assert!(ExternalFileMapping::open(forged).is_err());
    }

    #[test]
    fn checked_mapping_rejects_overflow_and_unaligned_offsets() {
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(8192).unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let base = ExternalFileMappingDescriptor {
            path: file.path().to_path_buf(),
            device: metadata.dev(),
            inode: metadata.ino(),
            offset: 1,
            len: 4096,
            alignment: 4096,
        };
        assert!(ExternalFileMapping::open(base.clone()).is_err());
        assert!(
            ExternalFileMapping::open(ExternalFileMappingDescriptor {
                offset: 4096,
                len: u64::MAX,
                ..base
            })
            .is_err()
        );
    }
}
