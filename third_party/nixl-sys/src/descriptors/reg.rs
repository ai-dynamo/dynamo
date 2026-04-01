// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
use super::sync_manager::{BackendSyncable, SyncManager};
use std::ops::{Index, IndexMut};
use serde::{Serialize, Deserialize};

/// Public registration descriptor used for indexing and comparisons
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegDescriptor {
    pub addr: usize,
    pub len: usize,
    pub dev_id: u64,
    pub metadata: Vec<u8>,
}

/// Internal data structure for registration descriptors
#[derive(Debug, Serialize, Deserialize)]
struct RegDescData {
    mem_type: MemType,
    descriptors: Vec<RegDescriptor>,
}

impl BackendSyncable for RegDescData {
    type Backend = NonNull<bindings::nixl_capi_reg_dlist_s>;
    type Error = NixlError;

    fn sync_to_backend(&self, backend: &Self::Backend) -> Result<(), Self::Error> {
        // Clear backend
        let status = unsafe { nixl_capi_reg_dlist_clear(backend.as_ptr()) };
        match status {
            NIXL_CAPI_SUCCESS => {}
            NIXL_CAPI_ERROR_INVALID_PARAM => return Err(NixlError::InvalidParam),
            _ => return Err(NixlError::BackendError),
        }

        // Re-add all descriptors
        for desc in &self.descriptors {
            let status = unsafe {
                nixl_capi_reg_dlist_add_desc(
                    backend.as_ptr(),
                    desc.addr as uintptr_t,
                    desc.len,
                    desc.dev_id,
                    desc.metadata.as_ptr() as *const std::ffi::c_void,
                    desc.metadata.len(),
                )
            };
            match status {
                NIXL_CAPI_SUCCESS => {}
                NIXL_CAPI_ERROR_INVALID_PARAM => return Err(NixlError::InvalidParam),
                _ => return Err(NixlError::BackendError),
            }
        }

        Ok(())
    }
}

/// A safe wrapper around a NIXL registration descriptor list
pub struct RegDescList<'a> {
    sync_mgr: SyncManager<RegDescData>,
    _phantom: PhantomData<&'a dyn NixlDescriptor>,
    mem_type: MemType,
}

impl<'a> RegDescList<'a> {
    /// Creates a new registration descriptor list for the given memory type
    pub fn new(mem_type: MemType) -> Result<Self, NixlError> {
        let mut dlist = ptr::null_mut();
        let status = unsafe {
            nixl_capi_create_reg_dlist(mem_type as nixl_capi_mem_type_t, &mut dlist)
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                if dlist.is_null() {
                    tracing::error!("Failed to create registration descriptor list");
                    return Err(NixlError::RegDescListCreationFailed);
                }
                let backend = NonNull::new(dlist).ok_or(NixlError::RegDescListCreationFailed)?;

                let data = RegDescData {
                    mem_type,
                    descriptors: Vec::new(),
                };
                let sync_mgr = SyncManager::new(data, backend);

                Ok(Self {
                    sync_mgr,
                    _phantom: PhantomData,
                    mem_type,
                })
            }
            _ => Err(NixlError::RegDescListCreationFailed),
        }
    }

    pub fn get_type(&self) -> Result<MemType, NixlError> { Ok(self.mem_type) }

    /// Adds a descriptor to the list
    pub fn add_desc(&mut self, addr: usize, len: usize, dev_id: u64) {
        self.add_desc_with_meta(addr, len, dev_id, &[])
    }

    /// Add a descriptor with metadata
    pub fn add_desc_with_meta(
        &mut self,
        addr: usize,
        len: usize,
        dev_id: u64,
        metadata: &[u8],
    ) {
        self.sync_mgr.data_mut().descriptors.push(RegDescriptor {
            addr,
            len,
            dev_id,
            metadata: metadata.to_vec(),
        });
    }

    /// Returns true if the list is empty
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        Ok(self.len()? == 0)
    }

    /// Returns the number of descriptors in the list
    pub fn desc_count(&self) -> Result<usize, NixlError> { Ok(self.sync_mgr.data().descriptors.len()) }

    /// Returns the number of descriptors in the list
    pub fn len(&self) -> Result<usize, NixlError> { Ok(self.sync_mgr.data().descriptors.len()) }

    /// Trims the list to the given size
    pub fn trim(&mut self) {
        self.sync_mgr.data_mut().descriptors.shrink_to_fit();
    }

    /// Removes the descriptor at the given index
    pub fn rem_desc(&mut self, index: i32) -> Result<(), NixlError> {
        if index < 0 { return Err(NixlError::InvalidParam); }
        let idx = index as usize;

        let data = self.sync_mgr.data_mut();
        if idx >= data.descriptors.len() {
            return Err(NixlError::InvalidParam);
        }
        data.descriptors.remove(idx);
        Ok(())
    }

    /// Prints the list contents
    pub fn print(&self) -> Result<(), NixlError> {
        let backend = self.sync_mgr.backend()?;
        let status = unsafe { nixl_capi_reg_dlist_print(backend.as_ptr()) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Clears all descriptors from the list
    pub fn clear(&mut self) {
        self.sync_mgr.data_mut().descriptors.clear();
    }

    /// Resizes the list to the given size
    pub fn resize(&mut self, new_size: usize) {
        self.sync_mgr.data_mut().descriptors.resize(new_size, RegDescriptor {
            addr: 0,
            len: 0,
            dev_id: 0,
            metadata: Vec::new(),
        });
    }

    /// Safe immutable access to descriptor by index
    pub fn get(&self, index: usize) -> Result<&RegDescriptor, NixlError> {
        self.sync_mgr.data().descriptors
            .get(index)
            .ok_or(NixlError::InvalidParam)
    }

    /// Safe mutable access to descriptor by index
    pub fn get_mut(&mut self, index: usize) -> Result<&mut RegDescriptor, NixlError> {
        self.sync_mgr.data_mut().descriptors
            .get_mut(index)
            .ok_or(NixlError::InvalidParam)
    }

    /// Add a descriptor from a type implementing NixlDescriptor
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The descriptor remains valid for the lifetime of the list
    /// - The memory region pointed to by the descriptor remains valid
    pub fn add_storage_desc(&mut self, desc: &'a dyn NixlDescriptor) -> Result<(), NixlError> {
        // Validate memory type matches
        let desc_mem_type = desc.mem_type();
        let list_mem_type = if self.len()? > 0 {
            self.get_type()?
        } else {
            desc_mem_type
        };

        if desc_mem_type != list_mem_type && list_mem_type != MemType::Unknown {
            return Err(NixlError::InvalidParam);
        }

        // Get descriptor details
        let addr = unsafe { desc.as_ptr() } as usize;
        let len = desc.size();
        let dev_id = desc.device_id();

        // Add to list
        self.add_desc(addr, len, dev_id);
        Ok(())
    }

    pub(crate) fn handle(&self) -> *mut bindings::nixl_capi_reg_dlist_s {
        self.sync_mgr.backend().map(|b| b.as_ptr()).unwrap_or(ptr::null_mut())
    }

    /// Serializes the descriptor list to a byte vector using bincode
    pub fn serialize(&self) -> Result<Vec<u8>, NixlError> {
        // Serialize the RegDescData directly (contains mem_type + descriptors)
        bincode::serialize(self.sync_mgr.data()).map_err(|_| NixlError::BackendError)
    }

    /// Deserializes a descriptor list from a byte slice using bincode
    pub fn deserialize(bytes: &[u8]) -> Result<Self, NixlError> {
        let data: RegDescData = bincode::deserialize(bytes)
            .map_err(|_| NixlError::RegDescListCreationFailed)?;

        let mut list = RegDescList::new(data.mem_type)?;
        for desc in data.descriptors {
            list.add_desc_with_meta(desc.addr, desc.len, desc.dev_id, &desc.metadata);
        }

        // Force synchronization to validate backend can handle the data
        list.sync_mgr.backend()?;

        Ok(list)
    }
}

impl std::fmt::Debug for RegDescList<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mem_type = self.get_type().unwrap_or(MemType::Unknown);
        let len = self.len().unwrap_or(0);
        let desc_count = self.desc_count().unwrap_or(0);

        f.debug_struct("RegDescList")
            .field("mem_type", &mem_type)
            .field("len", &len)
            .field("desc_count", &desc_count)
            .finish()
    }
}

impl PartialEq for RegDescList<'_> {
    fn eq(&self, other: &Self) -> bool {
        // Compare memory types first
        if self.mem_type != other.mem_type {
            return false;
        }

        // Compare internal descriptor tracking
        self.sync_mgr.data().descriptors == other.sync_mgr.data().descriptors
    }
}

// Implement Index trait for immutable indexing (list[i])
impl Index<usize> for RegDescList<'_> {
    type Output = RegDescriptor;

    fn index(&self, index: usize) -> &Self::Output {
        &self.sync_mgr.data().descriptors[index]
    }
}

// Implement IndexMut trait for mutable indexing (list[i] = value)
impl IndexMut<usize> for RegDescList<'_> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // data_mut() automatically marks dirty
        &mut self.sync_mgr.data_mut().descriptors[index]
    }
}

impl Drop for RegDescList<'_> {
    fn drop(&mut self) {
        tracing::trace!("Dropping registration descriptor list");
        if let Ok(backend) = self.sync_mgr.backend() {
            unsafe {
                nixl_capi_destroy_reg_dlist(backend.as_ptr());
            }
        }
        tracing::trace!("Registration descriptor list dropped");
    }
}
