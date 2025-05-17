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

//! Block storage management.
//!
//! This module provides the implementation for managing collections of blocks
//! and their storage. It handles the relationship between storage, layout,
//! and individual blocks.

use super::{BlockData, BlockError, Storage};

pub trait Kind: std::marker::Sized + std::fmt::Debug + Clone + Copy + Send + Sync {}

#[derive(Debug, Clone, Copy)]
pub struct BlockKind;
impl Kind for BlockKind {}

#[derive(Debug, Clone, Copy)]
pub struct LayerKind;
impl Kind for LayerKind {}

pub type BlockView<'a, S> = MemoryView<'a, S, BlockKind>;
pub type BlockViewMut<'a, S> = MemoryViewMut<'a, S, BlockKind>;

pub type LayerView<'a, S> = MemoryView<'a, S, LayerKind>;
pub type LayerViewMut<'a, S> = MemoryViewMut<'a, S, LayerKind>;

/// Storage view that provides safe access to a region of storage
#[derive(Debug)]
pub struct MemoryView<'a, S: Storage, K: Kind> {
    _block_data: &'a BlockData<S>,
    addr: usize,
    size: usize,
    kind: std::marker::PhantomData<K>,
}

impl<'a, S, K> MemoryView<'a, S, K>
where
    S: Storage,
    K: Kind,
{
    /// Create a new storage view
    ///
    /// # Safety
    /// The caller must ensure:
    /// - addr + size <= storage.size()
    /// - The view does not outlive the storage
    pub(crate) unsafe fn new(
        _block_data: &'a BlockData<S>,
        addr: usize,
        size: usize,
    ) -> Result<Self, BlockError> {
        Ok(Self {
            _block_data,
            addr,
            size,
            kind: std::marker::PhantomData,
        })
    }

    /// Get a raw pointer to the view's data
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The pointer is not used after the view is dropped
    /// - Access patterns respect the storage's thread safety model
    pub unsafe fn as_ptr(&self) -> *const u8 {
        self.addr as *const u8
    }

    /// Size of the view in bytes
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Mutable storage view that provides exclusive access to a region of storage
#[derive(Debug)]
pub struct MemoryViewMut<'a, S: Storage, K: Kind> {
    _block_data: &'a mut BlockData<S>,
    addr: usize,
    size: usize,
    kind: std::marker::PhantomData<K>,
}

impl<'a, S: Storage, K: Kind> MemoryViewMut<'a, S, K> {
    /// Create a new mutable storage view
    ///
    /// # Safety
    /// The caller must ensure:
    /// - addr + size <= storage.size()
    /// - The view does not outlive the storage
    /// - No other views exist for this region
    pub(crate) unsafe fn new(
        _block_data: &'a mut BlockData<S>,
        addr: usize,
        size: usize,
    ) -> Result<Self, BlockError> {
        Ok(Self {
            _block_data,
            addr,
            size,
            kind: std::marker::PhantomData,
        })
    }

    /// Get a raw mutable pointer to the view's data
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The pointer is not used after the view is dropped
    /// - No other references exist while the pointer is in use
    /// - Access patterns respect the storage's thread safety model
    pub unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.addr as *mut u8
    }

    /// Size of the view in bytes
    pub fn size(&self) -> usize {
        self.size
    }
}

mod nixl {
    use super::*;

    use super::super::nixl::*;

    pub use nixl_sys::{MemType, MemoryRegion, NixlDescriptor};

    impl<S: Storage, K: Kind> MemoryRegion for MemoryView<'_, S, K> {
        unsafe fn as_ptr(&self) -> *const u8 {
            self.addr as *const u8
        }

        fn size(&self) -> usize {
            self.size()
        }
    }

    impl<S, K> NixlDescriptor for MemoryView<'_, S, K>
    where
        S: Storage + NixlDescriptor,
        K: Kind,
    {
        fn mem_type(&self) -> MemType {
            self._block_data.layout.storage_type().nixl_mem_type()
        }

        fn device_id(&self) -> u64 {
            self._block_data
                .layout
                .storage()
                .into_iter()
                .next()
                .unwrap()
                .device_id()
        }
    }

    impl<S: Storage, K: Kind> MemoryRegion for MemoryViewMut<'_, S, K> {
        unsafe fn as_ptr(&self) -> *const u8 {
            self.addr as *const u8
        }

        fn size(&self) -> usize {
            self.size()
        }
    }

    impl<S: Storage, K: Kind> NixlDescriptor for MemoryViewMut<'_, S, K>
    where
        S: Storage + NixlDescriptor,
        K: Kind,
    {
        fn mem_type(&self) -> MemType {
            self._block_data.layout.storage_type().nixl_mem_type()
        }

        fn device_id(&self) -> u64 {
            self._block_data
                .layout
                .storage()
                .into_iter()
                .next()
                .unwrap()
                .device_id()
        }
    }

    impl<'a, S, K> MemoryView<'a, S, K>
    where
        S: Storage + NixlDescriptor, // Ensure the underlying storage is a NixlDescriptor
        K: Kind,
    {
        /// Creates an immutable NIXL memory descriptor from this view.
        pub fn as_nixl_descriptor(&self) -> NixlMemoryDescriptor<'a, K, IsImmutable> {
            NixlMemoryDescriptor::new(
                self.addr as u64,                // Address from the view
                self.size(),                     // Size from the view
                NixlDescriptor::mem_type(self),  // Delegate to self's NixlDescriptor impl
                NixlDescriptor::device_id(self), // Delegate to self's NixlDescriptor impl
            )
        }
    }

    impl<'a, S, K> MemoryViewMut<'a, S, K>
    where
        S: Storage + NixlDescriptor,
        K: Kind,
    {
        /// Creates a mutable NIXL memory descriptor from this view.
        // Note: We return a mutable descriptor even from an immutable borrow (&self)
        // because the underlying memory region *can* be mutated.
        pub fn as_nixl_descriptor_mut(&mut self) -> NixlMemoryDescriptor<'a, K, IsMutable> {
            NixlMemoryDescriptor::new(
                self.addr as u64,
                self.size(),
                NixlDescriptor::mem_type(self), // Delegate to self's NixlDescriptor impl
                NixlDescriptor::device_id(self), // Delegate to self's NixlDescriptor impl
            )
        }
    }
}
