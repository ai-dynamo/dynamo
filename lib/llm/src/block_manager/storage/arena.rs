// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Arena Allocator
//!
//! This module provides an arena allocator for generally heap-like allocations.
//! An [`ArenaAllocator`] can be create by taking ownership of a [`Storage`] instance.
//!
//! The [`ArenaAllocator`] allocates memory contiguous regions using the [`offset_allocator`] crate,
//! which builds on  [Sebastian Aaltonen's ArenaAllocator](https://github.com/sebbbi/ArenaAllocator)
//!
//! ## Usage
//!
//! TODO: provide rust example

use super::{Storage, StorageError};
use offset_allocator::{Allocation, Allocator};
use std::sync::{Arc, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum ArenaError {
    #[error("Page size must be a power of 2")]
    PageSizeNotAligned,

    #[error("Allocation failed")]
    AllocationFailed,

    #[error("Failed to convert pages to u32")]
    PagesNotConvertible,

    #[error("Storage not registered with NIXL")]
    NotRegisteredWithNixl,

    #[error("Storage error: {0}")]
    StorageError(#[from] StorageError),
}

/// Arena allocator backed by an instance of a [`Storage`] object.
///
/// This struct wraps an [`Allocator`] from the [`offset_allocator`] crate,
/// and provides methods for allocating memory from the storage.
///
/// The allocator is thread-safe, and the storage is shared between the allocator and the buffers.
#[derive(Clone)]
pub struct ArenaAllocator<S: Storage> {
    storage: Arc<S>,
    allocator: Arc<Mutex<Allocator>>,
    page_size: u64,
}

impl<S: Storage> std::fmt::Debug for ArenaAllocator<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ArenaAllocator {{ storage: {:?}, page_size: {} }}",
            self.storage, self.page_size
        )
    }
}

/// A buffer allocated from an [`ArenaAllocator`].
///
/// This struct wraps an [`Allocation`] from the [`offset_allocator`] crate,
/// and provides methods for interacting with the allocated memory.
///
/// The buffer is backed by a [`Storage`] object, and the allocation is freed when the buffer is dropped.
pub struct ArenaBuffer<S: Storage> {
    offset: u64,
    address: u64,
    requested_size: usize,
    storage: Arc<S>,
    allocation: Allocation,
    allocator: Arc<Mutex<Allocator>>,
}

impl<S: Storage> ArenaAllocator<S> {
    /// Create a new [`ArenaAllocator`] from a [`Storage`] object and a page size.
    ///
    /// The page size must be a power of two.
    ///
    /// The allocator will divide the storage into pages and allocations will consist of a set of contiguous
    /// pages whose aggregate size is greater than or equal to the requested size.
    ///
    /// The allocator is thread-safe, and the storage is shared between the allocator and the buffers.
    pub fn new(storage: S, page_size: usize) -> Result<Self, ArenaError> {
        let storage = Arc::new(storage);

        if !page_size.is_power_of_two() {
            return Err(ArenaError::PageSizeNotAligned);
        }

        // divide storage into pages,
        // round down such that all pages are fully and any remaining bytes are discarded
        let pages = storage.size() / page_size;

        let allocator = Allocator::new(
            pages
                .try_into()
                .map_err(|_| ArenaError::PagesNotConvertible)?,
        );

        let allocator = Arc::new(Mutex::new(allocator));

        Ok(Self {
            storage,
            allocator,
            page_size: page_size as u64,
        })
    }

    /// Allocate a new [`ArenaBuffer`] from the allocator.
    pub fn allocate(&self, size: usize) -> Result<ArenaBuffer<S>, ArenaError> {
        let size = size as u64;
        let pages = size.div_ceil(self.page_size);

        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(pages.try_into().map_err(|_| ArenaError::AllocationFailed)?)
            .ok_or(ArenaError::AllocationFailed)?;

        let offset = allocation.offset as u64 * self.page_size;
        let address = self.storage.addr() + offset;

        debug_assert!(address + size <= self.storage.addr() + self.storage.size() as u64);

        Ok(ArenaBuffer {
            offset,
            address,
            requested_size: size as usize,
            allocation,
            storage: self.storage.clone(),
            allocator: self.allocator.clone(),
        })
    }
}

impl<S: Storage> std::fmt::Debug for ArenaBuffer<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ArenaBuffer {{ addr {}, size: {}, kind: {:?}, allocator: {:p}}}",
            self.address,
            self.requested_size,
            self.storage.storage_type(),
            Arc::as_ptr(&self.storage)
        )
    }
}

impl<S: Storage> ArenaBuffer<S> {
    /// Starting address of the buffer
    pub fn address(&self) -> u64 {
        self.address
    }

    /// Size of the buffer
    pub fn size(&self) -> usize {
        self.requested_size
    }
}

mod nixl {
    use super::super::nixl::*;
    use super::super::*;
    use super::*;

    impl<S: Storage> ArenaBuffer<S>
    where
        S: NixlRegisterableStorage,
    {
        /// Create a [`NixlRemoteDescriptor`] from the buffer.
        pub fn nixl_remote_descriptor(&self) -> Result<NixlRemoteDescriptor, ArenaError> {
            let agent = self.storage.nixl_agent_name();

            match agent {
                Some(agent) => {
                    // update storage with the buffer address and size
                    let storage = NixlStorage::from_storage_with_offset(
                        self.storage.as_ref(),
                        self.offset as usize,
                        self.requested_size,
                    )?;

                    Ok(NixlRemoteDescriptor::new(storage, agent))
                }
                _ => Err(ArenaError::NotRegisteredWithNixl),
            }
        }
    }

    impl<S: Storage> MemoryRegion for ArenaBuffer<S>
    where
        S: MemoryRegion,
    {
        unsafe fn as_ptr(&self) -> *const u8 {
            unsafe { Storage::as_ptr(self.storage.as_ref()) }
        }

        fn size(&self) -> usize {
            Storage::size(self.storage.as_ref())
        }
    }

    impl<S: Storage> NixlDescriptor for ArenaBuffer<S>
    where
        S: NixlDescriptor,
    {
        fn mem_type(&self) -> MemType {
            NixlDescriptor::mem_type(self.storage.as_ref())
        }

        fn device_id(&self) -> u64 {
            NixlDescriptor::device_id(self.storage.as_ref())
        }
    }
}

impl<S: Storage> Drop for ArenaBuffer<S> {
    fn drop(&mut self) {
        self.allocator.lock().unwrap().free(self.allocation);
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

    use super::*;
    use crate::block_manager::storage::SystemStorage;

    const PAGE_SIZE: usize = 4096;
    const PAGE_COUNT: usize = 10;
    const TOTAL_STORAGE_SIZE: usize = PAGE_SIZE * PAGE_COUNT;

    fn create_allocator() -> ArenaAllocator<SystemStorage> {
        let storage = SystemStorage::new(TOTAL_STORAGE_SIZE).unwrap();
        ArenaAllocator::new(storage, PAGE_SIZE).unwrap()
    }

    #[test]
    /// Tests successful creation of an `ArenaAllocator` with valid page size.
    /// Verifies that `ArenaAllocator::new` returns `Ok`.
    fn test_arena_allocator_new_success() {
        let storage = SystemStorage::new(TOTAL_STORAGE_SIZE).unwrap();
        let allocator_result = ArenaAllocator::new(storage, PAGE_SIZE);
        assert!(allocator_result.is_ok());
    }

    #[test]
    /// Tests `ArenaAllocator` creation with an invalid page size (not a power of 2).
    /// Verifies that `ArenaAllocator::new` returns an `ArenaError::PageSizeNotAligned` error.
    fn test_arena_allocator_new_invalid_page_size() {
        let storage = SystemStorage::new(TOTAL_STORAGE_SIZE).unwrap();
        let allocator_result = ArenaAllocator::new(storage, PAGE_SIZE + 1);
        assert!(allocator_result.is_err());
        assert_matches!(allocator_result, Err(ArenaError::PageSizeNotAligned));
    }

    #[test]
    /// Tests allocation of a single buffer that is a multiple of the page size.
    /// Verifies that the allocation is successful, the buffer has the correct size,
    /// and its address is the start of the storage area (as it's the first allocation).
    fn test_allocate_single_buffer() {
        let allocator = create_allocator();
        let buffer_size = PAGE_SIZE * 2;
        let buffer_result = allocator.allocate(buffer_size);
        assert!(buffer_result.is_ok());
        let buffer = buffer_result.unwrap();
        assert_eq!(buffer.size(), buffer_size);
        assert_eq!(buffer.address(), allocator.storage.addr()); // First allocation starts at addr
    }

    #[test]
    /// Tests allocation of multiple buffers of varying sizes (multiples of page size).
    /// Verifies that allocations are successful, buffers have correct sizes, and their
    /// addresses are correctly offset from each other based on previous allocations.
    fn test_allocate_multiple_buffers() {
        let allocator = create_allocator();
        let buffer_size1 = PAGE_SIZE * 2;
        let buffer1_result = allocator.allocate(buffer_size1);
        assert!(buffer1_result.is_ok());
        let buffer1 = buffer1_result.unwrap();
        assert_eq!(buffer1.size(), buffer_size1);
        assert_eq!(buffer1.address(), allocator.storage.addr());

        let buffer_size2 = PAGE_SIZE * 3;
        let buffer2_result = allocator.allocate(buffer_size2);
        assert!(buffer2_result.is_ok());
        let buffer2 = buffer2_result.unwrap();
        assert_eq!(buffer2.size(), buffer_size2);
        assert_eq!(
            buffer2.address(),
            allocator.storage.addr() + buffer_size1 as u64
        );
    }

    #[test]
    /// Tests allocation of a single buffer that consumes the entire storage space.
    /// Verifies that the allocation is successful and the buffer has the correct size.
    fn test_allocate_exact_size() {
        let allocator = create_allocator();
        let buffer_size = TOTAL_STORAGE_SIZE;
        let buffer_result = allocator.allocate(buffer_size);
        assert!(buffer_result.is_ok());
        let buffer = buffer_result.unwrap();
        assert_eq!(buffer.size(), buffer_size);
    }

    #[test]
    /// Tests an attempt to allocate a buffer larger than the total available storage.
    /// Verifies that the allocation fails with `ArenaError::AllocationFailed`.
    fn test_allocate_too_large() {
        let allocator = create_allocator();
        let buffer_size = TOTAL_STORAGE_SIZE + PAGE_SIZE;
        let buffer_result = allocator.allocate(buffer_size);
        assert!(buffer_result.is_err());
        assert_matches!(buffer_result, Err(ArenaError::AllocationFailed));
    }

    #[test]
    /// Tests the `Drop` implementation of `ArenaBuffer` for freeing allocated pages.
    /// It allocates a buffer, lets it go out of scope (triggering `drop`), and then
    /// attempts to reallocate a buffer of the same size. This second allocation should
    /// succeed and reuse the initially allocated space, starting at the storage address.
    fn test_buffer_drop_and_reallocate() {
        let allocator = create_allocator();
        // we can not allocate two buffers of `buffer_size` as it will exceed the total storage size
        // if the memory is properly returned, then we should be able to reallocate the same size buffer
        let buffer_size = PAGE_SIZE * 6;

        {
            let buffer1 = allocator.allocate(buffer_size).unwrap();
            assert_eq!(buffer1.size(), buffer_size);
            assert_eq!(buffer1.address(), allocator.storage.addr());
        } // buffer1 is dropped here, freeing its pages

        // Try to allocate a new buffer of the same size, it should succeed and reuse the space
        let buffer2_result = allocator.allocate(buffer_size);
        assert!(buffer2_result.is_ok());
        let buffer2 = buffer2_result.unwrap();
        assert_eq!(buffer2.size(), buffer_size);
        assert_eq!(buffer2.address(), allocator.storage.addr()); // Should be at the start again
    }

    #[test]
    /// Tests filling the arena with two buffers that together consume all available pages
    /// and then attempting one more small allocation, which should fail.
    /// Verifies that after the allocator is full, `ArenaError::AllocationFailed` is returned.
    fn test_allocate_fill_and_fail() {
        let allocator = create_allocator();
        let buffer_size_half = TOTAL_STORAGE_SIZE / 2; // Each takes 5 pages

        let buffer1 = allocator.allocate(buffer_size_half).unwrap();
        assert_eq!(buffer1.size(), buffer_size_half);

        let buffer2 = allocator.allocate(buffer_size_half).unwrap();
        assert_eq!(buffer2.size(), buffer_size_half);
        assert_eq!(
            buffer2.address(),
            allocator.storage.addr() + buffer_size_half as u64
        );

        // Now try to allocate one more page, should fail
        let buffer3_result = allocator.allocate(PAGE_SIZE);
        assert!(buffer3_result.is_err());
        assert_matches!(buffer3_result, Err(ArenaError::AllocationFailed));
    }

    #[test]
    /// Tests allocation of a single byte.
    /// Verifies that the allocation is successful and the buffer reports its size as 1.
    /// The actual page consumption is tested behaviorally in exhaustion tests.
    fn test_allocate_non_page_aligned_single_byte() {
        let allocator = create_allocator();
        let buffer = allocator.allocate(1).unwrap();
        assert_eq!(buffer.size(), 1);
        // Internal page allocation is behaviorally tested by exhaustion tests
    }

    #[test]
    /// Tests allocation of a size that is one byte less than a full page.
    /// Verifies that the allocation is successful and the buffer reports the correct size.
    /// The actual page consumption is tested behaviorally in exhaustion tests.
    fn test_allocate_non_page_aligned_almost_full_page() {
        let allocator = create_allocator();
        let buffer = allocator.allocate(PAGE_SIZE - 1).unwrap();
        assert_eq!(buffer.size(), PAGE_SIZE - 1);
    }

    #[test]
    /// Tests allocation of a size that is one byte more than a full page.
    /// Verifies that the allocation is successful and the buffer reports the correct size.
    /// This will consume two pages, which is tested behaviorally in exhaustion tests.
    fn test_allocate_non_page_aligned_just_over_one_page() {
        let allocator = create_allocator();
        let buffer = allocator.allocate(PAGE_SIZE + 1).unwrap();
        assert_eq!(buffer.size(), PAGE_SIZE + 1);
    }

    #[test]
    /// Tests a specific scenario of non-page-aligned allocations leading to arena exhaustion.
    /// Allocates `(PAGE_COUNT / 2 * PAGE_SIZE) + 1` bytes. This requires `(PAGE_COUNT / 2) + 1` pages.
    /// The first allocation should succeed. The second allocation of the same size should fail
    /// because not enough pages remain, verifying the page rounding and consumption logic.
    fn test_allocate_half_plus_one_byte_twice_exhausts_arena() {
        let allocator = create_allocator();
        let allocation_size = (PAGE_COUNT / 2 * PAGE_SIZE) + 1;
        // This allocation will require (PAGE_COUNT / 2) + 1 pages.
        // For PAGE_COUNT = 10, this is 5 * PAGE_SIZE + 1 bytes, requiring 6 pages.

        let buffer1_result = allocator.allocate(allocation_size);
        assert!(buffer1_result.is_ok(), "First allocation should succeed");
        let buffer1 = buffer1_result.unwrap();
        assert_eq!(buffer1.size(), allocation_size);
        let pages_for_first_alloc = (allocation_size as u64).div_ceil(allocator.page_size);
        assert_eq!(pages_for_first_alloc, (PAGE_COUNT / 2 + 1) as u64);

        // Second allocation of the same size should fail because we don't have enough pages left.
        // Remaining pages = PAGE_COUNT - pages_for_first_alloc
        // For PAGE_COUNT = 10, remaining = 10 - 6 = 4 pages.
        // We need (PAGE_COUNT / 2 + 1) = 6 pages.
        let buffer2_result = allocator.allocate(allocation_size);
        assert!(
            buffer2_result.is_err(),
            "Second allocation should fail due to insufficient pages"
        );
        assert_matches!(buffer2_result, Err(ArenaError::AllocationFailed));
    }

    #[test]
    /// Tests filling the arena with multiple non-page-aligned allocations that each consume more
    /// than one page due to rounding (specifically, `PAGE_SIZE + 1` bytes, consuming 2 pages each).
    /// After filling the arena based on this consumption, it verifies that a subsequent small
    /// allocation fails with `ArenaError::AllocationFailed`.
    fn test_fill_with_non_aligned_and_fail() {
        let allocator = create_allocator();
        // This test verifies that multiple small allocations, each consuming slightly more than one page
        // (thus taking two pages from the underlying offset_allocator), correctly fill the arena.
        // Let's allocate (PAGE_SIZE + 1) multiple times. Each will take 2 pages.

        let single_alloc_size = PAGE_SIZE + 1; // Will take 2 pages
        let num_possible_allocs = PAGE_COUNT / 2; // e.g., 10 / 2 = 5 such allocations

        let mut allocated_buffers = Vec::with_capacity(num_possible_allocs);

        for i in 0..num_possible_allocs {
            let buffer_result = allocator.allocate(single_alloc_size);
            assert!(buffer_result.is_ok(), "Allocation {} should succeed", i + 1);
            let buffer = buffer_result.unwrap();
            assert_eq!(buffer.size(), single_alloc_size);
            allocated_buffers.push(buffer);
        }

        // At this point, all pages should be consumed (num_possible_allocs * 2 pages)
        // So, allocating even 1 byte should fail.
        let final_alloc_result = allocator.allocate(1);
        assert!(
            final_alloc_result.is_err(),
            "Final allocation of 1 byte should fail as arena is full"
        );
        assert_matches!(final_alloc_result, Err(ArenaError::AllocationFailed));
    }
}
