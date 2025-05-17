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

// TODO: Add docs.
#![allow(missing_docs)]

//! # Storage Management
//!
//! This module provides a unified interface for managing different types of memory storage used in the block manager.
//! It handles various memory types including system memory, pinned memory, device memory, and remote storage through NIXL.
//!
//! ## Core Concepts
//!
//! ### Storage Types
//! The module defines [`Storage`] trait which is implemented for all storage types. The primary module provide a
//! [`Storage`] implementation for system memory via [`SystemStorage`].
//!
//! CUDA support is provided via the [`cuda`] module.
//! NIXL support is provided via the [`nixl`] module.
//!
//! ### Memory Registration
//! Storage objects can be registered with external libraries (like NIXL) through the [`RegisterableStorage`] trait.
//! This registration process:
//! - Creates a registration handle that ties the external library's state to the storage's lifetime
//! - Ensures proper cleanup through the [`Drop`] implementation of [`RegistrationHandles`]
//! - Provides a safe way to manage external library resources
//!
//! ### Safety and Performance
//! The module emphasizes:
//! - Memory safety through proper lifetime management
//! - Thread safety with appropriate trait bounds
//! - Performance optimization for different memory types
//! - Automatic resource cleanup
//!
//! ## Usage
//!
//! Storage objects are typically created through their respective allocators:
//! ```rust
//! use dynamo_llm::block_manager::storage::{SystemAllocator, StorageAllocator};
//!
//! let system_allocator = SystemAllocator::default();
//! let storage = system_allocator.allocate(1024).unwrap();
//! ```
//!
//! For registering with external libraries:
//! ```rust
//! use dynamo_llm::block_manager::storage::{
//!     PinnedAllocator, StorageAllocator,
//!     nixl::NixlRegisterableStorage
//! };
//! use nixl_sys::Agent as NixlAgent;
//!
//! // Create a NIXL agent
//! let agent = NixlAgent::new("my_agent").unwrap();
//!
//! let mut storage = PinnedAllocator::default().allocate(1024).unwrap();
//! storage.nixl_register(&agent, None).unwrap();
//! ```
//!
//! ## Implementation Details
//!
//! The module uses several key traits to provide a unified interface:
//! - [`Storage`] - Core trait for memory access
//! - [`RegisterableStorage`] - Support for external library registration
//! - [`StorageMemset`] - Memory initialization operations
//! - [`StorageAllocator`] - Factory for creating storage instances

pub mod cuda;
pub mod disk;
pub mod nixl;

pub use cuda::*;
pub use disk::*;

use std::{
    alloc::{alloc_zeroed, dealloc, Layout},
    collections::HashMap,
    fmt::Debug,
    ptr::NonNull,
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Result type for storage operations
pub type StorageResult<T> = std::result::Result<T, StorageError>;

/// Represents the type of storage used for a block
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageType {
    /// System memory
    System,

    /// CUDA device memory
    Device(u32),

    /// CUDA page-locked host memory
    Pinned,

    /// Disk memory
    Disk,

    /// Remote memory accessible through NIXL
    Nixl,

    /// Null storage
    Null,
}

/// A block that is local to the current worker
pub trait Local {}

/// A block that is remote to the current worker
pub trait Remote {}

/// Marker trait for [`Storage`] types that can be accessed by the standard
/// mechanisms of the system, e.g. `memcpy`, `memset`, etc.
pub trait SystemAccessible {}
pub trait CudaAccessible {}

/// Errors that can occur during storage operations
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum StorageError {
    #[error("Storage allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Storage not accessible: {0}")]
    NotAccessible(String),

    #[error("Invalid storage configuration: {0}")]
    InvalidConfig(String),

    #[error("Storage operation failed: {0}")]
    OperationFailed(String),

    #[error("CUDA error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),

    #[error("Registration key already exists: {0}")]
    RegistrationKeyExists(String),

    #[error("Handle not found for key: {0}")]
    HandleNotFound(String),

    #[error("NIXL error: {0}")]
    NixlError(#[from] nixl_sys::NixlError),
}

/// Core storage trait that provides access to memory regions
pub trait Storage: Debug + Send + Sync + 'static {
    /// Returns the type of storage
    fn storage_type(&self) -> StorageType;

    /// Returns the address of the storage
    fn addr(&self) -> u64;

    /// Returns the total size of the storage in bytes
    fn size(&self) -> usize;

    /// Get a raw pointer to the storage
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The pointer is not used after the storage is dropped
    /// - Access patterns respect the storage's thread safety model
    unsafe fn as_ptr(&self) -> *const u8;

    /// Get a raw mutable pointer to the storage
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The pointer is not used after the storage is dropped
    /// - No other references exist while the pointer is in use
    /// - Access patterns respect the storage's thread safety model
    unsafe fn as_mut_ptr(&mut self) -> *mut u8;
}

/// Extension trait for storage types that support memory setting operations
pub trait StorageMemset: Storage {
    /// Sets a region of memory to a specific value
    ///
    /// # Arguments
    /// * `value` - The value to set (will be truncated to u8)
    /// * `offset` - Offset in bytes from the start of the storage
    /// * `size` - Number of bytes to set
    ///
    /// # Safety
    /// The caller must ensure:
    /// - offset + size <= self.size()
    /// - No other references exist to the memory region being set
    fn memset(&mut self, value: u8, offset: usize, size: usize) -> Result<(), StorageError>;
}

/// Registerable storage is a [Storage] that can be associated with one or more
/// [RegistationHandle]s.
///
/// The core concept here is that the storage might be registered with a library
/// like NIXL or some other custom library which might make some system calls on
/// viritual addresses of the storage.
///
/// Before the [Storage] is dropped, the [RegistationHandle]s should be released.
///
/// The behavior is enforced via the [Drop] implementation for [RegistrationHandles].
pub trait RegisterableStorage: Storage + Send + Sync + 'static {
    /// Register a handle with a key
    /// If a handle with the same key already exists, an error is returned
    fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError>;

    /// Check if a handle is registered with a key
    fn is_registered(&self, key: &str) -> bool;

    /// Get a reference to the registration handle for a key
    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle>;
}

/// Designed to be implemented by any type that can be used as a handle to a
/// [RegisterableStorage].
///
/// See [RegisterableStorage] for more details.
pub trait RegistationHandle: std::any::Any + Send + Sync + 'static {
    /// Release the [RegistationHandle].
    /// This should be called when the external registration of this storage
    /// is no longer needed.
    ///
    /// Note: All [RegistrationHandle]s should be explicitly released before
    /// the [Storage] is dropped.
    fn release(&mut self);
}

/// A collection of [RegistrationHandle]s for a [RegisterableStorage].
///
/// This is used to ensure that all [RegistrationHandle]s are explicitly released
/// before the [RegisterableStorage] is dropped.
#[derive(Default)]
pub struct RegistrationHandles {
    handles: HashMap<String, Box<dyn RegistationHandle>>,
}

impl RegistrationHandles {
    /// Create a new [RegistrationHandles] instance
    pub fn new() -> Self {
        Self {
            handles: HashMap::new(),
        }
    }

    /// Register a handle with a key
    /// If a handle with the same key already exists, an error is returned
    pub fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        let key = key.to_string();
        if self.handles.contains_key(&key) {
            return Err(StorageError::RegistrationKeyExists(key));
        }
        self.handles.insert(key, handle);
        Ok(())
    }

    /// Release all handles
    fn release(&mut self) {
        for handle in self.handles.values_mut() {
            handle.release();
        }
        self.handles.clear();
    }

    /// Check if a handle is registered with a key
    fn is_registered(&self, key: &str) -> bool {
        self.handles.contains_key(key)
    }

    /// Get a reference to the registration handle for a key
    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.get(key).map(|h| h.as_ref())
    }
}

impl std::fmt::Debug for RegistrationHandles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RegistrationHandles {{ count: {:?} }}",
            self.handles.len()
        )
    }
}

impl Drop for RegistrationHandles {
    fn drop(&mut self) {
        if !self.handles.is_empty() {
            panic!("RegistrationHandles dropped with {} handles remaining; RegistrationHandles::release() needs to be explicitly called", self.handles.len());
        }
    }
}

/// Trait for types that can allocate specific Storage implementations.
pub trait StorageAllocator<S: Storage>: Send + Sync {
    /// Allocate storage of the specific type `S` with the given size in bytes.
    fn allocate(&self, size: usize) -> Result<S, StorageError>;
}

/// System memory storage implementation using pinned memory
#[derive(Debug)]
pub struct SystemStorage {
    ptr: NonNull<u8>,
    layout: Layout,
    len: usize,
    handles: RegistrationHandles,
}

unsafe impl Send for SystemStorage {}
unsafe impl Sync for SystemStorage {}

impl Local for SystemStorage {}
impl SystemAccessible for SystemStorage {}

impl SystemStorage {
    /// Create a new system storage with the given size
    ///
    /// # Safety
    /// This function allocates memory that will be freed when the SystemStorage is dropped.
    pub fn new(size: usize) -> Result<Self, StorageError> {
        // Create layout for the allocation, ensuring proper alignment
        let layout =
            Layout::array::<u8>(size).map_err(|e| StorageError::AllocationFailed(e.to_string()))?;

        // Allocate zeroed memory
        let ptr = unsafe {
            NonNull::new(alloc_zeroed(layout))
                .ok_or_else(|| StorageError::AllocationFailed("memory allocation failed".into()))?
        };

        Ok(Self {
            ptr,
            layout,
            len: size,
            handles: RegistrationHandles::new(),
        })
    }
}

impl Drop for SystemStorage {
    fn drop(&mut self) {
        self.handles.release();
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

impl Storage for SystemStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::System
    }

    fn addr(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }

    fn size(&self) -> usize {
        self.len
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl StorageMemset for SystemStorage {
    fn memset(&mut self, value: u8, offset: usize, size: usize) -> Result<(), StorageError> {
        if offset + size > self.len {
            return Err(StorageError::OperationFailed(
                "memset: offset + size > storage size".into(),
            ));
        }
        unsafe {
            let ptr = self.ptr.as_ptr().add(offset);
            std::ptr::write_bytes(ptr, value, size);
        }
        Ok(())
    }
}

impl RegisterableStorage for SystemStorage {
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

/// Allocator for SystemStorage
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemAllocator;

impl StorageAllocator<SystemStorage> for SystemAllocator {
    fn allocate(&self, size: usize) -> Result<SystemStorage, StorageError> {
        SystemStorage::new(size)
    }
}

#[allow(missing_docs)]
pub mod tests {
    use super::*;

    #[derive(Debug)]
    pub struct NullDeviceStorage {
        size: u64,
    }

    impl NullDeviceStorage {
        pub fn new(size: u64) -> Self {
            Self { size }
        }
    }

    impl Storage for NullDeviceStorage {
        fn storage_type(&self) -> StorageType {
            StorageType::Null
        }

        fn addr(&self) -> u64 {
            0
        }

        fn size(&self) -> usize {
            self.size as usize
        }

        unsafe fn as_ptr(&self) -> *const u8 {
            std::ptr::null()
        }

        unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
            std::ptr::null_mut()
        }
    }

    pub struct NullDeviceAllocator;

    impl StorageAllocator<NullDeviceStorage> for NullDeviceAllocator {
        fn allocate(&self, size: usize) -> Result<NullDeviceStorage, StorageError> {
            Ok(NullDeviceStorage::new(size as u64))
        }
    }

    #[derive(Debug)]
    pub struct NullHostStorage {
        size: u64,
    }

    impl NullHostStorage {
        pub fn new(size: u64) -> Self {
            Self { size }
        }
    }

    impl Storage for NullHostStorage {
        fn storage_type(&self) -> StorageType {
            StorageType::Null
        }

        fn addr(&self) -> u64 {
            0
        }

        fn size(&self) -> usize {
            self.size as usize
        }

        unsafe fn as_ptr(&self) -> *const u8 {
            std::ptr::null()
        }

        unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
            std::ptr::null_mut()
        }
    }

    pub struct NullHostAllocator;

    impl StorageAllocator<NullHostStorage> for NullHostAllocator {
        fn allocate(&self, size: usize) -> Result<NullHostStorage, StorageError> {
            Ok(NullHostStorage::new(size as u64))
        }
    }
}
