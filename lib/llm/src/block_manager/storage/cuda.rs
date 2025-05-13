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

//! # CUDA Storage Support
//!
//! This module provides CUDA-specific storage implementations for the block manager.
//! It is conditionally compiled based on the `cuda` feature flag.
//!
//! ## Features
//!
//! The following types are available when the `cuda` feature is enabled:
//! - [`PinnedStorage`] - Page-locked host memory for efficient GPU transfers
//! - [`DeviceStorage`] - Direct GPU memory allocation
//!
//! ## Storage Allocators
//!
//! The module provides allocators for each storage type:
//! - [`PinnedAllocator`] - Creates pinned host memory allocations
//! - [`DeviceAllocator`] - Creates device memory allocations
//!
//! ## CUDA Context Management
//!
//! The module provides a singleton [`Cuda`] type for managing CUDA contexts:
//! - Thread-safe context management
//! - Lazy initialization of device contexts
//! - Automatic cleanup of resources
//!
//! ## Usage
//!
//! ### Using Allocators
//! ```rust
//! use dynamo_llm::block_manager::storage::{DeviceAllocator, PinnedAllocator, StorageAllocator};
//!
//! // Create a pinned memory allocator
//! let pinned_allocator = PinnedAllocator::default();
//! let pinned_storage = pinned_allocator.allocate(1024).unwrap();
//!
//! // Create a device memory allocator for a specific device
//! let device_allocator = DeviceAllocator::new(1).unwrap();  // Use device 1
//! let device_storage = device_allocator.allocate(1024).unwrap();
//! ```
//!
//! ### Memory Operations
//! ```rust
//! use dynamo_llm::block_manager::storage::{
//!     PinnedAllocator, StorageAllocator, Storage, StorageMemset
//! };
//!
//! // Initialize memory
//! let mut storage = PinnedAllocator::default().allocate(1024).unwrap();
//!
//! // Initialize memory
//! storage.memset(0, 0, 1024).unwrap();
//!
//! // Access memory through raw pointers (requires unsafe)
//! unsafe {
//!     let ptr = storage.as_mut_ptr();
//!     // Use the pointer...
//! }
//! ```
//!
//! ## Safety
//!
//! All CUDA operations are wrapped in safe Rust interfaces that ensure:
//! - Proper resource cleanup
//! - Thread safety
//! - Memory alignment requirements
//! - Error handling for CUDA operations

use super::*;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use cudarc::driver::{sys, CudaContext};

/// Trait for [Storage] types that can be accessed by CUDA
pub trait CudaAccessible: Storage {}

/// Trait for types that can provide a CUDA context.
pub trait CudaContextProivder {
    /// Get a referene to the [`CudaContext`].
    fn cuda_context(&self) -> &Arc<CudaContext>;
}

/// Singleton for managing CUDA contexts.
pub struct Cuda {
    contexts: HashMap<usize, Arc<CudaContext>>,
}

impl Cuda {
    // Private constructor
    fn new() -> Self {
        Self {
            contexts: HashMap::new(),
        }
    }

    /// Get a CUDA context for a specific device_id.
    /// If the context does not exist, it will return None.
    ///
    /// This will not lazily instantiate a context for a device. Use
    /// [Cuda::get_or_init_device]
    pub fn device(device_id: usize) -> Option<Arc<CudaContext>> {
        Cuda::instance()
            .lock()
            .unwrap()
            .get_existing_context(device_id)
    }

    /// Get or initialize a CUDA context for a specific device_id.
    /// If the context does not exist, it will be created or fail.
    ///
    /// This will lazily instantiate a context for a device. Use
    /// [CudaContextManager::device] to get an existing context.
    pub fn get_or_init_device(device_id: usize) -> Result<Arc<CudaContext>, StorageError> {
        Cuda::instance().lock().unwrap().get_context(device_id)
    }

    /// Check if a CUDA context exists for a specific device_id.
    pub fn is_initialized(device_id: usize) -> bool {
        Cuda::instance().lock().unwrap().has_context(device_id)
    }

    // Get the singleton instance
    fn instance() -> &'static Mutex<Cuda> {
        static INSTANCE: OnceLock<Mutex<Cuda>> = OnceLock::new();
        INSTANCE.get_or_init(|| Mutex::new(Cuda::new()))
    }

    // Get or create a CUDA context for a specific device
    fn get_context(&mut self, device_id: usize) -> Result<Arc<CudaContext>, StorageError> {
        // Check if we already have a context for this device
        if let Some(ctx) = self.contexts.get(&device_id) {
            return Ok(ctx.clone());
        }

        // Create a new context for this device
        let ctx = CudaContext::new(device_id)?;

        // Store the context
        self.contexts.insert(device_id, ctx.clone());

        Ok(ctx)
    }

    // Get a context if it exists, but don't create one
    fn get_existing_context(&self, device_id: usize) -> Option<Arc<CudaContext>> {
        self.contexts.get(&device_id).cloned()
    }

    // Check if a context exists for a device
    fn has_context(&self, device_id: usize) -> bool {
        self.contexts.contains_key(&device_id)
    }
}

/// Pinned host memory storage using CUDA page-locked memory
#[derive(Debug)]
pub struct PinnedStorage {
    ptr: u64,
    size: usize,
    handles: RegistrationHandles,
    ctx: Arc<CudaContext>,
}

impl Local for PinnedStorage {}
impl SystemAccessible for PinnedStorage {}
impl CudaAccessible for PinnedStorage {}

impl PinnedStorage {
    /// Create a new pinned storage with the given size
    pub fn new(ctx: &Arc<CudaContext>, size: usize) -> Result<Self, StorageError> {
        unsafe {
            ctx.bind_to_thread().map_err(StorageError::CudaError)?;

            let ptr = cudarc::driver::result::malloc_host(size, sys::CU_MEMHOSTALLOC_WRITECOMBINED)
                .map_err(StorageError::CudaError)?;

            let ptr = ptr as *mut u8;
            assert!(!ptr.is_null(), "Failed to allocate pinned memory");
            assert!(ptr.is_aligned(), "Pinned memory is not aligned");
            assert!(size < isize::MAX as usize);

            let ptr = ptr as u64;
            Ok(Self {
                ptr,
                size,
                handles: RegistrationHandles::new(),
                ctx: ctx.clone(),
            })
        }
    }
}

impl Drop for PinnedStorage {
    fn drop(&mut self) {
        self.handles.release();
        unsafe { cudarc::driver::result::free_host(self.ptr as _) }.unwrap();
    }
}

impl Storage for PinnedStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Pinned
    }

    fn addr(&self) -> u64 {
        self.ptr
    }

    fn size(&self) -> usize {
        self.size
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr as *mut u8
    }
}

impl CudaContextProivder for PinnedStorage {
    fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

impl RegisterableStorage for PinnedStorage {
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

impl StorageMemset for PinnedStorage {
    fn memset(&mut self, value: u8, offset: usize, size: usize) -> Result<(), StorageError> {
        if offset + size > self.size {
            return Err(StorageError::OperationFailed(
                "memset: offset + size > storage size".into(),
            ));
        }
        unsafe {
            let ptr = (self.ptr as *mut u8).add(offset);
            std::ptr::write_bytes(ptr, value, size);
        }
        Ok(())
    }
}

/// Allocator for PinnedStorage
pub struct PinnedAllocator {
    ctx: Arc<CudaContext>,
}

impl Default for PinnedAllocator {
    fn default() -> Self {
        Self {
            ctx: Cuda::get_or_init_device(0).expect("Failed to create CUDA context"),
        }
    }
}

impl PinnedAllocator {
    /// Create a new pinned allocator
    pub fn new() -> Result<Self, StorageError> {
        Ok(Self {
            ctx: Cuda::get_or_init_device(0)?,
        })
    }
}

impl StorageAllocator<PinnedStorage> for PinnedAllocator {
    fn allocate(&self, size: usize) -> Result<PinnedStorage, StorageError> {
        PinnedStorage::new(&self.ctx, size)
    }
}

/// CUDA device memory storage
#[derive(Debug)]
pub struct DeviceStorage {
    ptr: u64,
    size: usize,
    ctx: Arc<CudaContext>,
    handles: RegistrationHandles,
}

impl Local for DeviceStorage {}
impl CudaAccessible for DeviceStorage {}

impl DeviceStorage {
    /// Create a new device storage with the given size
    pub fn new(ctx: &Arc<CudaContext>, size: usize) -> Result<Self, StorageError> {
        ctx.bind_to_thread().map_err(StorageError::CudaError)?;
        let ptr =
            unsafe { cudarc::driver::result::malloc_sync(size).map_err(StorageError::CudaError)? };

        Ok(Self {
            ptr,
            size,
            ctx: ctx.clone(),
            handles: RegistrationHandles::new(),
        })
    }

    /// Get the CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

impl Storage for DeviceStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Device(self.ctx.cu_device() as u32)
    }

    fn addr(&self) -> u64 {
        self.ptr
    }

    fn size(&self) -> usize {
        self.size
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr as *mut u8
    }
}

impl CudaContextProivder for DeviceStorage {
    fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

impl Drop for DeviceStorage {
    fn drop(&mut self) {
        self.handles.release();
        unsafe { cudarc::driver::result::free_sync(self.ptr as _) }.unwrap();
    }
}

impl RegisterableStorage for DeviceStorage {
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

/// Allocator for DeviceStorage
pub struct DeviceAllocator {
    ctx: Arc<CudaContext>,
}

impl Default for DeviceAllocator {
    fn default() -> Self {
        Self {
            ctx: CudaContext::new(0).expect("Failed to create CUDA context"),
        }
    }
}

impl DeviceAllocator {
    /// Create a new device allocator
    pub fn new(device_id: usize) -> Result<Self, StorageError> {
        Ok(Self {
            ctx: Cuda::get_or_init_device(device_id)?,
        })
    }

    /// Get the CUDA context
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

impl StorageAllocator<DeviceStorage> for DeviceAllocator {
    fn allocate(&self, size: usize) -> Result<DeviceStorage, StorageError> {
        DeviceStorage::new(&self.ctx, size)
    }
}
