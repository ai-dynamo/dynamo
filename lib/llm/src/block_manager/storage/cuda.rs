// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # CUDA Storage Support
//!
//! This module provides CUDA-specific storage implementations for the block manager.
//!
//! ## Types
//!
//! - [`PinnedStorage`] - Page-locked host memory for efficient GPU transfers
//! - [`DeviceContext`] - Unified device context for multi-backend support
//! - [`StorageBackendOps`] - Trait abstracting backend-specific allocation
//! - [`CudaAccessible`] - Trait for CUDA-accessible storage types
//!
//! ## Storage Allocators
//!
//! - [`PinnedAllocator`] - Creates pinned host memory allocations
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
//! ```rust,ignore
//! use dynamo_llm::block_manager::storage::{PinnedAllocator, StorageAllocator};
//!
//! // Create a pinned memory allocator
//! let pinned_allocator = PinnedAllocator::default();
//! let pinned_storage = pinned_allocator.allocate(1024).unwrap();
//! ```

use super::*;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use cudarc::driver::CudaContext;
pub use dynamo_memory::StorageBackendOps;

/// Reuse `dynamo_memory::DeviceContext` — a type-erased wrapper around
/// `Arc<dyn StorageBackendOps>`. Construct via `DeviceContext::new(backend)`.
pub type DeviceContext = dynamo_memory::DeviceContext;

// ---------------------------------------------------------------------------
// CUDA traits & context management
// ---------------------------------------------------------------------------

/// Trait for [Storage] types that can be accessed by CUDA
pub trait CudaAccessible: Storage {}

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
    /// [Cuda::device_or_create]
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
    pub fn device_or_create(device_id: usize) -> Result<Arc<CudaContext>, StorageError> {
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
    pub fn get_existing_context(&self, device_id: usize) -> Option<Arc<CudaContext>> {
        self.contexts.get(&device_id).cloned()
    }

    // Check if a context exists for a device
    pub fn has_context(&self, device_id: usize) -> bool {
        self.contexts.contains_key(&device_id)
    }
}



#[cfg(all(test, feature = "testing-cuda"))]
mod tests {
    use super::*;

    /// Test PinnedStorage::new (deprecated) allocates usable pinned memory.
    #[allow(deprecated)]
    #[test]
    fn test_pinned_storage_new_without_numa() {
        let ctx = Cuda::device_or_create(0).expect("Failed to create CUDA context");
        let size = 8192;

        let mut storage =
            PinnedStorage::new(&ctx, size).expect("PinnedStorage::new should succeed");

        // Verify storage properties
        assert_eq!(storage.size(), size);
        assert_eq!(storage.storage_type(), StorageType::Pinned);
        assert_ne!(storage.addr(), 0, "Address should be non-zero");

        // Verify memory is accessible
        unsafe {
            let ptr = storage.as_mut_ptr();
            assert!(!ptr.is_null(), "Pointer should not be null");

            // Write a pattern to verify memory is usable
            for i in 0..size {
                std::ptr::write_volatile(ptr.add(i), (i & 0xFF) as u8);
            }

            // Read back and verify
            for i in 0..size {
                let val = std::ptr::read_volatile(ptr.add(i));
                assert_eq!(
                    val,
                    (i & 0xFF) as u8,
                    "Memory content mismatch at offset {}",
                    i
                );
            }
        }
    }
}
