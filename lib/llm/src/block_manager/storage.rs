// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

pub mod arena;
pub mod cuda;
pub mod disk;
pub mod nixl;
pub mod object;
pub mod torch;
pub mod ze;

pub use cuda::*;
pub use disk::*;
pub use object::ObjectStorage;
pub use ze::Ze;
use torch::*;

/// Check if Level Zero is available on this system (without loading the library)
/// Returns true only if the loader library file exists
pub fn is_ze_available() -> bool {
    std::fs::metadata("/usr/lib/x86_64-linux-gnu/libze_loader.so.1")
        .or_else(|_| std::fs::metadata("/usr/lib/x86_64-linux-gnu/libze_loader.so"))
        .or_else(|_| std::fs::metadata("/usr/local/lib/libze_loader.so.1"))
        .or_else(|_| std::fs::metadata("/usr/local/lib/libze_loader.so"))
        .is_ok()
}

use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    collections::HashMap,
    fmt::Debug,
    ptr::NonNull,
    sync::Arc,
};

use cudarc::driver::CudaContext;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::block_manager::numa_allocator;

use self::hpu::{SynapseContext, free_host_synapse, malloc_host_pinned_synapse};
use self::ze::{ZeContext, free_host_ze, malloc_host_prefer_writecombined_ze};

/// Result type for storage operations
pub type StorageResult<T> = std::result::Result<T, StorageError>;

/// Represents the type of storage used for a block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum StorageType {
    /// System memory
    System,

    /// CUDA device memory
    Device(u32),

    /// CUDA page-locked host memory
    Pinned,

    /// Disk memory
    Disk(u64),

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

    #[error("Out of bounds: {0}")]
    OutOfBounds(String),
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

pub trait StorageTypeProvider {
    type StorageType: Storage;

    fn storage_type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<Self::StorageType>()
    }
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
            panic!(
                "RegistrationHandles dropped with {} handles remaining; RegistrationHandles::release() needs to be explicitly called",
                self.handles.len()
            );
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

/// Backend operations for device memory allocation.
///
/// This trait abstracts backend-specific allocation and deallocation operations
/// for both pinned (host) and device memory. Implementations for CUDA and Ze
/// backends are provided in their respective modules.
pub trait StorageBackendOps {
    /// Allocate pinned host memory
    ///
    /// # Safety
    /// Caller must ensure proper context binding if required by the backend
    unsafe fn alloc_pinned(&self, size: usize) -> Result<*mut u8, StorageError>;

    /// Free pinned host memory
    ///
    /// # Safety
    /// Caller must ensure ptr was allocated by this backend and size matches
    unsafe fn free_pinned(&self, ptr: u64, size: usize) -> Result<(), StorageError>;

    /// Allocate device memory, returns (ptr, device_id, metadata)
    ///
    /// # Safety
    /// Caller must ensure proper context binding if required by the backend
    unsafe fn alloc_device(
        &self,
        size: usize,
    ) -> Result<(u64, u32, DeviceStorageType), StorageError>;

    /// Free device memory
    ///
    /// # Safety
    /// Caller must ensure ptr was allocated by this backend
    unsafe fn free_device(&self, ptr: u64) -> Result<(), StorageError>;

    /// Get the device ID
    fn device_id(&self) -> u32;
}



#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceBackend {
    Cuda,
    Ze,
    Hpu,
}

#[derive(Clone)]
pub enum DeviceContext {
    Cuda(Arc<CudaContext>),
    Ze(Arc<ZeContext>),
    Synapse(Arc<SynapseContext>),
}

/// Trait for types that can provide a device context.
pub trait DeviceContextProvider {
    /// Get a reference-counted device context (CUDA or ZE).
    fn device_context(&self) -> Arc<DeviceContext>;
}

/// Pinned host memory storage using page-locked memory
#[derive(Debug)]
pub struct PinnedStorage {
    ptr: u64,
    size: usize,
    handles: RegistrationHandles,
    ctx: DeviceContext,
}

impl Local for PinnedStorage {}
impl SystemAccessible for PinnedStorage {}
impl CudaAccessible for PinnedStorage {}

impl PinnedStorage {
    /// Create a new pinned storage for the provided backend context.
    fn new(ctx: &DeviceContext, size: usize) -> Result<Self, StorageError> {
        let ptr = unsafe { ctx.alloc_pinned(size)? };

        assert!(!ptr.is_null(), "Failed to allocate pinned memory");
        assert!(ptr.is_aligned(), "Pinned memory is not aligned");
        assert!(size < isize::MAX as usize);

        Ok(Self {
            ptr: ptr as u64,
            size,
            handles: RegistrationHandles::new(),
            ctx: ctx.clone(),
        })
    }
}

impl Drop for PinnedStorage {
    fn drop(&mut self) {
        self.handles.release();
        if let Err(e) = unsafe { self.ctx.free_pinned(self.ptr, self.size) } {
            tracing::error!(
                "Failed to free pinned storage at 0x{:x} (size={}): {}",
                self.ptr,
                self.size,
                e
            );
        }
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

impl DeviceContextProvider for PinnedStorage {
    fn device_context(&self) -> Arc<DeviceContext> {
        match &self.ctx {
            DeviceContext::Cuda(ctx) => Arc::new(DeviceContext::Cuda(ctx.clone())),
            DeviceContext::Ze(ctx) => Arc::new(DeviceContext::Ze(ctx.clone())),
            DeviceContext::Synapse(ctx) => Arc::new(DeviceContext::Synapse(ctx.clone())),
        }
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
    ctx: DeviceContext,
}

impl Default for PinnedAllocator {
    fn default() -> Self {
        Self::new(0, DeviceBackend::Cuda).expect("Failed to create CUDA context")
    }
}

impl PinnedAllocator {
    /// Create a new pinned allocator
    pub fn new(device_id: usize, backend: DeviceBackend) -> Result<Self, StorageError> {
        match backend {
            DeviceBackend::Cuda => Ok(Self {
                ctx: DeviceContext::Cuda(cuda::Cuda::device_or_create(device_id)?),
            }),
            DeviceBackend::Ze => Ok(Self {
                ctx: DeviceContext::Ze(Ze::device_or_create(device_id)?),
            }),
            DeviceBackend::Hpu => Ok(Self {
                ctx: DeviceContext::Synapse(Synapse::device_or_create(device_id)?),
            }),
        }
    }

    pub fn backend(&self) -> DeviceBackend {
        match &self.ctx {
            DeviceContext::Cuda(_) => DeviceBackend::Cuda,
            DeviceContext::Ze(_) => DeviceBackend::Ze,
            DeviceContext::Synapse(_) => DeviceBackend::Hpu,
        }
    }

    pub fn ctx(&self) -> &DeviceContext {
        &self.ctx
    }
}

impl StorageAllocator<PinnedStorage> for PinnedAllocator {
    fn allocate(&self, size: usize) -> Result<PinnedStorage, StorageError> {
        PinnedStorage::new(&self.ctx, size)
    }
}


impl std::fmt::Debug for DeviceContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda(_) => write!(f, "DeviceContext::Cuda"),
            Self::Ze(_) => write!(f, "DeviceContext::Ze"),
            Self::Synapse(_) => write!(f, "DeviceContext::Synapse"),
        }
    }
}

impl StorageBackendOps for DeviceContext {
    unsafe fn alloc_pinned(&self, size: usize) -> Result<*mut u8, StorageError> {
        match self {
            Self::Cuda(ctx) => StorageBackendOps::alloc_pinned(ctx, size),
            Self::Ze(ctx) => StorageBackendOps::alloc_pinned(ctx, size),
            Self::Synapse(ctx) => StorageBackendOps::alloc_pinned(ctx, size),
        }
    }

    unsafe fn free_pinned(&self, ptr: u64, size: usize) -> Result<(), StorageError> {
        match self {
            Self::Cuda(ctx) => StorageBackendOps::free_pinned(ctx, ptr, size),
            Self::Ze(ctx) => StorageBackendOps::free_pinned(ctx, ptr, size),
            Self::Synapse(ctx) => StorageBackendOps::free_pinned(ctx, ptr, size),
        }
    }

    unsafe fn alloc_device(
        &self,
        size: usize,
    ) -> Result<(u64, u32, DeviceStorageType), StorageError> {
        match self {
            Self::Cuda(ctx) => StorageBackendOps::alloc_device(ctx, size),
            Self::Ze(ctx) => StorageBackendOps::alloc_device(ctx, size),
            Self::Synapse(ctx) => StorageBackendOps::alloc_device(ctx, size),
        }
    }

    unsafe fn free_device(&self, ptr: u64) -> Result<(), StorageError> {
        match self {
            Self::Cuda(ctx) => StorageBackendOps::free_device(ctx, ptr),
            Self::Ze(ctx) => StorageBackendOps::free_device(ctx, ptr),
            Self::Synapse(ctx) => StorageBackendOps::free_device(ctx, ptr),
        }
    }

    fn device_id(&self) -> u32 {
        match self {
            Self::Cuda(ctx) => StorageBackendOps::device_id(ctx),
            Self::Ze(ctx) => StorageBackendOps::device_id(ctx),
            Self::Synapse(ctx) => StorageBackendOps::device_id(ctx),
        }
    }
}

/// An enum indicating the type of device storage.
/// This is needed to ensure ownership of memory is correctly handled.
/// When building a [`DeviceStorage`] from a torch tensor, we need to ensure that
/// the torch tensor is not GCed until the [`DeviceStorage`] is dropped.
/// Because of this, we need to store a reference to the torch tensor in the [`DeviceStorage`]
#[derive(Debug)]
enum DeviceStorageType {
    Owned {
        _ze_device_buffer: Option<level_zero::DeviceBuffer>,
    }, // Memory that we allocated ourselves.
    Torch { _tensor: Arc<dyn TorchTensor> }, // Memory that came from a torch tensor.
}

/// Device memory storage for CUDA and ZE backends.
#[derive(Debug)]
pub struct DeviceStorage {
    pub(crate) ptr: u64,
    pub(crate) size: usize,
    pub(crate) ctx: DeviceContext,
    pub(crate) handles: RegistrationHandles,
    pub(crate) storage_type: DeviceStorageType,
}

impl Local for DeviceStorage {}
impl CudaAccessible for DeviceStorage {}

impl DeviceStorage {
    /// Create a new device storage with the given size.
    pub fn new(ctx: &DeviceContext, size: usize) -> Result<Self, StorageError> {
        let (ptr, _device_id, storage_type) = unsafe { ctx.alloc_device(size)? };

        Ok(Self {
            ptr,
            size,
            ctx: ctx.clone(),
            handles: RegistrationHandles::new(),
            storage_type: storage_type,
        })
    }

    /// Create a device storage from a torch tensor by dispatching on context backend.
    pub fn new_from_torch(
        ctx: &DeviceContext,
        tensor: Arc<dyn TorchTensor>,
    ) -> Result<Self, StorageError> {
        match ctx {
            DeviceContext::Cuda(ctx) => Self::new_from_torch_cuda(ctx, tensor),
            DeviceContext::Ze(ctx) => Self::new_from_torch_ze(ctx, tensor),
            DeviceContext::Synapse(ctx) => Self::new_from_torch_synapse(ctx, tensor),
        }
    }

    pub fn backend(&self) -> DeviceBackend {
        match &self.ctx {
            DeviceContext::Cuda(_) => DeviceBackend::Cuda,
            DeviceContext::Ze(_) => DeviceBackend::Ze,
            DeviceContext::Synapse(_) => DeviceBackend::Hpu,
        }
    }

    pub fn context(&self) -> Arc<DeviceContext> {
        match &self.ctx {
            DeviceContext::Cuda(ctx) => Arc::new(DeviceContext::Cuda(ctx.clone())),
            DeviceContext::Ze(ctx) => Arc::new(DeviceContext::Ze(ctx.clone())),
            DeviceContext::Synapse(ctx) => Arc::new(DeviceContext::Synapse(ctx.clone())),
        }
    }
    pub fn device_storage_type(&self) -> &DeviceStorageType {
        &self.storage_type
    }
}

impl Storage for DeviceStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Device(self.ctx.device_id())
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

impl Drop for DeviceStorage {
    fn drop(&mut self) {
        self.handles.release();
        match &self.storage_type {
            DeviceStorageType::Owned { .. } => {
                if let Err(e) = unsafe { self.ctx.free_device(self.ptr) } {
                    tracing::error!(
                        "Failed to free device storage at 0x{:x} (size={}): {}",
                        self.ptr,
                        self.size,
                        e
                    );
                }
            }
            DeviceStorageType::Torch { .. } => {
                // Do nothing. The torch storage is responsible for cleaning up itself.
            }
        }
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
    ctx: DeviceContext,
}

impl Default for DeviceAllocator {
    fn default() -> Self {
        Self::new(0, DeviceBackend::Cuda).expect("Failed to create CUDA context")
    }
}

impl DeviceAllocator {
    /// Create a new device allocator
    pub fn new(device_id: usize, backend: DeviceBackend) -> Result<Self, StorageError> {
        let ctx = match backend {
            DeviceBackend::Cuda => DeviceContext::Cuda(cuda::Cuda::device_or_create(device_id)?),
            DeviceBackend::Ze => DeviceContext::Ze(Ze::device_or_create(device_id)?),
            DeviceBackend::Hpu => DeviceContext::Synapse(Synapse::device_or_create(device_id)?),
        };
        Ok(Self { ctx })
    }

    pub fn ctx(&self) -> Arc<DeviceContext> {
        match &self.ctx {
            DeviceContext::Cuda(ctx) => Arc::new(DeviceContext::Cuda(ctx.clone())),
            DeviceContext::Ze(ctx) => Arc::new(DeviceContext::Ze(ctx.clone())),
            DeviceContext::Synapse(ctx) => Arc::new(DeviceContext::Synapse(ctx.clone())),
        }
    }
}

impl StorageAllocator<DeviceStorage> for DeviceAllocator {
    fn allocate(&self, size: usize) -> Result<DeviceStorage, StorageError> {
        DeviceStorage::new(&self.ctx, size)
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

#[cfg(all(test, feature = "testing-cuda"))]
mod cuda_tests {
    use std::sync::Arc;

    use super::*;

    #[derive(Debug, Clone)]
    struct MockTensor {
        device: TorchDevice,
        data_ptr: u64,
        size_bytes: usize,
    }

    impl MockTensor {
        pub fn new(device: TorchDevice, data_ptr: u64, size_bytes: usize) -> Self {
            Self {
                device,
                data_ptr,
                size_bytes,
            }
        }
    }

    impl TorchTensor for MockTensor {
        fn device(&self) -> TorchDevice {
            self.device.clone()
        }

        fn data_ptr(&self) -> u64 {
            self.data_ptr
        }

        fn size_bytes(&self) -> usize {
            self.size_bytes
        }

        fn shape(&self) -> Vec<usize> {
            vec![self.size_bytes]
        }

        fn stride(&self) -> Vec<usize> {
            vec![1]
        }
    }

    #[test]
    fn test_device_storage_from_torch_valid_tensor() {
        let ctx = cuda::Cuda::device_or_create(0).expect("Failed to create CUDA context");
        let allocator = DeviceAllocator::new(0, DeviceBackend::Cuda).expect("Failed to create CUDA allocator");
        let size_bytes = 1024;
        let device_ctx = DeviceContext::Cuda(ctx.clone());

        let actual_storage =
            std::mem::ManuallyDrop::new(DeviceStorage::new(&device_ctx, size_bytes).unwrap());

        let tensor = MockTensor::new(TorchDevice::Cuda(0), actual_storage.addr(), size_bytes);

        let storage =
            DeviceStorage::new_from_torch(allocator.ctx().as_ref(), Arc::new(tensor)).unwrap();

        assert_eq!(storage.size(), size_bytes);
        assert_eq!(storage.storage_type(), StorageType::Device(0));
        assert_eq!(storage.addr(), actual_storage.addr());
    }

    #[test]
    fn test_device_storage_from_torch_cpu_tensor_fails() {
        let ctx = cuda::Cuda::device_or_create(0).expect("Failed to create CUDA context");
        let allocator = DeviceAllocator::new(0, DeviceBackend::Cuda).expect("Failed to create CUDA allocator");
        let size_bytes = 1024;
        let device_ctx = DeviceContext::Cuda(ctx.clone());

        let actual_storage = DeviceStorage::new(&device_ctx, size_bytes).unwrap();

        let tensor = MockTensor::new(
            TorchDevice::Other("cpu".to_string()),
            actual_storage.addr(),
            size_bytes,
        );

        let result = DeviceStorage::new_from_torch(allocator.ctx().as_ref(), Arc::new(tensor));
        assert!(result.is_err());

        if let Err(StorageError::InvalidConfig(msg)) = result {
            assert!(msg.contains("Tensor is not CUDA"));
        } else {
            panic!("Expected InvalidConfig error for CPU tensor");
        }
    }

    #[test]
    fn test_device_storage_wrong_device() {
        let ctx = cuda::Cuda::device_or_create(0).expect("Failed to create CUDA context");
        let allocator = DeviceAllocator::new(0, DeviceBackend::Cuda).expect("Failed to create CUDA allocator");
        let size_bytes = 1024;
        let device_ctx = DeviceContext::Cuda(ctx.clone());

        let actual_storage = DeviceStorage::new(&device_ctx, size_bytes).unwrap();

        let tensor = MockTensor::new(TorchDevice::Cuda(1), actual_storage.addr(), size_bytes);

        let result = DeviceStorage::new_from_torch(allocator.ctx().as_ref(), Arc::new(tensor));
        assert!(result.is_err());
    }

    #[test]
    fn test_malloc_host_prefer_writecombined_allocates_memory() {
        let ctx = cuda::Cuda::device_or_create(0).expect("Failed to create CUDA context");
        let size = 4096;

        unsafe {
            ctx.bind_to_thread().expect("Failed to bind CUDA context");

            let ptr = cuda::malloc_host_prefer_writecombined(size)
                .expect("malloc_host_prefer_writecombined should succeed");

            assert!(!ptr.is_null(), "Allocated pointer should not be null");

            std::ptr::write_volatile(ptr, 0xAB);
            let val = std::ptr::read_volatile(ptr);
            assert_eq!(val, 0xAB, "Should be able to write and read pinned memory");

            cudarc::driver::result::free_host(ptr as _).expect("Failed to free pinned memory");
        }
    }

    #[test]
    fn test_pinned_storage_new_without_numa() {
        assert!(
            !numa_allocator::is_numa_enabled(),
            "NUMA should be disabled for this test"
        );

        let size = 8192;
        let allocator =
            PinnedAllocator::new(0, DeviceBackend::Cuda).expect("Failed to create pinned allocator");
        let mut storage = allocator
            .allocate(size)
            .expect("PinnedStorage allocation should succeed");

        assert_eq!(storage.size(), size);
        assert_eq!(storage.storage_type(), StorageType::Pinned);
        assert_ne!(storage.addr(), 0, "Address should be non-zero");

        unsafe {
            let ptr = storage.as_mut_ptr();
            assert!(!ptr.is_null(), "Pointer should not be null");

            for i in 0..size {
                std::ptr::write_volatile(ptr.add(i), (i & 0xFF) as u8);
            }

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

// Comment out Nixl-related code for now
/*
pub trait NixlDescriptor: Storage {
    fn as_nixl_descriptor(&self) -> NixlMemoryDescriptor<'_, BlockKind, IsImmutable>;
    fn as_nixl_descriptor_mut(&mut self) -> NixlMemoryDescriptor<'_, BlockKind, IsMutable>;
}

impl NixlDescriptor for SystemStorage {
    fn as_nixl_descriptor(&self) -> NixlMemoryDescriptor<'_, BlockKind, IsImmutable> {
        NixlMemoryDescriptor::new(self.as_ptr() as *const u8, self.size())
    }

    fn as_nixl_descriptor_mut(&mut self) -> NixlMemoryDescriptor<'_, BlockKind, IsMutable> {
        NixlMemoryDescriptor::new_mut(self.as_mut_ptr() as *mut u8, self.size())
    }
}

impl NixlDescriptor for PinnedStorage {
    fn as_nixl_descriptor(&self) -> NixlMemoryDescriptor<'_, BlockKind, IsImmutable> {
        NixlMemoryDescriptor::new(self.as_ptr() as *const u8, self.size())
    }

    fn as_nixl_descriptor_mut(&mut self) -> NixlMemoryDescriptor<'_, BlockKind, IsMutable> {
        NixlMemoryDescriptor::new_mut(self.as_mut_ptr() as *mut u8, self.size())
    }
}

impl NixlDescriptor for DeviceStorage {
    fn as_nixl_descriptor(&self) -> NixlMemoryDescriptor<'_, BlockKind, IsImmutable> {
        NixlMemoryDescriptor::new(self.as_ptr() as *const u8, self.size())
    }

    fn as_nixl_descriptor_mut(&mut self) -> NixlMemoryDescriptor<'_, BlockKind, IsMutable> {
        NixlMemoryDescriptor::new_mut(self.as_mut_ptr() as *mut u8, self.size())
    }
}
*/
