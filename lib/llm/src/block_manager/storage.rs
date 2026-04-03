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
//! ```rust,ignore
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
use torch::*;

use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    collections::HashMap,
    fmt::Debug,
    ptr::NonNull,
    sync::Arc,
};

use cudarc::driver::CudaContext;
use dynamo_memory::MemoryDescriptor as _;
use serde::{Deserialize, Serialize};
use thiserror::Error;

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

impl From<dynamo_memory::StorageError> for StorageError {
    fn from(e: dynamo_memory::StorageError) -> Self {
        match e {
            dynamo_memory::StorageError::AllocationFailed(s) => StorageError::AllocationFailed(s),
            dynamo_memory::StorageError::OperationFailed(s) => StorageError::OperationFailed(s),
            dynamo_memory::StorageError::Cuda(e) => StorageError::Cuda(e),
            dynamo_memory::StorageError::Nixl(e) => StorageError::NixlError(e),
            e => StorageError::OperationFailed(e.to_string()),
        }
    }
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

// ---------------------------------------------------------------------------
// DeviceStorage (multi-backend)
// ---------------------------------------------------------------------------

/// An enum indicating the type of device storage.
/// This is needed to ensure ownership of memory is correctly handled.
/// When building a [`DeviceStorage`] from a torch tensor, we need to ensure that
/// the torch tensor is not GCed until the [`DeviceStorage`] is dropped.
/// For ZE backend, the device buffer must be kept alive so its Drop impl frees
/// the memory.
pub(crate) enum DeviceStorageType {
    Owned {
        /// Backend-specific ownership handle. For Ze, holds a `DeviceBuffer`
        /// whose `Drop` frees memory. `None` for CUDA (freed via driver API).
        _backend_handle: Option<Box<dyn std::any::Any + Send + Sync>>,
    },
    Torch {
        _tensor: Arc<dyn TorchTensor>,
    },
}

impl std::fmt::Debug for DeviceStorageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned { .. } => write!(f, "Owned"),
            Self::Torch { .. } => write!(f, "Torch"),
        }
    }
}

/// Device memory storage supporting CUDA and ZE backends.
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
    /// Create a new device storage using a [`DeviceContext`].
    pub fn new(ctx: &DeviceContext, size: usize) -> Result<Self, StorageError> {
        let (ptr, _device_id, metadata) =
            unsafe { ctx.backend.alloc_device(size)? };
        Ok(Self {
            ptr,
            size,
            ctx: ctx.clone(),
            handles: RegistrationHandles::new(),
            storage_type: DeviceStorageType::Owned {
                _backend_handle: metadata,
            },
        })
    }

    pub fn new_from_torch(
        ctx: &DeviceContext,
        tensor: Arc<dyn TorchTensor>,
    ) -> Result<Self, StorageError> {
        let device = tensor.device();
        let desc = dynamo_memory::TorchTensorDescriptor {
            data_ptr: tensor.data_ptr(),
            size_bytes: tensor.size_bytes(),
            is_cuda: matches!(device, TorchDevice::Cuda(_)),
            is_xpu: torch::is_ze(tensor.as_ref()),
            device_id: match device {
                TorchDevice::Cuda(id) => id as u32,
                _ => 0,
            },
        };

        let (ptr, size) = ctx.backend.new_from_torch(&desc)?;

        Ok(Self {
            ptr,
            size,
            ctx: ctx.clone(),
            handles: RegistrationHandles::new(),
            storage_type: DeviceStorageType::Torch { _tensor: tensor },
        })
    }

    /// Get the device context.
    pub fn device_ctx(&self) -> &DeviceContext {
        &self.ctx
    }
}

impl Storage for DeviceStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Device(self.ctx.backend.device_id())
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
            DeviceStorageType::Owned { _backend_handle } => {
                if _backend_handle.is_some() {
                    // ZE: memory is freed by the DeviceBuffer's Drop impl
                    // (stored in the _backend_handle box).
                } else {
                    // CUDA: free via driver API.
                    unsafe { cudarc::driver::result::free_sync(self.ptr as _) }.unwrap()
                }
            }
            DeviceStorageType::Torch { _tensor } => {
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

/// Backend selector for device allocation.
#[derive(Debug, Clone, Copy)]
pub enum DeviceBackend {
    Cuda,
    Ze,
}

impl DeviceBackend {
    /// Create the backend context for the given device.
    pub fn create(&self, device_id: usize) -> Result<DeviceContext, StorageError> {
        match self {
            DeviceBackend::Cuda => Ok(DeviceContext::new(cuda::Cuda::device_or_create(device_id)?)),
            DeviceBackend::Ze => Ok(DeviceContext::new(dynamo_memory::Ze::device_or_create(device_id)?)),
        }
    }

    /// Parse a device type string into a DeviceBackend.
    pub fn from_str(s: &str) -> Result<Self, StorageError> {
        match s {
            "cuda" => Ok(DeviceBackend::Cuda),
            "xpu" | "ze" => Ok(DeviceBackend::Ze),
            other => Err(StorageError::OperationFailed(format!(
                "Unsupported device_type: {}",
                other
            ))),
        }
    }
}

/// Allocator for DeviceStorage
pub struct DeviceAllocator {
    ctx: DeviceContext,
}

impl DeviceAllocator {
    /// Create a new device allocator with the default CUDA backend.
    pub fn new(device_id: usize) -> Result<Self, StorageError> {
        Self::new_with_backend(device_id, DeviceBackend::Cuda)
    }

    /// Create a new device allocator with the specified backend.
    pub fn new_with_backend(device_id: usize, backend: DeviceBackend) -> Result<Self, StorageError> {
        Ok(Self {
            ctx: backend.create(device_id)?,
        })
    }

    pub fn ctx(&self) -> &DeviceContext {
        &self.ctx
    }
}

impl Default for DeviceAllocator {
    fn default() -> Self {
        Self {
            ctx: DeviceContext::new(CudaContext::new(0).expect("Failed to create CUDA context")),
        }
    }
}

impl StorageAllocator<DeviceStorage> for DeviceAllocator {
    fn allocate(&self, size: usize) -> Result<DeviceStorage, StorageError> {
        DeviceStorage::new(&self.ctx, size)
    }
}

// ---------------------------------------------------------------------------
// SystemStorage
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// PinnedStorage
// ---------------------------------------------------------------------------

/// Pinned host memory storage using CUDA page-locked memory.
/// Wraps [`dynamo_memory::PinnedStorage`] and adds registration handle support.
#[derive(Debug)]
pub struct PinnedStorage {
    inner: dynamo_memory::PinnedStorage,
    handles: RegistrationHandles,
}

impl Local for PinnedStorage {}
impl SystemAccessible for PinnedStorage {}
impl CudaAccessible for PinnedStorage {}

impl PinnedStorage {
    /// Create a new pinned storage with the given size.
    ///
    /// Uses write-combined allocation with NUMA-awareness when enabled.
    /// Prefer [`new_for_device`](Self::new_for_device) for new code.
    ///
    /// TODO(KVBM-336): remove PinnedStorage::new in the future
    #[deprecated(since = "1.0.0", note = "Use PinnedStorage::new_for_device instead")]
    pub fn new(ctx: &Arc<CudaContext>, size: usize) -> Result<Self, StorageError> {
        let inner =
            dynamo_memory::PinnedStorage::new_for_device(size, Some(ctx.cu_device() as u32))?;
        Ok(Self {
            inner,
            handles: RegistrationHandles::new(),
        })
    }

    /// Create a new pinned storage, optionally NUMA-aware for a specific GPU.
    ///
    /// Delegates NUMA-aware allocation and write-combined selection to
    /// [`dynamo_memory::PinnedStorage::new_for_device`].
    ///
    /// When `device_id` is `None`, allocates on device 0 without NUMA awareness.
    pub fn new_for_device(size: usize, device_id: Option<u32>) -> Result<Self, StorageError> {
        // Warn once if the legacy opt-in env var is still set.
        static DEPRECATION_WARN: std::sync::Once = std::sync::Once::new();
        if std::env::var("DYN_KVBM_ENABLE_NUMA").is_ok() {
            DEPRECATION_WARN.call_once(|| {
                tracing::warn!(
                    "DYN_KVBM_ENABLE_NUMA is deprecated for PinnedStorage::new_for_device; \
                     NUMA is now enabled by default. Use DYN_MEMORY_DISABLE_NUMA=1 to disable."
                );
            });
        }
        let inner = dynamo_memory::PinnedStorage::new_for_device(size, device_id)?;
        Ok(Self {
            inner,
            handles: RegistrationHandles::new(),
        })
    }

    /// Create a new pinned storage using an existing [`DeviceContext`].
    pub fn new_with_context(size: usize, ctx: &DeviceContext) -> Result<Self, StorageError> {
        let inner = dynamo_memory::PinnedStorage::new_with_context(size, ctx)?;
        Ok(Self {
            inner,
            handles: RegistrationHandles::new(),
        })
    }
}

impl Drop for PinnedStorage {
    fn drop(&mut self) {
        self.handles.release();
        // inner Drop handles free_host
    }
}

impl Storage for PinnedStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Pinned
    }

    fn addr(&self) -> u64 {
        self.inner.addr() as u64
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        unsafe { self.inner.as_ptr() }
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        unsafe { self.inner.as_mut_ptr() }
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
        if offset + size > self.inner.size() {
            return Err(StorageError::OperationFailed(
                "memset: offset + size > storage size".into(),
            ));
        }
        unsafe {
            let ptr = self.inner.as_mut_ptr().add(offset);
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
        Self {
            ctx: DeviceContext::new(Cuda::device_or_create(0).expect("Failed to create CUDA context")),
        }
    }
}

impl PinnedAllocator {
    /// Create a new pinned allocator for the specified device (CUDA backend).
    ///
    /// The device_id determines which NUMA node pinned memory will be allocated
    /// on when NUMA-aware allocation is enabled.
    pub fn new(device_id: usize) -> Result<Self, StorageError> {
        Ok(Self {
            ctx: DeviceContext::new(Cuda::device_or_create(device_id)?),
        })
    }

    /// Create a new pinned allocator from an existing DeviceContext.
    ///
    /// This supports both CUDA and ZE backends.
    pub fn new_with_context(ctx: DeviceContext) -> Self {
        Self { ctx }
    }
}

impl StorageAllocator<PinnedStorage> for PinnedAllocator {
    fn allocate(&self, size: usize) -> Result<PinnedStorage, StorageError> {
        PinnedStorage::new_with_context(size, &self.ctx)
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
mod device_storage_tests {
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
        let cuda_ctx = cuda::Cuda::device_or_create(0).expect("Failed to create CUDA context");
        let ctx = DeviceContext::new(cuda_ctx);
        let size_bytes = 1024;

        let actual_storage =
            std::mem::ManuallyDrop::new(DeviceStorage::new(&ctx, size_bytes).unwrap());

        let tensor = MockTensor::new(TorchDevice::Cuda(0), actual_storage.addr(), size_bytes);

        let storage = DeviceStorage::new_from_torch(&ctx, Arc::new(tensor)).unwrap();

        assert_eq!(storage.size(), size_bytes);
        assert_eq!(storage.storage_type(), StorageType::Device(0));
        assert_eq!(storage.addr(), actual_storage.addr());
    }

    #[test]
    fn test_device_storage_from_torch_cpu_tensor_fails() {
        let cuda_ctx = cuda::Cuda::device_or_create(0).expect("Failed to create CUDA context");
        let ctx = DeviceContext::new(cuda_ctx);
        let size_bytes = 1024;

        let actual_storage = DeviceStorage::new(&ctx, size_bytes).unwrap();

        let tensor = MockTensor::new(
            TorchDevice::Other("cpu".to_string()),
            actual_storage.addr(),
            size_bytes,
        );

        let result = DeviceStorage::new_from_torch(&ctx, Arc::new(tensor));
        assert!(result.is_err());

        if let Err(StorageError::InvalidConfig(msg)) = result {
            assert!(msg.contains("Tensor is not CUDA"));
        } else {
            panic!("Expected InvalidConfig error for CPU tensor");
        }
    }

    #[test]
    fn test_device_storage_wrong_device() {
        let cuda_ctx = cuda::Cuda::device_or_create(0).expect("Failed to create CUDA context");
        let ctx = DeviceContext::new(cuda_ctx);
        let size_bytes = 1024;

        let actual_storage = DeviceStorage::new(&ctx, size_bytes).unwrap();

        let tensor = MockTensor::new(TorchDevice::Cuda(1), actual_storage.addr(), size_bytes);

        let result = DeviceStorage::new_from_torch(&ctx, Arc::new(tensor));
        assert!(result.is_err());
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
