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

//! Storage object representing large single slabs of bytes.
//!
//! There are three types denoted by [StorageType]
//!
//! - [StorageType::Device]: A pointer to a device memory allocation
//! - [StorageType::Pinned]: A pointer to a pinned memory allocation from cudaMallocHost
//! - [StorageType::System]: A pointer to a system memory allocation from malloc/calloc or
//!   other forms of heap allocation.
//!
//! Use [StorageType::System] Grace and other embedded platforms.
//!
//! Use [StorageType::Pinned] and [StorageType::Device] on traditional x86 platforms.
//!
//! WARNING: [Storage] and [OwnedStorage] are not Rust safe objects. For KV blocks, we use
//! [Storage]-like stabs to form [KvLayers][super::layer::KvLayer], both of which do not
//! conform to Rust's ownership or safety guarantees.
//!
//! As the underlying cuda kernels have ownership policies, they are not guarantees, nor are
//! they enforceable at this level by the Rust compiler.
//!
//! The first unit of ownership that will be Rust safe is the [KvBlock][super::KvBlock].

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use dynemo_runtime::{error, raise, Result};
use ndarray::{ArrayViewMut, Dimension, IxDyn, ShapeBuilder};
use std::any::Any;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageType {
    Device(usize),
    Pinned,
    System, // todo: for grace
}

/// Represents the data type of tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
}

impl DType {
    /// Get the size of the data type in bytes
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::U8 => 1,
            DType::U16 => 2,
            DType::U32 => 4,
            DType::U64 => 8,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
        }
    }
}

extern "C" {
    fn cuda_malloc_host(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cuda_free_host(ptr: *mut c_void) -> i32;
    fn cuda_memcpy_async(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        stream: *mut c_void,
    ) -> i32;
}

pub trait Storage: std::fmt::Debug {
    /// Get memory pointer as a u64 for direct indexing
    fn get_pointer(&self) -> u64;

    /// Get the total storage size in bytes
    fn storage_size(&self) -> usize;

    /// Get the storage type of the tensor
    fn storage_type(&self) -> StorageType;

    /// Create a view of the tensor
    fn view<const D: usize>(
        &self,
        shape: [usize; D],
        dtype: DType,
    ) -> Result<TensorView<'_, Self, D>>
    where
        Self: Sized,
    {
        TensorView::new(self, shape, dtype.size_in_bytes())
    }
}

#[derive(Clone)]
pub struct OwnedStorage {
    storage: Arc<dyn Storage>,
}

impl OwnedStorage {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self { storage }
    }

    pub fn create(bytes: usize, storage_type: StorageType) -> Result<Self> {
        match storage_type {
            StorageType::Device(device) => Self::create_device_array(bytes, device),
            StorageType::Pinned => Self::create_pinned_array(bytes),
            StorageType::System => {
                raise!("System memory not yet supported");
            }
        }
    }

    pub fn create_device_array(bytes: usize, device: usize) -> Result<Self> {
        let device_storage = DeviceStorageOwned::new(bytes, device)?;
        Ok(Self::new(Arc::new(device_storage)))
    }

    pub fn create_pinned_array(bytes: usize) -> Result<Self> {
        let pinned_memory = CudaPinnedMemory::new(bytes)?;
        Ok(Self::new(Arc::new(pinned_memory)))
    }

    pub fn byo_device_array(
        device_ptr: u64,
        bytes: usize,
        device: usize,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self> {
        let device_storage = DeviceStorageFromAny::new(owner, device_ptr, bytes, device);
        Ok(Self::new(Arc::new(device_storage)))
    }
}

impl Storage for OwnedStorage {
    fn get_pointer(&self) -> u64 {
        self.storage.get_pointer()
    }

    fn storage_size(&self) -> usize {
        self.storage.storage_size()
    }

    fn storage_type(&self) -> StorageType {
        self.storage.storage_type()
    }
}

impl std::fmt::Debug for OwnedStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedStorage")
            .field("storage_type", &self.storage.storage_type())
            .finish()
    }
}

pub struct DeviceStorageOwned {
    bytes: usize,
    cuda_device: Arc<CudaContext>,
    cuda_slice: Arc<CudaSlice<u8>>,
}

impl DeviceStorageOwned {
    pub fn new(bytes: usize, device: usize) -> Result<Self> {
        let cuda_device = CudaContext::new(device)?;
        let cuda_slice = cuda_device.default_stream().alloc_zeros::<u8>(bytes)?;
        cuda_device.default_stream().synchronize()?;

        Ok(Self {
            bytes,
            cuda_device,
            cuda_slice: Arc::new(cuda_slice),
        })
    }

    pub fn device_ptr(&self) -> *const c_void {
        let ptr = self.cuda_slice.device_ptr();
        (*ptr) as *const c_void
    }

    pub fn context(&self) -> Arc<CudaContext> {
        self.cuda_device.clone()
    }
}

impl Storage for DeviceStorageOwned {
    fn get_pointer(&self) -> u64 {
        self.device_ptr() as u64
    }

    fn storage_size(&self) -> usize {
        self.bytes
    }

    fn storage_type(&self) -> StorageType {
        StorageType::Device(self.cuda_device.ordinal())
    }
}

impl std::fmt::Debug for DeviceStorageOwned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("storage_type", &self.storage_type())
            .field("storage_size", &self.storage_size())
            .finish()
    }
}

/// Direct wrapper around CUDA pinned memory
pub struct CudaPinnedMemory {
    /// Raw pointer to the pinned memory
    ptr: NonNull<c_void>,
    /// Size in bytes
    bytes: usize,
}

impl CudaPinnedMemory {
    /// Allocate new pinned memory using CUDA
    pub fn new(bytes: usize) -> Result<Self> {
        if bytes == 0 {
            raise!("Bytes must be greater than 0");
        }

        let mut ptr: *mut c_void = std::ptr::null_mut();
        let result = unsafe { cuda_malloc_host(&mut ptr, bytes) };
        if result != 0 {
            raise!("Failed to allocate pinned memory");
        }

        // Safety: We just checked that the allocation succeeded
        let ptr =
            NonNull::new(ptr).ok_or_else(|| anyhow::anyhow!("Null pointer after allocation"))?;

        // Zero out the memory
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr() as *mut u8, 0, bytes);
        }

        Ok(Self { ptr, bytes })
    }

    /// Get raw pointer
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        self.bytes
    }
}

impl Drop for CudaPinnedMemory {
    fn drop(&mut self) {
        let result = unsafe { cuda_free_host(self.ptr.as_ptr()) };
        if result != 0 {
            eprintln!("Failed to free pinned memory");
        }
    }
}

// Implement Storage trait for the new CudaPinnedMemory
impl Storage for CudaPinnedMemory {
    fn get_pointer(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }

    fn storage_size(&self) -> usize {
        self.bytes
    }

    fn storage_type(&self) -> StorageType {
        StorageType::Pinned
    }
}

impl std::fmt::Debug for CudaPinnedMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaPinnedMemory")
            .field("ptr", &(self.ptr.as_ptr() as usize))
            .field("bytes", &self.bytes)
            .field("storage_type", &self.storage_type())
            .finish()
    }
}

/// A view into tensor data with statically-known dimension count
#[derive(Clone)]
pub struct TensorView<'a, T: Storage, const D: usize> {
    /// The underlying tensor storage
    storage: &'a T,
    /// Shape of the view (dimensions)
    shape: [usize; D],
    /// Strides for each dimension (in elements, not bytes)
    strides: [usize; D],
    /// Strides for each dimension (in bytes)
    byte_strides: [usize; D],
    /// Offset from the start of the storage, in bytes
    offset: usize,
    /// Element size in bytes
    element_size: usize,
    /// Total elements in this view
    total_elements: usize,
}

impl<'a, T: Storage, const D: usize> TensorView<'a, T, D> {
    /// Create a new tensor view from storage and shape
    pub fn new(storage: &'a T, shape: [usize; D], element_size: usize) -> Result<Self> {
        // Calculate row-major strides (in elements)
        let mut strides = [0; D];
        let mut byte_strides = [0; D];

        if D > 0 {
            strides[D - 1] = 1; // Rightmost dimension is contiguous (elements)
            byte_strides[D - 1] = element_size; // Rightmost dimension in bytes

            // Calculate remaining strides
            for i in (0..D - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
                byte_strides[i] = strides[i] * element_size;
            }
        }

        // Calculate total elements
        let total_elements = shape.iter().product();

        // Validate that the view fits within the storage
        if total_elements * element_size > storage.storage_size() {
            return Err(error!(
                "Shape {:?} requires {} bytes, but storage only has {} bytes",
                shape,
                total_elements * element_size,
                storage.storage_size()
            ));
        }

        Ok(Self {
            storage,
            shape,
            strides,
            byte_strides,
            offset: 0,
            element_size,
            total_elements,
        })
    }

    /// Create a new tensor view with custom strides
    pub fn with_strides(
        storage: &'a T,
        shape: [usize; D],
        strides: [usize; D],
        offset: usize,
        element_size: usize,
    ) -> Result<Self, String> {
        // Calculate byte strides
        let mut byte_strides = [0; D];
        for i in 0..D {
            byte_strides[i] = strides[i] * element_size;
        }

        // Calculate total elements
        let total_elements = shape.iter().product();

        // Validate that the view fits within the storage
        // Calculate the maximum offset this view will access
        let max_offset = if D > 0 {
            offset + Self::calculate_max_offset(&shape, &byte_strides)
        } else {
            offset
        };

        if max_offset > storage.storage_size() {
            return Err(format!(
                "View would access up to byte offset {}, but storage size is only {} bytes",
                max_offset,
                storage.storage_size()
            ));
        }

        Ok(Self {
            storage,
            shape,
            strides,
            byte_strides,
            offset,
            element_size,
            total_elements,
        })
    }

    /// Calculate the maximum byte offset that will be accessed by this view
    fn calculate_max_offset(shape: &[usize; D], byte_strides: &[usize; D]) -> usize {
        let mut max_offset = 0;

        // Calculate the maximum offset by positioning at the furthest element
        for i in 0..D {
            if shape[i] > 0 {
                max_offset += (shape[i] - 1) * byte_strides[i];
            }
        }

        max_offset
    }

    /// Get the shape of the tensor view
    pub fn shape(&self) -> &[usize; D] {
        &self.shape
    }

    /// Get the strides of the tensor view (in elements)
    pub fn strides(&self) -> &[usize; D] {
        &self.strides
    }

    /// Get the byte strides of the tensor view
    pub fn byte_strides(&self) -> &[usize; D] {
        &self.byte_strides
    }

    /// Get the element size in bytes
    pub fn element_size(&self) -> usize {
        self.element_size
    }

    /// Calculate flat index from multi-dimensional indices (in elements)
    pub fn flat_index(&self, indices: &[usize; D]) -> Result<usize, String> {
        // Validate indices
        for i in 0..D {
            if indices[i] >= self.shape[i] {
                return Err(format!(
                    "Index {} out of bounds for dimension {} with size {}",
                    indices[i], i, self.shape[i]
                ));
            }
        }

        // Calculate flat index
        let mut flat_idx = 0;
        for i in 0..D {
            flat_idx += indices[i] * self.strides[i];
        }

        Ok(flat_idx)
    }

    /// Calculate byte offset for indices
    pub fn byte_offset(&self, indices: &[usize; D]) -> Result<usize> {
        // Validate indices
        for i in 0..D {
            if indices[i] >= self.shape[i] {
                return Err(error!(
                    "Index {} out of bounds for dimension {} with size {}",
                    indices[i], i, self.shape[i]
                ));
            }
        }

        // Calculate byte offset directly using byte_strides
        let mut offset = self.offset;
        for i in 0..D {
            offset += indices[i] * self.byte_strides[i];
        }

        Ok(offset)
    }

    /// Get the absolute memory address for indices
    pub fn address(&self, indices: &[usize; D]) -> Result<u64> {
        let byte_offset = self.byte_offset(indices)?;
        Ok(self.storage.get_pointer() + byte_offset as u64)
    }

    /// Check if the tensor has a standard row-major contiguous layout
    pub fn is_contiguous(&self) -> bool {
        if D == 0 {
            return true;
        }

        let mut expected_stride = 1;
        let mut expected_byte_stride = self.element_size;

        for i in (0..D).rev() {
            if self.strides[i] != expected_stride || self.byte_strides[i] != expected_byte_stride {
                return false;
            }
            expected_stride *= self.shape[i];
            expected_byte_stride *= self.shape[i];
        }

        true
    }

    /// Get the total number of elements in the view
    pub fn num_elements(&self) -> usize {
        self.total_elements
    }

    /// Get the pointer to the data
    pub fn data(&self) -> u64 {
        self.storage.get_pointer()
    }

    /// Get the total size in bytes
    pub fn size_in_bytes(&self) -> usize {
        self.total_elements * self.element_size
    }

    pub fn copy_to<S: Storage>(
        &self,
        dst_view: &mut TensorView<'_, S, D>,
        stream: &CudaStream,
    ) -> Result<()> {
        // validate same shape and strides
        if self.shape != dst_view.shape || self.strides != dst_view.strides {
            raise!(
                "Shape or strides mismatch: {:?} vs {:?}",
                self.shape,
                dst_view.shape
            );
        }

        if !self.is_contiguous() {
            raise!("Source is not contiguous");
        }

        if !dst_view.is_contiguous() {
            raise!("Destination is not contiguous");
        }

        assert_eq!(self.size_in_bytes(), dst_view.size_in_bytes());

        tracing::debug!("Copying from {:?} to {:?}", self, dst_view);

        let stream_id = stream.cu_stream();

        let rc = unsafe {
            cuda_memcpy_async(
                dst_view.data() as *mut c_void,
                self.data() as *const c_void,
                self.size_in_bytes(),
                stream_id as *mut c_void,
            )
        };

        if rc != 0 {
            raise!("cudaMemcpyAsync failed");
        }

        Ok(())
    }

    /// Create a sliced view of this tensor along a dimension
    pub fn slice(&self, dim: usize, start: usize, end: Option<usize>) -> Result<Self, String> {
        if dim >= D {
            return Err(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim, D
            ));
        }

        let end_idx = end.unwrap_or(self.shape[dim]);
        if end_idx > self.shape[dim] {
            return Err(format!(
                "End index {} out of bounds for dimension {} with size {}",
                end_idx, dim, self.shape[dim]
            ));
        }
        if start >= end_idx {
            return Err(format!(
                "Invalid slice range: start={}, end={}",
                start, end_idx
            ));
        }

        // Create a new shape array with the sliced dimension
        let mut new_shape = self.shape;
        new_shape[dim] = end_idx - start;

        // Calculate the offset for the start of the slice (in bytes)
        let new_offset = self.offset + start * self.byte_strides[dim];

        // Create a new view with the same strides but updated shape and offset
        Ok(Self {
            storage: self.storage,
            shape: new_shape,
            strides: self.strides,
            byte_strides: self.byte_strides,
            offset: new_offset,
            element_size: self.element_size,
            total_elements: new_shape.iter().product(),
        })
    }

    pub fn as_ndarray_view<DT>(&self) -> Result<ndarray::ArrayView<'_, DT, IxDyn>>
    where
        DT: bytemuck::Pod,
    {
        match self.storage.storage_type() {
            StorageType::Device(_) => raise!("Cannot convert device tensor to ndarray"),
            StorageType::System | StorageType::Pinned => {}
        };

        // validate DT matches bytes per element
        if std::mem::size_of::<DT>() != self.element_size {
            return Err(anyhow::anyhow!(
                "Type size mismatch: {} vs {}",
                std::mem::size_of::<DT>(),
                self.element_size
            ));
        }

        if !self.is_contiguous() {
            raise!("Cannot convert non-contiguous tensor to ndarray");
        }
        // create a slice from the raw pointer
        let ptr = self.storage.get_pointer() as *const DT;
        let size = self.shape.iter().product::<usize>();

        // Create a slice from the raw pointer
        let slice = unsafe { std::slice::from_raw_parts::<DT>(ptr, size) };

        // Create an ndarray view from the slice
        // Convert our shape array to ndarray's Dim type
        let dim = ndarray::IxDyn(&self.shape);
        let array = ndarray::ArrayView::from_shape(dim, slice)?;
        Ok(array)
    }

    /// Convert to a mutable ndarray view
    pub fn as_ndarray_view_mut<DT>(&mut self) -> Result<ArrayViewMut<'_, DT, IxDyn>>
    where
        DT: bytemuck::Pod,
    {
        match self.storage.storage_type() {
            StorageType::Device(_) => {
                return Err(anyhow::anyhow!("Cannot convert device tensor to ndarray"))
            }
            StorageType::System | StorageType::Pinned => {}
        };

        // validate DT matches bytes per element
        if std::mem::size_of::<DT>() != self.element_size {
            return Err(anyhow::anyhow!(
                "Type size mismatch: {} vs {}",
                std::mem::size_of::<DT>(),
                self.element_size
            ));
        }

        if !self.is_contiguous() {
            return Err(anyhow::anyhow!(
                "Cannot convert non-contiguous tensor to ndarray"
            ));
        }

        // Get the pointer to the data plus offset
        let ptr =
            (self.storage.get_pointer() as *mut DT).wrapping_add(self.offset / self.element_size);
        let size = self.shape.iter().product::<usize>();

        // Create a mutable slice from the raw pointer
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };

        // Create an ndarray view from the slice - use the same pattern as the immutable version
        let dim = ndarray::IxDyn(&self.shape);
        let array = ndarray::ArrayViewMut::from_shape(dim, slice)?;
        Ok(array)
    }

    /// Returns the storage type of the underlying tensor
    pub fn storage_type(&self) -> StorageType {
        self.storage.storage_type()
    }
}

impl<T: Storage, const D: usize> std::fmt::Debug for TensorView<'_, T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorView")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("byte_strides", &self.byte_strides)
            .field("offset", &self.offset)
            .field("element_size", &self.element_size)
            .field("total_elements", &self.total_elements)
            .field("storage_type", &self.storage.storage_type())
            .finish()
    }
}

// Indexing helpers with updated byte stride handling
pub mod tensor_indexing {
    /// Converts a flat index to multidimensional indices for a given shape
    pub fn unflatten_index<const D: usize>(flat_idx: usize, shape: &[usize; D]) -> [usize; D] {
        let mut indices = [0; D];
        let mut remaining = flat_idx;

        // Calculate strides for the shape
        let mut strides = [0; D];

        if D > 0 {
            strides[D - 1] = 1;
            for i in (0..D - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }

        // Calculate indices
        for i in 0..D {
            indices[i] = remaining / strides[i];
            remaining %= strides[i];
        }

        indices
    }

    /// Calculates row-major strides for a given shape (element strides, not byte strides)
    pub fn calculate_strides<const D: usize>(shape: &[usize; D]) -> [usize; D] {
        let mut strides = [0; D];

        if D > 0 {
            strides[D - 1] = 1; // Rightmost dimension is contiguous
            for i in (0..D - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }

        strides
    }

    /// Calculates row-major byte strides for a given shape and element size
    pub fn calculate_byte_strides<const D: usize>(
        shape: &[usize; D],
        element_size: usize,
    ) -> [usize; D] {
        let mut byte_strides = [0; D];

        if D > 0 {
            byte_strides[D - 1] = element_size; // Rightmost dimension is contiguous
            for i in (0..D - 1).rev() {
                byte_strides[i] = byte_strides[i + 1] * shape[i + 1];
            }
        }

        byte_strides
    }
}

/// Storage that wraps external device memory with metadata provided externally
/// This is unsafe as it trusts that the provided device pointer and sizes are valid
#[derive(Debug)]
pub struct DeviceStorageFromAny {
    /// The original object that owns the memory (e.g., a PyObject)
    source: Arc<dyn Any + Send + Sync>,

    /// Device pointer to the data
    device_ptr: u64,

    /// Size of each element in bytes
    bytes: usize,

    /// CUDA device ordinal
    device_id: usize,
}

impl DeviceStorageFromAny {
    /// Create a new DeviceStorageFromAny wrapper
    ///
    /// # Safety
    ///
    /// This is unsafe because it trusts that:
    /// 1. The device_ptr is a valid CUDA device pointer
    /// 2. The device_ptr points to at least elements * bytes_per_element bytes of valid memory
    /// 3. The memory remains valid for the lifetime of this object
    /// 4. The device_id corresponds to the device where the memory is allocated
    pub fn new(
        source: Arc<dyn Any + Send + Sync>,
        device_ptr: u64,
        bytes: usize,
        device_id: usize,
    ) -> Self {
        Self {
            source,
            device_ptr,
            bytes,
            device_id,
        }
    }

    /// Get the original source object as Any
    pub fn source(&self) -> &Arc<dyn Any + Send + Sync> {
        &self.source
    }

    /// Try to downcast the source to a specific type
    pub fn downcast_source<T: 'static + Send + Sync>(&self) -> Option<&T> {
        self.source.downcast_ref::<T>()
    }
}

impl Storage for DeviceStorageFromAny {
    fn get_pointer(&self) -> u64 {
        self.device_ptr
    }

    fn storage_size(&self) -> usize {
        self.bytes
    }

    fn storage_type(&self) -> StorageType {
        StorageType::Device(self.device_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementation of Storage for testing
    #[derive(Debug)]
    struct MockTensor {
        data_ptr: u64,
        storage_size_bytes: usize,
    }

    impl Storage for MockTensor {
        fn get_pointer(&self) -> u64 {
            self.data_ptr
        }

        fn storage_size(&self) -> usize {
            self.storage_size_bytes
        }

        fn storage_type(&self) -> StorageType {
            StorageType::System
        }
    }

    #[test]
    fn test_tensor_view_creation() {
        // Create a mock tensor with sufficient storage
        let mock_tensor = MockTensor {
            data_ptr: 0x1000,
            storage_size_bytes: 96, // 24 elements * 4 bytes
        };

        // Create a 3D tensor view with F32 elements
        let shape = [2, 3, 4];
        let element_size = 4; // F32 size
        let view = TensorView::<_, 3>::new(&mock_tensor, shape, element_size).unwrap();

        // Verify shape and strides
        assert_eq!(view.shape(), &[2, 3, 4]);
        assert_eq!(view.strides(), &[12, 4, 1]);
        assert_eq!(view.byte_strides(), &[48, 16, 4]);
        assert_eq!(view.num_elements(), 24);
        assert_eq!(view.size_in_bytes(), 96);
        assert!(view.is_contiguous());
    }

    #[test]
    fn test_tensor_view_indexing() {
        // Error shows: "Shape [2, 3, 4] requires 96 bytes, but storage only has 24 bytes"
        let mock_tensor = MockTensor {
            data_ptr: 0x1000,
            storage_size_bytes: 96, // Increase from 24 to 96 bytes
        };

        // Create a 3D tensor view
        let shape = [2, 3, 4];
        let view = TensorView::<_, 3>::new(&mock_tensor, shape, 4).unwrap();

        // Rest of test unchanged
        // Test flat index calculations
        assert_eq!(view.flat_index(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(view.flat_index(&[0, 0, 1]).unwrap(), 1);
        assert_eq!(view.flat_index(&[0, 1, 0]).unwrap(), 4);
        assert_eq!(view.flat_index(&[1, 0, 0]).unwrap(), 12);
        assert_eq!(view.flat_index(&[1, 2, 3]).unwrap(), 23);

        // Test byte offset calculations
        assert_eq!(view.byte_offset(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(view.byte_offset(&[0, 0, 1]).unwrap(), 4);
        assert_eq!(view.byte_offset(&[0, 1, 0]).unwrap(), 16);
        assert_eq!(view.byte_offset(&[1, 0, 0]).unwrap(), 48);

        // Test absolute address calculations
        assert_eq!(view.address(&[0, 0, 0]).unwrap(), 0x1000);
        assert_eq!(view.address(&[0, 0, 1]).unwrap(), 0x1004);
        assert_eq!(view.address(&[1, 2, 3]).unwrap(), 0x1000 + 23 * 4);
    }

    #[test]
    fn test_tensor_view_slicing() {
        // Error shows: "Shape [2, 3, 4] requires 96 bytes, but storage only has 24 bytes"
        let mock_tensor = MockTensor {
            data_ptr: 0x1000,
            storage_size_bytes: 96, // Increase from 24 to 96 bytes
        };

        // Create a 3D tensor view
        let shape = [2, 3, 4];
        let view = TensorView::<_, 3>::new(&mock_tensor, shape, 4).unwrap();

        // Rest of test unchanged
        // Create a slice along dimension 1 (the middle dimension)
        let sliced = view.slice(1, 1, Some(3)).unwrap();

        // Verify the slice properties
        assert_eq!(sliced.shape(), &[2, 2, 4]); // Dimension 1 reduced from 3 to 2
        assert_eq!(sliced.strides(), &[12, 4, 1]); // Strides remain the same
        assert_eq!(sliced.byte_strides(), &[48, 16, 4]); // Byte strides remain the same
        assert_eq!(sliced.offset, 16); // Offset is now 4 elements (16 bytes)

        // Test addressing in the slice
        assert_eq!(sliced.address(&[0, 0, 0]).unwrap(), 0x1000 + 16);
        assert_eq!(
            sliced.address(&[1, 1, 3]).unwrap(),
            0x1000 + 16 + 48 + 16 + 12
        );
    }

    #[test]
    fn test_tensor_views_with_custom_strides() {
        // Error shows: "Shape [2, 3, 4] requires 96 bytes, but storage only has 24 bytes"
        let mock_tensor = MockTensor {
            data_ptr: 0x1000,
            storage_size_bytes: 96, // Increase from 24 to 96 bytes
        };

        // Total storage: 24 elements * 4 bytes = 96 bytes
        let shape = [2, 3, 4];

        // CASE 1: Standard contiguous view
        let contiguous_view = TensorView::<_, 3>::new(&mock_tensor, shape, 4).unwrap();
        assert!(contiguous_view.is_contiguous());
        assert_eq!(contiguous_view.strides(), &[12, 4, 1]);
        assert_eq!(contiguous_view.byte_strides(), &[48, 16, 4]);

        // CASE 2: Non-contiguous but within bounds
        let smaller_shape = [2, 2, 4]; // 16 elements instead of 24

        // These are the contiguous strides for shape [2, 3, 4] but NOT for [2, 2, 4]
        // For shape [2, 2, 4], contiguous strides would be [8, 4, 1]
        let non_contiguous_strides = [12, 4, 1];

        let non_contiguous = TensorView::<_, 3>::with_strides(
            &mock_tensor,
            smaller_shape,
            non_contiguous_strides,
            0,
            4,
        )
        .unwrap();

        // It should NOT be contiguous since the strides don't match the shape
        assert!(!non_contiguous.is_contiguous());
        assert_eq!(non_contiguous.strides(), &[12, 4, 1]);
        assert_eq!(non_contiguous.byte_strides(), &[48, 16, 4]);

        // Test accessing the last element to confirm it's within bounds
        let last_index = [1, 1, 3];
        let byte_offset = non_contiguous.byte_offset(&last_index).unwrap();
        assert_eq!(byte_offset, (12 + 4 + 3) * 4);
        assert!(
            byte_offset < mock_tensor.storage_size(),
            "Byte offset {} should be less than storage size {}",
            byte_offset,
            mock_tensor.storage_size()
        );

        // CASE 3: Non-contiguous that exceeds bounds
        // Using strides that will exceed the tensor's storage
        // 1*16 + 2*4 + 3*1 = 16 + 8 + 3 = 27 elements, which is beyond our 24 elements
        let invalid_custom_strides = [16, 4, 1];
        let result =
            TensorView::<_, 3>::with_strides(&mock_tensor, shape, invalid_custom_strides, 0, 4);

        // Verify we get the expected error
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(
            error_msg.contains("would access up to byte offset 108"),
            "Expected error about exceeding storage, got: {}",
            error_msg
        );
    }

    #[test]
    fn test_tensor_view_with_offset() {
        // Error shows: "View would access up to byte offset 108, but storage size is only 40 bytes"
        let mock_tensor = MockTensor {
            data_ptr: 0x1000,
            storage_size_bytes: 120, // Increase from 40 to 120 bytes
        };

        // Shape is smaller than the full tensor to allow for offset
        let shape = [2, 3, 4]; // 24 elements total

        // Create a view with an offset of 4 elements (16 bytes)
        let offset_view =
            TensorView::<_, 3>::with_strides(&mock_tensor, shape, [12, 4, 1], 16, 4).unwrap();

        // The view should still be contiguous
        assert!(offset_view.is_contiguous());

        // Check offset is preserved
        assert_eq!(offset_view.offset, 16);

        // Test accessing the first element
        let first_byte_offset = offset_view.byte_offset(&[0, 0, 0]).unwrap();
        assert_eq!(first_byte_offset, 16); // Should be at the offset

        // Test accessing the last element
        let last_byte_offset = offset_view.byte_offset(&[1, 2, 3]).unwrap();
        assert_eq!(last_byte_offset, 16 + (12 + 2 * 4 + 3) * 4);

        // Creating a view with an offset that would exceed the tensor size should fail
        let result = TensorView::<_, 3>::with_strides(
            &mock_tensor,
            shape,
            [12, 4, 1],
            80, // 40 elements - 24 + a bit more
            4,
        );

        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(
            error_msg.contains("would access up to byte offset"),
            "Expected error about exceeding storage, got: {}",
            error_msg
        );
    }

    #[test]
    fn test_ndarray_view_with_real_data() {
        use std::sync::{Arc, Mutex};

        // Create a mock tensor with real memory for testing
        #[derive(Debug)]
        struct RealDataMock {
            // Using Vec<u8> to store the raw bytes
            data: Arc<Mutex<Vec<u8>>>,
            element_size_bytes: usize,
        }

        impl Storage for RealDataMock {
            fn get_pointer(&self) -> u64 {
                // Get the raw pointer to the start of our vector's data
                self.data.lock().unwrap().as_ptr() as u64
            }

            fn storage_size(&self) -> usize {
                self.data.lock().unwrap().len()
            }

            fn storage_type(&self) -> StorageType {
                StorageType::System
            }
        }

        impl RealDataMock {
            fn new(num_elements: usize, element_size: usize) -> Self {
                // Create a zeroed buffer of the right size
                let buffer = vec![0u8; num_elements * element_size];
                Self {
                    data: Arc::new(Mutex::new(buffer)),
                    element_size_bytes: element_size,
                }
            }

            // Helper to set a specific element's value
            fn set_element_value(&self, index: usize, value: u32) {
                let mut data = self.data.lock().unwrap();
                let bytes = value.to_ne_bytes();
                let offset = index * self.element_size_bytes;

                for i in 0..std::mem::size_of::<u32>() {
                    data[offset + i] = bytes[i];
                }
            }
        }

        // Create a mock tensor with u32 elements (4 bytes each)
        let shape = [2, 3, 4]; // 24 elements total
        let num_elements = shape.iter().product();
        let element_size = std::mem::size_of::<u32>();

        let mock_tensor = RealDataMock::new(num_elements, element_size);

        // Create a tensor view
        let view = TensorView::<_, 3>::new(&mock_tensor, shape, 4).unwrap();

        // Create an ndarray view
        let ndarray_view = view.as_ndarray_view::<u32>().unwrap();

        // Verify all elements are zero (initial state)
        for &value in ndarray_view.iter() {
            assert_eq!(value, 0, "Expected all initial values to be 0");
        }

        // Create a mutable clone of the data to work with
        let data_arc = mock_tensor.data.clone();

        // Set all elements to 42 by modifying the underlying storage
        {
            let mut data = data_arc.lock().unwrap();
            for i in 0..num_elements {
                let offset = i * element_size;
                let bytes = 42u32.to_ne_bytes();
                for j in 0..element_size {
                    data[offset + j] = bytes[j];
                }
            }
        }

        // Create a new ndarray view to see the changes
        let updated_view = view.as_ndarray_view::<u32>().unwrap();

        // Verify all elements are now 42
        for &value in updated_view.iter() {
            assert_eq!(value, 42, "Expected all values to be 42 after update");
        }

        // See that ndarray_view is also updated
        for &value in ndarray_view.iter() {
            assert_eq!(value, 42, "Expected all values to be 42 after update");
        }

        // Change just the first element back to 0
        mock_tensor.set_element_value(0, 0);

        // Create another ndarray view to see the effect of our change
        let final_view = view.as_ndarray_view::<u32>().unwrap();

        // The first element should be 0, others should remain 42
        assert_eq!(final_view[[0, 0, 0]], 0, "First element should be 0");
        assert_eq!(updated_view[[0, 0, 0]], 0, "First element should be 0");
        assert_eq!(ndarray_view[[0, 0, 0]], 0, "First element should be 0");

        // Check some of the other elements to ensure they're still 42
        assert_eq!(
            final_view[[0, 0, 1]],
            42,
            "Element [0,0,1] should still be 42"
        );
        assert_eq!(final_view[[1, 2, 3]], 42, "Last element should still be 42");

        // Count the number of zeros (should be exactly 1)
        let zero_count = final_view.iter().filter(|&&x| x == 0).count();
        assert_eq!(zero_count, 1, "There should be exactly one zero element");

        // Count the number of 42s (should be num_elements - 1)
        let forty_two_count = final_view.iter().filter(|&&x| x == 42).count();
        assert_eq!(
            forty_two_count,
            num_elements - 1,
            "All other elements should be 42"
        );
    }
}
