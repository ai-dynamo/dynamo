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

use derive_builder::Builder;
use std::sync::Arc;

use super::storage::Storage;

#[derive(Debug, Clone)]
pub enum KvLayout {
    /// Tensor is laid out as [kv, block, head, head_dim]
    KvFirst,

    /// Tensor is laid out as [block, kv, head, head_dim]
    BlockFirst,
}

#[derive(Debug, Clone, Builder)]
pub struct KvLayer {
    /// The layout of the tensor
    layout: KvLayout,

    /// The storage of the tensor
    storage: Arc<dyn Storage>,

    /// The number of blocks in the tensor
    number_of_blocks: usize,

    /// The size of each block in the tensor
    block_size: usize,

    /// The number of heads in the tensor
    number_of_heads: usize,

    /// The size of each head in the tensor
    head_size: usize,

    /// The tensor parallel size (default is 1)
    #[builder(default = 1)]
    tp_size: usize,

    /// The tensor parallel rank (default is 0)
    #[builder(default = 0)]
    tp_rank: usize,
}

// pub struct KvLayer {
//     data: Tensor,
//     tp_size: usize,
//     block_dim_idx: usize,
// }

// impl KvLayer {
//     pub fn new(
//         tp_size: usize,
//         nblocks: usize,
//         block_size: usize,
//         n_heads: usize,
//         head_size: usize,
//         dtype: DType,
//         device: Device,
//     ) -> Result<Self> {
//         let data = Tensor::zeros((2, nblocks, block_size, n_heads, head_size), dtype, &device)?;
//         Ok(Self {
//             data,
//             tp_size,
//             block_dim_idx: 1,
//         })
//     }

//     pub fn nblocks(&self) -> usize {
//         self.data.dim(1).unwrap()
//     }

//     pub fn block_size(&self) -> usize {
//         self.data.dim(2).unwrap()
//     }

//     pub fn n_heads(&self) -> usize {
//         self.data.dim(3).unwrap()
//     }

//     pub fn head_size(&self) -> usize {
//         self.data.dim(4).unwrap()
//     }

//     pub fn copy_blocks(
//         &self,
//         src_blocks: &[usize],
//         dst_blocks: &[usize],
//         dst_layer: &mut Self,
//     ) -> Result<()> {
//         if self.tp_size != dst_layer.tp_size {
//             return Err(candle_core::Error::Msg(format!(
//                 "TP size mismatch: {} != {}",
//                 self.tp_size, dst_layer.tp_size
//             )));
//         }

//         if src_blocks.len() != dst_blocks.len() {
//             return Err(candle_core::Error::Msg(format!(
//                 "Block length mismatch: {} != {}",
//                 src_blocks.len(),
//                 dst_blocks.len()
//             )));
//         }

//         if self.block_dim_idx != dst_layer.block_dim_idx {
//             return Err(candle_core::Error::Msg(format!(
//                 "Block dimension index mismatch: {} != {}",
//                 self.block_dim_idx, dst_layer.block_dim_idx
//             )));
//         }

//         // todo - implement a kernel to copy the blocks if either are gpu tensors
//         if self.data.device().is_cuda() || dst_layer.data.device().is_cuda() {
//             // call the cuda kernel to copy the blocks
//             copy_blocks_between_layers(
//                 &self.data,
//                 &dst_layer.data,
//                 src_blocks,
//                 dst_blocks,
//                 self.block_dim_idx,
//             )?;
//         } else {
//             for kv in 0..2 {
//                 for (src_idx, dst_idx) in src_blocks.iter().zip(dst_blocks.iter()) {
//                     let src_block = self.data.get(kv)?.get(*src_idx)?;
//                     let dst_block = dst_layer.data.get(kv)?.get(*dst_idx)?;

//                     // Fix: Use a proper dimension value instead of a slice
//                     dst_block.slice_set(&src_block, 0, 0)?;
//                 }
//             }
//         }

//         Ok(())
//     }

//     /// Split the data into tp_size parts
//     /// The target tp_size greater than and evenly divisible by the current tp_size
//     pub fn tp_split(&self, tp_size: usize) -> Result<Tensor> {
//         if tp_size <= self.tp_size {
//             return Err(candle_core::Error::Msg(format!(
//                 "Target tp_size ({}) must be greater than current tp_size ({})",
//                 tp_size, self.tp_size
//             )));
//         }
//         if tp_size % self.tp_size != 0 {
//             return Err(candle_core::Error::Msg(format!(
//                 "Target tp_size ({}) must be evenly divisible by current tp_size ({})",
//                 tp_size, self.tp_size
//             )));
//         }
//         let data = transform_blocks_inplace_v2(&self.data, tp_size)?;
//         Ok(data)
//     }

//     pub fn required_memory(
//         nblocks: usize,
//         block_size: usize,
//         n_heads: usize,
//         head_size: usize,
//         dtype: DType,
//     ) -> usize {
//         let n_elements = 2 * nblocks * block_size * n_heads * head_size;
//         n_elements * dtype.size_in_bytes()
//     }
// }

// /// Groups contiguous block indices into separate vectors
// ///
// /// For example: [0, 1, 2, 3, 7, 9] -> [[0, 1, 2, 3], [7], [9]]
// pub fn group_contiguous_blocks(blocks: &[usize]) -> Vec<Vec<usize>> {
//     if blocks.is_empty() {
//         return vec![];
//     }

//     let mut result = Vec::new();
//     let mut current_group = vec![blocks[0]];

//     for i in 1..blocks.len() {
//         if blocks[i] == blocks[i - 1] + 1 {
//             // Block is contiguous with previous
//             current_group.push(blocks[i]);
//         } else {
//             // Block starts a new group
//             result.push(current_group);
//             current_group = vec![blocks[i]];
//         }
//     }

//     // Add the last group
//     result.push(current_group);

//     result
// }

// /// Performs a simplified in-place transformation for tensor parallelism
// ///
// /// Takes a tensor of shape [n_blocks, block_size, n_heads, head_size] and returns
// /// a tensor of shape [tp_size, n_blocks, block_size, n_heads/tp_size, head_size]
// ///
// /// This uses reshape + transpose operations for maximum efficiency and minimal
// /// memory overhead. The tensor transformation happens directly on the GPU without
// /// unnecessary copying.
// pub fn transform_blocks_inplace_v2(source: &Tensor, tp_size: usize) -> Result<Tensor> {
//     // Get the dimensions of the source tensor
//     let kv_dim = source.dim(0)?;
//     assert_eq!(kv_dim, 2);

//     let n_blocks = source.dim(1)?;
//     let block_size = source.dim(2)?;
//     let n_heads = source.dim(3)?;
//     let head_size = source.dim(4)?;

//     // Check that n_heads is divisible by tp_size
//     if n_heads % tp_size != 0 {
//         return Err(candle_core::Error::Msg(format!(
//             "Number of heads ({}) must be divisible by tp_size ({})",
//             n_heads, tp_size
//         )));
//     }

//     if !source.is_contiguous() {
//         return Err(candle_core::Error::Msg(
//             "Source tensor must be contiguous".into(),
//         ));
//     }

//     // Calculate heads per rank
//     let heads_per_rank = n_heads / tp_size;

//     // Reshape to split the heads dimension
//     // [n_blocks, block_size, n_heads, head_size] ->
//     // [n_blocks, block_size, tp_size, heads_per_rank, head_size]
//     let reshaped = source.reshape((2, n_blocks, block_size, tp_size, heads_per_rank, head_size))?;

//     // Now permute to move tp_size to the first dimension
//     // [n_blocks, block_size, tp_size, heads_per_rank, head_size] ->
//     // [tp_size, n_blocks, block_size, heads_per_rank, head_size]
//     let result = reshaped.permute((3, 0, 1, 2, 4, 5))?;

//     Ok(result)
// }

// /// Accepts an arbitrary shape and reshapes it to be 3d on the block dimension, e.g.
// ///
// /// Let `X` be the block dimension.
// /// (a, b, X, c, d, e) becomes (a*b, X, c*d*e)
// /// (X, a, b, c, d) becomes (1, X, a*b*c*d)
// /// (a, X, b, c) becomes (a, X, b*c)
// pub fn reshape_to_3d_on_block_dim(source: &Tensor, block_dim: usize) -> Result<Tensor> {
//     let shape = source.shape().dims();
//     let ndim = source.rank();

//     // Ensure block_dim is valid
//     if block_dim >= ndim {
//         return Err(candle_core::Error::Msg(format!(
//             "Block dimension {} is out of bounds for tensor with {} dimensions",
//             block_dim, ndim
//         )));
//     }

//     if !source.is_contiguous() {
//         return Err(candle_core::Error::Msg(
//             "Source tensor must be contiguous".into(),
//         ));
//     }

//     // Handle different cases based on block_dim position
//     if block_dim == 0 {
//         // Case: (X, a, b, c, d) becomes (1, X, a*b*c*d)
//         let prefix_dim = 1;
//         let block_size = shape[block_dim];
//         let suffix_dim: usize = shape.iter().skip(block_dim + 1).product();

//         // Reshape to 3D
//         source.reshape((prefix_dim, block_size, suffix_dim))
//     } else if block_dim == ndim - 1 {
//         // Special case when block dim is the last dimension
//         // (a, b, c, X) becomes (a*b*c, X, 1)
//         let prefix_dim: usize = shape.iter().take(block_dim).product();
//         let block_size = shape[block_dim];
//         let suffix_dim = 1;

//         // Reshape to 3D
//         source.reshape((prefix_dim, block_size, suffix_dim))
//     } else {
//         // General case: (a, b, X, c, d, e) becomes (a*b, X, c*d*e)
//         let prefix_dim: usize = shape.iter().take(block_dim).product();
//         let block_size = shape[block_dim];
//         let suffix_dim: usize = shape.iter().skip(block_dim + 1).product();

//         // Reshape to 3D
//         source.reshape((prefix_dim, block_size, suffix_dim))
//     }
// }

// /// Create a pinned memory tensor
// pub fn create_pinned_tensor(shape: &[usize], dtype: DType) -> Result<Tensor> {
//     let dimensions = shape.len();

//     let cuda_device = match Device::cuda_if_available(0) {
//         Ok(dev) => match dev {
//             Device::Cuda(cuda_dev) => cuda_dev,
//             _ => unreachable!(),
//         },
//         Err(e) => return Err(e),
//     };

//     // compute the number of elements in the tensor
//     let num_elements = shape.iter().product::<usize>();
//     let num_bytes = num_elements * dtype.size_in_bytes();

//     // Allocate pinned memory
//     let data = unsafe {
//         let pinned_slice = cuda_device.alloc_pinned::<u8>(num_bytes).map_err(|e| {
//             candle_core::Error::Msg(format!("Failed to allocate pinned memory: {}", e))
//         })?;
//         let data = Vec::from(std::slice::from_raw_parts(
//             pinned_slice.as_ptr(),
//             num_elements,
//         ));
//         data
//     };

//     // Create a tensor from this pinned memory
//     Tensor::from_vec(data, dimensions, &Device::Cpu)
// }

// extern "C" {
//     fn copy_blocks_3d(
//         src_data: *const c_void,
//         dst_data: *mut c_void,
//         h_src_block_ids: *const c_int,
//         h_dst_block_ids: *const c_int,
//         num_block_pairs: c_int,
//         prefix_dim: c_int,
//         src_blocks: c_int,
//         dst_blocks: c_int,
//         suffix_dim: c_int,
//         elem_size: c_int,
//     ) -> c_int;
// }
// /// Copy blocks between tensors - works with any combination of
// /// device/host tensors as long as host memory is pinned
// pub fn copy_blocks_between_layers(
//     src_tensor: &Tensor,
//     dst_tensor: &Tensor,
//     src_block_ids: &[usize],
//     dst_block_ids: &[usize],
//     block_dim: usize,
// ) -> Result<()> {
//     // Validation logic (same as before)
//     if src_block_ids.len() != dst_block_ids.len() {
//         return Err(candle_core::Error::Msg(
//             "Source and destination block ID arrays must have the same length".into(),
//         ));
//     }

//     if src_tensor.rank() != dst_tensor.rank() {
//         return Err(candle_core::Error::Msg(
//             "Source and destination tensors must have the same rank".into(),
//         ));
//     }

//     if block_dim >= src_tensor.rank() {
//         return Err(candle_core::Error::Msg(
//             "Block dimension is out of bounds".into(),
//         ));
//     }
//     if src_block_ids.is_empty() {
//         return Ok(()); // Nothing to do
//     }

//     // Add checks for contiguous tensors
//     if !src_tensor.is_contiguous() {
//         return Err(candle_core::Error::Msg(
//             "Source tensor must be contiguous for zero-copy operations".into(),
//         ));
//     }
//     if !dst_tensor.is_contiguous() {
//         return Err(candle_core::Error::Msg(
//             "Destination tensor must be contiguous for zero-copy operations".into(),
//         ));
//     }

//     // Reshape the tensors to match the 3d tensor copy kernel
//     let src_tensor = reshape_to_3d_on_block_dim(src_tensor, block_dim)?;
//     let dst_tensor = reshape_to_3d_on_block_dim(dst_tensor, block_dim)?;

//     let src_block_ids: Vec<u32> = src_block_ids.iter().map(|&id| id as u32).collect();
//     let dst_block_ids: Vec<u32> = dst_block_ids.iter().map(|&id| id as u32).collect();

//     // Get dimensions from the tensors
//     let src_dims = src_tensor.dims();
//     let dst_dims = dst_tensor.dims();

//     // Extract dimensions for the 3D tensor format
//     let prefix_dim = src_dims[0] as i32; // First dimension
//     let src_n_blocks = src_dims[1] as i32; // Number of blocks in source
//     let dst_n_blocks = dst_dims[1] as i32; // Number of blocks in destination
//     let suffix_dim = src_dims[2] as i32; // Last dimension (combined dimensions)

//     // Validate dimension compatibility
//     if src_dims[0] != dst_dims[0] || src_dims[2] != dst_dims[2] {
//         return Err(candle_core::Error::Msg(format!(
//             "Incompatible tensor dimensions: src={:?}, dst={:?}",
//             src_dims, dst_dims
//         )));
//     }

//     // Call the CUDA kernel without stride arguments
//     let result = unsafe {
//         let src_ptr = get_tensor_ptr(&src_tensor)?;
//         let dst_ptr = get_tensor_ptr_mut(&dst_tensor)?;

//         // Add debugging info
//         println!(
//             "Source shape: {:?}, Dest shape: {:?}",
//             src_tensor.dims(),
//             dst_tensor.dims()
//         );
//         println!("Element size: {} bytes", src_tensor.dtype().size_in_bytes());

//         copy_blocks_3d(
//             src_ptr,
//             dst_ptr,
//             src_block_ids.as_ptr() as *const i32,
//             dst_block_ids.as_ptr() as *const i32,
//             src_block_ids.len() as i32,
//             prefix_dim,
//             src_n_blocks,
//             dst_n_blocks,
//             suffix_dim,
//             src_tensor.dtype().size_in_bytes() as i32,
//         )
//     };

//     if result != 0 {
//         return Err(candle_core::Error::Msg(format!(
//             "CUDA error while copying blocks: {}",
//             result
//         )));
//     }

//     Ok(())
// }

// /// Returns the raw device pointer to the start of a tensor.
// ///
// /// # Arguments
// ///
// /// * `tensor` - The tensor to get the pointer for
// ///
// /// # Returns
// ///
// /// * `Result<*const c_void>` - Raw pointer to the start of the tensor's memory
// ///   - For CUDA tensors: returns the device pointer
// ///   - For CPU tensors: returns the host memory pointer
// ///
// /// # Errors
// ///
// /// Returns an error if:
// /// - The tensor's storage type is unsupported
// /// - Cannot retrieve the pointer from storage
// pub fn get_tensor_ptr(tensor: &Tensor) -> Result<*const c_void> {
//     let (storage, _) = tensor.storage_and_layout(); // Minimal use of storage_and_layout
//     let dtype = tensor.dtype();

//     // Get pointer based on storage type and dtype
//     match &*storage {
//         Storage::Cuda(cuda_storage) => {
//             // Match on dtype to get the correct pointer type
//             match dtype {
//                 DType::F32 => {
//                     let slice = cuda_storage.as_cuda_slice::<f32>()?;
//                     Ok(*slice.device_ptr() as *const c_void)
//                 }
//                 DType::F16 => {
//                     // Make sure to include the half crate in your dependencies
//                     let slice = cuda_storage.as_cuda_slice::<half::f16>()?;
//                     Ok(*slice.device_ptr() as *const c_void)
//                 }
//                 DType::BF16 => {
//                     let slice = cuda_storage.as_cuda_slice::<half::bf16>()?;
//                     Ok(*slice.device_ptr() as *const c_void)
//                 }
//                 // Add other dtypes you need to support
//                 _ => Err(candle_core::Error::Msg(format!(
//                     "Unsupported dtype for CUDA operations: {:?}",
//                     dtype
//                 ))),
//             }
//         }
//         Storage::Cpu(cpu_storage) => {
//             // Similar pattern for CPU storage
//             match dtype {
//                 DType::F32 => {
//                     let slice = cpu_storage.as_slice::<f32>()?;
//                     Ok(slice.as_ptr() as *const c_void)
//                 }
//                 // Add other dtypes
//                 _ => Err(candle_core::Error::Msg(format!(
//                     "Unsupported dtype for CPU operations: {:?}",
//                     dtype
//                 ))),
//             }
//         }
//         _ => Err(candle_core::Error::Msg(
//             "Unsupported storage type - only CPU and CUDA tensors are supported".into(),
//         )),
//     }
// }

// /// Returns the raw device pointer and strides for a tensor.
// /// This is useful for operations that need direct memory access with proper striding.
// ///
// /// # Arguments
// ///
// /// * `tensor` - The tensor to get pointer and strides for
// ///
// /// # Returns
// ///
// /// * `Result<(*const c_void, Vec<usize>)>` - Tuple containing:
// ///   - Raw pointer to the start of the tensor's memory
// ///   - Vector of strides for each dimension
// pub fn get_tensor_ptr_and_strides(tensor: &Tensor) -> Result<(*const c_void, Vec<usize>)> {
//     let (storage, layout) = tensor.storage_and_layout();
//     let strides = layout.stride().to_vec();

//     // Get pointer based on storage type
//     let ptr = match &*storage {
//         Storage::Cuda(cuda_storage) => {
//             let slice = cuda_storage.as_cuda_slice::<f32>()?;
//             *slice.device_ptr() as *const c_void
//         }
//         Storage::Cpu(cpu_storage) => {
//             let slice = cpu_storage.as_slice::<f32>()?;
//             slice.as_ptr() as *const c_void
//         }
//         _ => {
//             return Err(candle_core::Error::Msg(
//                 "Unsupported storage type - only CPU and CUDA tensors are supported".into(),
//             ))
//         }
//     };

//     Ok((ptr, strides))
// }

// /// Returns a mutable raw device pointer to the start of a tensor.
// /// Use with caution as this allows direct modification of tensor data.
// ///
// /// # Arguments
// ///
// /// * `tensor` - The tensor to get the mutable pointer for
// ///
// /// # Returns
// ///
// /// * `Result<*mut c_void>` - Mutable raw pointer to the start of the tensor's memory
// ///
// /// # Safety
// ///
// /// This function is unsafe because it returns a mutable pointer that allows
// /// direct modification of the tensor's memory, potentially leading to data races
// /// and undefined behavior if not used correctly.
// pub unsafe fn get_tensor_ptr_mut(tensor: &Tensor) -> Result<*mut c_void> {
//     let (storage, _layout) = tensor.storage_and_layout();

//     // Get pointer based on storage type
//     match &*storage {
//         Storage::Cuda(cuda_storage) => {
//             let ptr = *cuda_storage.as_cuda_slice::<f32>()?.device_ptr();
//             Ok(ptr as *mut c_void)
//         }
//         Storage::Cpu(cpu_storage) => {
//             // For CPU storage, get the host memory pointer
//             let slice = cpu_storage.as_slice::<f32>()?;
//             let ptr = slice.as_ptr() as *mut c_void;
//             Ok(ptr)
//         }
//         _ => Err(candle_core::Error::Msg(
//             "Unsupported storage type - only CPU and CUDA tensors are supported".into(),
//         )),
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use candle_core::{DType, Device, Result, Storage, Tensor};
//     use cudarc::driver::DevicePtr;

//     #[test]
//     fn test_group_contiguous_blocks() {
//         use super::group_contiguous_blocks;

//         // Test empty input
//         assert_eq!(group_contiguous_blocks(&[]), Vec::<Vec<usize>>::new());

//         // Test single element
//         assert_eq!(group_contiguous_blocks(&[5]), vec![vec![5]]);

//         // Test all contiguous
//         assert_eq!(
//             group_contiguous_blocks(&[0, 1, 2, 3]),
//             vec![vec![0, 1, 2, 3]]
//         );

//         // Test multiple groups
//         assert_eq!(
//             group_contiguous_blocks(&[0, 1, 2, 3, 7, 9]),
//             vec![vec![0, 1, 2, 3], vec![7], vec![9]]
//         );

//         // Test all separate
//         assert_eq!(
//             group_contiguous_blocks(&[2, 5, 8]),
//             vec![vec![2], vec![5], vec![8]]
//         );
//     }

//     #[test]
//     fn test_transform_blocks_inplace_v2() -> Result<()> {
//         use super::transform_blocks_inplace_v2;

//         // Set up the device - first try CUDA, fall back to CPU
//         let device = match Device::cuda_if_available(0) {
//             Ok(dev) => dev,
//             Err(_) => {
//                 println!("CUDA not available, falling back to CPU");
//                 Device::Cpu
//             }
//         };

//         println!("Using device: {:?}", device);

//         // Define tensor dimensions
//         let n_blocks = 4; // Small number of blocks for easy testing
//         let block_size = 2; // Small block size for easy testing
//         let n_heads = 8; // Number of heads
//         let head_size = 4; // Head size (small for testing)
//         let tp_size = 2; // Tensor parallel size
//         let heads_per_rank = n_heads / tp_size;

//         // Create tensor with sequential values for easy validation
//         let total_elements = 2 * n_blocks * block_size * n_heads * head_size;
//         let mut data = Vec::with_capacity(total_elements);
//         for i in 0..total_elements {
//             data.push(i as f32);
//         }

//         // Create the tensor on the selected device
//         let source =
//             Tensor::from_vec(data, (2, n_blocks, block_size, n_heads, head_size), &device)?;

//         match source.device() {
//             Device::Cpu => {
//                 println!("Source tensor shape: {:?}", source.shape());
//             }
//             Device::Cuda(_) => {
//                 println!("Source tensor shape: {:?}", source.shape());
//                 let (storage, layout) = source.storage_and_layout();
//                 println!("Storage: {:?}", storage);
//                 println!("Layout: {:?}", layout);
//                 match &*storage {
//                     Storage::Cuda(storage) => {
//                         println!("Cuda storage: {:?}", storage);
//                         let slice = storage.as_cuda_slice::<f32>()?;
//                         let ptr = *slice.device_ptr();
//                         println!("Ptr: {:?}", ptr);
//                     }
//                     _ => {
//                         println!("Other storage: {:?}", storage);
//                     }
//                 }
//             }
//             _ => {
//                 println!("Source tensor shape: {:?}", source.shape());
//             }
//         }

//         // Print the source tensor shape
//         println!("Source tensor shape: {:?}", source.shape());

//         // Perform the in-place transformation
//         let start_time = std::time::Instant::now();
//         let result_tensor = transform_blocks_inplace_v2(&source, tp_size)?;
//         let elapsed = start_time.elapsed();

//         println!("In-place transformation completed in {:?}", elapsed);
//         println!("Result tensor shape: {:?}", result_tensor.shape());

//         // Verify the results - shape should be [tp_size, n_blocks, block_size, heads_per_rank, head_size]
//         assert_eq!(result_tensor.dim(0)?, tp_size);
//         assert_eq!(result_tensor.dim(1)?, 2);
//         assert_eq!(result_tensor.dim(2)?, n_blocks);
//         assert_eq!(result_tensor.dim(3)?, block_size);
//         assert_eq!(result_tensor.dim(4)?, heads_per_rank);
//         assert_eq!(result_tensor.dim(5)?, head_size);

//         // Move tensors to CPU for validation
//         let cpu_source = if source.device().is_cuda() {
//             source.to_device(&Device::Cpu)?
//         } else {
//             source.clone()
//         };

//         let cpu_result = if result_tensor.device().is_cuda() {
//             result_tensor.to_device(&Device::Cpu)?
//         } else {
//             result_tensor.clone()
//         };

//         // Validate values - check a few selected positions
//         // For each rank, test a few values to ensure they were correctly transformed
//         for rank in 0..tp_size {
//             // For each block in the source tensor
//             for block_idx in 0..n_blocks {
//                 // Check the first position, first head, first element
//                 let position = 0;
//                 let head = 0;
//                 let element = 0;

//                 // Compute the expected value from source
//                 // The heads are now split by rank
//                 let src_head = rank * heads_per_rank + head;
//                 let expected = cpu_source
//                     .get(0)?
//                     .get(block_idx)?
//                     .get(position)?
//                     .get(src_head)?
//                     .get(element)?
//                     .to_scalar::<f32>()?;

//                 // Get the actual value from the result tensor
//                 // Shape is [tp_size, n_blocks, block_size, heads_per_rank, head_size]
//                 let actual = cpu_result
//                     .get(rank)?
//                     .get(0)?
//                     .get(block_idx)?
//                     .get(position)?
//                     .get(head)?
//                     .get(element)?
//                     .to_scalar::<f32>()?;

//                 // Compare values
//                 assert_eq!(
//                     expected, actual,
//                     "Value mismatch at rank {}, block_idx {}, position {}, head {}, element {}",
//                     rank, block_idx, position, head, element
//                 );
//             }
//         }

//         // Print some values for visual verification
//         println!("\nSample values from result tensor:");
//         for rank in 0..tp_size {
//             println!("Rank {}:", rank);
//             for block_idx in 0..1 {
//                 // Just print the first block
//                 println!("  Block {}:", block_idx);
//                 for position in 0..1 {
//                     // Just print the first position
//                     println!("    Position {}:", position);
//                     for head in 0..2 {
//                         // Print a couple of heads
//                         let head_values = cpu_result
//                             .get(rank)?
//                             .get(0)?
//                             .get(block_idx)?
//                             .get(position)?
//                             .get(head)?
//                             .to_vec1::<f32>()?;
//                         println!("      Head {}: {:?}", head, head_values);
//                     }
//                 }
//             }
//         }

//         println!("Transform blocks inplace v2 verification passed!");

//         Ok(())
//     }

//     #[test]
//     fn test_copy_blocks() -> Result<()> {
//         let tp_size = 1;
//         let nblocks = 10;
//         let block_size = 4;
//         let n_heads = 2;
//         let head_size = 8;
//         let dtype = DType::F32;
//         let device = Device::Cpu;

//         // Create source layer filled with ones
//         let mut src_layer = KvLayer::new(
//             tp_size,
//             nblocks,
//             block_size,
//             n_heads,
//             head_size,
//             dtype,
//             device.clone(),
//         )?;
//         let ones = Tensor::ones((2, nblocks, block_size, n_heads, head_size), dtype, &device)?;
//         src_layer.data = ones;

//         // Create destination layer filled with zeros
//         let mut dst_layer = KvLayer::new(
//             tp_size, nblocks, block_size, n_heads, head_size, dtype, device,
//         )?;

//         // Copy every even block from src to dst
//         let mut src_blocks = Vec::new();
//         let mut dst_blocks = Vec::new();
//         for i in 0..(nblocks / 2) {
//             let block_idx = i * 2; // 0, 2, 4, 6, 8
//             src_blocks.push(block_idx);
//             dst_blocks.push(block_idx);
//         }

//         println!("src_blocks: {:?}", src_blocks);
//         println!("dst_blocks: {:?}", dst_blocks);

//         // Perform the copy
//         src_layer.copy_blocks(&src_blocks, &dst_blocks, &mut dst_layer)?;

//         println!("dst_layer: {:?}", dst_layer.data);
//         println!("dst_layer values: {}", dst_layer.data);

//         // Verify the copy worked correctly
//         for block_idx in 0..nblocks {
//             let expected_value = if src_blocks.contains(&block_idx) {
//                 1.0f32
//             } else {
//                 0.0f32
//             };

//             // Check a sample position in each block
//             let position = 0;
//             let head = 0;
//             let element = 0;

//             // Check both K and V tensors (index 0 and 1)
//             for kv in 0..2 {
//                 let actual = dst_layer
//                     .data
//                     .get(kv)?
//                     .get(block_idx)?
//                     .get(position)?
//                     .get(head)?
//                     .get(element)?
//                     .to_scalar::<f32>()?;

//                 assert_eq!(
//                     expected_value, actual,
//                     "Value mismatch at kv {}, block_idx {}, expected {} but got {}",
//                     kv, block_idx, expected_value, actual
//                 );
//             }
//         }

//         // Print some values for visual verification
//         println!("\nSample values from destination tensor after copy:");
//         for block_idx in 0..nblocks {
//             let position = 0;
//             let head = 0;
//             let element = 0;
//             let k_value = dst_layer
//                 .data
//                 .get(0)?
//                 .get(block_idx)?
//                 .get(position)?
//                 .get(head)?
//                 .get(element)?
//                 .to_scalar::<f32>()?;

//             let v_value = dst_layer
//                 .data
//                 .get(1)?
//                 .get(block_idx)?
//                 .get(position)?
//                 .get(head)?
//                 .get(element)?
//                 .to_scalar::<f32>()?;

//             println!("  Block {}: K={}, V={}", block_idx, k_value, v_value);
//         }

//         println!("KvLayer copy_blocks verification passed!");

//         Ok(())
//     }

//     #[test]
//     fn test_block_copy_kernel_zero_copy() -> Result<()> {
//         use std::time::Instant;

//         // Setup CUDA device
//         let device = Device::cuda_if_available(0)?;
//         let cuda_device = match &device {
//             Device::Cuda(dev) => dev,
//             _ => panic!("Expected CUDA device"),
//         };

//         // Define tensor dimensions
//         let kv_size = 2; // 2 for key and value
//         let src_n_blocks = 10; // Source has 10 blocks
//         let dst_n_blocks = 20; // Destination has 20 blocks
//         let block_size = 2;
//         let heads_per_rank = 4;
//         let head_size = 8;

//         println!("Allocating source tensor on device...");

//         // Create source tensor on device with block ID values
//         // Each element in block 'n' will have the value 'n'
//         let mut src_data = Vec::new();
//         for _kv in 0..kv_size {
//             for block_id in 0..src_n_blocks {
//                 for _pos in 0..block_size {
//                     for _head in 0..heads_per_rank {
//                         for _i in 0..head_size {
//                             // Use block_id as the value for easy verification
//                             src_data.push(block_id as f32);
//                         }
//                     }
//                 }
//             }
//         }

//         let src_shape = (kv_size, src_n_blocks, block_size, heads_per_rank, head_size);
//         let src_tensor = Tensor::from_vec(src_data, src_shape, &device)?;

//         println!("Allocating destination tensor with pinned memory...");

//         // Allocate pinned memory for destination tensor
//         let total_dst_elements = kv_size * dst_n_blocks * block_size * heads_per_rank * head_size;
//         let mut pinned_mem =
//             unsafe { cuda_device.alloc_pinned::<f32>(total_dst_elements).unwrap() };

//         // Initialize pinned memory to zeros
//         unsafe {
//             let slice = std::slice::from_raw_parts_mut(pinned_mem.as_mut_ptr(), total_dst_elements);
//             slice.fill(0.0);
//         }

//         // Create a tensor that directly references the pinned memory (NO COPY!)
//         let dst_shape = (kv_size, dst_n_blocks, block_size, heads_per_rank, head_size);
//         let dst_tensor = unsafe {
//             let slice = std::slice::from_raw_parts(pinned_mem.as_ptr(), total_dst_elements);
//             Tensor::from_slice(slice, dst_shape, &Device::Cpu)?
//         };

//         // Define block mapping
//         let src_blocks = vec![1, 3, 5, 7, 9]; // Source blocks
//         let dst_blocks = vec![0, 1, 2, 3, 4]; // Destination blocks

//         println!("Performing block copy (device to host)...");

//         // Time the operation
//         let start = Instant::now();
//         copy_blocks_between_layers(&src_tensor, &dst_tensor, &src_blocks, &dst_blocks, 1)?;
//         let elapsed = start.elapsed();

//         println!("dst_tensor: {}", dst_tensor);

//         println!("Copy completed in {:?}", elapsed);

//         // Verify results directly from pinned memory to avoid any copying
//         println!("Verifying results...");

//         // validate the dst_tensor
//         // Validate only the blocks that were transferred
//         for (idx, &dst_block) in dst_blocks.iter().enumerate() {
//             let src_block = src_blocks[idx];

//             for kv in 0..2 {
//                 for pos in 0..block_size {
//                     for head in 0..heads_per_rank {
//                         for i in 0..head_size {
//                             // Get the value at this position in the destination tensor
//                             let value = dst_tensor
//                                 .get(kv)?
//                                 .get(dst_block)?
//                                 .get(pos)?
//                                 .get(head)?
//                                 .get(i)?;

//                             // The value should match the source block ID
//                             let value_f32 = value.to_scalar::<f32>()?;
//                             assert_eq!(
//                                 value_f32, src_block as f32,
//                                 "Value mismatch at kv={}, dst_block={}, pos={}, head={}, i={}, expected={}",
//                                 kv, dst_block, pos, head, i, src_block
//                             );
//                         }
//                     }
//                 }
//             }
//         }

//         println!("All tests passed successfully!");

//         Ok(())
//     }

//     #[test]
//     fn test_reshape_to_3d_on_block_dim_comprehensive() -> Result<()> {
//         use super::reshape_to_3d_on_block_dim;

//         // Set up the device - first try CUDA, fall back to CPU
//         let device = match Device::cuda_if_available(0) {
//             Ok(dev) => dev,
//             Err(_) => Device::Cpu,
//         };

//         // Test case 1: Block dimension in the middle
//         // (a, b, X, c, d, e) becomes (a*b, X, c*d*e)
//         let tensor1 = Tensor::zeros((2, 3, 4, 5, 6, 7), DType::F32, &device)?;
//         let reshaped1 = reshape_to_3d_on_block_dim(&tensor1, 2)?;
//         assert_eq!(reshaped1.shape().dims(), &[2 * 3, 4, 5 * 6 * 7]);

//         // Test case 2: Block dimension at the beginning
//         // (X, a, b, c, d) becomes (1, X, a*b*c*d)
//         let tensor2 = Tensor::zeros((4, 3, 2, 5), DType::F32, &device)?;
//         let reshaped2 = reshape_to_3d_on_block_dim(&tensor2, 0)?;
//         assert_eq!(reshaped2.shape().dims(), &[1, 4, 3 * 2 * 5]);

//         // Test case 3: Block dimension at the end
//         // (a, b, c, X) becomes (a*b*c, X, 1)
//         let tensor3 = Tensor::zeros((2, 3, 4, 5), DType::F32, &device)?;
//         let reshaped3 = reshape_to_3d_on_block_dim(&tensor3, 3)?;
//         assert_eq!(reshaped3.shape().dims(), &[2 * 3 * 4, 5, 1]);

//         // Test case 4: Block dimension with single dimensions
//         // (1, X, 1) becomes (1, X, 1)
//         let tensor4 = Tensor::zeros((1, 4, 1), DType::F32, &device)?;
//         let reshaped4 = reshape_to_3d_on_block_dim(&tensor4, 1)?;
//         assert_eq!(reshaped4.shape().dims(), &[1, 4, 1]);

//         // Test case 5: Single dimension tensor
//         // (X) becomes (1, X, 1)
//         let tensor5 = Tensor::zeros((5,), DType::F32, &device)?;
//         let reshaped5 = reshape_to_3d_on_block_dim(&tensor5, 0)?;
//         assert_eq!(reshaped5.shape().dims(), &[1, 5, 1]);

//         // Test case 6: Error case - block_dim out of bounds
//         let tensor6 = Tensor::zeros((2, 3), DType::F32, &device)?;
//         let result6 = reshape_to_3d_on_block_dim(&tensor6, 2);
//         assert!(result6.is_err());

//         Ok(())
//     }

//     #[test]
//     fn test_simple_block_copy() -> Result<()> {
//         // Setup CUDA device
//         let device = Device::cuda_if_available(0)?;

//         // Simple tensor shapes
//         let shape = &[2, 4, 64]; // Small power-of-2 shape for good alignment

//         // Create source tensor on device
//         let src_tensor = Tensor::ones(shape, DType::F32, &device)?;

//         // Create destination tensor using create_pinned_tensor
//         let dst_tensor = create_pinned_tensor(shape, DType::F32)?;

//         // Simple block mapping - just copy 1 block
//         let src_blocks = vec![1];
//         let dst_blocks = vec![0];

//         // Debug info
//         println!(
//             "Source tensor: device={:?}, shape={:?}, dtype={:?}",
//             src_tensor.device(),
//             src_tensor.shape(),
//             src_tensor.dtype()
//         );
//         println!(
//             "Destination tensor: device={:?}, shape={:?}, dtype={:?}",
//             dst_tensor.device(),
//             dst_tensor.shape(),
//             dst_tensor.dtype()
//         );

//         // Perform the copy
//         copy_blocks_between_layers(&src_tensor, &dst_tensor, &src_blocks, &dst_blocks, 1)?;

//         // This should succeed without misalignment errors
//         Ok(())
//     }
// }
