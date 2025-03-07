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

/// Utility functions for tensor parallel operations
use candle_core::{DType, Device, Result, Tensor};

pub struct KvLayer {
    data: Tensor,
    tp_size: usize,
}

impl KvLayer {
    pub fn new(
        tp_size: usize,
        nblocks: usize,
        block_size: usize,
        n_heads: usize,
        head_size: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let data = Tensor::zeros((2, nblocks, block_size, n_heads, head_size), dtype, &device)?;
        Ok(Self { data, tp_size })
    }

    pub fn nblocks(&self) -> usize {
        self.data.dim(1).unwrap()
    }

    pub fn block_size(&self) -> usize {
        self.data.dim(2).unwrap()
    }

    pub fn n_heads(&self) -> usize {
        self.data.dim(3).unwrap()
    }

    pub fn head_size(&self) -> usize {
        self.data.dim(4).unwrap()
    }

    pub fn copy_blocks(
        &self,
        src_blocks: &[usize],
        dst_blocks: &[usize],
        dst_layer: &mut Self,
    ) -> Result<()> {
        if self.tp_size != dst_layer.tp_size {
            return Err(candle_core::Error::Msg(format!(
                "TP size mismatch: {} != {}",
                self.tp_size, dst_layer.tp_size
            )));
        }

        if src_blocks.len() != dst_blocks.len() {
            return Err(candle_core::Error::Msg(format!(
                "Block length mismatch: {} != {}",
                src_blocks.len(),
                dst_blocks.len()
            )));
        }

        // todo - implement a kernel to copy the blocks if either are gpu tensors
        if self.data.device().is_cuda() || dst_layer.data.device().is_cuda() {
            // todo - implement a kernel to copy the blocks
        } else {
            for kv in 0..2 {
                for (src_idx, dst_idx) in src_blocks.iter().zip(dst_blocks.iter()) {
                    let src_block = self.data.get(kv)?.get(*src_idx)?;
                    let dst_block = dst_layer.data.get(kv)?.get(*dst_idx)?;

                    // Fix: Use a proper dimension value instead of a slice
                    dst_block.slice_set(&src_block, 0, 0)?;
                }
            }
        }

        Ok(())
    }

    /// Split the data into tp_size parts
    /// The target tp_size greater than and evenly divisible by the current tp_size
    pub fn tp_split(&self, tp_size: usize) -> Result<Tensor> {
        if tp_size <= self.tp_size {
            return Err(candle_core::Error::Msg(format!(
                "Target tp_size ({}) must be greater than current tp_size ({})",
                tp_size, self.tp_size
            )));
        }
        if tp_size % self.tp_size != 0 {
            return Err(candle_core::Error::Msg(format!(
                "Target tp_size ({}) must be evenly divisible by current tp_size ({})",
                tp_size, self.tp_size
            )));
        }
        let data = transform_blocks_inplace_v2(&self.data, tp_size)?;
        Ok(data)
    }

    pub fn required_memory(
        nblocks: usize,
        block_size: usize,
        n_heads: usize,
        head_size: usize,
        dtype: DType,
    ) -> usize {
        let n_elements = 2 * nblocks * block_size * n_heads * head_size;
        n_elements * dtype.size_in_bytes()
    }
}

/// Groups contiguous block indices into separate vectors
///
/// For example: [0, 1, 2, 3, 7, 9] -> [[0, 1, 2, 3], [7], [9]]
pub fn group_contiguous_blocks(blocks: &[usize]) -> Vec<Vec<usize>> {
    if blocks.is_empty() {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current_group = vec![blocks[0]];

    for i in 1..blocks.len() {
        if blocks[i] == blocks[i - 1] + 1 {
            // Block is contiguous with previous
            current_group.push(blocks[i]);
        } else {
            // Block starts a new group
            result.push(current_group);
            current_group = vec![blocks[i]];
        }
    }

    // Add the last group
    result.push(current_group);

    result
}

/// Performs a simplified in-place transformation for tensor parallelism
///
/// Takes a tensor of shape [n_blocks, block_size, n_heads, head_size] and returns
/// a tensor of shape [tp_size, n_blocks, block_size, n_heads/tp_size, head_size]
///
/// This uses reshape + transpose operations for maximum efficiency and minimal
/// memory overhead. The tensor transformation happens directly on the GPU without
/// unnecessary copying.
pub fn transform_blocks_inplace_v2(source: &Tensor, tp_size: usize) -> Result<Tensor> {
    // Get the dimensions of the source tensor
    let kv_dim = source.dim(0)?;
    assert_eq!(kv_dim, 2);

    let n_blocks = source.dim(1)?;
    let block_size = source.dim(2)?;
    let n_heads = source.dim(3)?;
    let head_size = source.dim(4)?;

    // Check that n_heads is divisible by tp_size
    if n_heads % tp_size != 0 {
        return Err(candle_core::Error::Msg(format!(
            "Number of heads ({}) must be divisible by tp_size ({})",
            n_heads, tp_size
        )));
    }

    // Calculate heads per rank
    let heads_per_rank = n_heads / tp_size;

    // First, ensure the tensor is contiguous for reshape
    let contiguous_source = source.contiguous()?;

    // Reshape to split the heads dimension
    // [n_blocks, block_size, n_heads, head_size] ->
    // [n_blocks, block_size, tp_size, heads_per_rank, head_size]
    let reshaped =
        contiguous_source.reshape((2, n_blocks, block_size, tp_size, heads_per_rank, head_size))?;

    // Now permute to move tp_size to the first dimension
    // [n_blocks, block_size, tp_size, heads_per_rank, head_size] ->
    // [tp_size, n_blocks, block_size, heads_per_rank, head_size]
    let result = reshaped.permute((3, 0, 1, 2, 4, 5))?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Result, Storage, Tensor};
    use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};

    #[test]
    fn test_group_contiguous_blocks() {
        use super::group_contiguous_blocks;

        // Test empty input
        assert_eq!(group_contiguous_blocks(&[]), Vec::<Vec<usize>>::new());

        // Test single element
        assert_eq!(group_contiguous_blocks(&[5]), vec![vec![5]]);

        // Test all contiguous
        assert_eq!(
            group_contiguous_blocks(&[0, 1, 2, 3]),
            vec![vec![0, 1, 2, 3]]
        );

        // Test multiple groups
        assert_eq!(
            group_contiguous_blocks(&[0, 1, 2, 3, 7, 9]),
            vec![vec![0, 1, 2, 3], vec![7], vec![9]]
        );

        // Test all separate
        assert_eq!(
            group_contiguous_blocks(&[2, 5, 8]),
            vec![vec![2], vec![5], vec![8]]
        );
    }

    #[test]
    fn test_transform_blocks_inplace_v2() -> Result<()> {
        use super::transform_blocks_inplace_v2;

        // Set up the device - first try CUDA, fall back to CPU
        let device = match Device::cuda_if_available(0) {
            Ok(dev) => dev,
            Err(_) => {
                println!("CUDA not available, falling back to CPU");
                Device::Cpu
            }
        };

        println!("Using device: {:?}", device);

        // Define tensor dimensions
        let n_blocks = 4; // Small number of blocks for easy testing
        let block_size = 2; // Small block size for easy testing
        let n_heads = 8; // Number of heads
        let head_size = 4; // Head size (small for testing)
        let tp_size = 2; // Tensor parallel size
        let heads_per_rank = n_heads / tp_size;

        // Create tensor with sequential values for easy validation
        let total_elements = 2 * n_blocks * block_size * n_heads * head_size;
        let mut data = Vec::with_capacity(total_elements);
        for i in 0..total_elements {
            data.push(i as f32);
        }

        // Create the tensor on the selected device
        let source =
            Tensor::from_vec(data, (2, n_blocks, block_size, n_heads, head_size), &device)?;

        match source.device() {
            Device::Cpu => {
                println!("Source tensor shape: {:?}", source.shape());
            }
            Device::Cuda(_) => {
                println!("Source tensor shape: {:?}", source.shape());
                let (storage, layout) = source.storage_and_layout();
                println!("Storage: {:?}", storage);
                println!("Layout: {:?}", layout);
                match &*storage {
                    Storage::Cuda(storage) => {
                        println!("Cuda storage: {:?}", storage);
                        let slice = storage.as_cuda_slice::<f32>()?;
                        let ptr = *slice.device_ptr();
                        println!("Ptr: {:?}", ptr);
                    }
                    _ => {
                        println!("Other storage: {:?}", storage);
                    }
                }
            }
            _ => {
                println!("Source tensor shape: {:?}", source.shape());
            }
        }

        // Print the source tensor shape
        println!("Source tensor shape: {:?}", source.shape());

        // Perform the in-place transformation
        let start_time = std::time::Instant::now();
        let result_tensor = transform_blocks_inplace_v2(&source, tp_size)?;
        let elapsed = start_time.elapsed();

        println!("In-place transformation completed in {:?}", elapsed);
        println!("Result tensor shape: {:?}", result_tensor.shape());

        // Verify the results - shape should be [tp_size, n_blocks, block_size, heads_per_rank, head_size]
        assert_eq!(result_tensor.dim(0)?, tp_size);
        assert_eq!(result_tensor.dim(1)?, 2);
        assert_eq!(result_tensor.dim(2)?, n_blocks);
        assert_eq!(result_tensor.dim(3)?, block_size);
        assert_eq!(result_tensor.dim(4)?, heads_per_rank);
        assert_eq!(result_tensor.dim(5)?, head_size);

        // Move tensors to CPU for validation
        let cpu_source = if source.device().is_cuda() {
            source.to_device(&Device::Cpu)?
        } else {
            source.clone()
        };

        let cpu_result = if result_tensor.device().is_cuda() {
            result_tensor.to_device(&Device::Cpu)?
        } else {
            result_tensor.clone()
        };

        // Validate values - check a few selected positions
        // For each rank, test a few values to ensure they were correctly transformed
        for rank in 0..tp_size {
            // For each block in the source tensor
            for block_idx in 0..n_blocks {
                // Check the first position, first head, first element
                let position = 0;
                let head = 0;
                let element = 0;

                // Compute the expected value from source
                // The heads are now split by rank
                let src_head = rank * heads_per_rank + head;
                let expected = cpu_source
                    .get(0)?
                    .get(block_idx)?
                    .get(position)?
                    .get(src_head)?
                    .get(element)?
                    .to_scalar::<f32>()?;

                // Get the actual value from the result tensor
                // Shape is [tp_size, n_blocks, block_size, heads_per_rank, head_size]
                let actual = cpu_result
                    .get(rank)?
                    .get(0)?
                    .get(block_idx)?
                    .get(position)?
                    .get(head)?
                    .get(element)?
                    .to_scalar::<f32>()?;

                // Compare values
                assert_eq!(
                    expected, actual,
                    "Value mismatch at rank {}, block_idx {}, position {}, head {}, element {}",
                    rank, block_idx, position, head, element
                );
            }
        }

        // Print some values for visual verification
        println!("\nSample values from result tensor:");
        for rank in 0..tp_size {
            println!("Rank {}:", rank);
            for block_idx in 0..1 {
                // Just print the first block
                println!("  Block {}:", block_idx);
                for position in 0..1 {
                    // Just print the first position
                    println!("    Position {}:", position);
                    for head in 0..2 {
                        // Print a couple of heads
                        let head_values = cpu_result
                            .get(rank)?
                            .get(0)?
                            .get(block_idx)?
                            .get(position)?
                            .get(head)?
                            .to_vec1::<f32>()?;
                        println!("      Head {}: {:?}", head, head_values);
                    }
                }
            }
        }

        println!("Transform blocks inplace v2 verification passed!");

        Ok(())
    }

    #[test]
    fn test_copy_blocks() -> Result<()> {
        let tp_size = 1;
        let nblocks = 10;
        let block_size = 4;
        let n_heads = 2;
        let head_size = 8;
        let dtype = DType::F32;
        let device = Device::Cpu;

        // Create source layer filled with ones
        let mut src_layer = KvLayer::new(
            tp_size,
            nblocks,
            block_size,
            n_heads,
            head_size,
            dtype,
            device.clone(),
        )?;
        let ones = Tensor::ones((2, nblocks, block_size, n_heads, head_size), dtype, &device)?;
        src_layer.data = ones;

        // Create destination layer filled with zeros
        let mut dst_layer = KvLayer::new(
            tp_size, nblocks, block_size, n_heads, head_size, dtype, device,
        )?;

        // Copy every even block from src to dst
        let mut src_blocks = Vec::new();
        let mut dst_blocks = Vec::new();
        for i in 0..(nblocks / 2) {
            let block_idx = i * 2; // 0, 2, 4, 6, 8
            src_blocks.push(block_idx);
            dst_blocks.push(block_idx);
        }

        println!("src_blocks: {:?}", src_blocks);
        println!("dst_blocks: {:?}", dst_blocks);

        // Perform the copy
        src_layer.copy_blocks(&src_blocks, &dst_blocks, &mut dst_layer)?;

        println!("dst_layer: {:?}", dst_layer.data);
        println!("dst_layer values: {}", dst_layer.data.to_string());

        // Verify the copy worked correctly
        for block_idx in 0..nblocks {
            let expected_value = if src_blocks.contains(&block_idx) {
                1.0f32
            } else {
                0.0f32
            };

            // Check a sample position in each block
            let position = 0;
            let head = 0;
            let element = 0;

            // Check both K and V tensors (index 0 and 1)
            for kv in 0..2 {
                let actual = dst_layer
                    .data
                    .get(kv)?
                    .get(block_idx)?
                    .get(position)?
                    .get(head)?
                    .get(element)?
                    .to_scalar::<f32>()?;

                assert_eq!(
                    expected_value, actual,
                    "Value mismatch at kv {}, block_idx {}, expected {} but got {}",
                    kv, block_idx, expected_value, actual
                );
            }
        }

        // Print some values for visual verification
        println!("\nSample values from destination tensor after copy:");
        for block_idx in 0..nblocks {
            let position = 0;
            let head = 0;
            let element = 0;
            let k_value = dst_layer
                .data
                .get(0)?
                .get(block_idx)?
                .get(position)?
                .get(head)?
                .get(element)?
                .to_scalar::<f32>()?;

            let v_value = dst_layer
                .data
                .get(1)?
                .get(block_idx)?
                .get(position)?
                .get(head)?
                .get(element)?
                .to_scalar::<f32>()?;

            println!("  Block {}: K={}, V={}", block_idx, k_value, v_value);
        }

        println!("KvLayer copy_blocks verification passed!");

        Ok(())
    }
}
