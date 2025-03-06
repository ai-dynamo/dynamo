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
pub mod tensor_parallel {
    use candle_core::Tensor as T;
    use candle_core::{DType, Device, Result, Tensor, D};

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

    /// Performs transformation of tensor blocks for tensor parallelism using candle-core
    ///
    /// Transforms blocks from (num_blocks, block_size, n_heads, head_size) to
    /// tensors with shape (num_blocks, block_size, n_heads/tp_size, head_size) for each rank
    ///
    /// Returns a vector of transformed tensors for each rank
    pub fn transform_blocks_for_tp(
        source: &Tensor,
        src_blocks: &[usize],
        block_size: usize,
        n_heads: usize,
        head_size: usize,
        tp_size: usize,
    ) -> Result<Vec<Tensor>> {
        let device = source.device();
        let dtype = source.dtype();

        // Calculate dimensions
        let heads_per_rank = n_heads / tp_size;
        let block_elements = block_size * n_heads * head_size;

        // Group contiguous blocks
        let block_groups = group_contiguous_blocks(src_blocks);

        // Create a tensor for each rank
        let mut rank_tensors = Vec::with_capacity(tp_size);
        for _ in 0..tp_size {
            // Create tensor with shape [num_src_blocks, block_size, heads_per_rank, head_size]
            let shape = (src_blocks.len(), block_size, heads_per_rank, head_size);
            let rank_tensor = Tensor::zeros(shape, dtype, device)?;
            rank_tensors.push(rank_tensor);
        }

        // Track the number of blocks processed in the result tensor
        let mut result_block_idx = 0;

        // Process each group of contiguous blocks
        for group in block_groups {
            let group_size = group.len();
            let start_block = group[0];

            // Extract the contiguous blocks from the source tensor
            // Shape: [group_size, block_size, n_heads, head_size]
            let blocks_slice = source.narrow(0, start_block, group_size)?;

            // Process each rank
            for rank in 0..tp_size {
                // Extract this rank's heads from the blocks
                // Starting head index for this rank
                let head_start = rank * heads_per_rank;

                // Extract just the heads for this rank
                // Shape: [group_size, block_size, heads_per_rank, head_size]
                let heads_for_rank = blocks_slice.narrow(2, head_start, heads_per_rank)?;

                // Copy to the result tensor at the right position
                for block_idx in 0..group_size {
                    // Extract this block
                    let block = heads_for_rank.narrow(0, block_idx, 1)?;

                    // Target position in the rank tensor
                    let target_idx = result_block_idx + block_idx;

                    // Create an updated version for this position
                    // Get the current tensor to be updated
                    let mut rank_tensor = rank_tensors[rank].clone();

                    // For each position in the block
                    for bs_idx in 0..block_size {
                        // Get the source position
                        let src_pos = block.narrow(1, bs_idx, 1)?;
                        // Reshape to remove the singleton dimensions for easier manipulation
                        let src_pos = src_pos.reshape((heads_per_rank, head_size))?;

                        // Get the target position in the rank tensor
                        let target_pos =
                            rank_tensor.narrow(0, target_idx, 1)?.narrow(1, bs_idx, 1)?;
                        let target_pos = target_pos.reshape((heads_per_rank, head_size))?;

                        // In candle-core, we need to update the entire tensor
                        // This can be done by using narrow to get the appropriate slices
                        // and then concatenating them back together

                        // First, get the parts before the update
                        let before = if target_idx > 0 {
                            Some(rank_tensor.narrow(0, 0, target_idx)?)
                        } else {
                            None
                        };

                        // Create the updated block
                        let updated_block = rank_tensor.narrow(0, target_idx, 1)?;

                        // Get parts before the position update
                        let before_pos = if bs_idx > 0 {
                            Some(updated_block.narrow(1, 0, bs_idx)?)
                        } else {
                            None
                        };

                        // Create the updated position with new values
                        let updated_pos = Tensor::cat(&[&src_pos.unsqueeze(0)?.unsqueeze(0)?], 0)?;

                        // Get parts after the position update
                        let after_pos = if bs_idx < block_size - 1 {
                            Some(updated_block.narrow(1, bs_idx + 1, block_size - bs_idx - 1)?)
                        } else {
                            None
                        };

                        // Concatenate the positions
                        let mut updated_parts = Vec::new();
                        if let Some(ref before_pos) = before_pos {
                            updated_parts.push(before_pos.clone());
                        }
                        updated_parts.push(updated_pos);
                        if let Some(ref after_pos) = after_pos {
                            updated_parts.push(after_pos.clone());
                        }

                        // Create the updated block
                        let updated_block = Tensor::cat(&updated_parts, 1)?;

                        // Get parts after the block update
                        let after = if target_idx < src_blocks.len() - 1 {
                            Some(rank_tensor.narrow(
                                0,
                                target_idx + 1,
                                src_blocks.len() - target_idx - 1,
                            )?)
                        } else {
                            None
                        };

                        // Concatenate all parts to recreate the full tensor
                        let mut final_parts = Vec::new();
                        if let Some(ref before) = before {
                            final_parts.push(before.clone());
                        }
                        final_parts.push(updated_block);
                        if let Some(ref after) = after {
                            final_parts.push(after.clone());
                        }

                        // Create the final tensor
                        rank_tensor = Tensor::cat(&final_parts, 0)?;
                    }

                    // Update the rank tensor
                    rank_tensors[rank] = rank_tensor;
                }
            }

            result_block_idx += group_size;
        }

        Ok(rank_tensors)
    }

    /// Performs a more efficient transformation using reshape and permute operations
    /// This is closer to an in-place transformation as it minimizes data movement
    pub fn transform_blocks_for_tp_efficient(
        source: &Tensor,
        src_blocks: &[usize],
        n_blocks: usize,
        block_size: usize,
        n_heads: usize,
        head_size: usize,
        tp_size: usize,
    ) -> Result<Vec<Tensor>> {
        let device = source.device();

        // Group contiguous blocks
        let block_groups = group_contiguous_blocks(src_blocks);

        // Calculate dimensions
        let heads_per_rank = n_heads / tp_size;

        // For each rank, create a result tensor
        let mut rank_results = Vec::with_capacity(tp_size);
        for _ in 0..tp_size {
            rank_results.push(Vec::new());
        }

        // Process each group of contiguous blocks
        for group in block_groups {
            let group_size = group.len();
            let start_block = group[0];

            // Extract the contiguous blocks from the source tensor
            let blocks = source.narrow(0, start_block, group_size)?;

            // Reshape to expose the heads dimension for splitting
            // From [group_size, block_size, n_heads, head_size] to
            // [group_size, block_size, tp_size, heads_per_rank, head_size]

            // First ensure the tensor is contiguous for reshape
            let blocks = blocks.contiguous()?;

            // Reshape to split heads dimension into (tp_size, heads_per_rank)
            let reshaped =
                blocks.reshape((group_size, block_size, tp_size, heads_per_rank, head_size))?;

            // For each rank, extract its portion
            for rank in 0..tp_size {
                // Extract this rank's data - this is the most efficient approach
                // Shape: [group_size, block_size, 1, heads_per_rank, head_size]
                let rank_data = reshaped.narrow(2, rank, 1)?;

                // Remove the singleton dimension
                // Shape: [group_size, block_size, heads_per_rank, head_size]
                let rank_data = rank_data.squeeze(2)?;

                // Save this group's tensor for this rank
                rank_results[rank].push(rank_data);
            }
        }

        // For each rank, concatenate its groups
        let mut final_results = Vec::with_capacity(tp_size);
        for rank in 0..tp_size {
            if rank_results[rank].is_empty() {
                // If no blocks were processed for this rank, create an empty tensor
                let shape = (0, block_size, heads_per_rank, head_size);
                final_results.push(Tensor::zeros(shape, source.dtype(), device)?);
            } else if rank_results[rank].len() == 1 {
                // If only one group, no need to concatenate
                final_results.push(rank_results[rank][0].clone());
            } else {
                // Concatenate all groups for this rank
                let rank_tensor = Tensor::cat(&rank_results[rank], 0)?;
                final_results.push(rank_tensor);
            }
        }

        Ok(final_results)
    }

    /// Represents a target destination for transformed tensor data
    pub struct TensorDestination {
        /// The rank to send data to (0 = local)
        pub rank: usize,
        /// The block indices to place data on the destination rank
        pub dst_blocks: Vec<usize>,
    }

    /// Performs tensor transformation and direct copy to destination blocks
    ///
    /// This simulates RDMA operations for tensor parallelism
    pub fn transform_and_copy_blocks(
        source: &Tensor,
        target: &mut Tensor,
        src_blocks: &[usize],
        destinations: &[TensorDestination],
        block_size: usize,
        n_heads: usize,
        head_size: usize,
        tp_size: usize,
    ) -> Result<()> {
        // Transform the blocks using the efficient implementation
        let n_blocks = source.dim(0)?;
        let rank_tensors = transform_blocks_for_tp_efficient(
            source, src_blocks, n_blocks, block_size, n_heads, head_size, tp_size,
        )?;

        // Now perform the copy operations to target blocks
        for dest in destinations {
            // Only continue if we have a matching rank tensor
            if dest.rank < rank_tensors.len() {
                let rank_tensor = &rank_tensors[dest.rank];

                // For each destination block, copy the corresponding transformed block
                for (i, &dst_block) in dest.dst_blocks.iter().enumerate() {
                    if i < src_blocks.len() && i < rank_tensor.dim(0)? {
                        // Extract the source block from the transformed tensor
                        let src_block = rank_tensor.narrow(0, i, 1)?;

                        // In a distributed system, this would be an RDMA PUT operation
                        println!(
                            "PUT: Rank {}, src block {} -> dst block {}",
                            dest.rank, i, dst_block
                        );

                        // For testing purposes, we'll print the shape
                        println!("Block shape: {:?}", src_block.shape());
                    }
                }
            }
        }

        Ok(())
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
        let n_blocks = source.dim(0)?;
        let block_size = source.dim(1)?;
        let n_heads = source.dim(2)?;
        let head_size = source.dim(3)?;

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
        let reshaped = contiguous_source.reshape((
            n_blocks,
            block_size,
            tp_size,
            heads_per_rank,
            head_size,
        ))?;

        // Now permute to move tp_size to the first dimension
        // [n_blocks, block_size, tp_size, heads_per_rank, head_size] ->
        // [tp_size, n_blocks, block_size, heads_per_rank, head_size]
        let result = reshaped.permute((2, 0, 1, 3, 4))?;

        Ok(result)
    }

    /// Performs an optimized RDMA simulation using in-place transformations
    ///
    /// This function:
    /// 1. Transforms the source blocks using the in-place transformation
    /// 2. Simulates RDMA PUTs to the destination blocks
    /// 3. Returns the result tensors for each rank
    pub fn transform_and_rdma_put(
        source: &Tensor,
        src_blocks: &[usize],
        destinations: &[TensorDestination],
        tp_size: usize,
    ) -> Result<Tensor> {
        // Extract the specified blocks
        let mut selected_blocks = Vec::new();
        for &block_idx in src_blocks {
            selected_blocks.push(source.get(block_idx)?);
        }

        // Stack the selected blocks
        let blocks_tensor = Tensor::cat(&selected_blocks, 0)?;

        // Transform the blocks using the in-place transformation
        let rank_tensors = transform_blocks_inplace_v2(&blocks_tensor, tp_size)?;

        // Get the dimensions
        let block_size = source.dim(1)?;
        let n_heads = source.dim(2)?;
        let head_size = source.dim(3)?;
        let heads_per_rank = n_heads / tp_size;

        println!("Transformed tensors:");
        println!("  Shape: {:?}", rank_tensors.shape());

        // Print RDMA PUT operations
        println!("\nRDMA PUT operations:");
        for dest in destinations {
            // Only continue if we have a matching rank
            if dest.rank < tp_size {
                // For each destination block, simulate a PUT
                for (i, &dst_block) in dest.dst_blocks.iter().enumerate() {
                    if i < src_blocks.len() {
                        // Get the source block for this rank
                        let src_block = rank_tensors.get(dest.rank)?.get(i)?;

                        println!(
                            "  PUT: Rank {}, src block {} -> dst block {}",
                            dest.rank, src_blocks[i], dst_block
                        );

                        // In a real implementation, this would be an RDMA PUT
                        // For visualization:
                        let block_elements = block_size * heads_per_rank * head_size;
                        println!("    Transfer size: {} elements", block_elements);
                    }
                }
            }
        }

        // In a real implementation, we would modify the target tensor
        // For this simulation, we just return the transformed tensors
        Ok(rank_tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Result, Tensor};
    use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};

    #[test]
    fn test_create_cuda_memory() -> Result<()> {
        let device = CudaDevice::new(0).unwrap();
        let mut ptr = device.alloc_zeros::<u8>(1024 * 1024 * 1024).unwrap();

        let ptr_mut = ptr.device_ptr_mut();

        // sleep thread for 30 sec
        std::thread::sleep(std::time::Duration::from_secs(30));

        Ok(())
    }

    #[test]
    fn test_group_contiguous_blocks() {
        use super::tensor_parallel::group_contiguous_blocks;

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
    fn test_transform_blocks_for_tp() -> Result<()> {
        use super::tensor_parallel::transform_blocks_for_tp_efficient;

        // Set up the device - first try CUDA, fall back to CPU
        let device = match Device::cuda_if_available(0) {
            Ok(dev) => dev,
            Err(_) => {
                println!("CUDA not available, falling back to CPU");
                Device::Cpu
            }
        };

        // Define tensor dimensions
        let n_blocks = 128; // Total number of blocks
        let block_size = 64; // Block size
        let n_heads = 8; // Number of heads
        let head_size = 4; // Head size (reduced for testing)
        let tp_size = 2; // Tensor parallel size

        // Create tensor with iota pattern (sequential numbers)
        let mut data = Vec::with_capacity(n_blocks * block_size * n_heads * head_size);
        for i in 0..(n_blocks * block_size * n_heads * head_size) {
            data.push(i as f32);
        }

        // Create the tensor
        let source = Tensor::from_vec(data, (n_blocks, block_size, n_heads, head_size), &device)?;

        // Define blocks to transform: [0, 1, 2, 3, 7, 9]
        let src_blocks = vec![0, 1, 2, 3, 7, 9];

        // Transform blocks
        let rank_tensors = transform_blocks_for_tp_efficient(
            &source,
            &src_blocks,
            n_blocks,
            block_size,
            n_heads,
            head_size,
            tp_size,
        )?;

        // Verify we got exactly tp_size results
        assert_eq!(rank_tensors.len(), tp_size);

        // Move tensors back to CPU for validation if needed
        let cpu_source = if source.device().is_cuda() {
            source.to_device(&Device::Cpu)?
        } else {
            source.clone()
        };

        let mut cpu_rank_tensors = Vec::with_capacity(tp_size);
        for tensor in &rank_tensors {
            let cpu_tensor = if tensor.device().is_cuda() {
                tensor.to_device(&Device::Cpu)?
            } else {
                tensor.clone()
            };
            cpu_rank_tensors.push(cpu_tensor);
        }

        // Check each rank tensor's shape
        for rank in 0..tp_size {
            let heads_per_rank = n_heads / tp_size;
            let expected_shape = [src_blocks.len(), block_size, heads_per_rank, head_size];
            assert_eq!(
                cpu_rank_tensors[rank].shape().dims(),
                &expected_shape,
                "Rank {} tensor has incorrect shape",
                rank
            );
        }

        // Validate values for a few sample positions
        let heads_per_rank = n_heads / tp_size;

        // Check the first block, first position in each rank
        for rank in 0..tp_size {
            let block_idx = 0; // First block
            let pos_idx = 0; // First position

            for head in 0..heads_per_rank {
                let src_head = rank * heads_per_rank + head;

                // Calculate indices in source tensor
                let src_block = src_blocks[block_idx]; // Block 0

                // Check first element in each head
                let hs_idx = 0;

                // Get the expected value from source
                let expected = cpu_source
                    .get(src_block)?
                    .get(pos_idx)?
                    .get(src_head)?
                    .get(hs_idx)?
                    .to_scalar::<f32>()?;

                // Get the actual value from the result tensor
                let actual = cpu_rank_tensors[rank]
                    .get(block_idx)?
                    .get(pos_idx)?
                    .get(head)?
                    .get(hs_idx)?
                    .to_scalar::<f32>()?;

                // Compare values
                assert_eq!(
                    expected, actual,
                    "Mismatch at rank {}, block {}, position {}, head {}, element {}",
                    rank, block_idx, pos_idx, head, hs_idx
                );
            }
        }

        println!("Tensor transformation verification passed!");

        Ok(())
    }

    #[test]
    fn test_print_transformed_tensor() -> Result<()> {
        use super::tensor_parallel::transform_blocks_for_tp_efficient;

        // Set up the device - first try CUDA, fall back to CPU
        let device = match Device::cuda_if_available(0) {
            Ok(dev) => dev,
            Err(_) => {
                println!("CUDA not available, falling back to CPU");
                Device::Cpu
            }
        };

        // Define smaller dimensions for easier visualization
        let n_blocks = 4; // Just a few blocks
        let block_size = 2; // Small block size
        let n_heads = 4; // 4 heads
        let head_size = 2; // Head size of 2
        let tp_size = 2; // Split into 2 ranks

        // Create tensor with sequential values for easy visualization
        let mut data = Vec::with_capacity(n_blocks * block_size * n_heads * head_size);
        for i in 0..(n_blocks * block_size * n_heads * head_size) {
            data.push(i as f32);
        }

        // Create the tensor
        let source = Tensor::from_vec(data, (n_blocks, block_size, n_heads, head_size), &device)?;

        // Define blocks to transform: [0, 1, 3] (0,1 contiguous, 3 separate)
        let src_blocks = vec![0, 1, 3];

        // Transform blocks
        let rank_tensors = transform_blocks_for_tp_efficient(
            &source,
            &src_blocks,
            n_blocks,
            block_size,
            n_heads,
            head_size,
            tp_size,
        )?;

        // Move tensors to CPU for printing
        let cpu_source = if source.device().is_cuda() {
            source.to_device(&Device::Cpu)?
        } else {
            source.clone()
        };

        let mut cpu_rank_tensors = Vec::with_capacity(tp_size);
        for tensor in &rank_tensors {
            let cpu_tensor = if tensor.device().is_cuda() {
                tensor.to_device(&Device::Cpu)?
            } else {
                tensor.clone()
            };
            cpu_rank_tensors.push(cpu_tensor);
        }

        // Print rank tensors
        for (rank, tensor) in cpu_rank_tensors.iter().enumerate() {
            println!("Rank {} tensor:", rank);

            // Print shape
            println!("  Shape: {:?}", tensor.shape());

            // Print sample values
            for block in 0..src_blocks.len() {
                println!("  Block {}:", block);
                let block_tensor = tensor.get(block)?;

                for bs in 0..block_size {
                    println!("    Position {}:", bs);
                    let pos_tensor = block_tensor.get(bs)?;

                    for head in 0..n_heads / tp_size {
                        let head_tensor = pos_tensor.get(head)?;
                        let head_values = head_tensor.to_vec1::<f32>()?;
                        println!("      Head {}: {:?}", head, head_values);
                    }
                }
            }
        }

        // Also print original tensor for comparison
        println!("\nOriginal tensor (blocks 0, 1, 3):");
        for &block in &src_blocks {
            println!("  Block {}:", block);
            let block_tensor = cpu_source.get(block)?;

            for bs in 0..block_size {
                println!("    Position {}:", bs);
                let pos_tensor = block_tensor.get(bs)?;

                for head in 0..n_heads {
                    let head_tensor = pos_tensor.get(head)?;
                    let head_values = head_tensor.to_vec1::<f32>()?;
                    println!("      Head {}: {:?}", head, head_values);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_transform_and_copy_blocks() -> Result<()> {
        use super::tensor_parallel::{transform_and_copy_blocks, TensorDestination};

        // Set up the device - first try CUDA, fall back to CPU
        let device = match Device::cuda_if_available(0) {
            Ok(dev) => dev,
            Err(_) => {
                println!("CUDA not available, falling back to CPU");
                Device::Cpu
            }
        };

        // Define smaller dimensions for easier visualization
        let n_blocks = 10; // 10 blocks total
        let block_size = 2; // Small block size
        let n_heads = 4; // 4 heads
        let head_size = 2; // Head size of 2
        let tp_size = 2; // Split into 2 ranks

        // Create tensor with sequential values for easy visualization
        let mut data = Vec::with_capacity(n_blocks * block_size * n_heads * head_size);
        for i in 0..(n_blocks * block_size * n_heads * head_size) {
            data.push(i as f32);
        }

        // Create the source tensor
        let source = Tensor::from_vec(
            data.clone(),
            (n_blocks, block_size, n_heads, head_size),
            &device,
        )?;

        // Create target tensor (same size as source)
        let mut target = Tensor::zeros(
            (n_blocks, block_size, n_heads, head_size),
            DType::F32,
            &device,
        )?;

        // Source blocks: [0, 1, 2, 3, 7, 9]
        let src_blocks = vec![0, 1, 2, 3, 7, 9];

        // Define destination blocks:
        // Rank 0: blocks [21, 22, 31, 32]
        // Rank 1: blocks [41, 42]
        let destinations = vec![
            TensorDestination {
                rank: 0,
                dst_blocks: vec![21, 22, 31, 32],
            },
            TensorDestination {
                rank: 1,
                dst_blocks: vec![41, 42],
            },
        ];

        // Perform the transformation and copy
        transform_and_copy_blocks(
            &source,
            &mut target,
            &src_blocks,
            &destinations,
            block_size,
            n_heads,
            head_size,
            tp_size,
        )?;

        println!("Transformation and copy completed!");

        Ok(())
    }

    #[test]
    fn test_transform_blocks_inplace() -> Result<()> {
        use super::tensor_parallel::transform_blocks_inplace_v2;

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
        let n_blocks = 128;        // Total number of blocks
        let block_size = 64;       // Block size
        let n_heads = 8;           // Number of heads
        let head_size = 4;         // Head size (reduced for testing)
        let tp_size = 2;           // Tensor parallel size
        let heads_per_rank = n_heads / tp_size;

        // Create tensor with sequential values
        let mut data = Vec::with_capacity(n_blocks * block_size * n_heads * head_size);
        for i in 0..(n_blocks * block_size * n_heads * head_size) {
            data.push(i as f32);
        }

        // Create the tensor on the selected device
        let source = Tensor::from_vec(
            data,
            (n_blocks, block_size, n_heads, head_size),
            &device
        )?;

        // Define blocks to transform: [0, 1, 2, 3, 7, 9]
        let src_blocks = vec![0, 1, 2, 3, 7, 9];

        // Extract only the specified blocks
        let mut selected_blocks = Vec::new();
        for &block_idx in &src_blocks {
            selected_blocks.push(source.get(block_idx)?);
        }

        // Stack the selected blocks into a single tensor
        let blocks_tensor = Tensor::cat(&selected_blocks, 0)?;

        // Perform the in-place transformation
        let start_time = std::time::Instant::now();
        let result_tensor = transform_blocks_inplace_v2(&blocks_tensor, tp_size)?;
        let elapsed = start_time.elapsed();

        println!("In-place transformation completed in {:?}", elapsed);

        // Verify the results - shape should be [tp_size, num_blocks, block_size, heads_per_rank, head_size]
        assert_eq!(result_tensor.dim(0)?, tp_size);
        assert_eq!(result_tensor.dim(1)?, src_blocks.len());
        assert_eq!(result_tensor.dim(2)?, block_size);
        assert_eq!(result_tensor.dim(3)?, heads_per_rank);
        assert_eq!(result_tensor.dim(4)?, head_size);

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

        // Verify values for a sample of positions
        for rank in 0..tp_size {
            // Test each source block and its corresponding position in the result
            for (idx, &src_block) in src_blocks.iter().enumerate() {
                // Test just the first position, head, and element for simplicity
                let position = 0;
                let head = 0;
                let element = 0;

                // Compute the expected value from source
                let src_head = rank * heads_per_rank + head;
                let expected = cpu_source.get(src_block)?
                                       .get(position)?
                                       .get(src_head)?
                                       .get(element)?
                                       .to_scalar::<f32>()?;

                // Get the actual value from the result tensor
                // Shape is [tp_size, num_blocks, block_size, heads_per_rank, head_size]
                let actual = cpu_result.get(rank)?
                                     .get(idx)?
                                     .get(position)?
                                     .get(head)?
                                     .get(element)?
                                     .to_scalar::<f32>()?;

                // Compare values
                assert_eq!(
                    expected,
                    actual,
                    "Value mismatch at rank {}, idx {} (src_block {}), position {}, head {}, element {}",
                    rank, idx, src_block, position, head, element
                );
            }
        }

        println!("In-place transformation verification passed!");

        Ok(())
    }

    #[test]
    fn test_transform_and_rdma_put() -> Result<()> {
        use super::tensor_parallel::{transform_and_rdma_put, TensorDestination};

        // Set up the device - first try CUDA, fall back to CPU
        let device = match Device::cuda_if_available(0) {
            Ok(dev) => dev,
            Err(_) => {
                println!("CUDA not available, falling back to CPU");
                Device::Cpu
            }
        };

        println!("Using device: {:?}", device);

        // Define dimensions
        let n_blocks = 10; // 10 blocks total
        let block_size = 2; // Small block size
        let n_heads = 4; // 4 heads
        let head_size = 2; // Head size of 2
        let tp_size = 2; // Split into 2 ranks

        // Create tensor with sequential values
        let mut data = Vec::with_capacity(n_blocks * block_size * n_heads * head_size);
        for i in 0..(n_blocks * block_size * n_heads * head_size) {
            data.push(i as f32);
        }

        // Create the source tensor
        let source = Tensor::from_vec(data, (n_blocks, block_size, n_heads, head_size), &device)?;

        // Source blocks: [0, 1, 2, 3, 7, 9]
        let src_blocks = vec![0, 1, 2, 3, 7, 9];

        // Define destination blocks:
        // Rank 0: blocks [21, 22, 31, 32]
        // Rank 1: blocks [41, 42]
        let destinations = vec![
            TensorDestination {
                rank: 0,
                dst_blocks: vec![21, 22, 31, 32],
            },
            TensorDestination {
                rank: 1,
                dst_blocks: vec![41, 42],
            },
        ];

        // Perform the transformation and RDMA PUT simulation
        let result_tensor = transform_and_rdma_put(&source, &src_blocks, &destinations, tp_size)?;

        // Verify the shape of the tensor - [tp_size, num_blocks, block_size, heads_per_rank, head_size]
        assert_eq!(result_tensor.dim(0)?, tp_size);
        assert_eq!(result_tensor.dim(1)?, src_blocks.len());
        assert_eq!(result_tensor.dim(2)?, block_size);
        assert_eq!(result_tensor.dim(3)?, n_heads / tp_size);
        assert_eq!(result_tensor.dim(4)?, head_size);

        println!("RDMA PUT simulation completed successfully!");

        Ok(())
    }

    #[test]
    fn test_transform_blocks_inplace_v2() -> Result<()> {
        use super::tensor_parallel::transform_blocks_inplace_v2;

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
        let total_elements = n_blocks * block_size * n_heads * head_size;
        let mut data = Vec::with_capacity(total_elements);
        for i in 0..total_elements {
            data.push(i as f32);
        }

        // Create the tensor on the selected device
        let source = Tensor::from_vec(data, (n_blocks, block_size, n_heads, head_size), &device)?;

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
        assert_eq!(result_tensor.dim(1)?, n_blocks);
        assert_eq!(result_tensor.dim(2)?, block_size);
        assert_eq!(result_tensor.dim(3)?, heads_per_rank);
        assert_eq!(result_tensor.dim(4)?, head_size);

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
                    .get(block_idx)?
                    .get(position)?
                    .get(src_head)?
                    .get(element)?
                    .to_scalar::<f32>()?;

                // Get the actual value from the result tensor
                // Shape is [tp_size, n_blocks, block_size, heads_per_rank, head_size]
                let actual = cpu_result
                    .get(rank)?
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
    fn benchmark_transform_methods() -> Result<()> {
        use super::tensor_parallel::{transform_blocks_for_tp, transform_blocks_for_tp_efficient, transform_blocks_inplace_v2};
        use std::time::Instant;

        // Set up the device - first try CUDA, fall back to CPU
        let device = match Device::cuda_if_available(0) {
            Ok(dev) => {
                println!("Using CUDA device");
                dev
            },
            Err(_) => {
                println!("CUDA not available, falling back to CPU");
                Device::Cpu
            }
        };

        // Define tensor dimensions
        let n_blocks = 128;     // Total number of blocks
        let block_size = 64;    // Block size
        let n_heads = 8;        // Number of heads
        let head_size = 128;    // Head size for LLaMA 8B model
        let tp_size = 2;        // Tensor parallel size
        let heads_per_rank = n_heads / tp_size;

        println!("Benchmarking tensor transformation methods with dimensions:");
        println!("  Blocks: {}, Block size: {}, Heads: {}, Head size: {}, TP: {}",
                 n_blocks, block_size, n_heads, head_size, tp_size);

        // Create tensor with sequential values
        println!("Creating tensor...");
        let mut data = Vec::with_capacity(n_blocks * block_size * n_heads * head_size);
        for i in 0..(n_blocks * block_size * n_heads * head_size) {
            data.push(i as f32);
        }

        // Create the tensor
        let source = Tensor::from_vec(
            data,
            (n_blocks, block_size, n_heads, head_size),
            &device
        )?;

        // Define blocks to transform: [0, 1, 2, 3, 7, 9]
        let src_blocks = vec![0, 1, 2, 3, 7, 9];
        let num_src_blocks = src_blocks.len();

        // Extract only the specified blocks for the V2 test
        let mut selected_blocks = Vec::new();
        for &block_idx in &src_blocks {
            selected_blocks.push(source.get(block_idx)?);
        }

        // Stack the selected blocks into a single tensor
        let blocks_tensor = Tensor::cat(&selected_blocks, 0)?;
        println!("Extracted blocks tensor shape: {:?}", blocks_tensor.shape());

        // Warm up the GPU
        println!("Warming up...");
        {
            let _ = transform_blocks_inplace_v2(&blocks_tensor, tp_size)?;
        }

        // Benchmark the different implementations
        println!("\nBenchmarking implementations:");

        // Basic implementation
        let start = Instant::now();
        let basic_result = transform_blocks_for_tp(
            &source,
            &src_blocks,
            block_size,
            n_heads,
            head_size,
            tp_size,
        )?;
        let basic_time = start.elapsed();
        println!("  Basic implementation: {:?}", basic_time);

        // Efficient implementation
        let start = Instant::now();
        let efficient_result = transform_blocks_for_tp_efficient(
            &source,
            &src_blocks,
            n_blocks,
            block_size,
            n_heads,
            head_size,
            tp_size,
        )?;
        let efficient_time = start.elapsed();
        println!("  Efficient implementation: {:?}", efficient_time);

        // In-place implementation
        let start = Instant::now();
        let inplace_result = transform_blocks_inplace_v2(&blocks_tensor, tp_size)?;
        let inplace_time = start.elapsed();
        println!("  In-place implementation: {:?}", inplace_time);
        println!("  Inplace result shape: {:?}", inplace_result.shape());

        // Compare to the basic implementation
        println!("\nPerformance comparisons:");
        println!("  Efficient vs Basic: {:.2}x speedup",
                 basic_time.as_micros() as f64 / efficient_time.as_micros() as f64);
        println!("  In-place vs Basic: {:.2}x speedup",
                 basic_time.as_micros() as f64 / inplace_time.as_micros() as f64);
        println!("  In-place vs Efficient: {:.2}x speedup",
                 efficient_time.as_micros() as f64 / inplace_time.as_micros() as f64);

        // Verify that all implementations produce the same results
        println!("\nVerifying results are equivalent...");
        assert_eq!(basic_result.len(), tp_size);
        assert_eq!(efficient_result.len(), tp_size);
        assert_eq!(inplace_result.dim(0)?, tp_size);

        // For validation, we need to check specific elements by accounting for how the tensors are organized
        // Move tensors to CPU for validation
        println!("Moving tensors to CPU for validation...");

        // Convert the basic and efficient results (Vec<Tensor>) to CPU
        let mut basic_cpu_tensors = Vec::with_capacity(tp_size);
        let mut efficient_cpu_tensors = Vec::with_capacity(tp_size);

        for i in 0..tp_size {
            let basic_cpu = if basic_result[i].device().is_cuda() {
                basic_result[i].to_device(&Device::Cpu)?
            } else {
                basic_result[i].clone()
            };
            basic_cpu_tensors.push(basic_cpu);

            let efficient_cpu = if efficient_result[i].device().is_cuda() {
                efficient_result[i].to_device(&Device::Cpu)?
            } else {
                efficient_result[i].clone()
            };
            efficient_cpu_tensors.push(efficient_cpu);
        }

        // Convert the inplace result (single Tensor) to CPU
        let inplace_cpu = if inplace_result.device().is_cuda() {
            inplace_result.to_device(&Device::Cpu)?
        } else {
            inplace_result.clone()
        };

        // Sample test indices - but only for the blocks that we've actually extracted
        let test_indices = [
            (0, 0, 0, 0),  // First block, first position, first head, first element
            (1, 1, 1, 0),  // Second block, second position, second head, first element
            (2, 0, 0, 1),  // Third block, first position, first head, second element
        ];

        // For each rank, check some sample values
        for rank in 0..tp_size {
            println!("Checking rank {}...", rank);

            // Check a few sample values
            for &(block_idx, pos, head_within_rank, elem) in &test_indices {
                if block_idx < num_src_blocks {
                    let src_block_id = src_blocks[block_idx];
                    let global_head = head_within_rank + rank * heads_per_rank;

                    // Get expected value from original tensor
                    let expected_val = source.get(src_block_id)?
                                            .get(pos)?
                                            .get(global_head)?
                                            .get(elem)?
                                            .to_scalar::<f32>()?;

                    // Get value from basic implementation
                    let basic_val = basic_cpu_tensors[rank].get(block_idx)?
                                                         .get(pos)?
                                                         .get(head_within_rank)?
                                                         .get(elem)?
                                                         .to_scalar::<f32>()?;

                    // Get value from efficient implementation
                    let efficient_val = efficient_cpu_tensors[rank].get(block_idx)?
                                                             .get(pos)?
                                                             .get(head_within_rank)?
                                                             .get(elem)?
                                                             .to_scalar::<f32>()?;

                    // Get value from inplace implementation
                    let inplace_val = inplace_cpu.get(rank)?
                                               .get(block_idx)?
                                               .get(pos)?
                                               .get(head_within_rank)?
                                               .get(elem)?
                                               .to_scalar::<f32>()?;

                    // Print sample values for debugging
                    println!("  Rank {}, Block {} (original ID {}), Pos {}, Head {}, Elem {}: Expected {}, Basic {}, Efficient {}, Inplace {}",
                             rank, block_idx, src_block_id, pos, head_within_rank, elem,
                             expected_val, basic_val, efficient_val, inplace_val);

                    // Compare values
                    assert_eq!(
                        expected_val, basic_val,
                        "Value mismatch between expected and basic at rank {}, position ({}, {}, {}, {})",
                        rank, block_idx, pos, head_within_rank, elem
                    );

                    assert_eq!(
                        expected_val, efficient_val,
                        "Value mismatch between expected and efficient at rank {}, position ({}, {}, {}, {})",
                        rank, block_idx, pos, head_within_rank, elem
                    );

                    assert_eq!(
                        expected_val, inplace_val,
                        "Value mismatch between expected and inplace at rank {}, position ({}, {}, {}, {})",
                        rank, block_idx, pos, head_within_rank, elem
                    );
                }
            }
        }

        println!("All implementations produce equivalent results!");

        Ok(())
    }
}
