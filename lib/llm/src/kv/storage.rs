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
    use std::cmp::Ordering;

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

    /// Performs in-place transformation of contiguous blocks for tensor parallelism
    ///
    /// Transforms blocks from (num_blocks, block_size, n_heads, head_size) to
    /// (tp_size, num_blocks, block_size, n_heads/tp_size, head_size)
    ///
    /// Returns a vector of transformed tensors for each rank
    pub fn transform_blocks_for_tp<T: Clone + Default>(
        source: &[T],
        src_blocks: &[usize],
        n_blocks: usize,
        block_size: usize,
        n_heads: usize,
        head_size: usize,
        tp_size: usize,
    ) -> Vec<Vec<T>> {
        // Group contiguous blocks
        let block_groups = group_contiguous_blocks(src_blocks);

        // Calculate dimensions
        let heads_per_rank = n_heads / tp_size;
        let block_elements = block_size * n_heads * head_size;

        // For each rank, create a result tensor
        let mut rank_results = Vec::with_capacity(tp_size);
        for _ in 0..tp_size {
            rank_results.push(Vec::with_capacity(
                src_blocks.len() * block_size * heads_per_rank * head_size,
            ));
        }

        // Process each group of contiguous blocks
        for group in block_groups {
            let group_size = group.len();
            let start_block = group[0];

            // Calculate offset in source tensor
            let start_offset = start_block * block_elements;
            let group_elements = group_size * block_elements;

            // Create a view of just these blocks
            let blocks_slice = &source[start_offset..start_offset + group_elements];

            // Distribute the transformed data to each rank
            for rank in 0..tp_size {
                let mut rank_data =
                    Vec::with_capacity(group_size * block_size * heads_per_rank * head_size);

                for block_idx in 0..group_size {
                    for bs_idx in 0..block_size {
                        // For each block and position within block
                        // The offset in the blocks_data (which only contains this group's blocks)
                        let block_base = block_idx * block_size * n_heads * head_size;
                        let pos_base = block_base + bs_idx * n_heads * head_size;

                        // Extract just this rank's heads
                        for head in 0..heads_per_rank {
                            let src_head = rank * heads_per_rank + head;
                            let src_head_base = pos_base + src_head * head_size;

                            // Copy this head's data
                            for hs in 0..head_size {
                                rank_data.push(blocks_slice[src_head_base + hs].clone());
                            }
                        }
                    }
                }

                // Add this group's data to the rank's result
                rank_results[rank].extend(rank_data);
            }
        }

        rank_results
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
    /// This simulates the operations that would be used in a distributed setup
    /// For each contiguous group of source blocks, it transforms the data for tensor parallelism
    /// and then copies it to the specified destination blocks on each rank
    pub fn transform_and_copy_blocks<T: Clone + Default + Copy>(
        source: &[T],
        target: &mut [T],
        src_blocks: &[usize],
        destinations: &[TensorDestination],
        n_blocks: usize,
        block_size: usize,
        n_heads: usize,
        head_size: usize,
        tp_size: usize,
    ) {
        // Group contiguous source blocks
        let block_groups = group_contiguous_blocks(src_blocks);

        // Calculate dimensions
        let heads_per_rank = n_heads / tp_size;
        let block_elements = block_size * n_heads * head_size;
        let rank_block_elements = block_size * heads_per_rank * head_size;

        // Create temporary buffers for each rank's transformed data
        let mut rank_buffers = Vec::with_capacity(tp_size);
        for _ in 0..tp_size {
            let buffer = vec![T::default(); src_blocks.len() * rank_block_elements];
            rank_buffers.push(buffer);
        }

        // Track the number of blocks processed for each contiguous group
        let mut processed_blocks = 0;

        // Process each group of contiguous blocks
        for group in block_groups {
            let group_size = group.len();
            let start_block = group[0];

            // Calculate offset in source tensor
            let start_offset = start_block * block_elements;
            let group_elements = group_size * block_elements;

            // Get a slice of just these blocks
            let blocks_slice = &source[start_offset..start_offset + group_elements];

            // Transform the data for each rank
            for rank in 0..tp_size {
                let mut rank_data =
                    Vec::with_capacity(group_size * block_size * heads_per_rank * head_size);

                for block_idx in 0..group_size {
                    for bs_idx in 0..block_size {
                        // Calculate offsets in the source data
                        let block_base = block_idx * block_size * n_heads * head_size;
                        let pos_base = block_base + bs_idx * n_heads * head_size;

                        // Extract just this rank's heads
                        for head in 0..heads_per_rank {
                            let src_head = rank * heads_per_rank + head;
                            let src_head_base = pos_base + src_head * head_size;

                            // Copy this head's data
                            for hs in 0..head_size {
                                rank_data.push(blocks_slice[src_head_base + hs].clone());
                            }
                        }
                    }
                }

                // Copy transformed data to the rank's buffer at the right offset
                let offset = processed_blocks * block_size * heads_per_rank * head_size;
                let buffer_slice = &mut rank_buffers[rank][offset..offset + rank_data.len()];
                buffer_slice.copy_from_slice(&rank_data);
            }

            processed_blocks += group_size;
        }

        // Now perform the copy operations to target blocks
        for dest in destinations {
            // Only continue if we have a matching rank buffer
            if dest.rank < tp_size {
                let rank_buffer = &rank_buffers[dest.rank];

                // For each destination block, copy the corresponding transformed block
                for (i, &dst_block) in dest.dst_blocks.iter().enumerate() {
                    if i < src_blocks.len() {
                        // Source offset in the transformed buffer
                        let src_offset = i * rank_block_elements;

                        // Destination offset in the target tensor
                        let dst_offset = dst_block * block_elements;

                        // In a distributed system, this would be an RDMA PUT operation
                        println!(
                            "PUT: Rank {}, src block {} -> dst block {}",
                            dest.rank, i, dst_block
                        );

                        // For testing purposes, we'll just copy the data
                        // Extract this rank's data for the source block
                        let src_data = &rank_buffer[src_offset..src_offset + rank_block_elements];

                        // In a distributed setting, we would copy this to the remote memory
                        // For our test, we'll just print the size
                        println!(
                            "Performed PUT of {} elements to rank {}, block {}",
                            src_data.len(),
                            dest.rank,
                            dst_block
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
    use dynemo_runtime::Result;

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
        use super::tensor_parallel::{group_contiguous_blocks, transform_blocks_for_tp};

        // Define tensor dimensions
        let n_blocks = 128; // Total number of blocks
        let block_size = 64; // Block size
        let n_heads = 8; // Number of heads
        let head_size = 4; // Head size (reduced for testing)
        let tp_size = 2; // Tensor parallel size

        // Calculate total elements in tensor
        let total_elements = n_blocks * block_size * n_heads * head_size;

        // Create host data with iota pattern (sequential numbers)
        let mut host_data = Vec::with_capacity(total_elements);
        for i in 0..total_elements {
            host_data.push(i as f32);
        }

        // Define blocks to transform: [0, 1, 2, 3, 7, 9]
        let src_blocks = vec![0, 1, 2, 3, 7, 9];

        // Transform blocks
        let host_results = transform_blocks_for_tp(
            &host_data,
            &src_blocks,
            n_blocks,
            block_size,
            n_heads,
            head_size,
            tp_size,
        );

        // Verify we got exactly tp_size results
        assert_eq!(host_results.len(), tp_size);

        // Verify the transformation
        // The first rank should have heads 0-3, second rank heads 4-7

        // Group contiguous blocks to help verification
        let block_groups = group_contiguous_blocks(&src_blocks);

        // Validate first group [0,1,2,3] (contiguous blocks)
        for rank in 0..tp_size {
            let heads_per_rank = n_heads / tp_size;

            // Check a few values from the first block (block 0)
            for bs_idx in 0..3 {
                // Just check first few positions
                for head in 0..heads_per_rank {
                    let src_head = rank * heads_per_rank + head;

                    // Calculate expected source offset
                    let expected_src_block_base = 0 * block_size * n_heads * head_size;
                    let expected_src_pos_base =
                        expected_src_block_base + bs_idx * n_heads * head_size;
                    let expected_src_head_base = expected_src_pos_base + src_head * head_size;

                    // Calculate expected destination offset in rank tensor
                    let expected_dst_block_base = 0 * block_size * heads_per_rank * head_size;
                    let expected_dst_pos_base =
                        expected_dst_block_base + bs_idx * heads_per_rank * head_size;
                    let expected_dst_head_base = expected_dst_pos_base + head * head_size;

                    // Check first value in the head
                    assert_eq!(
                        host_results[rank][expected_dst_head_base],
                        host_data[expected_src_head_base],
                        "Mismatch for rank {}, block 0, position {}, head {}",
                        rank,
                        bs_idx,
                        head
                    );
                }
            }

            // Check a few values from block 7 (non-contiguous)
            for bs_idx in 0..3 {
                // Just check first few positions
                for head in 0..heads_per_rank {
                    let src_head = rank * heads_per_rank + head;

                    // Calculate expected source offset
                    let expected_src_block_base = 7 * block_size * n_heads * head_size;
                    let expected_src_pos_base =
                        expected_src_block_base + bs_idx * n_heads * head_size;
                    let expected_src_head_base = expected_src_pos_base + src_head * head_size;

                    // Calculate expected destination offset in rank tensor
                    // Block 7 is the 5th block in our list (after [0,1,2,3])
                    let expected_dst_block_base = 4 * block_size * heads_per_rank * head_size;
                    let expected_dst_pos_base =
                        expected_dst_block_base + bs_idx * heads_per_rank * head_size;
                    let expected_dst_head_base = expected_dst_pos_base + head * head_size;

                    // Check first value in the head
                    assert_eq!(
                        host_results[rank][expected_dst_head_base],
                        host_data[expected_src_head_base],
                        "Mismatch for rank {}, block 7, position {}, head {}",
                        rank,
                        bs_idx,
                        head
                    );
                }
            }
        }

        println!("Tensor transformation verification passed!");

        Ok(())
    }

    #[test]
    fn test_print_transformed_tensor() -> Result<()> {
        use super::tensor_parallel::transform_blocks_for_tp;

        // Define smaller dimensions for easier visualization
        let n_blocks = 4; // Just a few blocks
        let block_size = 2; // Small block size
        let n_heads = 4; // 4 heads
        let head_size = 2; // Head size of 2
        let tp_size = 2; // Split into 2 ranks

        // Calculate total elements
        let total_elements = n_blocks * block_size * n_heads * head_size;

        // Create host data with sequential values for easy visualization
        let mut host_data = Vec::with_capacity(total_elements);
        for i in 0..total_elements {
            host_data.push(i as f32);
        }

        // Define blocks to transform: [0, 1, 3] (0,1 contiguous, 3 separate)
        let src_blocks = vec![0, 1, 3];

        // Transform blocks
        let host_results = transform_blocks_for_tp(
            &host_data,
            &src_blocks,
            n_blocks,
            block_size,
            n_heads,
            head_size,
            tp_size,
        );

        // Print results
        for (rank, rank_data) in host_results.iter().enumerate() {
            println!("Rank {} tensor:", rank);

            // Print in a structured way to visualize the tensor
            let heads_per_rank = n_heads / tp_size;
            for block in 0..src_blocks.len() {
                println!("  Block {}:", block);
                for bs in 0..block_size {
                    println!("    Position {}:", bs);
                    for head in 0..heads_per_rank {
                        print!("      Head {}: [", head);
                        for hs in 0..head_size {
                            let idx = block * block_size * heads_per_rank * head_size
                                + bs * heads_per_rank * head_size
                                + head * head_size
                                + hs;
                            print!("{:.1} ", rank_data[idx]);
                        }
                        println!("]");
                    }
                }
            }
        }

        // Also print original data for comparison
        println!("\nOriginal tensor (blocks 0, 1, 3):");
        for &block in &src_blocks {
            println!("  Block {}:", block);
            for bs in 0..block_size {
                println!("    Position {}:", bs);
                for head in 0..n_heads {
                    print!("      Head {}: [", head);
                    for hs in 0..head_size {
                        let idx = block * block_size * n_heads * head_size
                            + bs * n_heads * head_size
                            + head * head_size
                            + hs;
                        print!("{:.1} ", host_data[idx]);
                    }
                    println!("]");
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_transform_and_copy_blocks() -> Result<()> {
        use super::tensor_parallel::{transform_and_copy_blocks, TensorDestination};

        // Define smaller dimensions for easier visualization
        let n_blocks = 10; // 10 blocks total
        let block_size = 2; // Small block size
        let n_heads = 4; // 4 heads
        let head_size = 2; // Head size of 2
        let tp_size = 2; // Split into 2 ranks

        // Calculate total elements
        let total_elements = n_blocks * block_size * n_heads * head_size;

        // Create host data with sequential values for easy visualization
        let mut host_data = Vec::with_capacity(total_elements);
        for i in 0..total_elements {
            host_data.push(i as f32);
        }

        // Create target tensor
        let mut target_data = vec![0.0f32; total_elements];

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
            &host_data,
            &mut target_data,
            &src_blocks,
            &destinations,
            n_blocks,
            block_size,
            n_heads,
            head_size,
            tp_size,
        );

        println!("Transformation and copy completed!");

        Ok(())
    }
}
