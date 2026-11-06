// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multimodal block and request metadata: which multimodal objects a block or request references, and where.

use serde::{Deserialize, Serialize};

/// Multimodal object information within a block.
/// Offsets are relative to the block (0 to block_size-1).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockMmObjectInfo {
    /// Hash identifying this multimodal object
    pub mm_hash: u64,
    /// Token offset ranges where this MM object's placeholders appear within THIS block
    /// Each tuple is (start_offset, end_offset) relative to block start
    pub offsets: Vec<(usize, usize)>,
}

/// Extra metadata for a block containing multimodal objects
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockExtraInfo {
    /// All multimodal objects referenced in this block
    pub mm_objects: Vec<BlockMmObjectInfo>,
}

/// Request-level multimodal object information.
/// Offsets are relative to the entire request token sequence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RequestMmObjectInfo {
    /// Hash identifying this multimodal object
    pub mm_hash: u64,
    /// Token offset ranges where this MM object's placeholders appear in the ENTIRE request
    /// Each tuple is (start_offset, end_offset) relative to request start
    pub offsets: Vec<(usize, usize)>,
}

/// Request-level multimodal metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RequestExtraInfo {
    /// All multimodal objects in this request
    pub mm_objects: Vec<RequestMmObjectInfo>,
}

impl RequestExtraInfo {
    /// Convert request-level MM info to block-level MM info for a sequence of blocks.
    ///
    /// This function splits request-level offsets (relative to the entire request token sequence)
    /// into block-level offsets (relative to each block).
    ///
    /// # Arguments
    /// * `block_size` - The size of each block in tokens
    /// * `total_tokens` - Total number of tokens in the request
    ///
    /// # Returns
    /// A vector of `Option<BlockExtraInfo>` where each element corresponds to a block.
    /// `None` indicates a block with no multimodal objects.
    pub fn to_block_level(
        &self,
        block_size: usize,
        total_tokens: usize,
    ) -> Vec<Option<BlockExtraInfo>> {
        let num_blocks = total_tokens.div_ceil(block_size);
        let mut block_infos: Vec<Option<BlockExtraInfo>> = vec![None; num_blocks];

        for req_mm_obj in &self.mm_objects {
            for (req_start, req_end) in &req_mm_obj.offsets {
                // Find which blocks this offset range spans
                let start_block = req_start / block_size;
                let end_block = (req_end.saturating_sub(1)) / block_size;

                let upper_bound = end_block.min(num_blocks - 1) + 1;
                for (block_idx, block_info_opt) in block_infos
                    .iter_mut()
                    .enumerate()
                    .take(upper_bound)
                    .skip(start_block)
                {
                    let block_start_global = block_idx * block_size;
                    let block_end_global = ((block_idx + 1) * block_size).min(total_tokens);

                    // Calculate the intersection of this MM object's range with this block
                    let local_start = (*req_start).max(block_start_global) - block_start_global;
                    let local_end = (*req_end).min(block_end_global) - block_start_global;

                    if local_start < local_end {
                        let block_info = block_info_opt
                            .get_or_insert_with(|| BlockExtraInfo { mm_objects: vec![] });

                        // Check if we already have this mm_hash in this block
                        if let Some(existing) = block_info
                            .mm_objects
                            .iter_mut()
                            .find(|obj| obj.mm_hash == req_mm_obj.mm_hash)
                        {
                            // Add the offset range to existing object
                            existing.offsets.push((local_start, local_end));
                        } else {
                            // Create new MM object entry for this block
                            block_info.mm_objects.push(BlockMmObjectInfo {
                                mm_hash: req_mm_obj.mm_hash,
                                offsets: vec![(local_start, local_end)],
                            });
                        }
                    }
                }
            }
        }

        block_infos
    }
}
