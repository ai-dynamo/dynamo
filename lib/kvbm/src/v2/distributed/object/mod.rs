// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Object storage module for distributed block management.
//!
//! This module provides traits and implementations for storing KV cache blocks
//! in object storage systems like S3/MinIO.

use async_trait::async_trait;

use crate::v2::physical::layout::LayoutConfig;
use crate::v2::physical::transfer::PhysicalLayout;
use crate::{BlockId, SequenceHash};

#[cfg(feature = "s3")]
pub mod s3;

/// Extension methods for LayoutConfig to support object storage operations.
pub trait LayoutConfigExt {
    /// Compute the size of a single block in bytes.
    fn block_size_bytes(&self) -> usize;

    /// Compute the size of a single memory region in bytes.
    fn region_size(&self) -> usize;
}

impl LayoutConfigExt for LayoutConfig {
    fn block_size_bytes(&self) -> usize {
        self.num_layers
            .saturating_mul(self.outer_dim)
            .saturating_mul(self.page_size)
            .saturating_mul(self.inner_dim)
            .saturating_mul(self.dtype_width_bytes)
    }

    fn region_size(&self) -> usize {
        self.page_size
            .saturating_mul(self.inner_dim)
            .saturating_mul(self.dtype_width_bytes)
    }
}

/// Low-level object storage client trait.
pub trait ObjectClient: Send + Sync {
    /// Check if an object exists.
    fn has_object(&self, key: &[u8]) -> anyhow::Result<bool>;

    /// Put an object.
    fn put_object(&self, key: &[u8], data: &[&[u8]]) -> anyhow::Result<()>;

    /// Get an object.
    fn get_object(&self, key: &[u8], data: &mut [&mut [u8]]) -> anyhow::Result<()>;
}

/// Block-level object storage client trait.
///
/// This trait provides high-level operations for storing and retrieving
/// KV cache blocks in object storage (e.g., S3, MinIO).
///
/// Unlike handle-based operations, methods take a `PhysicalLayout` directly.
/// Handle resolution (LogicalLayoutHandle → LayoutHandle → PhysicalLayout)
/// is done by the caller (e.g., DirectWorker).
#[async_trait]
pub trait ObjectBlockClient: Send + Sync {
    /// Check if blocks exist in object storage.
    ///
    /// Returns a vector of (hash, size_option) pairs where:
    /// - Some(size) indicates the block exists with the given size in bytes
    /// - None indicates the block does not exist or an error occurred
    async fn has_blocks(&self, keys: &[SequenceHash]) -> Vec<(SequenceHash, Option<usize>)>;

    /// Put blocks to object storage.
    ///
    /// # Arguments
    /// * `keys` - Sequence hashes identifying each block
    /// * `layout` - Physical layout containing the block data
    /// * `block_ids` - Block IDs within the layout to upload
    ///
    /// Returns a vector of results for each block:
    /// - Ok(hash) indicates the block was successfully stored
    /// - Err(hash) indicates the block failed to store
    async fn put_blocks(
        &self,
        keys: &[SequenceHash],
        layout: &PhysicalLayout,
        block_ids: &[BlockId],
    ) -> Vec<Result<SequenceHash, SequenceHash>>;

    /// Get blocks from object storage.
    ///
    /// # Arguments
    /// * `keys` - Sequence hashes identifying each block
    /// * `layout` - Physical layout to write the block data into
    /// * `block_ids` - Block IDs within the layout to download into
    ///
    /// Returns a vector of results for each block:
    /// - Ok(hash) indicates the block was successfully retrieved
    /// - Err(hash) indicates the block failed to retrieve
    async fn get_blocks(
        &self,
        keys: &[SequenceHash],
        layout: &PhysicalLayout,
        block_ids: &[BlockId],
    ) -> Vec<Result<SequenceHash, SequenceHash>>;
}
