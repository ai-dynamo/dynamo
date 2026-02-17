// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test pool setup builder.

use derive_builder::Builder;

use crate::blocks::BlockMetadata;
use crate::pools::{
    InactivePool, ResetPool,
    backends::{FifoReusePolicy, HashMapBackend},
};

use super::blocks::create_reset_blocks;

/// Configuration for setting up test pools.
#[derive(Builder)]
#[builder(pattern = "owned")]
pub(crate) struct TestPoolSetup {
    #[builder(default = "10")]
    pub(crate) block_count: usize,

    #[builder(default = "4")]
    pub(crate) block_size: usize,
}

impl TestPoolSetup {
    /// Build a reset pool with the configured settings.
    pub(crate) fn build_reset_pool<T: BlockMetadata>(&self) -> ResetPool<T> {
        let blocks = create_reset_blocks::<T>(self.block_count, self.block_size);
        ResetPool::new(blocks, self.block_size, None)
    }

    /// Build both inactive and reset pools with the configured settings.
    pub(crate) fn build_pools<T: BlockMetadata>(&self) -> (InactivePool<T>, ResetPool<T>) {
        let reset_pool = self.build_reset_pool::<T>();
        let reuse_policy = Box::new(FifoReusePolicy::new());
        let backend = Box::new(HashMapBackend::new(reuse_policy));
        let inactive_pool = InactivePool::new(backend, &reset_pool, None);
        (inactive_pool, reset_pool)
    }
}
