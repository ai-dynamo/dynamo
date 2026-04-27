// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test pool setup builder.

use std::sync::Arc;

use derive_builder::Builder;

use crate::blocks::BlockMetadata;
use crate::pools::{
    BlockStore,
    backends::{FifoReusePolicy, HashMapBackend},
};

/// Configuration for setting up a test [`BlockStore`].
#[derive(Builder)]
#[builder(pattern = "owned")]
pub(crate) struct TestPoolSetup {
    #[builder(default = "10")]
    pub(crate) block_count: usize,

    #[builder(default = "4")]
    pub(crate) block_size: usize,
}

impl TestPoolSetup {
    /// Build a unified [`BlockStore`] backed by a HashMap+FIFO inactive index.
    pub(crate) fn build_store<T: BlockMetadata + Sync>(&self) -> Arc<BlockStore<T>> {
        let reuse_policy = Box::new(FifoReusePolicy::new());
        let backend = Box::new(HashMapBackend::new(reuse_policy));
        BlockStore::new(self.block_count, self.block_size, backend, None)
    }
}
