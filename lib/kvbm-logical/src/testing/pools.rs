// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test pool setup builder.

#![allow(dead_code)]

use std::sync::Arc;

use derive_builder::Builder;

use crate::blocks::BlockMetadata;
use crate::metrics::{BlockPoolMetrics, short_type_name};
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

    #[builder(default = "false")]
    pub(crate) default_reset_on_release: bool,
}

impl TestPoolSetup {
    /// Build a unified [`BlockStore`] backed by a HashMap+FIFO inactive index.
    pub(crate) fn build_store<T: BlockMetadata + Sync>(&self) -> Arc<BlockStore<T>> {
        let reuse_policy = Box::new(FifoReusePolicy::new());
        let backend = Box::new(HashMapBackend::new(reuse_policy));
        let metrics = Arc::new(BlockPoolMetrics::new(short_type_name::<T>()));
        BlockStore::new(
            self.block_count,
            self.block_size,
            backend,
            metrics,
            self.default_reset_on_release,
        )
    }
}
