// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::{DistributedRuntime, Runtime};

use derive_getters::Getters;
use serde::{Deserialize, Serialize};

pub const ZMQ_PING_MESSAGE: &str = "ping";
pub const ZMQ_TRANSFER_BLOCKS_MESSAGE: &str = "transfer_blocks";

/// Build a new DistributedRuntime.
/// This is a helper function that builds a new tokio runtime and a new DistributedRuntime.
/// It also blocks on the runtime to ensure the DistributedRuntime is initialized.
pub fn build_drt() -> anyhow::Result<(DistributedRuntime, tokio::runtime::Runtime)> {
    let rt = tokio::runtime::Runtime::new()?;

    let runtime = Runtime::from_handle(rt.handle().clone())?;

    let drt = rt
        .handle()
        .block_on(async move { DistributedRuntime::from_settings(runtime.clone()).await })?;

    Ok((drt, rt))
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum BlockTransferPool {
    Device,
    Host,
    Disk,
}

#[derive(Serialize, Deserialize, Debug, Getters, Clone)]
pub struct BlockTransferRequest {
    from_pool: BlockTransferPool,
    to_pool: BlockTransferPool,
    blocks: Vec<(usize, usize)>,
}

impl BlockTransferRequest {
    #[allow(dead_code)]
    pub fn new(
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            from_pool,
            to_pool,
            blocks,
        }
    }
}
