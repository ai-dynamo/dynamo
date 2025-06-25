// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_getters::Getters;
use serde::{Deserialize, Serialize};

pub const ZMQ_PING_MESSAGE: &str = "ping";
pub const ZMQ_TRANSFER_BLOCKS_MESSAGE: &str = "transfer_blocks";

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
