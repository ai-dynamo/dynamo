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

use std::sync::Weak;

use crate::block_manager::block::{BlockMetadata, ImmutableBlock, MutableBlock};
use crate::block_manager::pool::BlockPoolError;
use crate::block_manager::storage::Storage;

/// Data needed to offload a block.
/// While the block is in the offload queue, we hold a weak reference to it.
/// This way, we don't prevent the block from being reused if needed.
#[derive(Debug)]
pub struct OffloadRequest<S: Storage, M: BlockMetadata> {
    pub priority: u64,
    pub block: Weak<MutableBlock<S, M>>,
    pub sequence_hash: u64,
    pub parent_sequence_hash: Option<u64>,
}

pub type BlockResult<Target, Metadata> =
    Result<Vec<ImmutableBlock<Target, Metadata>>, BlockPoolError>;

/// Data needed for onboarding.
/// Unlike offloading, we need a means to return the resulting blocks to the caller.
pub struct OnboardRequest<Source: Storage, Target: Storage, M: BlockMetadata> {
    pub blocks: Vec<ImmutableBlock<Source, M>>,
    pub response_tx:
        oneshot::Sender<std::result::Result<Vec<ImmutableBlock<Target, M>>, BlockPoolError>>,
}

impl<Source: Storage, Target: Storage, M: BlockMetadata> OnboardRequest<Source, Target, M> {
    pub fn new(
        blocks: Vec<ImmutableBlock<Source, M>>,
        response_tx: oneshot::Sender<Result<Vec<ImmutableBlock<Target, M>>, BlockPoolError>>,
    ) -> Self {
        Self {
            blocks,
            response_tx,
        }
    }
}
