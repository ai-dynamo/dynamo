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

use std::sync::Arc;
use std::thread::spawn;
use tokio::sync::mpsc;

use crate::block_manager::block::{BlockMetadata, ImmutableBlock, MutableBlock};
use crate::block_manager::pool::BlockPoolError;
use crate::block_manager::storage::Storage;
use crate::block_manager::BlockPool;
use anyhow::Result;
use cudarc::driver::CudaEvent;

type OnboardResult<Target, Metadata> =
    Result<Vec<ImmutableBlock<Target, Metadata>>, BlockPoolError>;

/// Manage a set of pending transfers.
pub struct PendingTransfer<Source: Storage, Target: Storage, Metadata: BlockMetadata> {
    /// The block being copied from.
    _sources: Vec<Arc<MutableBlock<Source, Metadata>>>,
    /// The block being copied to.
    targets: Vec<MutableBlock<Target, Metadata>>,
    /// The Cuda event that indicates the completion of the transfer.
    event: CudaEvent,
    /// The oneshot sender that optionally returns the registered blocks once the transfer is complete.
    completion_indicator: Option<oneshot::Sender<OnboardResult<Target, Metadata>>>,
    /// The target pool that will receive the registered block.
    target_pool: Arc<Option<BlockPool<Target, Metadata>>>,
}

impl<Source: Storage, Target: Storage, Metadata: BlockMetadata>
    PendingTransfer<Source, Target, Metadata>
{
    pub fn new(
        sources: Vec<Arc<MutableBlock<Source, Metadata>>>,
        targets: Vec<MutableBlock<Target, Metadata>>,
        event: CudaEvent,
        completion_indicator: Option<oneshot::Sender<OnboardResult<Target, Metadata>>>,
        target_pool: Arc<Option<BlockPool<Target, Metadata>>>,
    ) -> Self {
        Self {
            _sources: sources,
            targets,
            event,
            completion_indicator,
            target_pool,
        }
    }
}

pub struct TransferManager<Source: Storage, Target: Storage, Metadata: BlockMetadata> {
    pending_transfer_q: mpsc::Sender<PendingTransfer<Source, Target, Metadata>>,
}

impl<Source: Storage, Target: Storage, Metadata: BlockMetadata>
    TransferManager<Source, Target, Metadata>
{
    pub fn new(max_depth: usize) -> Self {
        let (tx, mut rx) = mpsc::channel::<PendingTransfer<Source, Target, Metadata>>(max_depth);

        spawn(move || {
            while let Some(pending_transfer) = rx.blocking_recv() {
                // Wait for the event.
                pending_transfer.event.synchronize()?;

                let PendingTransfer {
                    targets,
                    target_pool,
                    ..
                } = pending_transfer;

                if let Some(target_pool) = target_pool.as_ref() {
                    // Register the blocks in the new pool only AFTER the transfers have been completed.
                    // This way, we maintain the invariant that blocks that are registered in a pool
                    // are always available in that pool.
                    let blocks = target_pool.register_blocks_blocking(targets)?;

                    if let Some(completion_indicator) = pending_transfer.completion_indicator {
                        completion_indicator.send(Ok(blocks))?;
                    }
                }
            }
            Ok::<(), anyhow::Error>(())
        });

        Self {
            pending_transfer_q: tx,
        }
    }

    pub async fn handle_pending_transfer(
        &self,
        pending_transfer: PendingTransfer<Source, Target, Metadata>,
    ) -> Result<()> {
        self.pending_transfer_q.send(pending_transfer).await?;

        Ok(())
    }
}
