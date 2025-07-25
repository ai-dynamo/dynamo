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

use crate::block_manager::{
    block::{registry::BlockRegistrationError, BlockState, PrivateBlockExt},
    events::{PublishHandle, Publisher},
};

use super::*;

use active::ActiveBlockPool;
use inactive::InactiveBlockPool;

impl<S: Storage, L: LocalityProvider + 'static, M: BlockMetadata> State<S, L, M> {
    pub fn new(
        event_manager: Arc<dyn EventManager>,
        return_tx: tokio::sync::mpsc::UnboundedSender<Block<S, L, M>>,
        global_registry: GlobalRegistry,
        async_runtime: Handle,
        metrics: Arc<PoolMetrics>,
        default_duplication_setting: BlockRegistrationDuplicationSetting,
    ) -> Self {
        Self {
            active: ActiveBlockPool::new(),
            inactive: InactiveBlockPool::new(),
            registry: BlockRegistry::new(event_manager.clone(), global_registry, async_runtime),
            return_tx,
            event_manager,
            metrics,
            default_duplication_setting,
        }
    }

    pub fn allocate_blocks(
        &mut self,
        count: usize,
    ) -> Result<Vec<MutableBlock<S, L, M>>, BlockPoolError> {
        let available_blocks = self.inactive.available_blocks() as usize;

        if available_blocks < count {
            tracing::debug!(
                "not enough blocks available, requested: {}, available: {}",
                count,
                available_blocks
            );
            return Err(BlockPoolError::NotEnoughBlocksAvailable(
                count,
                available_blocks,
            ));
        }

        let mut blocks = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(block) = self.inactive.acquire_free_block() {
                blocks.push(MutableBlock::new(block, self.return_tx.clone()));
            }
        }

        self.metrics
            .counter("blocks_allocated")
            .inc_by(count as u64);

        Ok(blocks)
    }

    pub async fn register_blocks(
        &mut self,
        mut blocks: Vec<MutableBlock<S, L, M>>,
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
    ) -> Result<Vec<ImmutableBlock<S, L, M>>, BlockPoolError> {
        assert!(!blocks.is_empty(), "no blocks to register");
        let expected_len = blocks.len();

        let mut immutable_blocks = Vec::with_capacity(expected_len);
        let mut publish_handles = self.publisher();
        let mut start_position = 0;

        while !blocks.is_empty() {
            match self.try_register_blocks_direct(
                &mut blocks,
                &mut immutable_blocks,
                &mut publish_handles,
                start_position,
            ) {
                Ok(()) => {
                    break;
                }
                Err(e) => match e.retry_or_err() {
                    Ok(restart_position) => {
                        start_position = restart_position;
                    }
                    Err(pool_err) => return Err(pool_err),
                },
            }

            // caught a retry error, so we need to process the remaining blocks
            // however, we can not continue to process until the first block is returned
            if let Some(block) = blocks.first() {
                self.process_return_channel_until_block_is_returned(
                    block.sequence_hash()?,
                    return_rx,
                )
                .await;
            }
        }

        Ok(immutable_blocks)
    }

    pub fn try_register_blocks_direct(
        &mut self,
        blocks: &mut [MutableBlock<S, L, M>],
        immutable_blocks: &mut Vec<ImmutableBlock<S, L, M>>,
        publish_handles: &mut Publisher,
        start_position: usize,
    ) -> Result<(), PoolRegisterBlocksError> {
        assert!(!blocks.is_empty(), "no blocks to register");
        assert!(
            start_position < blocks.len(),
            "start position is out of bounds"
        );

        let mut restart_position = start_position;

        for block in &mut blocks[start_position..] {
            let raw_block = block.try_take_block(private::PrivateToken);
            let local_block = MutableBlock::new(raw_block, self.return_tx.clone());

            match self.register_block(local_block) {
                Ok((immutable, publish_handle)) => {
                    immutable_blocks.push(immutable);
                    if let Some(publish_handle) = publish_handle {
                        publish_handles.take_handle(publish_handle);
                    }
                    restart_position += 1;
                }
                Err(e) => match e.retry_or_err() {
                    Ok(mut partial) => {
                        std::mem::swap(block, &mut partial);
                        return Err(PoolRegisterBlocksError::retry(restart_position));
                    }
                    Err(pool_err) => return Err(pool_err.into()),
                },
            }
        }

        assert_eq!(immutable_blocks.len(), blocks.len());
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub fn register_block(
        &mut self,
        mut block: MutableBlock<S, L, M>,
    ) -> Result<(ImmutableBlock<S, L, M>, Option<PublishHandle>), PoolRegisterBlockError<S, L, M>>
    {
        let sequence_hash = block.sequence_hash()?;

        // If the block is already registered, acquire a clone of the immutable block
        if let Some(immutable) = self.active.match_sequence_hash(sequence_hash) {
            match self.default_duplication_setting {
                BlockRegistrationDuplicationSetting::Allowed => {
                    return Ok((ImmutableBlock::make_duplicate(block, immutable)?, None));
                }
                BlockRegistrationDuplicationSetting::Disabled => {
                    // immediate return the block to the pool if duplicates are disabled
                    self.inactive
                        .return_block(block.try_take_block(private::PrivateToken));
                    return Ok((immutable, None));
                }
            }
        }

        if let Some(raw_block) = self.inactive.match_sequence_hash(sequence_hash) {
            // We already have a match, so our block is a duplicate.
            assert!(matches!(raw_block.state(), BlockState::Registered(_, _)));
            let primary = self
                .active
                .register(MutableBlock::new(raw_block, self.return_tx.clone()))?;
            return Ok((ImmutableBlock::make_duplicate(block, primary)?, None));
        }

        // Attempt to register the block
        // On the very rare chance that the block is registered, but in the process of being returned,
        // we will wait for it to be returned and then register it.
        let publish_handle = match block.register(&mut self.registry) {
            Ok(handle) => handle,
            Err(BlockRegistrationError::BlockAlreadyRegistered(_)) => {
                return Err(PoolRegisterBlockError::retry(block));
            }
            Err(e) => return Err(BlockPoolError::RegistrationFailed(e.to_string()).into()),
        };

        let immutable = self.active.register(block)?;

        if let Some(priority) = immutable.metadata().offload_priority() {
            immutable.enqueue_offload(priority).unwrap();
        }

        self.metrics.counter("blocks_registered").inc();

        Ok((immutable, publish_handle))
    }

    pub async fn match_sequence_hashes(
        &mut self,
        sequence_hashes: Vec<SequenceHash>,
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
    ) -> Vec<ImmutableBlock<S, L, M>> {
        let mut immutable_blocks = Vec::new();
        let mut start_position = 0;

        loop {
            match self.try_match_sequence_hashes_direct(
                &sequence_hashes,
                &mut immutable_blocks,
                start_position,
            ) {
                Ok(()) => {
                    break;
                }
                Err(e) => match e.retry_or_err() {
                    Ok(restart_position) => {
                        start_position = restart_position;

                        self.process_return_channel_until_block_is_returned(
                            sequence_hashes[start_position],
                            return_rx,
                        )
                        .await
                    }
                    Err(_) => {
                        unreachable!("this should be unreachable");
                    }
                },
            }
        }

        immutable_blocks
    }

    pub fn try_match_sequence_hashes_direct(
        &mut self,
        sequence_hashes: &[SequenceHash],
        immutable_blocks: &mut Vec<ImmutableBlock<S, L, M>>,
        start_position: usize,
    ) -> Result<(), PoolMatchHashesError> {
        let mut current_position = start_position;
        for sequence_hash in &sequence_hashes[start_position..] {
            // O(1) per operation
            match self.match_sequence_hash(*sequence_hash) {
                Ok(Some(immutable)) => immutable_blocks.push(immutable),
                Ok(None) => break,
                Err(e) => match e.retry_or_err() {
                    Ok(sequence_hash) => {
                        assert_eq!(sequence_hash, sequence_hashes[current_position]);
                        return Err(PoolMatchHashesError::retry(current_position));
                    }
                    Err(pool_err) => return Err(pool_err.into()),
                },
            }
            current_position += 1;
        }

        self.metrics
            .counter("cache_hits")
            .inc_by(immutable_blocks.len() as u64);
        self.metrics
            .counter("cache_misses")
            .inc_by(sequence_hashes.len() as u64 - immutable_blocks.len() as u64);

        Ok(())
    }

    fn match_sequence_hash(
        &mut self,
        sequence_hash: SequenceHash,
    ) -> Result<Option<ImmutableBlock<S, L, M>>, PoolMatchHashError> {
        if !self.registry.is_registered(sequence_hash) {
            self.metrics.counter("cache_misses").inc();
            return Ok(None);
        }

        // the block is registered, so to get it from either the:
        // 1. active pool
        // 2. inactive pool
        // 3. retry -- allow the return channel to be processed

        if let Some(immutable) = self.active.match_sequence_hash(sequence_hash) {
            self.metrics.counter("cache_hits").inc();
            return Ok(Some(immutable));
        }

        if let Some(raw_block) = self.inactive.match_sequence_hash(sequence_hash) {
            assert!(matches!(raw_block.state(), BlockState::Registered(_, _)));

            let mutable = MutableBlock::new(raw_block, self.return_tx.clone());

            let immutable = self
                .active
                .register(mutable)
                .expect("unable to register block; should never happen");

            self.metrics.counter("cache_hits").inc();
            return Ok(Some(immutable));
        }

        tracing::debug!("match_sequence_hash: {} retry required", sequence_hash);
        Err(PoolMatchHashError::retry(sequence_hash))
    }

    pub async fn touch_blocks(
        &mut self,
        sequence_hashes: &[SequenceHash],
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
    ) {
        for sequence_hash in sequence_hashes {
            if !self.registry.is_registered(*sequence_hash) {
                break;
            }

            let block = if let Some(block) = self.inactive.match_sequence_hash(*sequence_hash) {
                block
            } else if self.active.match_sequence_hash(*sequence_hash).is_none() {
                self.wait_for_returned_block(*sequence_hash, return_rx)
                    .await
            } else {
                continue;
            };

            self.inactive.return_block(block);
        }
    }

    /// Returns a block to the inactive pool
    pub fn return_block(&mut self, mut block: Block<S, L, M>) {
        if let Some(registered_block) = block.detach_registered_block() {
            if let Some(mut raw) = Arc::try_unwrap(registered_block)
                .ok()
                .map(|mut b| b.try_take_block(private::PrivateToken))
            {
                self.active.remove(&mut raw);
                self.inactive.return_block(raw);
            }
        }

        self.active.remove(&mut block);
        self.inactive.return_block(block);
    }

    pub(crate) fn publisher(&self) -> Publisher {
        Publisher::new(self.event_manager.clone())
    }

    pub fn status(&self) -> BlockPoolStatus {
        let active = self.active.status();
        let (inactive, empty) = self.inactive.status();
        BlockPoolStatus {
            active_blocks: active,
            inactive_blocks: inactive,
            empty_blocks: empty,
        }
    }

    pub fn try_reset_blocks(&mut self, sequence_hashes: &[SequenceHash]) -> ResetBlocksResponse {
        let mut reset_blocks = Vec::new();
        let mut not_found = Vec::new();
        let mut not_reset = Vec::new();

        for sequence_hash in sequence_hashes {
            if !self.registry.is_registered(*sequence_hash) {
                not_found.push(*sequence_hash);
                continue;
            }

            if let Some(mut block) = self.inactive.match_sequence_hash(*sequence_hash) {
                reset_blocks.push(*sequence_hash);
                block.reset();
                self.inactive.return_block(block);
            } else {
                not_reset.push(*sequence_hash);
            }
        }

        ResetBlocksResponse {
            reset_blocks,
            not_found,
            not_reset,
        }
    }

    /// We have a strong guarantee that the block will be returned to the pool in the near future.
    /// The caller must take ownership of the block
    async fn wait_for_returned_block(
        &mut self,
        sequence_hash: SequenceHash,
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
    ) -> Block<S, L, M> {
        while let Some(block) = return_rx.recv().await {
            if matches!(block.state(), BlockState::Registered(handle, _) if handle.sequence_hash() == sequence_hash)
            {
                return block;
            }
            self.return_block(block);
        }

        unreachable!("this should be unreachable");
    }

    /// Process return channel until a specific block is returned
    async fn process_return_channel_until_block_is_returned(
        &mut self,
        sequence_hash: SequenceHash,
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
    ) {
        let mut done = false;
        while let Some(block) = return_rx.recv().await {
            if matches!(block.state(), BlockState::Registered(handle, _) if handle.sequence_hash() == sequence_hash)
            {
                done = true;
            }
            self.return_block(block);

            if done {
                break;
            }
        }
    }
}
