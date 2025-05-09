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
    block::{registry::BlockRegistationError, BlockState, PrivateBlockExt},
    events::Publisher,
};

use super::*;

impl<S: Storage, M: BlockMetadata> State<S, M> {
    fn new(
        event_manager: Arc<dyn EventManager>,
        return_tx: tokio::sync::mpsc::UnboundedSender<Block<S, M>>,
    ) -> Self {
        Self {
            active: ActiveBlockPool::new(),
            inactive: InactiveBlockPool::new(),
            registry: BlockRegistry::new(event_manager.clone()),
            return_tx,
            event_manager,
        }
    }

    async fn handle_priority_request(
        &mut self,
        req: PriorityRequest<S, M>,
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, M>>,
    ) {
        match req {
            PriorityRequest::AllocateBlocks(req) => {
                let (count, resp_tx) = req.dissolve();
                let blocks = self.allocate_blocks(count);
                if resp_tx.send(blocks).is_err() {
                    tracing::error!("failed to send response to allocate blocks");
                }
            }
            PriorityRequest::RegisterBlocks(req) => {
                let (blocks, resp_tx) = req.dissolve();
                let immutable_blocks = self.register_blocks(blocks, return_rx).await;
                if resp_tx.send(immutable_blocks).is_err() {
                    tracing::error!("failed to send response to register blocks");
                }
            }
            PriorityRequest::MatchSequenceHashes(req) => {
                let (sequence_hashes, resp_tx) = req.dissolve();
                let immutable_blocks = self.match_sequence_hashes(sequence_hashes, return_rx).await;
                if resp_tx.send(immutable_blocks).is_err() {
                    tracing::error!("failed to send response to match sequence hashes");
                }
            }
        }
    }

    fn handle_control_request(&mut self, req: ControlRequest<S, M>) {
        match req {
            ControlRequest::AddBlocks(blocks) => {
                let (blocks, resp_rx) = blocks.dissolve();
                self.inactive.add_blocks(blocks);
                if resp_rx.send(()).is_err() {
                    tracing::error!("failed to send response to add blocks");
                }
            }
        }
    }

    fn handle_return_block(&mut self, block: Block<S, M>) {
        self.return_block(block);
    }

    /// We have a strong guarantee that the block will be returned to the pool in the near future.
    /// The caller must take ownership of the block
    async fn wait_for_returned_block(
        &mut self,
        sequence_hash: SequenceHash,
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, M>>,
    ) -> Block<S, M> {
        while let Some(block) = return_rx.recv().await {
            if matches!(block.state(), BlockState::Registered(handle) if handle.sequence_hash() == sequence_hash)
            {
                return block;
            }
            self.handle_return_block(block);
        }

        unreachable!("this should be unreachable");
    }

    pub fn allocate_blocks(
        &mut self,
        count: usize,
    ) -> Result<Vec<MutableBlock<S, M>>, BlockPoolError> {
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

        Ok(blocks)
    }

    pub async fn register_blocks(
        &mut self,
        blocks: Vec<MutableBlock<S, M>>,
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, M>>,
    ) -> Result<Vec<ImmutableBlock<S, M>>, BlockPoolError> {
        let expected_len = blocks.len();
        let mut immutable_blocks = Vec::new();

        // raii object that will collect all the publish handles and publish them when the object is dropped
        let mut publish_handles = self.publisher();

        for mut block in blocks.into_iter() {
            let sequence_hash = block.sequence_hash()?;

            // If the block is already registered, acquire a clone of the immutable block
            if let Some(immutable) = self.active.match_sequence_hash(sequence_hash) {
                immutable_blocks.push(immutable);
                continue;
            }

            let mutable = if let Some(raw_block) = self.inactive.match_sequence_hash(sequence_hash)
            {
                assert!(matches!(raw_block.state(), BlockState::Registered(_)));
                MutableBlock::new(raw_block, self.return_tx.clone())
            } else {
                // Attempt to register the block
                // On the very rare chance that the block is registered, but in the process of being returned,
                // we will wait for it to be returned and then register it.
                let result = block.register(&mut self.registry);

                match result {
                    Ok(handle) => {
                        publish_handles.take_handle(handle);
                        block
                    }
                    Err(BlockRegistationError::BlockAlreadyRegistered(_)) => {
                        // Block is already registered, wait for it to be returned
                        let raw_block =
                            self.wait_for_returned_block(sequence_hash, return_rx).await;
                        MutableBlock::new(raw_block, self.return_tx.clone())
                    }
                    Err(e) => {
                        return Err(BlockPoolError::FailedToRegisterBlock(e.to_string()));
                    }
                }
            };

            let immutable = self.active.register(mutable)?;

            immutable_blocks.push(immutable);
        }

        assert_eq!(immutable_blocks.len(), expected_len);

        Ok(immutable_blocks)
    }

    async fn match_sequence_hashes(
        &mut self,
        sequence_hashes: Vec<SequenceHash>,
        return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, M>>,
    ) -> Vec<ImmutableBlock<S, M>> {
        let mut immutable_blocks = Vec::new();
        for sequence_hash in sequence_hashes {
            if !self.registry.is_registered(sequence_hash) {
                return immutable_blocks;
            }

            // the block is registered, so to get it from either the:
            // 1. active pool
            // 2. inactive pool
            // 3. return channel

            if let Some(immutable) = self.active.match_sequence_hash(sequence_hash) {
                immutable_blocks.push(immutable);
                continue;
            }

            let raw_block =
                if let Some(raw_block) = self.inactive.match_sequence_hash(sequence_hash) {
                    raw_block
                } else {
                    self.wait_for_returned_block(sequence_hash, return_rx).await
                };

            // this assert allows us to skip the error checking on the active pool registration step
            assert!(matches!(raw_block.state(), BlockState::Registered(_)));

            let mutable = MutableBlock::new(raw_block, self.return_tx.clone());

            let immutable = self
                .active
                .register(mutable)
                .expect("unable to register block; should ever happen");

            immutable_blocks.push(immutable);
        }

        immutable_blocks
    }

    /// Returns a block to the inactive pool
    pub fn return_block(&mut self, mut block: Block<S, M>) {
        self.active.remove(&mut block);
        self.inactive.return_block(block);
    }

    fn publisher(&self) -> Publisher {
        Publisher::new(self.event_manager.clone())
    }
}

impl<S: Storage, M: BlockMetadata> ProgressEngine<S, M> {
    pub fn new(
        event_manager: Arc<dyn EventManager>,
        priority_rx: tokio::sync::mpsc::UnboundedReceiver<PriorityRequest<S, M>>,
        ctrl_rx: tokio::sync::mpsc::UnboundedReceiver<ControlRequest<S, M>>,
        cancel_token: CancellationToken,
        blocks: Vec<Block<S, M>>,
    ) -> Self {
        let (return_tx, return_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut state = State::<S, M>::new(event_manager, return_tx);

        tracing::debug!(count = blocks.len(), "adding blocks to inactive pool");
        state.inactive.add_blocks(blocks);

        Self {
            priority_rx,
            ctrl_rx,
            cancel_token,
            state,
            return_rx,
        }
    }

    pub async fn step(&mut self) -> bool {
        tokio::select! {
            biased;

            Some(priority_req) = self.priority_rx.recv(), if !self.priority_rx.is_closed() => {
                self.state.handle_priority_request(priority_req, &mut self.return_rx).await;
            }

            Some(req) = self.ctrl_rx.recv(), if !self.ctrl_rx.is_closed() => {
                self.state.handle_control_request(req);
            }

            Some(block) = self.return_rx.recv() => {
                self.state.handle_return_block(block);
            }

            _ = self.cancel_token.cancelled() => {
                return false;
            }
        }

        true
    }
}
// pub(crate) async fn progress_engine<S: Storage, M: BlockMetadata>(
//     event_manager: Arc<dyn EventManager>,
//     mut priority_rx: tokio::sync::mpsc::UnboundedReceiver<PriorityRequest<S, M>>,
//     mut ctrl_rx: tokio::sync::mpsc::UnboundedReceiver<ControlRequest<S, M>>,
//     cancel_token: CancellationToken,
// ) {
//     let (return_tx, mut return_rx) = tokio::sync::mpsc::unbounded_channel();
//     let mut state = State::<S, M>::new(event_manager, return_tx);

//     loop {
//         tokio::select! {
//             biased;

//             Some(priority_req) = priority_rx.recv(), if !priority_rx.is_closed() => {
//                 state.handle_priority_request(priority_req, &mut return_rx).await;
//             }

//             Some(req) = ctrl_rx.recv(), if !ctrl_rx.is_closed() => {
//                 state.handle_control_request(req);
//             }

//             Some(block) = return_rx.recv() => {
//                 state.handle_return_block(block);
//             }

//             _ = cancel_token.cancelled() => {
//                 break;
//             }
//         }
//     }
// }

// pub(crate) async fn progress_engine_v2<S: Storage, M: BlockMetadata>(
//     event_manager: Arc<dyn EventManager>,
//     priority_rx: tokio::sync::mpsc::UnboundedReceiver<PriorityRequest<S, M>>,
//     ctrl_rx: tokio::sync::mpsc::UnboundedReceiver<ControlRequest<S, M>>,
//     cancel_token: CancellationToken,
// ) {
//     let mut progress_engine =
//         ProgressEngine::<S, M>::new(event_manager, priority_rx, ctrl_rx, cancel_token);

//     while progress_engine.step().await {
//         tracing::trace!("progress engine step");
//     }
// }
