// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Calls the [`State`] methods via async channel operation. Requests are
/// processed by the progress engine.
pub struct Client<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    priority_tx: tokio::sync::mpsc::UnboundedSender<PriorityRequest<S, L, M>>,
    ctrl_tx: tokio::sync::mpsc::UnboundedSender<ControlRequest<S, L, M>>,
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Client<S, L, M> {
    pub fn new(
        priority_tx: tokio::sync::mpsc::UnboundedSender<PriorityRequest<S, L, M>>,
        ctrl_tx: tokio::sync::mpsc::UnboundedSender<ControlRequest<S, L, M>>,
    ) -> Self {
        Self {
            priority_tx,
            ctrl_tx,
        }
    }

    pub async fn allocate_blocks(&self, count: usize) -> BlockPoolResult<MutableBlocks<S, L, M>> {
        let (req, resp_rx) = AllocateBlocksReq::new(count);

        self.priority_tx
            .send(PriorityRequest::AllocateBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        resp_rx
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    pub async fn register_blocks(
        &self,
        blocks: Vec<MutableBlock<S, L, M>>,
    ) -> BlockPoolResult<ImmutableBlocks<S, L, M>> {
        if blocks.is_empty() {
            return Err(BlockPoolError::NoBlocksToRegister);
        }

        let (req, resp_rx) = RegisterBlocksReq::new(blocks);

        self.priority_tx
            .send(PriorityRequest::RegisterBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        resp_rx
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    pub async fn match_sequence_hashes(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> BlockPoolResult<ImmutableBlocks<S, L, M>> {
        let (req, resp_rx) = MatchHashesReq::new(sequence_hashes.into());

        self.priority_tx
            .send(PriorityRequest::MatchSequenceHashes(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        resp_rx
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    pub async fn reset(&self) -> BlockPoolResult<()> {
        let (req, resp_rx) = ResetReq::new(());

        self.priority_tx
            .send(PriorityRequest::Reset(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        resp_rx
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    pub async fn touch_blocks(&self, sequence_hashes: &[SequenceHash]) -> BlockPoolResult<()> {
        let (req, resp_rx) = TouchBlocksReq::new(sequence_hashes.into());

        self.priority_tx
            .send(PriorityRequest::TouchBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        resp_rx
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    pub async fn add_blocks(&self, blocks: Vec<Block<S, L, M>>) -> BlockPoolResult<()> {
        let (req, resp_rx) = AddBlocksReq::new(blocks);

        self.ctrl_tx
            .send(ControlRequest::AddBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        resp_rx
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    // Controller methods

    pub async fn status(&self) -> BlockPoolResult<BlockPoolStatus> {
        let (req, resp_rx) = StatusReq::new(());

        self.ctrl_tx
            .send(ControlRequest::Status(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        resp_rx
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    pub async fn reset_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> BlockPoolResult<ResetBlocksResponse> {
        let (req, resp_rx) = ResetBlocksReq::new(sequence_hashes.into());

        self.ctrl_tx
            .send(ControlRequest::ResetBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        resp_rx
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    pub async fn try_return_block(&self, mut block: OwnedBlock<S, L, M>) -> BlockPoolResult<()> {
        let raw_block = block
            .try_take_block(private::PrivateToken)
            .ok_or(BlockPoolError::NotReturnable)?;

        let (req, resp_rx) = ReturnBlockReq::new(vec![raw_block]);

        self.priority_tx
            .send(PriorityRequest::ReturnBlock(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        resp_rx
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }
}
