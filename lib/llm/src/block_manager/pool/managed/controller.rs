// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> ManagedBlockPool<S, L, M> {
    fn _status(&self) -> AsyncResponse<BlockPoolResult<BlockPoolStatus>> {
        let (req, resp_rx) = StatusReq::new(());

        self.ctrl_tx
            .send(ControlRequest::Status(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        Ok(resp_rx)
    }

    fn _reset_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> AsyncResponse<BlockPoolResult<ResetBlocksResponse>> {
        let (req, resp_rx) = ResetBlocksReq::new(sequence_hashes.into());

        self.ctrl_tx
            .send(ControlRequest::ResetBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        Ok(resp_rx)
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> BlockPoolController
    for ManagedBlockPool<S, L, M>
{
    fn status_blocking(&self) -> Result<BlockPoolStatus, BlockPoolError> {
        self._status()?
            .blocking_recv()
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    fn reset_blocking(&self) -> Result<(), BlockPoolError> {
        self._reset()?
            .blocking_recv()
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    fn reset_blocks_blocking(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<ResetBlocksResponse, BlockPoolError> {
        self._reset_blocks(sequence_hashes)?
            .blocking_recv()
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }
}

#[async_trait::async_trait]
impl<S: Storage, L: LocalityProvider, M: BlockMetadata> AsyncBlockPoolController
    for ManagedBlockPool<S, L, M>
{
    async fn status(&self) -> Result<BlockPoolStatus, BlockPoolError> {
        self._status()?
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    async fn reset(&self) -> Result<(), BlockPoolError> {
        self._reset()?
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    async fn reset_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<ResetBlocksResponse, BlockPoolError> {
        self._reset_blocks(sequence_hashes)?
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }
}
