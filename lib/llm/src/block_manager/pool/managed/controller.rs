// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> ManagedBlockPool<S, L, M> {}

#[async_trait::async_trait]
impl<S: Storage, L: LocalityProvider, M: BlockMetadata> AsyncBlockPoolController
    for ManagedBlockPool<S, L, M>
{
    async fn status(&self) -> Result<BlockPoolStatus, BlockPoolError> {
        self.client.status().await
    }

    async fn reset(&self) -> Result<(), BlockPoolError> {
        self.client.reset().await
    }

    async fn reset_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<ResetBlocksResponse, BlockPoolError> {
        self.client.reset_blocks(sequence_hashes).await
    }
}
