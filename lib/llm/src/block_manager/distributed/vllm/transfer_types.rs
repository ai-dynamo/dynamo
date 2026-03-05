// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::{
    BlockMetadata, DiskStorage, ImmutableBlock, PinnedStorage, Storage,
    block::{BlockId, locality::LocalityProvider},
    distributed::BlockTransferPool,
    pool::PinGuard,
};
use crate::tokens::TokenBlock;

pub trait AnyBlocks: Send {
    fn len(&self) -> usize;
    fn storage_pool(&self) -> BlockTransferPool;
    fn block_ids(&self) -> Vec<BlockId>;
}

pub struct AnyImmutableBlocks<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    blocks: Vec<ImmutableBlock<S, L, M>>,
    storage_pool: BlockTransferPool,
}

impl<L: LocalityProvider, M: BlockMetadata> AnyImmutableBlocks<PinnedStorage, L, M> {
    pub fn new(blocks: Vec<ImmutableBlock<PinnedStorage, L, M>>) -> Self {
        Self {
            blocks,
            storage_pool: BlockTransferPool::Host,
        }
    }
}

impl<L: LocalityProvider, M: BlockMetadata> AnyImmutableBlocks<DiskStorage, L, M> {
    pub fn new(blocks: Vec<ImmutableBlock<DiskStorage, L, M>>) -> Self {
        Self {
            blocks,
            storage_pool: BlockTransferPool::Disk,
        }
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> AnyImmutableBlocks<S, L, M> {
    pub fn storage_pool(&self) -> BlockTransferPool {
        self.storage_pool
    }

    pub fn block_ids(&self) -> Vec<BlockId> {
        self.blocks.iter().map(|b| b.block_id()).collect()
    }

    fn len(&self) -> usize {
        self.blocks.len()
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> AnyBlocks for AnyImmutableBlocks<S, L, M> {
    fn len(&self) -> usize {
        self.len()
    }

    fn storage_pool(&self) -> BlockTransferPool {
        self.storage_pool()
    }

    fn block_ids(&self) -> Vec<BlockId> {
        self.block_ids()
    }
}

pub enum LocalTransferRequest {
    Offload(LocalOffloadRequest),
    Onboard(LocalOnboardRequest),
    Remote(RemoteTransferRequest),
}

pub struct LocalOffloadRequest {
    pub request_id: String,
    pub block_ids: Vec<BlockId>,
    pub token_blocks: Vec<TokenBlock>,
    pub priorities: Vec<u32>,
    pub operation_id: uuid::Uuid,
    pub sequence_hashes: Vec<u64>,
    pub block_size: usize,
}

impl LocalOffloadRequest {
    pub fn new(
        request_id: String,
        block_ids: Vec<BlockId>,
        token_blocks: Vec<TokenBlock>,
        priorities: Vec<u32>,
        operation_id: uuid::Uuid,
        block_size: usize,
    ) -> Self {
        debug_assert!(block_ids.len() == token_blocks.len());
        debug_assert!(block_ids.len() == priorities.len());
        let sequence_hashes = token_blocks.iter().map(|tb| tb.sequence_hash()).collect();
        Self {
            request_id,
            block_ids,
            token_blocks,
            priorities,
            operation_id,
            sequence_hashes,
            block_size,
        }
    }
}

pub struct LocalOnboardRequest {
    pub request_id: String,
    pub src_blocks: Box<dyn AnyBlocks>,
    pub dst_block_ids: Vec<BlockId>,
    pub operation_id: uuid::Uuid,
}

impl LocalOnboardRequest {
    pub fn new(
        request_id: String,
        src_blocks: Box<dyn AnyBlocks>,
        dst_block_ids: Vec<BlockId>,
        operation_id: uuid::Uuid,
    ) -> Self {
        debug_assert!(src_blocks.len() == dst_block_ids.len());
        Self {
            request_id,
            src_blocks,
            dst_block_ids,
            operation_id,
        }
    }
}

pub struct RemoteTransferRequest {
    pub request_id: String,
    pub sequence_hashes: Vec<u64>,
    pub device_block_ids: Vec<BlockId>,
    pub host_block_ids: Option<Vec<BlockId>>,
    pub operation_id: uuid::Uuid,
    pub block_size: usize,
    pub is_onboard: bool,
    pub pin_id: Option<uuid::Uuid>,
    pub token_blocks: Option<Vec<TokenBlock>>,
    /// W3C traceparent for propagating trace context across async boundaries
    pub traceparent: Option<String>,
}

impl RemoteTransferRequest {
    pub fn from_g4_params(params: &super::integration::G4OnboardParams) -> Self {
        let traceparent = dynamo_runtime::logging::get_distributed_tracing_context()
            .map(|ctx| ctx.create_traceparent());
        Self {
            request_id: params.request_id.clone(),
            sequence_hashes: params.sequence_hashes.clone(),
            device_block_ids: params.device_block_ids.clone(),
            host_block_ids: None,
            operation_id: params.operation_id,
            block_size: params.block_size,
            is_onboard: true,
            pin_id: None,
            token_blocks: Some(params.token_blocks.clone()),
            traceparent,
        }
    }

    pub fn new_h2o(
        request_id: String,
        sequence_hashes: Vec<u64>,
        host_block_ids: Vec<BlockId>,
        operation_id: uuid::Uuid,
        block_size: usize,
        pin_id: uuid::Uuid,
    ) -> Self {
        debug_assert!(sequence_hashes.len() == host_block_ids.len());
        let traceparent = dynamo_runtime::logging::get_distributed_tracing_context()
            .map(|ctx| ctx.create_traceparent());
        Self {
            request_id,
            sequence_hashes,
            device_block_ids: vec![],
            host_block_ids: Some(host_block_ids),
            operation_id,
            block_size,
            is_onboard: false,
            pin_id: Some(pin_id),
            token_blocks: None,
            traceparent,
        }
    }

    pub fn is_h2o(&self) -> bool {
        self.host_block_ids.is_some() && !self.is_onboard
    }
}

/// Item pushed to the drain queue after D2H completes.
/// Holds Arc references to host blocks, preventing eviction until H2R finishes.
pub struct DrainItem {
    pub request_id: String,
    pub sequence_hashes: Vec<u64>,
    pub host_block_ids: Vec<BlockId>,
    pub pin_guard: PinGuard,
    pub block_size: usize,
}
