// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub mod distributed_leader_worker;
pub mod null;

use crate::block_manager::block::{
    BlockDataProvider, ReadableBlock, WritableBlock,
    transfer::{TransferContext, TransferError, WriteToStrategy},
};
use crate::block_manager::locality::Logical;
use crate::block_manager::storage::{self, nixl::NixlDescriptor};
use tokio::sync::oneshot;

pub enum LogicalKinds {
    Simple,
    Sharded,
}

pub trait LogicalResources: Clone + Send + Sync + 'static + std::fmt::Debug {
    fn handle_transfer<RB, WB>(
        &self,
        sources: &[RB],
        targets: &mut [WB],
        ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError>
    where
        RB: ReadableBlock + WriteToStrategy<WB> + storage::Local,
        <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
        <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
        RB: BlockDataProvider<Locality = Logical<Self>>,
        WB: WritableBlock + BlockDataProviderMut<Locality = Logical<Self>>;
}

/// Individual block storage - cannot be cloned to ensure uniqueness
#[derive(Debug)]
pub struct LogicalBlockData<S: Storage, R: LogicalResources> {
    block_id: BlockId,
    block_set_id: usize,
    worker_id: WorkerID,
    resources: Arc<R>,
    storage_type: StorageType,
    storage: std::marker::PhantomData<S>,
    page_size: usize,
}

impl<S: Storage, R: LogicalResources> LogicalBlockData<S, R> {
    pub fn new(
        block_id: BlockId,
        block_set_id: usize,
        worker_id: WorkerID,
        resources: Arc<R>,
        storage_type: StorageType,
        page_size: usize,
    ) -> Self {
        Self {
            block_id,
            block_set_id,
            worker_id,
            resources,
            storage_type,
            storage: std::marker::PhantomData,
            page_size,
        }
    }

    pub fn resources(&self) -> Arc<R> {
        self.resources.clone()
    }
}

impl<S: Storage, R: LogicalResources> BlockDataExt<S> for LogicalBlockData<S, R> {
    fn block_id(&self) -> BlockId {
        self.block_id
    }

    fn block_set_id(&self) -> usize {
        self.block_set_id
    }

    fn worker_id(&self) -> WorkerID {
        self.worker_id
    }

    fn storage_type(&self) -> &StorageType {
        &self.storage_type
    }

    fn is_fully_contiguous(&self) -> bool {
        unimplemented!()
    }

    fn num_layers(&self) -> usize {
        unimplemented!()
    }

    /// Even though the block is logical, we still need to know this for the token block stuff.
    fn page_size(&self) -> usize {
        self.page_size
    }

    fn num_outer_dims(&self) -> usize {
        unimplemented!()
    }

    fn num_inner_dims(&self) -> usize {
        unimplemented!()
    }

    fn is_local(&self) -> Option<&dyn BlockDataViews<S>> {
        None
    }

    fn is_local_mut(&mut self) -> Option<&mut dyn BlockDataViews<S>> {
        None
    }
}
