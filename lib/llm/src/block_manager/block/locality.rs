// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// todo: move this up one level to be on par with state and block
// locality is primarily focused on the locality of the block data; however,
// the choice of locality permeates the entire block manager.
//
// by moving up a level, it will make more sense use a kvbm level config object
// and kvbm state resources object to construct a locality aware block factory
//
// note: a block factory is also a block data factory
//
// factories can be turned into pools to implement the block pool and kvbm top-level
// interface; however, it can also be used to directly construct block data objects
// which can be used by leader-driven workers which do not have full block pools.

use super::*;
use crate::block_manager::block::transfer::{
    TransferContext, TransferError, WriteToStrategy, handle_local_transfer,
};
use crate::block_manager::storage::{self, nixl::NixlDescriptor};

use std::any::Any;
use tokio::sync::oneshot;

pub trait LocalityProvider: Send + Sync + 'static + std::fmt::Debug {
    // type Disk: BlockDataExt<DiskStorage>;
    // type Host: BlockDataExt<PinnedStorage>;
    // type Device: BlockDataExt<DeviceStorage>;

    type BlockData<S: Storage>: BlockDataExt<S>;

    fn handle_transfer<RB, WB>(
        _sources: &[RB],
        _targets: &mut [WB],
        _ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError>
    where
        RB: ReadableBlock + WriteToStrategy<WB> + storage::Local,
        <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
        <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
        RB: BlockDataProvider<Locality = Self>,
        WB: WritableBlock + BlockDataProviderMut<Locality = Self>;
}

/// Local locality provider for direct memory access
#[derive(Debug)]
pub struct Local;

impl LocalityProvider for Local {
    type BlockData<S: Storage> = BlockData<S>;

    fn handle_transfer<RB, WB>(
        sources: &[RB],
        targets: &mut [WB],
        ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError>
    where
        RB: ReadableBlock + WriteToStrategy<WB> + storage::Local,
        <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
        <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
        RB: BlockDataProvider<Locality = Self>,
        WB: WritableBlock + BlockDataProviderMut<Locality = Self>,
    {
        handle_local_transfer(sources, targets, ctx)
    }
}

pub use crate::block_manager::block::data::logical::{LogicalBlockData, LogicalResources};

/// General logical locality for future RPC-based transfers
#[derive(Debug)]
pub struct Logical<R: LogicalResources> {
    _resources: std::marker::PhantomData<R>,
}

impl<R: LogicalResources> Logical<R> {
    // TODO(jthomson04): Refactor these???
    fn load_resources<B: BlockDataProvider<Locality = Logical<R>>>(blocks: &[B]) -> Vec<Arc<R>> {
        blocks
            .iter()
            .map(|block| {
                let any_block = block.block_data() as &dyn Any;

                // TODO: Downcasting and unwrapping like this is atrocious...
                let logical_block = any_block
                    .downcast_ref::<LogicalBlockData<<B as StorageTypeProvider>::StorageType, R>>()
                    .unwrap();

                logical_block.resources()
            })
            .collect()
    }

    fn load_resources_mut<B: BlockDataProviderMut<Locality = Logical<R>>>(
        blocks: &mut [B],
    ) -> Vec<Arc<R>> {
        blocks
            .iter_mut()
            .map(|block| {
                let any_block = block.block_data_mut() as &mut dyn Any;

                let logical_block = any_block
                    .downcast_mut::<LogicalBlockData<<B as StorageTypeProvider>::StorageType, R>>()
                    .unwrap();

                logical_block.resources()
            })
            .collect()
    }
}

impl<R: LogicalResources> LocalityProvider for Logical<R> {
    type BlockData<S: Storage> = LogicalBlockData<S, R>;

    fn handle_transfer<RB, WB>(
        sources: &[RB],
        targets: &mut [WB],
        ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError>
    where
        RB: ReadableBlock + WriteToStrategy<WB> + storage::Local,
        <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
        <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
        RB: BlockDataProvider<Locality = Self>,
        WB: WritableBlock + BlockDataProviderMut<Locality = Self>,
    {
        // Check for empty slices and length mismatch early
        if sources.is_empty() && targets.is_empty() {
            tracing::warn!(
                "Logical::handle_transfer called with both sources and targets empty, skipping transfer"
            );
            let (tx, rx) = oneshot::channel();
            tx.send(()).unwrap();
            return Ok(rx);
        }

        if sources.len() != targets.len() {
            return Err(TransferError::CountMismatch(sources.len(), targets.len()));
        }

        let source_resources = Self::load_resources(sources);
        let target_resources = Self::load_resources_mut(targets);

        let all_resources = source_resources
            .into_iter()
            .chain(target_resources)
            .collect::<Vec<_>>();

        // For now, assert that all resources between the source and target are the same
        if !all_resources
            .iter()
            .all(|r| Arc::ptr_eq(r, &all_resources[0]))
        {
            return Err(anyhow::anyhow!("Resources used in a transfer must be the same!").into());
        }

        let common_resource = all_resources[0].clone();

        common_resource.handle_transfer(sources, targets, ctx)
    }
}
