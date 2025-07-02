// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    handle_local_transfer, TransferContext, TransferError, WriteToStrategy,
};
use crate::block_manager::storage::{self, nixl::NixlDescriptor};
use crate::block_manager::{DeviceStorage, DiskStorage, PinnedStorage};

use std::any::Any;
use tokio::sync::oneshot;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalityType {
    Local(StorageType),
    Remote,
    Logical(ParallelismType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelismType {
    WorkerSharded(usize),
}

pub trait LocalityProvider: Send + Sync + 'static + std::fmt::Debug {
    type Disk: BlockDataExt<DiskStorage>;
    type Host: BlockDataExt<PinnedStorage>;
    type Device: BlockDataExt<DeviceStorage>;

    type BlockData<S: Storage>: BlockDataExt<S>;

    fn handle_transfer<RB, WB>(
        _sources: &[RB],
        _targets: &mut [WB],
        _notify: bool,
        _ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError>
    where
        RB: ReadableBlock + WriteToStrategy<WB> + storage::Local,
        <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
        <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
        RB: BlockDataProvider<Locality = Self>,
        WB: WritableBlock + BlockDataProviderMut<Locality = Self>,
    {
        panic!("Transfers are not supported for this locality provider");
    }
}

/// Transfer mechanism describes how transfers should be performed between locality types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMechanism {
    /// Direct memory access (Local-to-Local transfers using memcpy, CUDA, etc.)
    DirectMemory,

    /// Remote RPC mechanism for cross-worker transfers (Logical localities)
    RemoteRpc,

    /// Unsupported transfer combination
    Unsupported,
}

/// Local locality provider for direct memory access
#[derive(Debug)]
pub struct Local;

impl LocalityProvider for Local {
    type Disk = Self::BlockData<DiskStorage>;
    type Host = Self::BlockData<PinnedStorage>;
    type Device = Self::BlockData<DeviceStorage>;

    type BlockData<S: Storage> = BlockData<S>;

    fn handle_transfer<RB, WB>(
        sources: &[RB],
        targets: &mut [WB],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError>
    where
        RB: ReadableBlock + WriteToStrategy<WB> + storage::Local,
        <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
        <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
        RB: BlockDataProvider<Locality = Self>,
        WB: WritableBlock + BlockDataProviderMut<Locality = Self>,
    {
        handle_local_transfer(sources, targets, notify, ctx)
    }
}

// /// Mock logical locality for testing - computes transfer sizes without moving data
// #[derive(Debug)]
// pub struct MockLogical;

// impl LocalityProvider for MockLogical {
//     type Disk = Self::BlockData<DiskStorage>;
//     type Host = Self::BlockData<PinnedStorage>;
//     type Device = Self::BlockData<DeviceStorage>;

//     type BlockData<S: Storage> = MockLogicalBlockData<S>;

//     fn is_transfer_compatible_with<Other: LocalityProvider>() -> bool {
//         // MockLogical is only compatible with MockLogical
//         std::any::TypeId::of::<Other>() == std::any::TypeId::of::<MockLogical>()
//     }

//     fn transfer_mechanism_to<Other: LocalityProvider>() -> TransferMechanism {
//         if Self::is_transfer_compatible_with::<Other>() {
//             TransferMechanism::MockCounting
//         } else {
//             TransferMechanism::Unsupported
//         }
//     }
// }

pub use crate::block_manager::block::data::logical::{LogicalBlockData, LogicalResources};

/// General logical locality for future RPC-based transfers
#[derive(Debug)]
pub struct Logical<R: LogicalResources> {
    _resources: std::marker::PhantomData<R>,
}

impl<R: LogicalResources> Logical<R> {
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
    type Disk = LogicalBlockData<DiskStorage, R>;
    type Host = LogicalBlockData<PinnedStorage, R>;
    type Device = LogicalBlockData<DeviceStorage, R>;

    type BlockData<S: Storage> = LogicalBlockData<S, R>;

    fn handle_transfer<RB, WB>(
        sources: &[RB],
        targets: &mut [WB],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError>
    where
        RB: ReadableBlock + WriteToStrategy<WB> + storage::Local,
        <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
        <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
        RB: BlockDataProvider<Locality = Self>,
        WB: WritableBlock + BlockDataProviderMut<Locality = Self>,
    {
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

        common_resource.handle_transfer(sources, targets, notify, ctx)
    }
}

// pub mod nixl {
//     use super::*;
//     use crate::block_manager::storage::{nixl::NixlStorage, StorageType};

//     #[derive(Debug)]
//     pub struct ActiveMessageClient {}

//     #[derive(Debug)]
//     pub struct WorkerReplicated;

//     #[derive(Debug)]
//     pub struct ReplicatedBlockDataParallel<S: Storage> {
//         layouts: Vec<Arc<dyn BlockLayout<StorageType = NixlStorage>>>,
//         am_client: Vec<ActiveMessageClient>,
//         storage: std::marker::PhantomData<S>,

//         // extracted from the first layout and validated for continuity
//         layout_config: LayoutConfig,
//         layout_type: LayoutType,
//         storage_type: StorageType,
//     }

//     impl<S: Storage> BlockDataExt<S> for ReplicatedBlockDataParallel<S> {
//         fn block_id(&self) -> BlockId {
//             self.block_id
//         }

//         fn is_fully_contiguous(&self) -> bool {
//             unimplemented!()
//         }

//         fn num_layers(&self) -> usize {
//             unimplemented!()
//         }

//         fn num_outer_dims(&self) -> usize {
//             unimplemented!()
//         }

//         fn layer_view(
//             &self,
//             layer_idx: usize,
//             outer_idx: usize,
//         ) -> BlockResult<view::LayerView<S>> {
//             unimplemented!()
//         }

//         fn layer_view_mut(
//             &mut self,
//             layer_idx: usize,
//             outer_idx: usize,
//         ) -> BlockResult<view::LayerViewMut<S>> {
//             unimplemented!()
//         }

//         fn block_view(&self) -> BlockResult<view::BlockView<S>> {
//             unimplemented!()
//         }

//         fn block_view_mut(&mut self) -> BlockResult<view::BlockViewMut<S>> {
//             unimplemented!()
//         }
//     }

//     impl Parallelism for WorkerReplicated {
//         type Output<S: Storage> = ReplicatedBlockDataParallel<S>;
//     }

//     impl<S: Storage> ReplicatedBlockDataParallel<S> {
//         pub fn new(
//             layouts: Vec<SerializedNixlBlockLayout>,
//             am_client: Vec<ActiveMessageClient>,
//         ) -> Result<Self, BlockError> {
//             // num of am_clients should be equal to the number of layouts
//             // there must be at least one am_client

//             assert!(layouts.len() > 0);
//             assert!(am_client.len() > 0);

//             if layouts.len() != am_client.len() {
//                 return Err(BlockError::MisconfiguredBlockDataParallelism(
//                     "Number of layouts must be equal to the number of am_clients".to_string(),
//                 ));
//             }

//             // deserialize the layouts
//             let layouts = layouts
//                 .into_iter()
//                 .map(|layout| layout.deserialize())
//                 .collect::<Result<Vec<_>, _>>()?;

//             // extract and validate for continuity
//             let storage_type = layouts[0].storage_type().clone();
//             let layout_config = layouts[0].config().clone();
//             let layout_type = layouts[0].layout_type();

//             for layout in layouts.iter().skip(1) {
//                 if layout.storage_type() != &storage_type {
//                     return Err(BlockError::MisconfiguredBlockDataParallelism(
//                         "All layouts must have the same storage type".to_string(),
//                     ));
//                 }

//                 if layout.config() != &layout_config {
//                     return Err(BlockError::MisconfiguredBlockDataParallelism(
//                         "All layouts must have the same config".to_string(),
//                     ));
//                 }

//                 if layout.layout_type() != layout_type {
//                     return Err(BlockError::MisconfiguredBlockDataParallelism(
//                         "All layouts must have the same layout type".to_string(),
//                     ));
//                 }
//             }

//             Ok(Self {
//                 layouts,
//                 layout_config,
//                 layout_type,
//                 storage_type: storage_type.clone(),
//                 am_client,
//                 storage: std::marker::PhantomData,
//             })
//         }
//     }

//     // impl<S: Storage> BlockDataStorage for ReplicatedBlockDataParallel<S> {
//     //     type StorageType = S;
//     //     type BlockDataType<T: Storage> = Self;

//     //     fn block_data(&self) -> &Self::BlockDataType<Self::StorageType> {
//     //         &self
//     //     }

//     //     fn block_data_mut(&mut self) -> &mut Self::BlockDataType<Self::StorageType> {
//     //         self
//     //     }
//     // }

//     // impl<S: Storage> BlockLayoutConfig for ReplicatedBlockDataParallel<S> {
//     //     fn layout_type(&self) -> LayoutType {
//     //         self.layout_type
//     //     }

//     //     fn num_blocks(&self) -> usize {
//     //         self.layout_config.num_blocks
//     //     }

//     //     fn num_layers(&self) -> usize {
//     //         self.layout_config.num_layers
//     //     }

//     //     fn outer_dim(&self) -> usize {
//     //         self.layout_config.outer_dim
//     //     }

//     //     fn page_size(&self) -> usize {
//     //         self.layout_config.page_size
//     //     }

//     //     fn inner_dim(&self) -> usize {
//     //         self.layout_config.inner_dim
//     //     }
//     // }
// }
