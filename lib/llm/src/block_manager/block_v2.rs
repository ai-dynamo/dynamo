// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nixl_sys::NixlDescriptor;

use crate::block_manager::{
    block::{
        private::PrivateToken,
        transfer::{TransferContext, TransferError, TransferStrategy, WriteTo, WriteToStrategy},
        BlockData, BlockDataExt, BlockDataProvider, BlockIdentifier,
    },
    layout::BlockLayout,
    storage::{nixl::NixlStorage, Local, Remote, Storage},
    DeviceStorage, NixlRegisterableStorage, PinnedStorage, WorkerID,
};

use super::block::{BlockId, BlockMetadata, BlockSetId, BlockState};

use std::sync::Arc;

pub trait BlockDataLocality: Send + Sync {
    fn block_id(&self) -> BlockId;
}

pub trait BlockDataLocalityProvider: Send + Sync {
    type Locality: BlockDataLocality;

    fn block_data_locality(&self, _token: PrivateToken) -> &Self::Locality;
}

pub trait BlockDataLocalityProviderMut: BlockDataLocalityProvider {
    fn block_data_locality_mut(&mut self, _token: PrivateToken) -> &mut Self::Locality;
}

pub struct LocalData<S: Storage + Local + NixlDescriptor>(BlockData<S>);

impl<S: Storage + Local + NixlDescriptor> BlockDataLocality for LocalData<S> {
    fn block_id(&self) -> BlockId {
        self.0.block_id()
    }
}

impl<S: Storage + Local + NixlDescriptor> BlockDataProvider for LocalData<S> {
    type StorageType = S;

    fn block_data(&self, _token: PrivateToken) -> &BlockData<S> {
        &self.0
    }
}

pub struct RemoteData<S: Storage + Remote + NixlDescriptor>(BlockData<S>);

impl<S: Storage + Remote + NixlDescriptor> BlockDataLocality for RemoteData<S> {
    fn block_id(&self) -> BlockId {
        self.0.block_id()
    }
}

impl<S: Storage + Remote + NixlDescriptor> BlockDataProvider for RemoteData<S> {
    type StorageType = S;

    fn block_data(&self, _token: PrivateToken) -> &BlockData<S> {
        &self.0
    }
}

pub trait StorageKind: Send + Sync {}

pub struct SystemMemory;
impl StorageKind for SystemMemory {}

pub struct PinnedMemory;
impl StorageKind for PinnedMemory {}

pub struct DeviceMemory;
impl StorageKind for DeviceMemory {}

pub trait ParallelKind: Send + Sync {}

pub struct ReplicatedWorkers {
    workers: Vec<RemoteData<NixlStorage>>,
}

impl ParallelKind for ReplicatedWorkers {}

pub struct LogicalData<S: StorageKind, P: ParallelKind> {
    block_id: BlockId,
    paralleism: P,
}

impl<S: StorageKind, P: ParallelKind> BlockDataLocality for LogicalData<S, P> {
    fn block_id(&self) -> BlockId {
        self.block_id
    }
}

// impl<S: StorageKind, D: StorageKind, P: ParallelKind> WriteTo<LogicalData<S, P>>
//     for Vec<LogicalData<D, P>>
// {
//     fn write_to(
//         &self,
//         dst: &mut Vec<WB>,
//         notify: bool,
//         ctx: Arc<TransferContext>,
//     ) -> Result<Option<oneshot::Receiver<()>>, TransferError> {
//         let (tx, rx) = oneshot::channel();

//         Ok(Some(rx))
//     }
// }
