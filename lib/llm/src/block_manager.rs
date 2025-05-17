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

//! Block Manager for LLM KV Cache
//!
//! This module provides functionality for managing KV blocks in LLM attention
//! mechanisms. It handles storage allocation, block management, and safe access
//! patterns for both system memory and remote (NIXL) storage.

mod config;
mod state;

pub mod block;
pub mod events;
pub mod layout;
pub mod offload;
pub mod pool;
pub mod storage;

pub use crate::common::dtype::DType;
pub use block::{
    nixl::{
        AsBlockDescriptorSet, BlockDescriptorList, IsImmutable, IsMutable, MutabilityKind,
        RemoteBlock,
    },
    transfer::{BlockTransferEngineV1, TransferRequestPut},
    BasicMetadata, BlockMetadata, Blocks, ImmutableBlock,
};
pub use config::*;
pub use layout::{nixl::NixlLayout, LayoutConfig, LayoutConfigBuilder, LayoutError, LayoutType};
pub use pool::BlockPool;
use pool::BlockPoolError;
pub use storage::{
    nixl::NixlRegisterableStorage, DeviceStorage, PinnedStorage, Storage, StorageAllocator,
};
pub use tokio_util::sync::CancellationToken;

use anyhow::{Context, Result};
use block::nixl::{BlockMutability, NixlBlockSet, RemoteBlocks, SerializedNixlBlockSet};
use derive_builder::Builder;
use nixl_sys::Agent as NixlAgent;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use storage::nixl::MemType;
use validator::Validate;

pub type WorkerID = u64;

pub type ReferenceBlockManager = KvBlockManager<BasicMetadata>;

/// Represents the different cache levels for KV blocks
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum CacheLevel {
    /// Represents KV blocks in GPU memory
    G1,

    /// Represents KV blocks in CPU memory
    G2,

    /// Represents KV blocks in Local NVMe storage
    G3,

    /// Represents KV blocks in Remote NVMe storage
    G4,
}

// When we construct the pool:
// 1. instantiate the runtime,
// 2. build layout::LayoutConfigs for each of the requested storage types
// 3. register the layouts with the NIXL agent if enabled
// 4. construct a Blocks object for each layout providing a unique block_set_idx
//    for each layout type.
// 5. initialize the pools for each set of blocks
pub struct KvBlockManager<Metadata: BlockMetadata> {
    state: Arc<state::KvBlockManagerState<Metadata>>,
    cancellation_token: CancellationToken,
}

impl<Metadata: BlockMetadata> KvBlockManager<Metadata> {
    /// Create a new [KvBlockManager]
    ///
    /// The returned object is a frontend to the [KvBlockManager] which owns the cancellation
    /// tokens. When this object gets drop, the cancellation token will be cancelled and begin
    /// the gracefully shutdown of the block managers internal state.
    pub fn new(config: KvBlockManagerConfig) -> Result<Self> {
        let mut config = config;

        // The frontend of the KvBlockManager will take ownership of the cancellation token
        // and will be responsible for cancelling the task when the KvBlockManager is dropped
        let cancellation_token = config.runtime.cancellation_token.clone();

        // The internal state will use a child token of the original token
        config.runtime.cancellation_token = cancellation_token.child_token();

        // Create the internal state
        let state = state::KvBlockManagerState::new(config)?;

        Ok(Self {
            state,
            cancellation_token,
        })
    }

    /// Exports the local blockset configuration as a serialized object.
    pub fn export_local_blockset(&self) -> Result<SerializedNixlBlockSet> {
        self.state.export_local_blockset()
    }

    /// Imports a remote blockset configuration from a serialized object.
    pub fn import_remote_blockset(
        &self,
        serialized_blockset: SerializedNixlBlockSet,
    ) -> Result<()> {
        self.state.import_remote_blockset(serialized_blockset)
    }

    /// Get a [`Vec<RemoteBlock<IsImmutable>>`] from a [`BlockDescriptorList`]
    pub fn get_remote_blocks_immutable(
        &self,
        bds: &BlockDescriptorList,
    ) -> Result<Vec<RemoteBlock<IsImmutable>>> {
        self.state.get_remote_blocks_immutable(bds)
    }

    /// Get a [`Vec<RemoteBlock<IsMutable>>`] from a [`BlockDescriptorList`]
    pub fn get_remote_blocks_mutable(
        &self,
        bds: &BlockDescriptorList,
    ) -> Result<Vec<RemoteBlock<IsMutable>>> {
        self.state.get_remote_blocks_mutable(bds)
    }

    /// Get a reference to the host block pool
    pub fn host(&self) -> Option<&BlockPool<PinnedStorage, Metadata>> {
        self.state.host()
    }

    /// Get a reference to the device block pool
    pub fn device(&self) -> Option<&BlockPool<DeviceStorage, Metadata>> {
        self.state.device()
    }

    pub async fn onboard_blocks(
        &self,
        blocks: Vec<ImmutableBlock<PinnedStorage, Metadata>>,
    ) -> std::result::Result<Vec<ImmutableBlock<DeviceStorage, Metadata>>, BlockPoolError> {
        self.state.onboard_blocks(blocks).await
    }

    /// Get the worker ID
    pub fn worker_id(&self) -> WorkerID {
        self.state.worker_id()
    }
}

impl<Metadata: BlockMetadata> Drop for KvBlockManager<Metadata> {
    fn drop(&mut self) {
        self.cancellation_token.cancel();
    }
}

#[cfg(all(test, feature = "testing-full"))]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicU64, Ordering};

    // Atomic Counter for Worker ID
    static WORKER_ID: AtomicU64 = AtomicU64::new(1337);

    fn create_reference_block_manager() -> ReferenceBlockManager {
        let worker_id = WORKER_ID.fetch_add(1, Ordering::SeqCst);
        let config = KvBlockManagerConfig::builder()
            .runtime(
                KvManagerRuntimeConfig::builder()
                    .worker_id(worker_id)
                    .build()
                    .unwrap(),
            )
            .model(
                KvManagerModelConfig::builder()
                    .num_layers(3)
                    .page_size(4)
                    .inner_dim(16)
                    .build()
                    .unwrap(),
            )
            .host_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(16)
                    .allocator(storage::PinnedAllocator::default())
                    .build()
                    .unwrap(),
            )
            .device_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(8)
                    .allocator(storage::DeviceAllocator::new(0).unwrap())
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();

        ReferenceBlockManager::new(config).unwrap()
    }

    #[tokio::test]
    async fn test_reference_block_manager_inherited_async_runtime() {
        dynamo_runtime::logging::init();
        let _block_manager = create_reference_block_manager();
    }

    #[test]
    fn test_reference_block_manager_blocking() {
        dynamo_runtime::logging::init();
        let _block_manager = create_reference_block_manager();
    }

    // This tests mimics the behavior of two unique kvbm workers exchanging blocksets
    // Each KvBlockManager is a unique worker in this test, each has its resources including
    // it's own worker_ids, nixl_agent, and block pools.
    //
    // This test is meant to mimic the behavior of the basic nixl integration test found here:
    // https://github.com/ai-dynamo/nixl/blob/main/src/bindings/rust/src/tests.rs
    #[tokio::test]
    async fn test_reference_block_managers() {
        dynamo_runtime::logging::init();

        // create two block managers - mimics two unique dynamo workers
        let kvbm_0 = create_reference_block_manager();
        let kvbm_1 = create_reference_block_manager();

        assert_ne!(kvbm_0.worker_id(), kvbm_1.worker_id());

        // in dynamo, we would exchange the blocksets via the discovery plane
        let blockset_0 = kvbm_0.export_local_blockset().unwrap();
        let blockset_1 = kvbm_1.export_local_blockset().unwrap();

        // in dynamo, we would be watching the discovery plane for remote blocksets
        kvbm_0.import_remote_blockset(blockset_1).unwrap();
        kvbm_1.import_remote_blockset(blockset_0).unwrap();

        // Worker 0
        // Allocate 4 mutable blocks on the host
        let blocks_0 = kvbm_0.host().unwrap().allocate_blocks(4).await.unwrap();

        // Create a BlockDescriptorList for the mutable blocks
        // let blockset_0 = BlockDescriptorList::from_mutable_blocks(&blocks_0).unwrap();
        let blockset_0 = blocks_0.as_block_descriptor_set().unwrap();

        // Worker 1
        // Create a RemoteBlock list from blockset_0
        let _blocks_1 = kvbm_1.host().unwrap().allocate_blocks(4).await.unwrap();
        let mut _remote_blocks_0 = kvbm_1.get_remote_blocks_mutable(&blockset_0).unwrap();

        // TODO(#967) - Enable with TransferEngine

        // // Create a TransferRequestPut for the mutable blocks
        // let transfer_request = TransferRequestPut::new(&blocks_0, &mut remote_blocks_0).unwrap();

        // // Validate blocks - this could be an expensive operation
        // // TODO: Create an ENV trigger debug flag which will call this on every transfer request
        // // In this case, we expect an error because we have overlapping blocks as we are sending to/from the same blocks
        // // because we are using the wrong target (artifact of the test setup allowing variable to cross what woudl be
        // // worker boundaries)
        // assert!(transfer_request.validate_blocks().is_err());

        // // This is proper request - PUT from worker 1 (local) to worker 0 (remote)
        // let transfer_request = TransferRequestPut::new(&blocks_1, &mut remote_blocks_0).unwrap();
        // assert!(transfer_request.validate_blocks().is_ok());

        // // Execute the transfer request
        // transfer_request.execute().unwrap();

        // let mut put_request = PutRequestBuilder::<_, _>::builder();

        // put_request.from(&blocks_1).to(&mut remote_blocks_0);

        // // Create a Put request direct between two local blocks
        // // split the blocks into two vecs each with 2 blocks
        // let mut blocks_1 = blocks_1;

        // let slice_0 = blocks_1.split_off(2);
        // let mut slice_1 = blocks_1;

        // let transfer_request = TransferRequestPut::new(&slice_0, &mut slice_1).unwrap();
        // assert!(transfer_request.validate_blocks().is_ok());

        // // Execute the transfer request
        // transfer_request.execute().unwrap();
    }
}
