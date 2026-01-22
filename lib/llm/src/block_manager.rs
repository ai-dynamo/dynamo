// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block Manager for LLM KV Cache
//!
//! This module provides functionality for managing KV blocks in LLM attention
//! mechanisms. It handles storage allocation, block management, and safe access
//! patterns for both system memory and remote (NIXL) storage.

pub mod config;
mod state;

pub mod block;
pub mod connector;
pub mod distributed;
pub mod events;
pub mod kv_consolidator;
pub mod layout;
pub mod metrics_kvbm;
pub mod numa_allocator;
pub mod offload;
pub mod pool;
pub mod storage;
pub mod v2;

// dynamo rt integration
pub mod controller;

pub use crate::common::dtype::DType;
pub use block::{
    BasicMetadata, BlockMetadata, Blocks, ImmutableBlock, MutableBlock,
    locality::{self, LocalityProvider, LogicalResources},
    nixl::{BlockDescriptorList, IsImmutable, IsMutable, MutabilityKind, RemoteBlock},
};
pub use config::*;

pub use layout::{LayoutConfig, LayoutConfigBuilder, LayoutError, LayoutType, nixl::NixlLayout};
pub use offload::{filter::OffloadFilter, request::BlockResult};
pub use pool::{BlockPool, ManagedBlockPool};
pub use storage::{
    DeviceStorage, DiskStorage, PinnedStorage, Storage, StorageAllocator,
    nixl::NixlRegisterableStorage,
};
pub use tokio_util::sync::CancellationToken;

use anyhow::{Context, Result};
use block::nixl::{BlockMutability, NixlBlockSet, RemoteBlocks, SerializedNixlBlockSet};
use derive_builder::Builder;
use nixl_sys::Agent as NixlAgent;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use storage::nixl::MemType;
use tokio::sync::oneshot;
use validator::Validate;

pub type WorkerID = u64;

pub type ReferenceBlockManager = KvBlockManager<locality::Local, BasicMetadata>;

/// Represents the different cache levels for KV blocks
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
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

/// Type of channel used to reset the block manager to a specific cache level
pub type BlockResetChannel = tokio::sync::broadcast::Receiver<CacheLevel>;

#[derive(Debug)]
struct CancelOnLastDrop {
    cancellation_token: CancellationToken,
}

impl Drop for CancelOnLastDrop {
    fn drop(&mut self) {
        self.cancellation_token.cancel();
    }
}

// When we construct the pool:
// 1. instantiate the runtime,
// 2. build layout::LayoutConfigs for each of the requested storage types
// 3. register the layouts with the NIXL agent if enabled
// 4. construct a Blocks object for each layout providing a unique block_set_idx
//    for each layout type.
// 5. initialize the pools for each set of blocks
#[derive(Debug)]
pub struct KvBlockManager<Locality: LocalityProvider, Metadata: BlockMetadata> {
    state: Arc<state::KvBlockManagerState<Locality, Metadata>>,
    _cancellation_token: Arc<CancelOnLastDrop>,
    block_size: usize,
}

impl<Locality: LocalityProvider, Metadata: BlockMetadata> Clone
    for KvBlockManager<Locality, Metadata>
{
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            _cancellation_token: self._cancellation_token.clone(),
            block_size: self.block_size,
        }
    }
}

impl<Locality: LocalityProvider, Metadata: BlockMetadata> KvBlockManager<Locality, Metadata> {
    /// Get the block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get a reference to the disk block pool
    pub fn disk(&self) -> Option<&dyn BlockPool<DiskStorage, Locality, Metadata>> {
        self.state.disk()
    }

    /// Get a reference to the host block pool
    pub fn host(&self) -> Option<&dyn BlockPool<PinnedStorage, Locality, Metadata>> {
        self.state.host()
    }

    /// Get a reference to the device block pool
    pub fn device(&self) -> Option<&dyn BlockPool<DeviceStorage, Locality, Metadata>> {
        self.state.device()
    }

    /// Get the worker ID
    pub fn worker_id(&self) -> WorkerID {
        self.state.worker_id()
    }

    /// Onboard a set of blocks to the device pool
    pub fn onboard_blocks<S: Storage>(
        &self,
        blocks: Vec<ImmutableBlock<S, Locality, Metadata>>,
        targets: Option<Vec<MutableBlock<DeviceStorage, Locality, Metadata>>>,
    ) -> oneshot::Receiver<BlockResult<DeviceStorage, Locality, Metadata>> {
        self.state.onboard_blocks(blocks, targets)
    }
}

fn build_cancel_token(config: &mut KvBlockManagerConfig) -> Arc<CancelOnLastDrop> {
    // The frontend of the KvBlockManager will take ownership of the cancellation token
    // and will be responsible for cancelling the task when the KvBlockManager is dropped
    let cancellation_token = config.runtime.cancellation_token.clone();

    // The internal state will use a child token of the original token
    config.runtime.cancellation_token = cancellation_token.child_token();

    Arc::new(CancelOnLastDrop { cancellation_token })
}

impl<Metadata: BlockMetadata> KvBlockManager<locality::Local, Metadata> {
    /// Create a new [KvBlockManager]
    ///
    /// The returned object is a frontend to the [KvBlockManager] which owns the cancellation
    /// tokens. When this object gets drop, the cancellation token will be cancelled and begin
    /// the gracefully shutdown of the block managers internal state.
    pub async fn new(mut config: KvBlockManagerConfig) -> Result<Self> {
        let _cancellation_token = build_cancel_token(&mut config);

        let block_size = config.model.page_size;

        // Create the internal state
        let state = state::KvBlockManagerState::<locality::Local, Metadata>::new(config).await?;

        Ok(Self {
            state,
            _cancellation_token,
            block_size,
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
}

impl<R: LogicalResources, Metadata: BlockMetadata> KvBlockManager<locality::Logical<R>, Metadata> {
    pub async fn new(mut config: KvBlockManagerConfig, logical_resources: R) -> Result<Self> {
        let block_size = config.model.page_size;

        let _cancellation_token = build_cancel_token(&mut config);

        let state = state::KvBlockManagerState::<locality::Logical<R>, Metadata>::new(
            config,
            logical_resources,
        )
        .await?;

        Ok(Self {
            state,
            _cancellation_token,
            block_size,
        })
    }
}

#[cfg(all(test, feature = "testing-full"))]
mod tests {

    use super::*;

    use crate::tokens::Tokens;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Atomic Counter for Worker ID
    static WORKER_ID: AtomicU64 = AtomicU64::new(1337);

    pub fn create_reference_block_manager_config_with_counts(
        device: usize,
        host: usize,
        disk: usize,
    ) -> KvBlockManagerConfig {
        let worker_id = WORKER_ID.fetch_add(1, Ordering::SeqCst);

        // Check if we're already in a Tokio runtime context
        let async_runtime = if tokio::runtime::Handle::try_current().is_ok() {
            None // If we're already in a runtime, don't create a new one
        } else {
            // Only create a new runtime if not already in one
            Some(Arc::new(tokio::runtime::Runtime::new().unwrap()))
        };

        let builder = KvBlockManagerConfig::builder()
            .runtime(
                KvManagerRuntimeConfig::builder()
                    .worker_id(worker_id)
                    .enable_nixl()
                    .async_runtime(async_runtime)
                    .build()
                    .unwrap(),
            )
            .model(
                KvManagerModelConfig::builder()
                    .num_layers(3)
                    .outer_dim(2)
                    .page_size(4)
                    .inner_dim(16)
                    .build()
                    .unwrap(),
            );

        let builder = if disk > 0 {
            builder.disk_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(disk)
                    .allocator(storage::DiskAllocator)
                    .build()
                    .unwrap(),
            )
        } else {
            builder
        };

        let builder = if host > 0 {
            builder.host_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(host)
                    .allocator(storage::PinnedAllocator::default())
                    .build()
                    .unwrap(),
            )
        } else {
            builder
        };

        let builder = if device > 0 {
            builder.device_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(device)
                    .allocator(storage::DeviceAllocator::new(0).unwrap())
                    .build()
                    .unwrap(),
            )
        } else {
            builder
        };

        builder.build().unwrap()
    }

    pub fn create_reference_block_manager_config() -> KvBlockManagerConfig {
        create_reference_block_manager_config_with_counts(8, 16, 16)
    }

    pub async fn create_reference_block_manager() -> ReferenceBlockManager {
        ReferenceBlockManager::new(create_reference_block_manager_config())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_reference_block_manager_inherited_async_runtime() {
        dynamo_runtime::logging::init();
        let _block_manager = create_reference_block_manager().await;
    }

    // This tests mimics the behavior of two unique kvbm workers exchanging blocksets
    // Each KvBlockManager is a unique worker in this test, each has its resources including
    // it's own worker_ids, nixl_agent, and block pools.
    //
    // This test is meant to mimic the behavior of the basic nixl integration test found here:
    // https://github.com/ai-dynamo/nixl/blob/main/src/bindings/rust/src/tests.rs
    // TODO: This test doesn't work because NIXL doesn't support partial metadata in the rust bindings.
    #[ignore]
    #[tokio::test]
    async fn test_reference_block_managers() {
        dynamo_runtime::logging::init();

        // create two block managers - mimics two unique dynamo workers
        let kvbm_0 = create_reference_block_manager().await;
        let kvbm_1 = create_reference_block_manager().await;

        assert_ne!(kvbm_0.worker_id(), kvbm_1.worker_id());

        // in dynamo, we would exchange the blocksets via the discovery plane
        let blockset_0 = kvbm_0.export_local_blockset().unwrap();
        let blockset_1 = kvbm_1.export_local_blockset().unwrap();

        // in dynamo, we would be watching the discovery plane for remote blocksets
        kvbm_0.import_remote_blockset(blockset_1).unwrap();
        kvbm_1.import_remote_blockset(blockset_0).unwrap();

        // Worker 0
        // Allocate 4 mutable blocks on the host
        let _blocks_0 = kvbm_0.host().unwrap().allocate_blocks(4).await.unwrap();

        // // Create a BlockDescriptorList for the mutable blocks
        // // let blockset_0 = BlockDescriptorList::from_mutable_blocks(&blocks_0).unwrap();
        // let blockset_0 = blocks_0.as_block_descriptor_set().unwrap();

        // // Worker 1
        // // Create a RemoteBlock list from blockset_0
        // let _blocks_1 = kvbm_1.host().unwrap().allocate_blocks(4).await.unwrap();
        // let mut _remote_blocks_0 = kvbm_1.get_remote_blocks_mutable(&blockset_0).unwrap();

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

    #[tokio::test]
    async fn test_offload() -> Result<()> {
        dynamo_runtime::logging::init();

        let block_manager = create_reference_block_manager().await;

        let device = block_manager.device().unwrap();

        let tokens = Tokens::from(vec![1, 2, 3, 4]);
        let token_sequence = tokens.into_sequence(4, Some(0));
        let token_block = token_sequence.blocks().first().unwrap();

        let mut device_block = device.allocate_blocks(1).await?.into_iter().next().unwrap();
        device_block.apply_token_block(token_block.clone())?;

        let immutable_device_blocks = device.register_blocks(vec![device_block]).await.unwrap();
        assert_eq!(immutable_device_blocks.len(), 1);

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // It should now be on host and disk.
        let host_blocks = block_manager
            .host()
            .unwrap()
            .match_sequence_hashes(vec![immutable_device_blocks[0].sequence_hash()].as_slice())
            .await
            .unwrap();
        assert_eq!(host_blocks.len(), 1);

        let disk_blocks = block_manager
            .disk()
            .unwrap()
            .match_sequence_hashes(vec![immutable_device_blocks[0].sequence_hash()].as_slice())
            .await
            .unwrap();
        assert_eq!(disk_blocks.len(), 1);

        Ok(())
    }
}
