// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod client;
pub mod handler;

use super::*;
use crate::tokens::SequenceHash;

use derive_getters::Dissolve;
use serde::{Deserialize, Serialize};

use dynamo_runtime::{
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn,
        async_trait, network::Ingress,
    },
    protocols::annotated::Annotated,
    traits::DistributedRuntimeProvider,
    utils::task::CriticalTaskExecutionHandle,
};

use crate::block_manager::pool::{BlockPoolStatus, ResetBlocksResponse};

pub type HandlerInput = SingleIn<ControlMessage>;
pub type HandlerOutput = ManyOut<Annotated<serde_json::Value>>;

/// Code that translates request/response messages to/from the block manager
#[derive(Clone)]
struct ControllerHandler<Locality: LocalityProvider, Metadata: BlockMetadata> {
    block_manager: KvBlockManager<Locality, Metadata>,
}

#[derive(Clone)]
pub struct Controller<Locality: LocalityProvider, Metadata: BlockMetadata> {
    _handler: Arc<ControllerHandler<Locality, Metadata>>,
}

impl<Locality: LocalityProvider, Metadata: BlockMetadata> Controller<Locality, Metadata> {
    pub async fn new(
        block_manager: KvBlockManager<Locality, Metadata>,
        component: dynamo_runtime::component::Component,
    ) -> anyhow::Result<Self> {
        let handler = ControllerHandler::new(block_manager.clone());
        let engine = Ingress::for_engine(handler.clone())?;

        let component_clone = component.clone();
        let reset_task = CriticalTaskExecutionHandle::new(
            |_cancel_token| async move {
                component_clone
                    .endpoint("controller")
                    .endpoint_builder()
                    .handler(engine)
                    .start()
                    .await
            },
            component.drt().primary_token(),
            "reset_cache_level",
        )?;

        reset_task.detach();

        Ok(Self { _handler: handler })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    Status(CacheLevel),
    ResetPool(CacheLevel),
    ResetBlocks(ResetRequest),
    ResetAll,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheLevel {
    G1,
    G2,
    G3,
}

#[derive(Debug, Clone, Serialize, Deserialize, Dissolve)]
pub struct ResetRequest {
    pub cache_level: CacheLevel,
    pub sequence_hashes: Vec<SequenceHash>,
}

pub type MaybeError = Option<String>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResetResponse {
    ResetAll(MaybeError),
    ResetPool(MaybeError),
    ResetBlocks(ResetBlocksResponse),
}

#[cfg(all(test, feature = "testing-etcd", feature = "testing-full"))]
mod tests {
    use crate::tokens::Tokens;

    use super::super::tests::create_reference_block_manager_with_counts;
    use super::*;

    #[tokio::test]
    async fn test_reset_cache_level() {
        dynamo_runtime::logging::init();

        let rt = dynamo_runtime::Runtime::from_current().unwrap();
        let drt = dynamo_runtime::DistributedRuntime::from_settings(rt)
            .await
            .unwrap();

        let worker_id = drt.connection_id();

        let block_manager = create_reference_block_manager_with_counts(8, 16, 0).await;

        let component = drt
            .namespace("test-kvbm")
            .unwrap()
            .component("kvbm")
            .unwrap();

        let _controller = Controller::new(block_manager.clone(), component.clone())
            .await
            .unwrap();

        let client = client::ControlClient::new(component.clone(), worker_id)
            .await
            .unwrap();

        let g1_status = client.status(CacheLevel::G1).await.unwrap();
        println!("G1 Status: {:?}", g1_status);

        assert_eq!(g1_status.active_blocks, 0);
        assert_eq!(g1_status.inactive_blocks, 0);
        let initial_block_count = g1_status.empty_blocks;

        match client.status(CacheLevel::G2).await.ok() {
            Some(status) => println!("G2 Status: {:?}", status),
            None => {
                println!("G2 Status: None");
            }
        }

        match client.status(CacheLevel::G3).await.ok() {
            Some(status) => println!("G3 Status: {:?}", status),
            None => {
                println!("G3 Status: None");
            }
        }

        let mut device_block = block_manager
            .device()
            .unwrap()
            .allocate_blocks(1)
            .await
            .unwrap();

        assert_eq!(device_block.len(), 1);
        let mut device_block = device_block.pop().unwrap();

        let tokens = Tokens::from(vec![1, 2, 3, 4]);
        let token_sequence = tokens.into_sequence(block_manager.block_size() as u32, Some(0));
        let token_block = token_sequence.blocks().first().unwrap();

        device_block.apply_token_block(token_block.clone()).unwrap();

        let mut immutable_device_blocks = block_manager
            .device()
            .unwrap()
            .register_blocks(vec![device_block])
            .await
            .unwrap();

        assert_eq!(immutable_device_blocks.len(), 1);
        let immutable_device_block = immutable_device_blocks.pop().unwrap();
        let sequence_hash = immutable_device_block.sequence_hash();

        let should_fail = client.reset_pool(CacheLevel::G1).await;
        assert!(should_fail.is_err());

        let one_allocated_status = client.status(CacheLevel::G1).await.unwrap();
        assert_eq!(one_allocated_status.active_blocks, 1);
        assert_eq!(one_allocated_status.inactive_blocks, 0);
        assert_eq!(one_allocated_status.empty_blocks, initial_block_count - 1);

        // try to reset the block by its sequence hash
        let reset_response = client
            .reset_blocks(CacheLevel::G1, vec![sequence_hash, 1337])
            .await
            .unwrap();

        assert_eq!(reset_response.reset_blocks.len(), 0);
        assert_eq!(reset_response.not_found.len(), 1);
        assert_eq!(reset_response.not_reset.len(), 1);

        println!("✅ Single allocation success");

        block_manager
            .device()
            .unwrap()
            .try_return_block(immutable_device_block.into())
            .await
            .unwrap();

        let after_drop_resposne = client.status(CacheLevel::G1).await.unwrap();
        assert_eq!(after_drop_resposne.active_blocks, 0);
        assert_eq!(after_drop_resposne.inactive_blocks, 1);
        assert_eq!(after_drop_resposne.empty_blocks, initial_block_count - 1);

        println!("✅ Single allocation drop success");

        // try to reset the block by its sequence hash
        let reset_response = client
            .reset_blocks(CacheLevel::G1, vec![sequence_hash, 1337])
            .await
            .unwrap();

        assert_eq!(reset_response.reset_blocks.len(), 1);
        assert_eq!(reset_response.not_found.len(), 1);
        assert_eq!(reset_response.not_reset.len(), 0);

        let g2_status = client.status(CacheLevel::G2).await.unwrap();
        assert_eq!(g2_status.active_blocks, 0);
        assert_eq!(g2_status.inactive_blocks, 1); // offloaded block

        client.reset_pool(CacheLevel::G2).await.unwrap();

        let g2_status = client.status(CacheLevel::G2).await.unwrap();
        assert_eq!(g2_status.active_blocks, 0);
        assert_eq!(g2_status.inactive_blocks, 0); // offloaded block
    }
}
