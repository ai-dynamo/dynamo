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

//! Block Manager Dynamo Integration Tests
//!
//! This module both the integration components in the `llm_kvbm` module
//! and the tests for the `llm_kvbm` module.
//!
//! The intent is to move [llm_kvbm] to a separate crate in the future.

pub mod llm_kvbm {
    // alias for the kvbm module to make the refactor to standalone crate easier
    use dynamo_llm::block_manager as kvbm;

    // kvbm specific imports
    use kvbm::{block::registry::RegistrationHandle, events::*};

    // std imports
    use std::collections::VecDeque;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use async_trait::async_trait;
    use serde::Serialize;

    use anyhow::{Result};
    use derive_builder::Builder;
    use derive_getters::Dissolve;
    use dynamo_runtime::DistributedRuntime;
    use std::sync::atomic::{AtomicU64, Ordering};
    use dynamo_llm::tokens::{SequenceHash, BlockHash};
    use dynamo_runtime::component::Namespace;
    use dynamo_runtime::prelude::DistributedRuntimeProvider;
    use tokio::sync::mpsc;
    pub const KV_EVENT_SUBJECT: &str = "kv_events";
    use dynamo_runtime::traits::events::EventPublisher;
    use kvbm::events::EventManager;
    use dynamo_llm::kv_router::{
        protocols::{
            ExternalSequenceBlockHash, LocalBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData,
        },
        indexer::RouterEvent,
    };

//-------------------------------- KVBM Runtime Component --------------------------------
    #[derive(Builder, Clone)]
    #[builder(pattern = "owned")]
    pub struct KVBMDynamoRuntimeComponent {
        #[builder(private)]
        drt: DistributedRuntime,

        /// Name of the Runtime Component
        #[builder(setter(into))]
        name: String,

        /// Namespace
        #[builder(setter(into))]
        namespace: Namespace,

        /// Buffer State
        #[builder(private)]
        buffer: Arc<Mutex<BufferState>>,
    }

    impl KVBMDynamoRuntimeComponent {

        pub fn namespace(&self) -> &Namespace {
            &self.namespace
        }

        pub fn name(&self) -> String {
            self.name.clone()
        }

        async fn publish_store(
            &self,
            event_name: impl AsRef<str> + Send + Sync,
            event: &(impl Serialize + Send + Sync),
        ) -> Result<()> {
            const MAX_BATCH_SIZE: usize = 1;
            let mut buffer = self.buffer.lock().await;
            buffer.push_store(event)?;
            if buffer.should_flush(MAX_BATCH_SIZE) {
                self.flush_batch(&mut buffer, event_name).await?;
            }
            Ok(())
        }

        async fn publish_remove(
            &self,
            event_name: impl AsRef<str> + Send + Sync,
            event: &(impl Serialize + Send + Sync),
        ) -> Result<()> {
            const MAX_BATCH_SIZE: usize = 1;
            let mut buffer = self.buffer.lock().await;
            buffer.push_remove(event)?;
            if buffer.should_flush(MAX_BATCH_SIZE) {
                self.flush_batch(&mut buffer, event_name).await?;
            }
            Ok(())
        }

        async fn flush_batch(&self, buffer: &mut BufferState, event_name: impl AsRef<str> + Send + Sync) -> Result<()> {
            let events = buffer.drain_batch();
            for event in events {
                match event {
                    BatchedEvent::Store { data } | BatchedEvent::Remove { data } => {
                        self.publish_bytes(&event_name, data).await?;
                    }
                }
            }
            Ok(())
        }
    }

    impl KVBMDynamoRuntimeComponentBuilder {

        pub fn from_runtime(drt: DistributedRuntime) -> Self {
            Self::default().drt(drt).buffer(Arc::new(Mutex::new(BufferState::default())))
        }
    }

    impl DistributedRuntimeProvider for KVBMDynamoRuntimeComponent {
        fn drt(&self) -> &DistributedRuntime {
            &self.drt
        }
    }

    #[derive(Default)]
    struct BufferState {
        buffer: VecDeque<BatchedEvent>,
    }

    impl BufferState {
        pub fn push_store(&mut self, event: &impl Serialize) -> serde_json::Result<()> {
            let data = serde_json::to_vec(event)?;
            self.buffer.push_back(BatchedEvent::Store { data });
            Ok(())
        }

        pub fn push_remove(&mut self, event: &impl Serialize) -> serde_json::Result<()> {
            let data = serde_json::to_vec(event)?;
            self.buffer.push_back(BatchedEvent::Remove { data });
            Ok(())
        }

        pub fn should_flush(&self, max_batch_size: usize) -> bool {
            self.buffer.len() >= max_batch_size
        }

        pub fn drain_batch(&mut self) -> Vec<BatchedEvent> {
            self.buffer.drain(..).collect()
        }
    }

    #[derive(Debug, Clone)]
    pub enum BatchedEvent {
        Store { data: Vec<u8> },
        Remove { data: Vec<u8> },
    }

    #[async_trait]
    impl EventPublisher for KVBMDynamoRuntimeComponent {
        fn subject(&self) -> String {
            format!("namespace.{}", self.namespace.name())
        }

        async fn publish(
            &self,
            event_name: impl AsRef<str> + Send + Sync,
            event: &(impl Serialize + Send + Sync),
        ) -> Result<()> {
            let bytes = serde_json::to_vec(event)?;
            self.publish_bytes(event_name, bytes).await
        }

        async fn publish_bytes(
            &self,
            event_name: impl AsRef<str> + Send + Sync,
            bytes: Vec<u8>,
        ) -> Result<()> {
            let subject = format!("{}.{}", self.subject(), event_name.as_ref());
            Ok(self
                .drt()
                .nats_client()
                .client()
                .publish(subject, bytes.into())
                .await?)
        }
    }
//-------------------------------- End Event Publisher --------------------------------

    // Event enum for background event processing
    pub enum Event {
        RegisterMultiple {
            blocks: Vec<(SequenceHash, BlockHash, Option<SequenceHash>)>,
            worker_identifier: u64,
        },
        Release {
            sequence_hash: SequenceHash,
            worker_identifier: u64,
        }
    }

    /// Translate the Dynamo [`DistributedRuntime`] to the [`kvbm::config::KvManagerRuntimeConfig`]
    #[derive(Clone, Builder, Dissolve)]
    #[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
    pub struct DynamoKvbmRuntimeConfig {
        pub runtime: DistributedRuntime,
        pub nixl: kvbm::config::NixlOptions,
    }

    impl DynamoKvbmRuntimeConfig {
        pub fn builder() -> DynamoKvbmRuntimeConfigBuilder {
            DynamoKvbmRuntimeConfigBuilder::default()
        }
    }

    impl DynamoKvbmRuntimeConfigBuilder {
        pub fn build(self) -> Result<kvbm::config::KvManagerRuntimeConfig> {
            let (runtime, nixl) = self.build_internal()?.dissolve();
            Ok(kvbm::config::KvManagerRuntimeConfig::builder()
                .worker_id(runtime.primary_lease().unwrap().id() as u64)
                .cancellation_token(runtime.primary_token().child_token())
                .nixl(nixl)
                .build()?)
        }
    }
//-------------------------------- Event Manager --------------------------------
    /// Implementation of the [`kvbm::events::EventManager`] for the Dynamo Runtime Event Plane + the
    /// Dynamo LLM KV router message protocol.
    #[derive(Clone)]
    pub struct DynamoEventManager {
        tx: mpsc::UnboundedSender<Event>,
        worker_identifier: u64
    }

    impl DynamoEventManager {

        pub fn new(component: KVBMDynamoRuntimeComponent, worker_identifier: u64) -> Self {
            let (tx, rx) = mpsc::unbounded_channel();
            let event_id_counter = Arc::new(AtomicU64::new(0));
            worker_task(component, rx, event_id_counter.clone());
            Self { tx, worker_identifier}
        }

        pub fn publisher(self: &Arc<Self>) -> Publisher {
            Publisher::new(self.clone())
        }

    }

    // Worker task to receive and process messages
    pub fn worker_task(
        component: KVBMDynamoRuntimeComponent,
        mut rx: mpsc::UnboundedReceiver<Event>,
        event_id_counter: Arc<AtomicU64>,
    ) {
        let component_clone = component.clone();
        _ = component.drt().runtime().secondary().spawn(async move {
        while let Some(event) = rx.recv().await {
            let event_id = event_id_counter.fetch_add(1, Ordering::SeqCst);
            match event {
                Event::RegisterMultiple { blocks, worker_identifier } => {
                    let parent_hash = blocks
                        .first()
                        .and_then(|(_, _, parent)| parent.clone());
                    let store_data = KvCacheStoreData {
                        blocks: blocks.iter().map(|(sequence_hash, block_hash, _parent_sequence_hash)|
                            KvCacheStoredBlockData {
                                block_hash: ExternalSequenceBlockHash(*sequence_hash),
                                tokens_hash: LocalBlockHash(*block_hash),
                            }
                        ).collect(),
                        parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                    };
                    let data = KvCacheEventData::Stored(store_data);
                    let event = KvCacheEvent {
                        event_id: event_id,
                        data,
                    };
                    let router_event = RouterEvent::new(worker_identifier as i64, event);
                    if let Err(e) = component_clone.publish_store(KV_EVENT_SUBJECT, &router_event).await {
                        tracing::warn!("Failed to publish registration event: {:?}", e);
                    }
                }
                Event::Release { sequence_hash, worker_identifier } => {
                    let event = KvCacheEvent {
                        event_id: event_id,
                        data: KvCacheEventData::Removed(KvCacheRemoveData {
                            block_hashes: vec![ExternalSequenceBlockHash(sequence_hash)],
                        }),
                    };
                    let router_event = RouterEvent::new(worker_identifier as i64, event);
                    if let Err(e) = component_clone.publish_remove(KV_EVENT_SUBJECT, &router_event).await {
                        tracing::warn!("Failed to publish block release event: {:?}", e);
                    }
                }
                }
            }
        });
    }

    impl EventManager for DynamoEventManager {}

    impl kvbm::events::EventPublisher for DynamoEventManager {
        fn publish(&self, handles: Vec<Arc<RegistrationHandle>>) {
            if !handles.is_empty() {
                let blocks = handles.iter()
                    .map(|h| (h.sequence_hash(), h.block_hash(), h.parent_sequence_hash()))
                    .collect();
                let _ = self.tx.send(Event::RegisterMultiple {
                    blocks,
                    worker_identifier: self.worker_identifier,
                });
            }
        }
    }

    impl kvbm::events::EventReleaseManager for DynamoEventManager {
        fn block_release(&self, registration_handle: &RegistrationHandle) {
            let _ = self.tx.send(Event::Release {
                sequence_hash: registration_handle.sequence_hash(),
                worker_identifier: self.worker_identifier,
            });
        }
    }
//-------------------------------- End Event Manager --------------------------------
}

#[cfg(test)]
mod tests {

    #[allow(unused_imports)]
    use super::llm_kvbm::*;
    use dynamo_llm::block_manager as kvbm;
    use kvbm::storage::tests::{NullDeviceAllocator, NullDeviceStorage};
    use kvbm::layout::{FullyContiguous, LayoutConfig, LayoutError};
    use kvbm::pool::BlockPool;
    use kvbm::block::{BasicMetadata, Blocks};
    use dynamo_llm::block_manager::DType;

    use dynamo_runtime::{DistributedRuntime, Runtime};
    use dynamo_llm::block_manager::NixlOptions;
    use dynamo_llm::block_manager::{KvBlockManager, /*KvManagerLayoutConfig,*/ KvBlockManagerConfig, KvManagerModelConfig};
    use dynamo_llm::block_manager::block::registry::BlockRegistry;

    use dynamo_llm::tokens::{TokenBlockSequence, Tokens};
    use std::sync::Arc;
    use dynamo_llm::block_manager::events::EventManager;
    use dynamo_runtime::traits::events::{EventPublisher, EventSubscriber};
    use futures::stream::StreamExt;
    use dynamo_llm::block_manager::block::BlockExt;
    //use dynamo_llm::block_manager::storage::{DeviceAllocator, PinnedAllocator};

    pub type ReferenceBlockManager = KvBlockManager<BasicMetadata>;

    //-------------------------------- Test Helpers --------------------------------
    pub fn setup_layout(
        alignment: Option<usize>, // Option to override default alignment
    ) -> Result<FullyContiguous<NullDeviceStorage>, LayoutError> {
        let config = LayoutConfig {
            num_blocks: 1,
            num_layers: 5,
            page_size: 4,
            inner_dim: 13,
            alignment: alignment.unwrap_or(1),
            dtype: DType::FP32,
        };

        FullyContiguous::allocate(config, &NullDeviceAllocator)
    }

    fn create_sequence() -> TokenBlockSequence {
        let tokens = Tokens::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        // NOTE: 1337 was the original seed, so we are temporarily using that here to prove the logic has not changed
        let sequence = TokenBlockSequence::new(tokens, 4, Some(1337_u64));

        assert_eq!(sequence.blocks().len(), 2);
        assert_eq!(sequence.current_block().len(), 2);

        assert_eq!(sequence.blocks()[0].tokens(), &vec![1, 2, 3, 4]);
        assert_eq!(sequence.blocks()[0].sequence_hash(), 14643705804678351452);

        assert_eq!(sequence.blocks()[1].tokens(), &vec![5, 6, 7, 8]);
        assert_eq!(sequence.blocks()[1].sequence_hash(), 4945711292740353085);

        assert_eq!(sequence.current_block().tokens(), &vec![9, 10]);

        sequence
    }

    async fn create_dynamo_block_manager() -> ReferenceBlockManager {
        let rt = Runtime::from_current().unwrap();
        let dtr = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
        let nixl = NixlOptions::Enabled;
        let ns = dtr.namespace("test".to_string()).unwrap();
        let _kvbm_component = KVBMDynamoRuntimeComponentBuilder::from_runtime(dtr.clone())
            .name("kvbm_component".to_string())
            .namespace(ns.clone())
            .build()
            .unwrap();

        let dyn_config = DynamoKvbmRuntimeConfig::builder()
            .runtime(dtr.clone())
            .nixl(nixl)
            .build()
            .unwrap();

        let config = KvBlockManagerConfig::builder()
            .runtime(
                dyn_config
            )
            .model(
                KvManagerModelConfig::builder()
                    .num_layers(3)
                    .page_size(4)
                    .inner_dim(16)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();

        ReferenceBlockManager::new(config).unwrap()
    }

    //-------------------------------- Test Cases --------------------------------
    #[tokio::test]
    async fn test_create_dynamo_block_manager() {
        let _block_manager = create_dynamo_block_manager();
    }

    #[tokio::test]
    async fn test_dynamo_kvbm_runtime_config_builder() {
        let rt = Runtime::from_current().unwrap();
        let runtime = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
        let nixl = NixlOptions::Enabled;

        let config = DynamoKvbmRuntimeConfig::builder()
            .runtime(runtime.clone())
            .nixl(nixl)
            .build()
            .unwrap();

        assert_eq!(config.worker_id, runtime.primary_lease().unwrap().id() as u64);
        assert!(matches!(config.nixl, NixlOptions::Enabled));
        rt.shutdown();
    }

    #[tokio::test]
    async fn test_event_manager_drop_vec() {
        dynamo_runtime::logging::init();
        let sequence = create_sequence();
        let rt = Runtime::from_current().unwrap();
        let dtr = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
        let namespace_name = "test_event_manager_drop_vec".to_string();
        let ns = dtr.namespace(namespace_name).unwrap();
        let kvbm_component = KVBMDynamoRuntimeComponentBuilder::from_runtime(dtr.clone())
            .name("kvbm_component".to_string())
            .namespace(ns.clone())
            .build()
            .unwrap();

        let manager = Arc::new(DynamoEventManager::new(kvbm_component.clone(), dtr.primary_lease().unwrap().id() as u64)); // Split the tuple
        let event_manager: Arc<dyn EventManager> = manager;
        // Create a subscriber
        let mut subscriber = ns.subscribe(KV_EVENT_SUBJECT.to_string()).await.unwrap();

        // Create a Vec of publish_handles
        let publish_handles: Vec<_> = sequence.blocks().iter()
            .map(|block| BlockRegistry::create_publish_handle(block, event_manager.clone()))
            .collect();

        // No event should have been triggered yet
        let timeout = tokio::time::timeout(std::time::Duration::from_millis(100), subscriber.next()).await;
        assert!(timeout.is_err(), "Unexpected event triggered before dropping publish_handles");

        drop(publish_handles);

        let expected_events = sequence.blocks().len() * 2; // 2 events per handle
        let mut event_count = 0;
        let timeout = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(msg) = subscriber.next().await {
                let _received = String::from_utf8(msg.payload.to_vec()).expect("Failed to decode message payload");
                event_count += 1;

                if event_count == expected_events {
                    break;
                }
            }
        }).await;

        if timeout.is_err() {
            panic!("Test timed out while waiting for events");
        }

        assert_eq!(event_count, expected_events, "Expected {} events to be triggered", expected_events);

        let timeout = tokio::time::timeout(std::time::Duration::from_millis(100), subscriber.next()).await;
        assert!(timeout.is_err(), "Unexpected event received after the expected events");
        rt.shutdown();
    }

    #[tokio::test]
    async fn test_block_pool_publishing() {
        const EXPECTED_SEQUENCE_HASH: u64 = 14643705804678351452;

        let rt = Runtime::from_current().unwrap();
        let dtr = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
        let namespace_name = "test_block_pool_publishing".to_string();
        let ns = dtr.namespace(namespace_name).unwrap();
        let kvbm_component = KVBMDynamoRuntimeComponentBuilder::from_runtime(dtr.clone())
            .name("kvbm_component".to_string())
            .namespace(ns.clone())
            .build()
            .unwrap();

        let manager = Arc::new(DynamoEventManager::new(kvbm_component.clone(), dtr.primary_lease().unwrap().id() as u64)); // Split the tuple
        let event_manager: Arc<dyn EventManager> = manager.clone();

        // Create a subscriber
        let mut subscriber = ns.subscribe(KV_EVENT_SUBJECT.to_string()).await.unwrap();


        // Create a new layout
        let layout = setup_layout(None).unwrap();

        // Create the Blocks
        let blocks = Blocks::<_, BasicMetadata>::new(layout, 42, dtr.primary_lease().unwrap().id() as u64)
            .unwrap()
            .into_blocks()
            .unwrap();

        // Create the BlockPool and add the blocks
        let pool = BlockPool::builder().blocks(blocks).event_manager(event_manager).build().unwrap();

        // All blocks should be in the Reset/Empty state
        // No blocks should match the expected sequence hash
        let matched_blocks = pool
            .match_sequence_hashes_blocking(&[EXPECTED_SEQUENCE_HASH])
            .unwrap();
        assert_eq!(matched_blocks.len(), 0);

        // Allocate a single block from the pool
        let mut mutable_blocks = pool.allocate_blocks_blocking(1).unwrap();
        assert_eq!(mutable_blocks.len(), 1);
        let mut block = mutable_blocks.pop().unwrap();

        // Initialize the sequence on the block with a salt hash
        block.init_sequence(1337).unwrap();

        // Add some tokens to the block - our page_size is 4
        block.add_token(1).unwrap();
        block.add_token(2).unwrap();
        block.add_token(3).unwrap();
        block.add_token(4).unwrap();

        // Should fail because we don't have space in the block
        assert!(block.add_token(5).is_err());

        // Commit the block - this will generate a sequence hash
        // This will put the block in a Complete state
        block.commit().unwrap();
        assert!(block.state().is_complete()); // perhaps renamed to Commited

        let sequence_hash = block.sequence_hash().unwrap();
        assert_eq!(sequence_hash, EXPECTED_SEQUENCE_HASH);

        // Register the block
        // We provide a mutable block to the register_blocks function
        // This will take ownership of the block and return an immutable block
        let mut immutable_blocks = pool.register_blocks_blocking(vec![block]).unwrap();
        let block = immutable_blocks.pop().unwrap();
        assert!(block.state().is_registered());
        assert_eq!(block.sequence_hash().unwrap(), sequence_hash);

        let expected_events = 1;
        let mut event_count = 0;
        let timeout = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(msg) = subscriber.next().await {
                let _received = String::from_utf8(msg.payload.to_vec()).expect("Failed to decode message payload");
                event_count += 1;

                if event_count == expected_events {
                    break;
                }
            }
        }).await;

        if timeout.is_err() {
            panic!("Test timed out while waiting for events");
        }
        // Dropping the immutable block should return the block to the pool
        // However, the block should remain in the BlockPool as an inactive block until it is reused
        // or promoted back to an immutable block by being matched with a sequence hash
        drop(block);

        // Get the list of ImmutableBlocks that match the sequence hash
        let matched = pool
            .match_sequence_hashes_blocking(&[sequence_hash])
            .unwrap();
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].sequence_hash().unwrap(), sequence_hash);

        // No more events should be received
        let timeout = tokio::time::timeout(std::time::Duration::from_millis(100), subscriber.next()).await;
        if let Ok(Some(msg)) = &timeout {
            let received = String::from_utf8(msg.payload.to_vec()).expect("Failed to decode message payload");
            panic!("Unexpected event received after batch event: {}", received);
        }
        assert!(timeout.is_err(), "Unexpected event received after the expected events");
        rt.shutdown();
    }

    #[tokio::test]
    async fn test_publisher() {
        let sequence = create_sequence();
        let rt = Runtime::from_current().unwrap();
        let dtr = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
        let namespace_name = "test_publisher".to_string();
        let ns = dtr.namespace(namespace_name).unwrap();
        let kvbm_component = KVBMDynamoRuntimeComponentBuilder::from_runtime(dtr.clone())
            .name("kvbm_component".to_string())
            .namespace(ns.clone())
            .build()
            .unwrap();

        let manager = Arc::new(DynamoEventManager::new(kvbm_component.clone(), dtr.primary_lease().unwrap().id() as u64)); // Split the tuple
        let event_manager: Arc<dyn EventManager> = manager.clone();
        let mut publisher = manager.publisher();

        // Create a subscriber
        let mut subscriber = ns.subscribe(KV_EVENT_SUBJECT.to_string()).await.unwrap();

        let publish_handle =
            BlockRegistry::create_publish_handle(&sequence.blocks()[0], event_manager.clone());

        let reg_handle = publish_handle.remove_handle();

        publisher.take_handle(publish_handle);

        // no event should have been triggered
        let timeout = tokio::time::timeout(std::time::Duration::from_millis(100), subscriber.next()).await;
        assert!(timeout.is_err(), "Unexpected event triggered before dropping publish_handle");

        // we should get two events when this is dropped, since we never took ownership of the RegistrationHandle
        drop(publisher);

        // Verify that two events are triggered
        let mut event_count = 0;
        // Add a timeout to prevent the test from hanging indefinitely
        let timeout = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(msg) = subscriber.next().await {
                let received = String::from_utf8(msg.payload.to_vec()).expect("Failed to decode message payload");
                event_count += 1;

                if event_count == 1 {
                    break; // Stop after receiving two events
                }
            }
        }).await;

        if timeout.is_err() {
            panic!("Test timed out while waiting for events");
        }

        drop(reg_handle);
        event_count = 0;
        let timeout = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(msg) = subscriber.next().await {
                let received = String::from_utf8(msg.payload.to_vec()).expect("Failed to decode message payload");
                event_count += 1;

                if event_count == 1 {
                    break; // Stop after receiving two events
                }
            }
        }).await;

        if timeout.is_err() {
            panic!("Test timed out while waiting for events");
        }
        rt.shutdown();
    }

    #[test]
    fn test_dynamo_block_manager_blocking() {
        //let event_manager = DynamoEventManager::new();
    }

    #[tokio::test]
    async fn test_dynamo_block_manager_async() {
        //let event_manager = DynamoEventManager::new();
    }

    #[tokio::test]
    async fn test_kvbm_component() {
        let rt = Runtime::from_current().unwrap();
        let dtr = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
        let namespace_name = "test_kvbm_component".to_string();
        let ns = dtr.namespace(namespace_name).unwrap();

        let kvbm_component = KVBMDynamoRuntimeComponentBuilder::from_runtime(dtr)
        .name("kvbm_component".to_string())
        .namespace(ns.clone())
        .build()
        .unwrap();

        // Create a subscriber
        let mut subscriber = ns.subscribe("testing_channel".to_string()).await.unwrap();
        if let Err(e) = kvbm_component.publish("testing_channel".to_string(), &"test_message".to_string()).await {
            tracing::warn!("Failed to publish registration event: {:?}", e);
        }
        // Receive the message
        if let Some(msg) = subscriber.next().await {
            let received = String::from_utf8(msg.payload.to_vec()).unwrap();
            assert_eq!(received, "\"test_message\"");
        }

        rt.shutdown();
    }
}
