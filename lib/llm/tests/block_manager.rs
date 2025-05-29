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

#[cfg(feature = "block-manager")]
pub mod llm_kvbm {
    // alias for the kvbm module to make the refactor to standalone crate easier
    use dynamo_llm::block_manager as kvbm;

    // kvbm specific imports
    use kvbm::{block::registry::RegistrationHandle, events::*};

    // std imports
    use async_trait::async_trait;
    use serde::Serialize;
    use std::collections::VecDeque;
    use std::sync::Arc;
    use tokio::time::{sleep, Duration, Instant};

    use anyhow::Result;
    use derive_builder::Builder;
    use derive_getters::Dissolve;
    use dynamo_llm::kv_router::{
        indexer::RouterEvent,
        protocols::{
            ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
            KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
        },
    };
    use dynamo_llm::tokens::{BlockHash, SequenceHash};
    use dynamo_runtime::component::Namespace;
    use dynamo_runtime::prelude::DistributedRuntimeProvider;
    use dynamo_runtime::traits::events::EventPublisher;
    use dynamo_runtime::DistributedRuntime;
    use kvbm::events::EventManager;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tokio::sync::mpsc;

    //-------------------------------- KVBM Runtime Component --------------------------------

    pub const KV_EVENT_SUBJECT: &str = "kv_events";

    #[derive(Debug, Clone)]
    pub enum PublisherEvent {
        Store(RouterEvent),
        Remove(RouterEvent),
    }

    pub async fn start_batching_publisher(
        component: Arc<KVBMDynamoRuntimeComponent>,
        mut rx: mpsc::Receiver<PublisherEvent>,
        max_batch_size: usize,
        deadline: Duration,
    ) {
        let mut buffer: VecDeque<RouterEvent> = VecDeque::new();
        let timer = sleep(deadline);
        tokio::pin!(timer);

        loop {
            tokio::select! {
                // Deadline fired
                _ = &mut timer, if !buffer.is_empty() => {
                    let events: Vec<RouterEvent> = buffer.drain(..).collect();
                    let _ = component.publish(KV_EVENT_SUBJECT, &events).await;
                    timer.as_mut().reset(Instant::now() + deadline);
                }

                // New event received
                maybe_evt = rx.recv() => {
                    match maybe_evt {
                        Some(PublisherEvent::Store(data)) => {
                            buffer.push_back(data);
                            if buffer.len() >= max_batch_size {
                                let events: Vec<RouterEvent> = buffer.drain(..).collect();
                                let _ = component.publish(KV_EVENT_SUBJECT, &events).await;
                                timer.as_mut().reset(Instant::now() + deadline);
                            }
                        }
                        Some(PublisherEvent::Remove(data)) => {
                            buffer.push_back(data);
                            // On remove, always flush immediately
                            let events: Vec<RouterEvent> = buffer.drain(..).collect();
                            let _ = component.publish(KV_EVENT_SUBJECT, &events).await;
                            timer.as_mut().reset(Instant::now() + deadline);
                        }
                        None => {
                            // Channel closed, flush remaining
                            if !buffer.is_empty() {
                                let events: Vec<RouterEvent> = buffer.drain(..).collect();
                                let _ = component.publish(KV_EVENT_SUBJECT, &events).await;
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

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
        batch_tx: mpsc::Sender<PublisherEvent>,
    }

    impl KVBMDynamoRuntimeComponent {
        pub fn new(
            drt: DistributedRuntime,
            name: String,
            namespace: Namespace,
            deadline: Duration,
            max_batch_size: usize,
        ) -> Arc<Self> {
            let (tx, rx) = mpsc::channel(2048);

            let component = Arc::new(Self {
                drt,
                name,
                namespace,
                batch_tx: tx,
            });

            let batching_component = Arc::clone(&component);
            batching_component
                .drt
                .runtime()
                .secondary()
                .spawn(async move {
                    start_batching_publisher(batching_component, rx, max_batch_size, deadline)
                        .await;
                });

            component
        }

        pub fn namespace(&self) -> &Namespace {
            &self.namespace
        }

        pub fn name(&self) -> String {
            self.name.clone()
        }

        #[cfg(test)]
        pub fn batch_tx(&self) -> mpsc::Sender<PublisherEvent> {
            self.batch_tx.clone()
        }
    }

    impl DistributedRuntimeProvider for KVBMDynamoRuntimeComponent {
        fn drt(&self) -> &DistributedRuntime {
            &self.drt
        }
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

    // Event enum for background event processing// Event enum for background event processing
    pub enum Event {
        RegisterMultiple {
            blocks: Vec<(SequenceHash, BlockHash, Option<SequenceHash>)>,
            worker_identifier: u64,
        },
        Release {
            sequence_hash: SequenceHash,
            worker_identifier: u64,
        },
    }

    /// Implementation of the [`kvbm::events::EventManager`] for the Dynamo Runtime Event Plane + the
    /// Dynamo LLM KV router message protocol.
    #[derive(Clone)]
    pub struct DynamoEventManager {
        tx: mpsc::UnboundedSender<Event>,
        worker_identifier: u64,
    }

    impl DynamoEventManager {
        pub fn new(component: Arc<KVBMDynamoRuntimeComponent>) -> Self {
            let (tx, rx) = mpsc::unbounded_channel();
            let event_id_counter = Arc::new(AtomicU64::new(0));
            let worker_identifier = component.drt().primary_lease().unwrap().id() as u64;
            worker_task(component, rx, event_id_counter.clone());
            Self {
                tx,
                worker_identifier,
            }
        }

        pub fn publisher(self: &Arc<Self>) -> Publisher {
            Publisher::new(self.clone())
        }
    }

    // Worker task to receive and process messages
    pub fn worker_task(
        component: Arc<KVBMDynamoRuntimeComponent>,
        mut rx: mpsc::UnboundedReceiver<Event>,
        event_id_counter: Arc<AtomicU64>,
    ) {
        let component_clone = component.clone();
        _ = component.drt().runtime().secondary().spawn(async move {
            while let Some(event) = rx.recv().await {
                let event_id = event_id_counter.fetch_add(1, Ordering::SeqCst);
                match event {
                    Event::RegisterMultiple {
                        blocks,
                        worker_identifier,
                    } => {
                        let parent_hash = blocks.first().and_then(|(_, _, parent)| *parent);
                        let store_data = KvCacheStoreData {
                            blocks: blocks
                                .iter()
                                .map(|(sequence_hash, block_hash, _parent_sequence_hash)| {
                                    KvCacheStoredBlockData {
                                        block_hash: ExternalSequenceBlockHash(*sequence_hash),
                                        tokens_hash: LocalBlockHash(*block_hash),
                                    }
                                })
                                .collect(),
                            parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                        };
                        let data = KvCacheEventData::Stored(store_data);
                        let event = KvCacheEvent { event_id, data };
                        let router_event = RouterEvent::new(worker_identifier as i64, event);
                        if let Err(e) = component_clone
                            .batch_tx
                            .send(PublisherEvent::Store(router_event))
                            .await
                        {
                            tracing::warn!("Failed to send event to batch channel: {:?}", e);
                        }
                    }
                    Event::Release {
                        sequence_hash,
                        worker_identifier,
                    } => {
                        let event = KvCacheEvent {
                            event_id,
                            data: KvCacheEventData::Removed(KvCacheRemoveData {
                                block_hashes: vec![ExternalSequenceBlockHash(sequence_hash)],
                            }),
                        };
                        let router_event = RouterEvent::new(worker_identifier as i64, event);
                        if let Err(e) = component_clone
                            .batch_tx
                            .send(PublisherEvent::Remove(router_event))
                            .await
                        {
                            tracing::warn!("Failed to send event to batch channel: {:?}", e);
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
                let blocks = handles
                    .iter()
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

#[cfg(all(test, feature = "testing-full"))]
mod tests {

    #[allow(unused_imports)]
    use super::llm_kvbm::*;
    use dynamo_llm::block_manager as kvbm;
    use kvbm::block::BasicMetadata;

    use dynamo_llm::block_manager::block::registry::BlockRegistry;
    use dynamo_llm::block_manager::NixlOptions;
    use dynamo_llm::block_manager::{
        KvBlockManager, KvBlockManagerConfig, KvManagerLayoutConfig, KvManagerModelConfig,
    };
    use dynamo_runtime::{DistributedRuntime, Runtime};

    use dynamo_llm::block_manager::events::EventManager;
    use dynamo_llm::block_manager::storage::{DeviceAllocator, DiskAllocator, PinnedAllocator};
    use dynamo_llm::kv_router::{
        indexer::RouterEvent,
        protocols::{
            ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
            KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
        },
    };
    use dynamo_llm::tokens::{TokenBlockSequence, Tokens};
    use dynamo_runtime::traits::events::{EventPublisher, EventSubscriber};
    use futures::stream::StreamExt;
    use std::sync::Arc;
    use tokio::time::Duration;

    pub type ReferenceBlockManager = KvBlockManager<BasicMetadata>;

    //-------------------------------- Test Helpers --------------------------------

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
        let kvbm_component = KVBMDynamoRuntimeComponent::new(
            dtr.clone(),
            "kvbm_component".to_string(),
            ns.clone(),
            Duration::from_secs(10),
            1, /*max_batch_size*/
        );
        let manager = Arc::new(DynamoEventManager::new(kvbm_component.clone()));

        let config = KvBlockManagerConfig::builder()
            .runtime(
                DynamoKvbmRuntimeConfig::builder()
                    .runtime(dtr.clone())
                    .nixl(nixl)
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
            .disk_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(16)
                    .allocator(DiskAllocator)
                    .build()
                    .unwrap(),
            )
            .host_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(16)
                    .allocator(PinnedAllocator::default())
                    .build()
                    .unwrap(),
            )
            .device_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(8)
                    .allocator(DeviceAllocator::new(0).unwrap())
                    .build()
                    .unwrap(),
            )
            .event_manager(Some(manager))
            .build()
            .unwrap();

        ReferenceBlockManager::new(config).unwrap()
    }

    async fn setup_kvbm_component(
        deadline: Duration,
        max_batch_size: usize,
    ) -> (Arc<KVBMDynamoRuntimeComponent>, Runtime) {
        let rt = Runtime::from_current().unwrap();
        let dtr = DistributedRuntime::from_settings(rt.clone()).await.unwrap();

        // Generate a random namespace name
        let namespace_name = format!("test_namespace_{}", rand::random::<u32>());
        let ns = dtr.namespace(namespace_name).unwrap();

        // Create component with small batch size and short deadline for testing
        let kvbm_component = KVBMDynamoRuntimeComponent::new(
            dtr.clone(),
            "kvbm_component".to_string(),
            ns.clone(),
            deadline,
            max_batch_size,
        );
        (kvbm_component, rt)
    }

    //-------------------------------- Test Cases --------------------------------
    #[test]
    fn test_dynamo_block_manager() {
        // Check if we're already in a Tokio runtime context
        let async_runtime = if tokio::runtime::Handle::try_current().is_ok() {
            None // If we're already in a runtime, don't create a new one
        } else {
            // Only create a new runtime if not already in one
            Some(Arc::new(tokio::runtime::Runtime::new().unwrap()))
        };

        let future = async {
            let rt = Runtime::from_current().unwrap();
            let dtr = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
            let namespace_name = "test_dynamo_block_manager".to_string();
            let ns = dtr.namespace(namespace_name).unwrap();
            let kvbm_component = KVBMDynamoRuntimeComponent::new(
                dtr.clone(),
                "kvbm_component".to_string(),
                ns.clone(),
                Duration::from_secs(10),
                1, /*max_batch_size*/
            );

            let _manager = Arc::new(DynamoEventManager::new(kvbm_component.clone()));
        };

        // If we're already in a runtime, just run the future
        if let Some(runtime) = async_runtime {
            runtime.block_on(future);
        } else {
            // If we're already in a runtime context, we can just await the future
            tokio::runtime::Handle::current().block_on(future);
        }
    }

    #[tokio::test]
    async fn test_dynamo_block_manager_async() {
        let (kvbm_component, rt) = setup_kvbm_component(Duration::from_secs(10), 1).await;
        let _manager = Arc::new(DynamoEventManager::new(kvbm_component.clone()));
        rt.shutdown();
    }

    #[tokio::test]
    async fn test_create_dynamo_block_manager() {
        let _block_manager = create_dynamo_block_manager();
    }

    #[tokio::test]
    async fn test_dynamo_event_manager_drop_vec() {
        dynamo_runtime::logging::init();
        let sequence = create_sequence();
        let (kvbm_component, rt) = setup_kvbm_component(Duration::from_secs(10), 1).await;
        let mut subscriber = kvbm_component
            .namespace()
            .subscribe(KV_EVENT_SUBJECT.to_string())
            .await
            .unwrap();
        let manager = Arc::new(DynamoEventManager::new(kvbm_component));
        let event_manager: Arc<dyn EventManager> = manager;

        // Create a Vec of publish_handles
        let publish_handles: Vec<_> = sequence
            .blocks()
            .iter()
            .map(|block| BlockRegistry::create_publish_handle(block, event_manager.clone()))
            .collect();

        // No event should have been triggered yet
        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(100), subscriber.next()).await;
        assert!(
            timeout.is_err(),
            "Unexpected event triggered before dropping publish_handles"
        );

        drop(publish_handles);

        let expected_events = sequence.blocks().len() * 2; // 2 events per handle
        let mut event_count = 0;
        let timeout = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(msg) = subscriber.next().await {
                let _received = String::from_utf8(msg.payload.to_vec())
                    .expect("Failed to decode message payload");
                event_count += 1;

                if event_count == expected_events {
                    break;
                }
            }
        })
        .await;

        if timeout.is_err() {
            panic!("Test timed out while waiting for events");
        }

        assert_eq!(
            event_count, expected_events,
            "Expected {} events to be triggered",
            expected_events
        );

        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(100), subscriber.next()).await;
        assert!(
            timeout.is_err(),
            "Unexpected event received after the expected events"
        );
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
        let kvbm_component = KVBMDynamoRuntimeComponent::new(
            dtr.clone(),
            "kvbm_component".to_string(),
            ns.clone(),
            Duration::from_millis(500),
            2, /*max_batch_size*/
        );

        let manager = Arc::new(DynamoEventManager::new(kvbm_component.clone()));
        let mut publisher = manager.publisher();
        let event_manager: Arc<dyn EventManager> = manager;
        // Create a subscriber
        let mut subscriber = ns.subscribe(KV_EVENT_SUBJECT.to_string()).await.unwrap();

        let publish_handle1 =
            BlockRegistry::create_publish_handle(&sequence.blocks()[0], event_manager.clone());
        let publish_handle2 =
            BlockRegistry::create_publish_handle(&sequence.blocks()[1], event_manager.clone());

        // Remove handles before adding to publisher
        let _reg_handle1 = publish_handle1.remove_handle();

        // Add disarmed handles to publisher
        publisher.take_handle(publish_handle1);

        publisher.publish();

        let _timeout = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(msg) = subscriber.next().await {
                let _received = String::from_utf8(msg.payload.to_vec())
                    .expect("Failed to decode message payload");
                break;
            }
        })
        .await;

        let _reg_handle2 = publish_handle2.remove_handle();
        publisher.take_handle(publish_handle2);

        // No event should have been triggered yet
        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(1000), subscriber.next()).await;
        assert!(
            timeout.is_err(),
            "Unexpected event triggered before dropping publish_handles"
        );

        drop(publisher);

        let expected_events = 1; // 2 events per handle, but should only get one
        let mut event_count = 0;
        let timeout = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(msg) = subscriber.next().await {
                let _received = String::from_utf8(msg.payload.to_vec())
                    .expect("Failed to decode message payload");
                event_count += 1;

                if event_count == expected_events {
                    break;
                }
            }
        })
        .await;

        if timeout.is_err() {
            panic!("Test timed out while waiting for events");
        }

        assert_eq!(
            event_count, expected_events,
            "Expected {} events to be triggered",
            expected_events
        );

        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(1000), subscriber.next()).await;
        assert!(
            timeout.is_err(),
            "Unexpected event received after the expected events"
        );
        rt.shutdown();
    }

    #[tokio::test]
    async fn test_kvbm_component_publish() {
        let rt = Runtime::from_current().unwrap();
        let dtr = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
        let namespace_name = "test_kvbm_component".to_string();
        let ns = dtr.namespace(namespace_name).unwrap();

        let kvbm_component = KVBMDynamoRuntimeComponent::new(
            dtr.clone(),
            "kvbm_component".to_string(),
            ns.clone(),
            Duration::from_secs(10),
            1, /*max_batch_size*/
        );

        // Create a subscriber
        let mut subscriber = ns.subscribe("testing_channel".to_string()).await.unwrap();
        if let Err(e) = kvbm_component
            .publish("testing_channel".to_string(), &"test_message".to_string())
            .await
        {
            tracing::warn!("Failed to publish registration event: {:?}", e);
        }
        // Receive the message
        if let Some(msg) = subscriber.next().await {
            let received = String::from_utf8(msg.payload.to_vec()).unwrap();
            assert_eq!(received, "\"test_message\"");
        }

        rt.shutdown();
    }

    #[tokio::test]
    async fn test_dynamo_component_batching_publisher_max_batch_size() {
        let (kvbm_component, rt) = setup_kvbm_component(Duration::from_millis(100), 2).await;

        // Create a subscriber
        let mut subscriber = kvbm_component
            .namespace()
            .subscribe(KV_EVENT_SUBJECT.to_string())
            .await
            .unwrap();
        let tx = kvbm_component.batch_tx();

        // Send two store events - should trigger batch due to max_batch_size
        let event1 = RouterEvent::new(
            1,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(1),
                        tokens_hash: LocalBlockHash(1),
                    }],
                    parent_hash: None,
                }),
            },
        );

        let event2 = RouterEvent::new(
            2,
            KvCacheEvent {
                event_id: 2,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(2),
                        tokens_hash: LocalBlockHash(2),
                    }],
                    parent_hash: None,
                }),
            },
        );

        tx.send(PublisherEvent::Store(event1)).await.unwrap();
        tx.send(PublisherEvent::Store(event2)).await.unwrap();

        // Should receive one batch with both events
        let msg = subscriber.next().await.unwrap();
        let received: Vec<RouterEvent> = serde_json::from_slice(&msg.payload).unwrap();
        assert_eq!(received.len(), 2, "Should receive both events in one batch");

        drop(tx); // Close the channel

        // No more events should be received
        let timeout = tokio::time::timeout(Duration::from_millis(200), subscriber.next()).await;
        assert!(
            timeout.is_err(),
            "Should not receive any more events after channel closure"
        );
        rt.shutdown();
    }

    #[tokio::test]
    async fn test_dynamo_component_batching_publisher_deadline() {
        let (kvbm_component, rt) = setup_kvbm_component(Duration::from_millis(100), 2).await;
        let mut subscriber = kvbm_component
            .namespace()
            .subscribe(KV_EVENT_SUBJECT.to_string())
            .await
            .unwrap();
        let tx = kvbm_component.batch_tx();

        let event3 = RouterEvent::new(
            3,
            KvCacheEvent {
                event_id: 3,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(3),
                        tokens_hash: LocalBlockHash(3),
                    }],
                    parent_hash: None,
                }),
            },
        );

        tx.send(PublisherEvent::Store(event3)).await.unwrap();

        // Wait for deadline to trigger
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should receive the event after deadline
        let msg = subscriber.next().await.unwrap();
        let received: Vec<RouterEvent> = serde_json::from_slice(&msg.payload).unwrap();
        assert_eq!(
            received.len(),
            1,
            "Should receive single event after deadline"
        );

        drop(tx);

        // No more events should be received
        let timeout = tokio::time::timeout(Duration::from_millis(200), subscriber.next()).await;
        assert!(
            timeout.is_err(),
            "Should not receive any more events after channel closure"
        );
        rt.shutdown();
    }

    #[tokio::test]
    async fn test_dynamo_component_batching_publisher_remove_event() {
        let (kvbm_component, rt) = setup_kvbm_component(Duration::from_millis(100), 2).await;

        // Create a subscriber
        let mut subscriber = kvbm_component
            .namespace()
            .subscribe(KV_EVENT_SUBJECT.to_string())
            .await
            .unwrap();
        let tx = kvbm_component.batch_tx();

        // Test 3: Immediate flush for Remove event
        let event4 = RouterEvent::new(
            4,
            KvCacheEvent {
                event_id: 4,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(4)],
                }),
            },
        );

        tx.send(PublisherEvent::Remove(event4)).await.unwrap();

        // Should receive remove event immediately
        let msg = subscriber.next().await.unwrap();
        let received: Vec<RouterEvent> = serde_json::from_slice(&msg.payload).unwrap();
        assert_eq!(received.len(), 1, "Should receive remove event immediately");
        println!("Received event: {:?}", received);

        drop(tx); // Close the channel

        // No more events should be received
        let timeout = tokio::time::timeout(Duration::from_millis(200), subscriber.next()).await;
        assert!(
            timeout.is_err(),
            "Should not receive any more events after channel closure"
        );
        rt.shutdown();
    }
}
