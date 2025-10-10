// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
    use tokio::time::Duration;

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
    use dynamo_runtime::DistributedRuntime;
    use dynamo_runtime::component::Namespace;
    use dynamo_runtime::prelude::DistributedRuntimeProvider;
    use dynamo_runtime::traits::events::EventPublisher;
    use kvbm::events::EventManager;
    use tokio::sync::mpsc;
    pub use tokio_util::sync::CancellationToken;

    pub const KV_EVENT_SUBJECT: &str = "kv_events";

    #[derive(Debug, Clone)]
    pub enum PublisherEvent {
        Store(RouterEvent),
        Remove(RouterEvent),
    }

    // TODO[oandreeva] The potential issue with the start_batching_publisher
    // same as with the worker task, it's potentially a slow subscriber.
    // Needs to be perf tested and improved.
    pub async fn start_batching_publisher(
        component: Arc<KVBMDynamoRuntimeComponent>,
        mut rx: mpsc::UnboundedReceiver<PublisherEvent>,
        max_batch_size: usize,
        deadline: Duration,
    ) {
        let mut buffer: VecDeque<RouterEvent> = VecDeque::new();
        let mut interval = tokio::time::interval(deadline);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {

                _ = interval.tick() => {
                    if !buffer.is_empty() {
                        let events: Vec<RouterEvent> = buffer.drain(..).collect();
                        if let Err(e) = component.publish(KV_EVENT_SUBJECT, &events).await {
                           tracing::warn!("Failed to publish events: {:?}", e);
                        }
                    }
                }

                maybe_evt = rx.recv() => {
                    match maybe_evt {
                        Some(PublisherEvent::Store(data)) => {
                            buffer.push_back(data);
                            if buffer.len() >= max_batch_size {
                                let events: Vec<RouterEvent> = buffer.drain(..).collect();
                                if let Err(e) = component.publish(KV_EVENT_SUBJECT, &events).await {
                                    tracing::warn!("Failed to publish events: {:?}", e);
                                 }
                            }
                        }
                        Some(PublisherEvent::Remove(data)) => {
                            buffer.push_back(data);
                            let events: Vec<RouterEvent> = buffer.drain(..).collect();
                            if let Err(e) = component.publish(KV_EVENT_SUBJECT, &events).await {
                                tracing::warn!("Failed to publish events: {:?}", e);
                            }
                        }
                        None => {
                            if !buffer.is_empty() {
                                let events: Vec<RouterEvent> = buffer.drain(..).collect();
                                if let Err(e) = component.publish(KV_EVENT_SUBJECT, &events).await {
                                    tracing::warn!("Failed to publish events: {:?}", e);
                                }
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

        #[builder(setter(into))]
        name: String,

        #[builder(setter(into))]
        namespace: Namespace,

        /// Buffer State
        #[builder(private)]
        batch_tx: mpsc::UnboundedSender<PublisherEvent>,
    }

    impl KVBMDynamoRuntimeComponent {
        pub fn new(
            drt: DistributedRuntime,
            name: String,
            namespace: Namespace,
            deadline: Duration,
            max_batch_size: usize,
        ) -> Arc<Self> {
            let (tx, rx) = mpsc::unbounded_channel();
            let component = Arc::new(Self {
                drt,
                name,
                namespace,
                batch_tx: tx,
            });

            let batching_component = Arc::clone(&component);
            component.drt().runtime().secondary().spawn(async move {
                start_batching_publisher(batching_component, rx, max_batch_size, deadline).await;
            });

            component
        }

        pub fn namespace(&self) -> &Namespace {
            &self.namespace
        }

        pub fn name(&self) -> &str {
            &self.name
        }

        #[cfg(test)]
        pub fn batch_tx(&self) -> mpsc::UnboundedSender<PublisherEvent> {
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
            self.drt()
                .nats_client()
                .client()
                .publish(subject, bytes.into())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to publish to NATS: {}", e))
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
            let worker_id = runtime.primary_lease().unwrap().id() as u64;
            Ok(kvbm::config::KvManagerRuntimeConfig::builder()
                .worker_id(worker_id)
                .cancellation_token(runtime.primary_token().child_token())
                .nixl(nixl)
                .build()?)
        }
    }

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
            let worker_id = component.drt().primary_lease().unwrap().id() as u64;
            component.drt().runtime().secondary().spawn(async move {
                worker_task(component, rx).await;
            });
            Self {
                tx,
                worker_identifier: worker_id,
            }
        }

        pub fn publisher(self: &Arc<Self>) -> Publisher {
            Publisher::new(self.clone())
        }
    }

    // Worker task to receive and process messages
    // TODO[oandreeva] The potential issue with the worker task
    // being a slow subscriber. Needs to be perf tested and improved.
    pub async fn worker_task(
        component: Arc<KVBMDynamoRuntimeComponent>,
        mut rx: mpsc::UnboundedReceiver<Event>,
    ) {
        let mut event_id_counter: u64 = 0;
        let component_clone = component.clone();
        loop {
            match rx.recv().await {
                Some(Event::RegisterMultiple {
                    blocks,
                    worker_identifier,
                }) => {
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
                    let event = KvCacheEvent {
                        data,
                        event_id: event_id_counter,
                    };
                    let router_event = RouterEvent::new(worker_identifier as i64, event);
                    event_id_counter += 1;
                    if let Err(e) = component_clone
                        .batch_tx
                        .send(PublisherEvent::Store(router_event))
                    {
                        tracing::warn!("Failed to send event to batch channel: {:?}", e);
                    }
                }
                Some(Event::Release {
                    sequence_hash,
                    worker_identifier,
                }) => {
                    let event = KvCacheEvent {
                        data: KvCacheEventData::Removed(KvCacheRemoveData {
                            block_hashes: vec![ExternalSequenceBlockHash(sequence_hash)],
                        }),
                        event_id: event_id_counter,
                    };
                    let router_event = RouterEvent::new(worker_identifier as i64, event);
                    event_id_counter += 1;
                    if let Err(e) = component_clone
                        .batch_tx
                        .send(PublisherEvent::Remove(router_event))
                    {
                        tracing::warn!("Failed to send event to batch channel: {:?}", e);
                    }
                }
                None => {
                    tracing::info!("Receiver is closed, stopping KVBM Dynamo Event Manager");
                    break;
                }
            }
        }
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
}

#[cfg(all(test, feature = "testing-full"))]
mod tests {

    #[allow(unused_imports)]
    use super::llm_kvbm::*;
    use futures::stream::StreamExt;
    use std::sync::Arc;
    use tokio::time::Duration;

    use dynamo_llm::block_manager as kvbm;

    use dynamo_llm::tokens::{TokenBlockSequence, Tokens};
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        traits::events::{EventPublisher, EventSubscriber},
    };
    use kvbm::{
        KvBlockManagerConfig, KvManagerLayoutConfig, KvManagerModelConfig, NixlOptions,
        ReferenceBlockManager,
        block::BlockState,
        block::GlobalRegistry,
        block::registry::BlockRegistry,
        block::state::CompleteState,
        events::EventManager,
        storage::{DeviceAllocator, DiskAllocator, PinnedAllocator},
    };

    use dynamo_llm::kv_router::{
        indexer::RouterEvent,
        protocols::{
            ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
            KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
        },
    };

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
                    .outer_dim(2)
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

        ReferenceBlockManager::new(config).await.unwrap()
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

    // There are issues with running the following 2 tests concurrently with others
    // With --test-threads=1 flag, they pass.
    #[tokio::test]
    #[ignore]
    async fn test_dynamo_block_manager_async() {
        dynamo_runtime::logging::init();
        let _block_manager = create_dynamo_block_manager().await;
    }

    #[test]
    #[ignore]
    fn test_create_dynamo_block_manager() {
        // Create a runtime for the test
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        // Run the async function in the runtime
        let _block_manager = rt.block_on(async { create_dynamo_block_manager().await });
    }

    #[tokio::test]
    async fn test_kvbm_component_publish() {
        dynamo_runtime::logging::init();
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

        tx.send(PublisherEvent::Store(event1)).unwrap();
        tx.send(PublisherEvent::Store(event2)).unwrap();

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

        tx.send(PublisherEvent::Store(event3)).unwrap();

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

        tx.send(PublisherEvent::Remove(event4)).unwrap();

        // Should receive remove event immediately
        let msg = subscriber.next().await.unwrap();
        let received: Vec<RouterEvent> = serde_json::from_slice(&msg.payload).unwrap();
        assert_eq!(received.len(), 1, "Should receive remove event immediately");

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
    async fn test_dynamo_event_manager() {
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

        let global_registry = GlobalRegistry::default();

        let mut block_registry =
            BlockRegistry::new(event_manager, global_registry.clone(), rt.primary().clone());

        // Register a block
        // Create CompleteState from TokenBlock
        let complete_state = CompleteState::new(sequence.blocks()[0].clone());
        let mut block_state = BlockState::Complete(complete_state);
        let publish_handle = block_registry.register_block(&mut block_state).unwrap();

        // No event should have been triggered yet
        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(100), subscriber.next()).await;
        assert!(
            timeout.is_err(),
            "Unexpected event triggered before dropping publish_handles"
        );

        // Dropping the publish handle should trigger a `Store` event
        drop(publish_handle);

        let timeout =
            tokio::time::timeout(std::time::Duration::from_secs(5), subscriber.next()).await;

        match timeout {
            Ok(Some(msg)) => {
                let _received = String::from_utf8(msg.payload.to_vec())
                    .expect("Failed to decode message payload");
            }
            Ok(None) => {
                panic!("Subscriber channel closed without receiving event");
            }
            Err(_) => {
                panic!("Timeout waiting for remove event");
            }
        }

        // Dropping the block state should trigger a `Remove` event
        drop(block_state);

        let timeout =
            tokio::time::timeout(std::time::Duration::from_secs(5), subscriber.next()).await;

        match timeout {
            Ok(Some(msg)) => {
                let _received = String::from_utf8(msg.payload.to_vec())
                    .expect("Failed to decode message payload");
            }
            Ok(None) => {
                panic!("Subscriber channel closed without receiving event");
            }
            Err(_) => {
                panic!("Timeout waiting for remove event");
            }
        }

        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(100), subscriber.next()).await;
        assert!(
            timeout.is_err(),
            "Unexpected event received after the expected events"
        );
        rt.shutdown();
    }

    #[tokio::test]
    async fn test_dynamo_event_manager_multiple_handles() {
        dynamo_runtime::logging::init();
        let sequence = create_sequence();
        let (kvbm_component, rt) = setup_kvbm_component(Duration::from_secs(10), 1).await;
        let mut subscriber = kvbm_component
            .namespace()
            .subscribe(KV_EVENT_SUBJECT.to_string())
            .await
            .unwrap();
        let manager = Arc::new(DynamoEventManager::new(kvbm_component));
        let mut publisher = manager.publisher();
        let event_manager: Arc<dyn EventManager> = manager;

        let global_registry = GlobalRegistry::default();

        let mut block_registry =
            BlockRegistry::new(event_manager, global_registry.clone(), rt.primary().clone());

        // Register first block
        let complete_state_1 = CompleteState::new(sequence.blocks()[0].clone());
        let mut block_state_1 = BlockState::Complete(complete_state_1);
        let publish_handle_1 = block_registry.register_block(&mut block_state_1).unwrap();

        // Register second block
        let complete_state_2 = CompleteState::new(sequence.blocks()[1].clone());
        let mut block_state_2 = BlockState::Complete(complete_state_2);
        let publish_handle_2 = block_registry.register_block(&mut block_state_2).unwrap();
        drop(block_state_2);
        drop(block_state_1);

        // No event should have been triggered yet
        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(100), subscriber.next()).await;
        assert!(
            timeout.is_err(),
            "Unexpected event triggered before dropping publish_handles"
        );

        publisher.take_handle(publish_handle_1.unwrap());
        publisher.take_handle(publish_handle_2.unwrap());

        publisher.publish();

        let timeout =
            tokio::time::timeout(std::time::Duration::from_secs(5), subscriber.next()).await;

        match timeout {
            Ok(Some(msg)) => {
                let _received = String::from_utf8(msg.payload.to_vec())
                    .expect("Failed to decode message payload");
            }
            Ok(None) => {
                panic!("Subscriber channel closed without receiving event");
            }
            Err(_) => {
                panic!("Timeout waiting for remove event");
            }
        }

        drop(publisher);

        let expected_events = 2; // 2 events per handle per remove
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
}
