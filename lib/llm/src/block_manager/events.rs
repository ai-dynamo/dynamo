// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use super::block::registry::RegistrationHandle;
use crate::kv_router::{
    protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
    },
    publisher::KvEventPublisher,
};

#[cfg(any(test, feature = "testing-full"))]
use tokio::sync::mpsc;

/// The [EventManager] is not responsible for managing the history of the blocks, nor what
/// events have been published.
///
/// The [EventManager] is only responsible for issuing events on state changes. In this case,
/// there are two states:
///
/// - Store: a dynamo event plane message will be published which defines the registration/storing
///   of the block. Details include, but are not limited to, the sequence/prefix hash, the local block
///   hash, the sequence position of the block, the block size, and the storage location/class which
///   the block is stored in.
///
/// - Remove: a dynamo event plane message will be published which defines the removal of the block
///   from the cache. This messasge will include enough information to identify the block within a
///   storage hierarchy; minmally, the sequence hash and the storage location/class.
///
/// The [RegistrationHandle] associated from [EventManager::block_register] call is an RAII object
/// which will trigger a `Remove` event on being dropped.
pub trait EventManager: EventPublisher + EventReleaseManager + Send + Sync {
    // fn register_block(&self, token_block: &TokenBlock) -> PublishHandle;
    // fn publisher(&self) -> Publisher;
}

pub trait EventPublisher: Send + Sync {
    fn publish(&self, handles: Vec<Arc<RegistrationHandle>>);
}

pub trait EventReleaseManager: Send + Sync {
    fn block_release(&self, registration_handle: &RegistrationHandle);
}

/// A handle to a registered block.
///
/// Ensures that the register event published before the release event by
/// holding an [Arc] to the [RegistrationHandle], which by extension holds
/// issues the release event when dropped.
///
/// Ownership of the [PublishHandle] transferred to a [Publisher] object
/// which is responsible for coordinating the publication of multiple
/// registration events.
pub struct PublishHandle {
    handle: Arc<RegistrationHandle>,
    publisher: Option<Arc<dyn EventPublisher>>,
}

impl PublishHandle {
    pub fn new(handle: RegistrationHandle, publisher: Arc<dyn EventPublisher>) -> Self {
        let handle = Arc::new(handle);
        let publisher = Some(publisher);
        Self { handle, publisher }
    }

    pub fn remove_handle(&self) -> Arc<RegistrationHandle> {
        self.handle.clone()
    }

    fn disarm(&mut self) {
        self.publisher = None;
    }
}

impl Drop for PublishHandle {
    fn drop(&mut self) {
        if let Some(publisher) = self.publisher.take() {
            publisher.publish(vec![self.handle.clone()]);
        }
    }
}

/// Responsible for publishing multiple registration events.
///
/// Because [EventPublisher::publish] takes a list of shared [RegistrationHandles][RegistrationHandle]
/// this allows the [EventPublisher] logic to optimize the number of events published
/// by consoldiate multiple registration events with additional sequence logic.
///
/// The behavior of the [EventPublisher] is left entirely up to the the implementor.
#[derive(Clone)]
pub struct Publisher {
    handles: Vec<Arc<RegistrationHandle>>,
    publisher: Arc<dyn EventPublisher>,
}

impl Publisher {
    pub fn new(publisher: Arc<dyn EventPublisher>) -> Self {
        Self {
            handles: Vec::new(),
            publisher,
        }
    }

    pub fn take_handle(&mut self, publish_handle: PublishHandle) -> Arc<RegistrationHandle> {
        let handle = publish_handle.remove_handle();
        self.handles.push(handle.clone());
        let mut publish_handle = publish_handle;
        publish_handle.disarm();
        handle
    }

    pub fn publish(&mut self) {
        let handles = std::mem::take(&mut self.handles);
        if !handles.is_empty() {
            self.publisher.publish(handles);
        }
    }
}

impl Drop for Publisher {
    fn drop(&mut self) {
        self.publish();
    }
}

// Implementation notes:
//
// - Removable events are per blocks. I think we will want to leverage a task to collect drop/remove
//   events so that we can batch them together.
//
// - Registration events are can be batched by the nature of the [EventManager::register_blocks] call.

pub struct NullEventManager;

impl NullEventManager {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {})
    }
}

impl EventManager for NullEventManager {}

impl EventPublisher for NullEventManager {
    fn publish(&self, _handles: Vec<Arc<RegistrationHandle>>) {}
}

impl EventReleaseManager for NullEventManager {
    fn block_release(&self, _registration_handle: &RegistrationHandle) {}
}

/// Event manager that emits KV cache events to the indexer.
pub struct DynamoEventManager {
    event_id_counter: AtomicU64,
    /// KV event publisher for publishing events to the indexer
    publisher: PublisherImpl,
}

/// Publisher implementation - can be real or mock for testing
enum PublisherImpl {
    Real(Arc<KvEventPublisher>),
    #[cfg(any(test, feature = "testing-full"))]
    Mock(mpsc::UnboundedSender<KvCacheEvent>),
}

impl PublisherImpl {
    fn publish(&self, event: KvCacheEvent) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match self {
            PublisherImpl::Real(publisher) => publisher
                .publish(event)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
            #[cfg(any(test, feature = "testing-full"))]
            PublisherImpl::Mock(tx) => tx
                .send(event)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
        }
    }
}

impl DynamoEventManager {
    /// Create a DynamoEventManager with a KV event publisher.
    pub fn new(publisher: Arc<KvEventPublisher>) -> Arc<Self> {
        Arc::new(Self {
            event_id_counter: AtomicU64::new(0),
            publisher: PublisherImpl::Real(publisher),
        })
    }

    /// Create a test DynamoEventManager that uses channels instead of NATS.
    /// Returns the manager and a receiver to verify emitted events.
    #[cfg(any(test, feature = "testing-full"))]
    pub fn new_test() -> (Arc<Self>, mpsc::UnboundedReceiver<KvCacheEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let manager = Arc::new(Self {
            event_id_counter: AtomicU64::new(0),
            publisher: PublisherImpl::Mock(tx),
        });
        (manager, rx)
    }

    fn next_event_id(&self) -> u64 {
        self.event_id_counter.fetch_add(1, Ordering::SeqCst)
    }
}

impl std::fmt::Debug for DynamoEventManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DynamoEventManager")
    }
}

impl EventManager for DynamoEventManager {}

impl EventPublisher for DynamoEventManager {
    fn publish(&self, handles: Vec<Arc<RegistrationHandle>>) {
        if handles.is_empty() {
            return;
        }

        let parent_hash = handles
            .first()
            .and_then(|h| h.parent_sequence_hash())
            .map(ExternalSequenceBlockHash);

        let blocks: Vec<KvCacheStoredBlockData> = handles
            .iter()
            .map(|handle| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(handle.sequence_hash()),
                tokens_hash: LocalBlockHash(handle.block_hash()),
            })
            .collect();

        let store_data = KvCacheStoreData {
            parent_hash,
            blocks,
        };

        let event = KvCacheEvent {
            event_id: self.next_event_id(),
            data: KvCacheEventData::Stored(store_data.clone()),
        };

        // Publish to the indexer
        if let Err(e) = self.publisher.publish(event) {
            tracing::error!("Failed to publish STORED event to indexer: {}", e);
        }
    }
}

impl EventReleaseManager for DynamoEventManager {
    fn block_release(&self, registration_handle: &RegistrationHandle) {
        let sequence_hash = registration_handle.sequence_hash();

        let remove_data = KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(sequence_hash)],
        };

        let event = KvCacheEvent {
            event_id: self.next_event_id(),
            data: KvCacheEventData::Removed(remove_data),
        };

        // Publish to the indexer
        if let Err(e) = self.publisher.publish(event) {
            tracing::error!("Failed to publish REMOVED event to indexer: {}", e);
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::tokens::SequenceHash;

    #[derive(Debug, PartialEq, Eq)]
    pub enum EventType {
        Register(SequenceHash),
        Remove(SequenceHash),
    }

    pub struct MockEventManager {
        tx: tokio::sync::mpsc::UnboundedSender<Vec<EventType>>,
    }

    impl MockEventManager {
        pub fn new() -> (
            Arc<Self>,
            tokio::sync::mpsc::UnboundedReceiver<Vec<EventType>>,
        ) {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            (Arc::new(Self { tx }), rx)
        }

        pub fn publisher(self: &Arc<Self>) -> Publisher {
            Publisher::new(self.clone())
        }
    }

    impl EventManager for MockEventManager {}

    impl EventPublisher for MockEventManager {
        fn publish(&self, handles: Vec<Arc<RegistrationHandle>>) {
            let events = handles
                .iter()
                .map(|handle| EventType::Register(handle.sequence_hash()))
                .collect::<Vec<_>>();
            self.tx.send(events).unwrap();
        }
    }

    impl EventReleaseManager for MockEventManager {
        fn block_release(&self, registration_handle: &RegistrationHandle) {
            let events = vec![EventType::Remove(registration_handle.sequence_hash())];
            self.tx.send(events).unwrap();
        }
    }
}
