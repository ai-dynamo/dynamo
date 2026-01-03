// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::block::registry::RegistrationHandle;

use crate::block_manager::kv_consolidator::EventSource;
use crate::block_manager::kv_consolidator::KvEventConsolidator;

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

/// Event manager that sends KVBM events to the kv event consolidator
pub struct DynamoEventManager {
    consolidator_handle: Arc<crate::block_manager::kv_consolidator::KvEventConsolidatorHandle>,
    #[allow(dead_code)]
    _consolidator: Option<Arc<crate::block_manager::kv_consolidator::KvEventConsolidator>>,
}

impl DynamoEventManager {
    /// Create a new DynamoEventManager with a consolidator handle
    pub fn new(
        consolidator_handle: Arc<crate::block_manager::kv_consolidator::KvEventConsolidatorHandle>,
    ) -> Arc<Self> {
        Arc::new(Self {
            consolidator_handle,
            _consolidator: None,
        })
    }

    /// Create a new DynamoEventManager with kv event consolidator configuration
    ///
    /// This creates and manages the kv event consolidator internally.
    /// The kv event consolidator will be started asynchronously.
    pub async fn new_with_config(
        config: crate::block_manager::kv_consolidator::KvEventConsolidatorConfig,
    ) -> anyhow::Result<Arc<Self>> {
        let mut kv_event_consolidator = KvEventConsolidator::new(config)?;
        kv_event_consolidator.start().await?;
        let handle = kv_event_consolidator.get_handle();

        Ok(Arc::new(Self {
            consolidator_handle: Arc::new(handle),
            _consolidator: Some(Arc::new(kv_event_consolidator)),
        }))
    }

    /// Send store events to the kv event consolidator
    ///
    /// Called when KVBM registers/stores blocks. Sends events to the kv event consolidator
    /// which will deduplicate them with vLLM events.
    ///
    fn publish_store_events(&self, handles: Vec<Arc<RegistrationHandle>>) {
        if handles.is_empty() {
            return;
        }

        tracing::debug!(
            "DynamoEventManager::publish_store_events called with {} blocks",
            handles.len()
        );

        // Send each block to the consolidator
        let kv_event_consolidator = self.consolidator_handle.clone();

        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move {
                for handle in handles {
                    // Extract block metadata from RegistrationHandle
                    let block_hash = handle.sequence_hash().to_string();
                    let parent_hash = handle.parent_sequence_hash().map(|h| h.to_string());

                    // Extract block_size and tokens from RegistrationHandle
                    let block_size = handle.block_size(); // usize
                    let tokens: Vec<u32> = handle.tokens().iter().copied().collect();

                    tracing::debug!(
                        "DynamoEventManager sending store event to kv event consolidator: block_hash={}, block_size={}, tokens={}",
                        block_hash,
                        block_size,
                        tokens.len()
                    );

                    // Send to consolidator with EventSource::Kvbm
                    kv_event_consolidator
                        .handle_store(
                            block_hash,
                            EventSource::Kvbm,
                            tokens,
                            parent_hash,
                            block_size,
                            None, // lora_id
                            None, // tier
                            None, // data_parallel_rank
                        )
                        .await;
                }
            });
        } else {
            tracing::error!(
                "No Tokio runtime in context; dropping store events for {} blocks",
                handles.len()
            );
        }
    }

    /// Send remove event to the kv event consolidator
    ///
    /// Called when a RegistrationHandle is dropped (block evicted from KVBM).
    fn publish_remove_event(&self, registration_handle: &RegistrationHandle) {
        let block_hash = registration_handle.sequence_hash().to_string();

        tracing::debug!(
            "DynamoEventManager::publish_remove_event called: block_hash={}",
            block_hash
        );

        let kv_event_consolidator = self.consolidator_handle.clone();

        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move {
                kv_event_consolidator
                    .handle_remove(&block_hash, EventSource::Kvbm)
                    .await;
            });
        } else {
            tracing::error!(
                "No Tokio runtime in context; dropping remove event for block {}",
                block_hash
            );
        }
    }
}

impl std::fmt::Debug for DynamoEventManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DynamoEventManager(kv_event_consolidator)")
    }
}

impl EventManager for DynamoEventManager {}

impl EventPublisher for DynamoEventManager {
    fn publish(&self, handles: Vec<Arc<RegistrationHandle>>) {
        self.publish_store_events(handles);
    }
}

impl EventReleaseManager for DynamoEventManager {
    fn block_release(&self, registration_handle: &RegistrationHandle) {
        self.publish_remove_event(registration_handle);
    }
}

#[cfg(test)]
pub mod tests {
    use crate::tokens::SequenceHash;

    use super::*;

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
