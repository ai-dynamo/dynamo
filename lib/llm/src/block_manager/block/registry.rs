// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # KV Cache Block Registration
//!
//! - This module is responsible for maintaining a registry of all blocks currently within a pool.
//!   This consists of two components: A global registry of all blocks, and a per-pool registry of blocks.
//! - The global registry is keyed by sequence hash and storage tier. If two blocks in different pools
//!   have the same sequence hash but live in different tiers, they keep distinct registration handles
//!   so KVBM can emit per-tier events. The global registry is shared across all pools.
//! - The per-pool registry is a mapping of sequence hashes to block handles. This is used to track which blocks are
//!   currently within a specific pool. The block handle is unique across pools, and is used to track the block's lifetime.
//! - When a block is in the registered state, it has a unique block handle and a possibly shared registration handle.
//!
//! ## Workflow
//!
//! 1. When a block is registered into a pool, we create a unique block handle.
//! 2. We then check the global registry to see if the block already exists in any other pool.
//! 3. If it does, we use the existing registration handle. Otherwise, we create a new one.
//! 4. When the block handle is dropped, it means that the block is no longer in the pool.
//! 5. When the registration handle is dropped, it means that the block is no longer in any pool.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
};

use super::super::events::{EventManager, EventReleaseManager, PublishHandle};
use super::state::BlockState;

use crate::block_manager::kv_consolidator::StorageTier;
use crate::tokens::{BlockHash, SequenceHash, TokenBlock};

use derive_getters::Getters;
use tokio::{runtime::Handle, sync::mpsc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegistrationKey {
    sequence_hash: SequenceHash,
    storage_tier: StorageTier,
}

impl RegistrationKey {
    fn new(sequence_hash: SequenceHash, storage_tier: StorageTier) -> Self {
        Self {
            sequence_hash,
            storage_tier,
        }
    }
}

pub type GlobalRegistry = Arc<Mutex<HashMap<RegistrationKey, Weak<RegistrationHandle>>>>;

#[derive(Debug, thiserror::Error)]
pub enum BlockRegistrationError {
    #[error("Block already registered")]
    BlockAlreadyRegistered(SequenceHash),

    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// A block entry is a handle to a block that is registered in the pool.
/// On drop, we need to notify the pool that the block has been unregistered.
/// This is different than the registration handle, which is only dropped when the block is no longer in ANY pool.
#[derive(Debug)]
pub struct BlockHandle {
    sequence_hash: SequenceHash,
    unregister_tx: mpsc::UnboundedSender<SequenceHash>,
}

impl BlockHandle {
    pub fn new(
        sequence_hash: SequenceHash,
        unregister_tx: mpsc::UnboundedSender<SequenceHash>,
    ) -> Self {
        Self {
            sequence_hash,
            unregister_tx,
        }
    }
}

impl Drop for BlockHandle {
    fn drop(&mut self) {
        let _ = self.unregister_tx.send(self.sequence_hash);
    }
}

pub struct BlockRegistry {
    blocks: Arc<Mutex<HashMap<SequenceHash, Weak<BlockHandle>>>>,
    storage_tier: StorageTier,
    event_manager: Arc<dyn EventManager>,
    global_registry: GlobalRegistry,
    unregister_tx: mpsc::UnboundedSender<SequenceHash>,
}

impl BlockRegistry {
    pub fn new(
        event_manager: Arc<dyn EventManager>,
        global_registry: GlobalRegistry,
        async_runtime: Handle,
        storage_tier: StorageTier,
    ) -> Self {
        let (unregister_tx, mut unregister_rx) = mpsc::unbounded_channel();

        let blocks: Arc<Mutex<HashMap<SequenceHash, Weak<BlockHandle>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let blocks_clone = blocks.clone();
        let global_registry_clone = global_registry.clone();
        async_runtime.spawn(async move {
            let blocks = blocks_clone;
            let global_registry = global_registry_clone;
            while let Some(sequence_hash) = unregister_rx.recv().await {
                {
                    let mut blocks = blocks.lock().unwrap();

                    if let Some(handle) = blocks.get(&sequence_hash)
                        && handle.upgrade().is_none()
                    {
                        blocks.remove(&sequence_hash);
                    }
                }

                let mut global_registry = global_registry.lock().unwrap();
                let registration_key = RegistrationKey::new(sequence_hash, storage_tier);

                if let Some(entry) = global_registry.get(&registration_key)
                    && entry.upgrade().is_none()
                {
                    global_registry.remove(&registration_key);
                }
            }
        });

        Self {
            blocks,
            storage_tier,
            event_manager,
            global_registry,
            unregister_tx,
        }
    }

    pub fn is_registered(&self, sequence_hash: SequenceHash) -> bool {
        let blocks = self.blocks.lock().unwrap();
        if let Some(handle) = blocks.get(&sequence_hash)
            && let Some(_handle) = handle.upgrade()
        {
            return true;
        }
        false
    }

    pub fn register_block(
        &mut self,
        block_state: &mut BlockState,
    ) -> Result<Option<PublishHandle>, BlockRegistrationError> {
        match block_state {
            BlockState::Reset => Err(BlockRegistrationError::InvalidState(
                "Block is in Reset state".to_string(),
            )),
            BlockState::Partial(_partial) => Err(BlockRegistrationError::InvalidState(
                "Block is in Partial state".to_string(),
            )),

            BlockState::Complete(state) => {
                let sequence_hash = state.token_block().sequence_hash();
                let mut blocks = self.blocks.lock().unwrap();

                // If an identical block already exists in this pool, return an error.
                if let Some(handle) = blocks.get(&sequence_hash)
                    && let Some(_handle) = handle.upgrade()
                {
                    return Err(BlockRegistrationError::BlockAlreadyRegistered(
                        sequence_hash,
                    ));
                }

                let mut publish_handle = None;

                let block_handle =
                    Arc::new(BlockHandle::new(sequence_hash, self.unregister_tx.clone()));

                let reg_handle = 'reg_block: {
                    // Now, check the global registry.
                    let mut global_registry = self.global_registry.lock().unwrap();
                    let registration_key = RegistrationKey::new(sequence_hash, self.storage_tier);

                    // If an identical block exists in other pool, use the same registration handle.
                    if let Some(handle) = global_registry.get(&registration_key)
                        && let Some(handle) = handle.upgrade()
                    {
                        break 'reg_block handle;
                    }

                    // Otherwise, create a new registration handle.
                    publish_handle = Some(Self::create_publish_handle(
                        state.token_block(),
                        self.event_manager.clone(),
                        self.storage_tier,
                    ));
                    let reg_handle = publish_handle.as_ref().unwrap().remove_handle();

                    // Insert the registration handle into the global registry.
                    global_registry.insert(registration_key, Arc::downgrade(&reg_handle));

                    reg_handle
                };

                blocks.insert(sequence_hash, Arc::downgrade(&block_handle));

                // Update the [BlockState] to [BlockState::Registered]
                let _ = std::mem::replace(
                    block_state,
                    BlockState::Registered(reg_handle, block_handle),
                );

                Ok(publish_handle)
            }
            BlockState::Registered(registered, _) => Err(
                BlockRegistrationError::BlockAlreadyRegistered(registered.sequence_hash()),
            ),
        }
    }

    fn create_publish_handle(
        token_block: &TokenBlock,
        event_manager: Arc<dyn EventManager>,
        storage_tier: StorageTier,
    ) -> PublishHandle {
        let reg_handle =
            RegistrationHandle::from_token_block(token_block, event_manager.clone(), storage_tier);

        PublishHandle::new(reg_handle, event_manager)
    }
}

#[derive(Getters)]
pub struct RegistrationHandle {
    #[getter(copy)]
    block_hash: BlockHash,

    #[getter(copy)]
    sequence_hash: SequenceHash,

    #[getter(copy)]
    parent_sequence_hash: Option<SequenceHash>,

    #[getter(copy)]
    external_sequence_hash: Option<SequenceHash>,

    #[getter(copy)]
    external_parent_sequence_hash: Option<SequenceHash>,

    #[getter(copy)]
    storage_tier: StorageTier,

    #[getter(skip)]
    release_manager: Arc<dyn EventReleaseManager>,

    token_block: TokenBlock,
}

impl RegistrationHandle {
    /// Returns the block size (number of tokens in the block)
    pub fn block_size(&self) -> usize {
        self.token_block.block_size()
    }

    /// Returns a reference to the tokens in this block
    pub fn tokens(&self) -> &crate::tokens::Tokens {
        self.token_block.tokens()
    }

    /// Returns the router-facing sequence hash for this block.
    pub fn published_sequence_hash(&self) -> SequenceHash {
        self.external_sequence_hash.unwrap_or(self.sequence_hash)
    }

    /// Returns the router-facing parent sequence hash for this block.
    pub fn published_parent_sequence_hash(&self) -> Option<SequenceHash> {
        self.external_parent_sequence_hash
            .or(self.parent_sequence_hash)
    }

    fn from_token_block(
        token_block: &TokenBlock,
        release_manager: Arc<dyn EventReleaseManager>,
        storage_tier: StorageTier,
    ) -> Self {
        Self {
            block_hash: token_block.block_hash(),
            sequence_hash: token_block.sequence_hash(),
            parent_sequence_hash: token_block.parent_sequence_hash(),
            external_sequence_hash: token_block.external_sequence_hash(),
            external_parent_sequence_hash: token_block.external_parent_sequence_hash(),
            storage_tier,
            release_manager,
            token_block: token_block.clone(),
        }
    }
}

impl std::fmt::Debug for RegistrationHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RegistrationHandle {{ sequence_hash: {}; block_hash: {}; parent_sequence_hash: {:?}; external_sequence_hash: {:?}; external_parent_sequence_hash: {:?}; storage_tier: {:?} }}",
            self.sequence_hash,
            self.block_hash,
            self.parent_sequence_hash,
            self.external_sequence_hash,
            self.external_parent_sequence_hash,
            self.storage_tier
        )
    }
}

impl Drop for RegistrationHandle {
    fn drop(&mut self) {
        self.release_manager.block_release(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::block_manager::events::NullEventManager;
    use crate::block_manager::events::tests::{EventType, MockEventManager};
    use crate::block_manager::kv_consolidator::StorageTier;
    use crate::tokens::{TokenBlockSequence, Tokens};

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

    #[test]
    fn test_mock_event_manager_with_single_publish_handle() {
        let sequence = create_sequence();

        let (event_manager, mut rx) = MockEventManager::new();

        let publish_handle = BlockRegistry::create_publish_handle(
            &sequence.blocks()[0],
            event_manager.clone(),
            StorageTier::Device,
        );

        // no event should have been triggered
        assert!(rx.try_recv().is_err());

        // we shoudl get two events when this is dropped, since we never took ownership of the RegistrationHandle
        drop(publish_handle);

        // the first event should be a Register event
        let events = rx.try_recv().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            EventType::Register(sequence.blocks()[0].sequence_hash(), StorageTier::Device)
        );

        // the second event should be a Remove event
        let events = rx.try_recv().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            EventType::Remove(sequence.blocks()[0].sequence_hash(), StorageTier::Device)
        );

        // there should be no more events
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_mock_event_manager_single_publish_handle_removed() {
        let sequence = create_sequence();
        let block_to_test = &sequence.blocks()[0];
        let expected_sequence_hash = block_to_test.sequence_hash();

        let (event_manager, mut rx) = MockEventManager::new();

        let publish_handle = BlockRegistry::create_publish_handle(
            block_to_test,
            event_manager.clone(),
            StorageTier::Device,
        );

        // Remove the registration handle before dropping the publish handle
        let reg_handle = publish_handle.remove_handle();

        // no event should have been triggered yet
        assert!(rx.try_recv().is_err());

        // Drop the publish handle - it SHOULD trigger a Register event now because remove_handle doesn't disarm
        drop(publish_handle);
        let register_events = rx.try_recv().unwrap();
        assert_eq!(
            register_events.len(),
            1,
            "Register event should be triggered on PublishHandle drop"
        );
        assert_eq!(
            register_events[0],
            EventType::Register(expected_sequence_hash, StorageTier::Device),
            "Expected Register event"
        );

        // Drop the registration handle - this SHOULD trigger the Remove event
        drop(reg_handle);

        let events = rx.try_recv().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            EventType::Remove(expected_sequence_hash, StorageTier::Device),
            "Only Remove event should be triggered"
        );

        // there should be no more events
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_mock_event_manager_publisher_multiple_handles_removed() {
        let sequence = create_sequence();
        let block1 = &sequence.blocks()[0];
        let block2 = &sequence.blocks()[1];
        let hash1 = block1.sequence_hash();
        let hash2 = block2.sequence_hash();

        let (event_manager, mut rx) = MockEventManager::new();
        let mut publisher = event_manager.publisher();

        let publish_handle1 = BlockRegistry::create_publish_handle(
            block1,
            event_manager.clone(),
            StorageTier::Device,
        );
        let publish_handle2 = BlockRegistry::create_publish_handle(
            block2,
            event_manager.clone(),
            StorageTier::Device,
        );

        // Remove handles before adding to publisher
        let reg_handle1 = publish_handle1.remove_handle();
        let reg_handle2 = publish_handle2.remove_handle();

        // Add disarmed handles to publisher
        publisher.take_handle(publish_handle1);
        publisher.take_handle(publish_handle2);

        // no events yet
        assert!(rx.try_recv().is_err());

        // Drop the publisher - should trigger a single Publish event with both Register events
        drop(publisher);

        let events = rx.try_recv().unwrap();
        assert_eq!(
            events.len(),
            2,
            "Should receive two Register events in one batch"
        );
        // Order isn't guaranteed, so check for both
        assert!(events.contains(&EventType::Register(hash1, StorageTier::Device)));
        assert!(events.contains(&EventType::Register(hash2, StorageTier::Device)));

        // no more events immediately after publish
        assert!(rx.try_recv().is_err());

        // Drop registration handles individually - should trigger Remove events
        drop(reg_handle1);
        let events1 = rx.try_recv().unwrap();
        assert_eq!(events1.len(), 1);
        assert_eq!(events1[0], EventType::Remove(hash1, StorageTier::Device));

        drop(reg_handle2);
        let events2 = rx.try_recv().unwrap();
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0], EventType::Remove(hash2, StorageTier::Device));

        // no more events
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_publisher_empty_drop() {
        let (event_manager, mut rx) = MockEventManager::new();
        let publisher = event_manager.publisher();

        drop(publisher);
        // No events should be sent
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_publisher_publish_multiple_times() {
        let sequence = create_sequence();
        let block1 = &sequence.blocks()[0];
        let hash1 = block1.sequence_hash();

        let (event_manager, mut rx) = MockEventManager::new();
        let mut publisher = event_manager.publisher();

        let publish_handle1 = BlockRegistry::create_publish_handle(
            block1,
            event_manager.clone(),
            StorageTier::Device,
        );

        publisher.take_handle(publish_handle1);

        // First publish call
        publisher.publish();
        let events = rx.try_recv().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], EventType::Register(hash1, StorageTier::Device));

        // The RegistrationHandle Arc was taken by the publisher and dropped after the publish call
        // So, the Remove event should follow immediately.
        let remove_events = rx.try_recv().unwrap();
        assert_eq!(
            remove_events.len(),
            1,
            "Remove event should be triggered after publish consumes the handle"
        );
        assert_eq!(
            remove_events[0],
            EventType::Remove(hash1, StorageTier::Device),
            "Expected Remove event"
        );

        // Second publish call (should do nothing as handles were taken)
        publisher.publish();
        assert!(rx.try_recv().is_err());

        // Drop publisher (should also do nothing)
        drop(publisher);
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_same_sequence_in_different_tiers_emits_distinct_events() {
        let sequence = create_sequence();
        let block = sequence.blocks()[0].clone();
        let sequence_hash = block.sequence_hash();

        let (event_manager, mut rx) = MockEventManager::new();
        let global_registry = GlobalRegistry::default();

        let mut host_registry = BlockRegistry::new(
            event_manager.clone(),
            global_registry.clone(),
            Handle::current(),
            StorageTier::HostPinned,
        );
        let mut disk_registry = BlockRegistry::new(
            event_manager.clone(),
            global_registry,
            Handle::current(),
            StorageTier::Disk,
        );

        let mut host_state = BlockState::Reset;
        host_state.apply_token_block(block.clone()).unwrap();
        let host_publish = host_registry
            .register_block(&mut host_state)
            .unwrap()
            .unwrap();
        drop(host_publish);

        assert_eq!(
            rx.recv().await.unwrap(),
            vec![EventType::Register(sequence_hash, StorageTier::HostPinned)]
        );

        let mut disk_state = BlockState::Reset;
        disk_state.apply_token_block(block).unwrap();
        let disk_publish = disk_registry
            .register_block(&mut disk_state)
            .unwrap()
            .unwrap();
        drop(disk_publish);

        assert_eq!(
            rx.recv().await.unwrap(),
            vec![EventType::Register(sequence_hash, StorageTier::Disk)]
        );

        drop(host_state);
        assert_eq!(
            rx.recv().await.unwrap(),
            vec![EventType::Remove(sequence_hash, StorageTier::HostPinned)]
        );

        drop(disk_state);
        assert_eq!(
            rx.recv().await.unwrap(),
            vec![EventType::Remove(sequence_hash, StorageTier::Disk)]
        );
    }

    #[test]
    fn test_registration_handle_prefers_external_hashes_for_publication() {
        let mut sequence = create_sequence();
        sequence.sync_external_sequence_hashes(&[50_001, 50_002]);

        let release_manager = NullEventManager::new();
        let registration_handle = RegistrationHandle::from_token_block(
            &sequence.blocks()[1],
            release_manager,
            StorageTier::HostPinned,
        );

        assert_eq!(registration_handle.external_sequence_hash(), Some(50_002));
        assert_eq!(
            registration_handle.external_parent_sequence_hash(),
            Some(50_001)
        );
        assert_eq!(registration_handle.published_sequence_hash(), 50_002);
        assert_eq!(
            registration_handle.published_parent_sequence_hash(),
            Some(50_001)
        );
    }
}
