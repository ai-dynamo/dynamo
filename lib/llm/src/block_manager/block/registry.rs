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

use std::{
    collections::HashMap,
    sync::{Arc, Weak},
};

use super::super::events::{EventManager, EventReleaseManager, PublishHandle};
use super::state::BlockState;

use crate::tokens::{BlockHash, SequenceHash, TokenBlock};

use derive_getters::Getters;

#[derive(Debug, thiserror::Error)]
pub enum BlockRegistationError {
    #[error("Block already registered")]
    BlockAlreadyRegistered(SequenceHash),

    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// Error returned when an attempt is made to unregister a block that is still active.
#[derive(Debug, thiserror::Error)]
#[error("Failed to unregister block: {0}")]
pub struct UnregisterFailure(SequenceHash);

#[derive()]
pub struct BlockRegistry {
    blocks: HashMap<SequenceHash, Weak<RegistrationHandle>>,
    event_manager: Arc<dyn EventManager>,
}

impl BlockRegistry {
    pub fn new(event_manager: Arc<dyn EventManager>) -> Self {
        Self {
            blocks: HashMap::new(),
            event_manager,
        }
    }

    pub fn is_registered(&self, sequence_hash: SequenceHash) -> bool {
        if let Some(handle) = self.blocks.get(&sequence_hash) {
            if let Some(_handle) = handle.upgrade() {
                return true;
            }
        }
        false
    }

    pub fn register_block(
        &mut self,
        block_state: &mut BlockState,
    ) -> Result<PublishHandle, BlockRegistationError> {
        match block_state {
            BlockState::Reset => Err(BlockRegistationError::InvalidState(
                "Block is in Reset state".to_string(),
            )),
            BlockState::Partial(_partial) => Err(BlockRegistationError::InvalidState(
                "Block is in Partial state".to_string(),
            )),

            BlockState::Complete(state) => {
                let sequence_hash = state.token_block().sequence_hash();
                if let Some(handle) = self.blocks.get(&sequence_hash) {
                    if let Some(_handle) = handle.upgrade() {
                        return Err(BlockRegistationError::BlockAlreadyRegistered(sequence_hash));
                    }
                }

                // Create the [RegistrationHandle] and [PublishHandle]
                let publish_handle =
                    Self::create_publish_handle(state.token_block(), self.event_manager.clone());
                let reg_handle = publish_handle.remove_handle();

                // Insert the [RegistrationHandle] into the registry
                self.blocks
                    .insert(sequence_hash, Arc::downgrade(&reg_handle));

                // Update the [BlockState] to [BlockState::Registered]
                let _ = std::mem::replace(block_state, BlockState::Registered(reg_handle));

                Ok(publish_handle)
            }
            BlockState::Registered(registered) => Err(
                BlockRegistationError::BlockAlreadyRegistered(registered.sequence_hash()),
            ),
        }
    }

    pub fn unregister_block(
        &mut self,
        sequence_hash: SequenceHash,
    ) -> Result<(), UnregisterFailure> {
        if let Some(handle) = self.blocks.get(&sequence_hash) {
            if handle.upgrade().is_none() {
                self.blocks.remove(&sequence_hash);
                return Ok(());
            } else {
                return Err(UnregisterFailure(sequence_hash));
            }
        }
        Ok(())
    }

    fn create_publish_handle(
        token_block: &TokenBlock,
        event_manager: Arc<dyn EventManager>,
    ) -> PublishHandle {
        let reg_handle = RegistrationHandle::from_token_block(token_block, event_manager.clone());

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

    #[getter(skip)]
    release_manager: Arc<dyn EventReleaseManager>,
}

impl RegistrationHandle {
    fn from_token_block(
        token_block: &TokenBlock,
        release_manager: Arc<dyn EventReleaseManager>,
    ) -> Self {
        Self {
            block_hash: token_block.block_hash(),
            sequence_hash: token_block.sequence_hash(),
            parent_sequence_hash: token_block.parent_sequence_hash(),
            release_manager,
        }
    }
}

impl std::fmt::Debug for RegistrationHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RegistrationHandle {{ sequence_hash: {}; block_hash: {}; parent_sequence_hash: {:?} }}",
            self.sequence_hash, self.block_hash, self.parent_sequence_hash
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

    use crate::block_manager::events::tests::{EventType, MockEventManager};
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

        let publish_handle =
            BlockRegistry::create_publish_handle(&sequence.blocks()[0], event_manager.clone());

        // no event should have been triggered
        assert!(rx.try_recv().is_err());

        // we shoudl get two events when this is dropped, since we never took ownership of the RegistrationHandle
        drop(publish_handle);

        // the first event should be a Register event
        let events = rx.try_recv().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            EventType::Register(sequence.blocks()[0].sequence_hash())
        );

        // the second event should be a Remove event
        let events = rx.try_recv().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            EventType::Remove(sequence.blocks()[0].sequence_hash())
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

        let publish_handle =
            BlockRegistry::create_publish_handle(block_to_test, event_manager.clone());

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
            EventType::Register(expected_sequence_hash),
            "Expected Register event"
        );

        // Drop the registration handle - this SHOULD trigger the Remove event
        drop(reg_handle);

        let events = rx.try_recv().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            EventType::Remove(expected_sequence_hash),
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

        let publish_handle1 = BlockRegistry::create_publish_handle(block1, event_manager.clone());
        let publish_handle2 = BlockRegistry::create_publish_handle(block2, event_manager.clone());

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
        assert!(events.contains(&EventType::Register(hash1)));
        assert!(events.contains(&EventType::Register(hash2)));

        // no more events immediately after publish
        assert!(rx.try_recv().is_err());

        // Drop registration handles individually - should trigger Remove events
        drop(reg_handle1);
        let events1 = rx.try_recv().unwrap();
        assert_eq!(events1.len(), 1);
        assert_eq!(events1[0], EventType::Remove(hash1));

        drop(reg_handle2);
        let events2 = rx.try_recv().unwrap();
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0], EventType::Remove(hash2));

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

        let publish_handle1 = BlockRegistry::create_publish_handle(block1, event_manager.clone());

        publisher.take_handle(publish_handle1);

        // First publish call
        publisher.publish();
        let events = rx.try_recv().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], EventType::Register(hash1));

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
            EventType::Remove(hash1),
            "Expected Remove event"
        );

        // Second publish call (should do nothing as handles were taken)
        publisher.publish();
        assert!(rx.try_recv().is_err());

        // Drop publisher (should also do nothing)
        drop(publisher);
        assert!(rx.try_recv().is_err());
    }
}
