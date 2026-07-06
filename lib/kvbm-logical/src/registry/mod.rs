// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global registry for block deduplication via weak references and sequence hash matching.
//!
//! The [`BlockRegistry`] is the central coordination point for block deduplication in the
//! KVBM system. It maps sequence hashes to registration handles using a
//! [`dynamo_tokens::PositionalRadixTree`], enabling efficient prefix-based lookups.
//!
//! # Architecture
//!
//! ```text
//! BlockRegistry
//!   └── PositionalRadixTree<Weak<BlockRegistrationHandleInner>>
//!         ├── seq_hash_1 → Handle → AttachmentStore (presence markers, weak refs, typed data)
//!         ├── seq_hash_2 → Handle → AttachmentStore
//!         └── ...
//! ```
//!
//! - **Handle**: One per sequence hash. Ties blocks across all pool tiers (active, inactive).
//! - **Attachments**: Arbitrary typed data stored on handles (unique or multiple per type).
//! - **Presence markers**: Track which `Block<T, Registered>` exist for a given handle.
//! - **Weak references**: Enable block resurrection during pool transitions.
//!
//! # Future directions
//!
//! - Delegate pattern to decouple EventsManager from BlockRegistry
//! - Cross-pool touch tracking
//! - RAII attachment guards

mod attachments;
mod handle;
mod registration;

#[cfg(test)]
pub(crate) mod tests;

// Re-export public types
pub use attachments::{AttachmentError, TypedAttachments};
pub use handle::BlockRegistrationHandle;

use crate::{events::EventsManager, tinylfu::FrequencyTracker};

use crate::blocks::SequenceHash;

use std::sync::{Arc, Weak};

use handle::BlockRegistrationHandleInner;

pub(crate) type PositionalRadixTree<V> = dynamo_tokens::PositionalRadixTree<V, SequenceHash>;

/// Builder for [`BlockRegistry`].
///
/// # Example
///
/// ```ignore
/// // Simple registry with no tracking
/// let registry = BlockRegistry::builder().build();
///
/// // With frequency tracking
/// let registry = BlockRegistry::builder()
///     .frequency_tracker(tracker)
///     .build();
///
/// // With both frequency tracking and event management
/// let registry = BlockRegistry::builder()
///     .frequency_tracker(tracker)
///     .event_manager(events_manager)
///     .build();
/// ```
#[derive(Default)]
pub struct BlockRegistryBuilder {
    frequency_tracker: Option<Arc<dyn FrequencyTracker<u128>>>,
    event_manager: Option<Arc<EventsManager>>,
}

impl BlockRegistryBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the frequency tracker for block access tracking.
    pub fn frequency_tracker(mut self, tracker: Arc<dyn FrequencyTracker<u128>>) -> Self {
        self.frequency_tracker = Some(tracker);
        self
    }

    /// Sets the events manager for distributed coordination.
    // TODO(delegate): Replace direct EventsManager coupling with a delegate/observer pattern.
    pub fn event_manager(mut self, manager: Arc<EventsManager>) -> Self {
        self.event_manager = Some(manager);
        self
    }

    /// Builds the BlockRegistry.
    pub fn build(self) -> BlockRegistry {
        BlockRegistry {
            frequency_tracker: self.frequency_tracker,
            event_manager: self.event_manager,
            prt: Arc::new(PositionalRadixTree::new()),
        }
    }
}

/// Global registry for managing block registrations.
/// Tracks canonical blocks and provides registration handles.
#[derive(Clone)]
pub struct BlockRegistry {
    pub(crate) prt: Arc<PositionalRadixTree<Weak<BlockRegistrationHandleInner>>>,
    frequency_tracker: Option<Arc<dyn FrequencyTracker<u128>>>,
    // TODO(delegate): Replace direct EventsManager field with a delegate/observer trait.
    event_manager: Option<Arc<EventsManager>>,
}

impl BlockRegistry {
    /// Creates a new builder for BlockRegistry.
    pub fn builder() -> BlockRegistryBuilder {
        BlockRegistryBuilder::new()
    }

    /// Creates a new BlockRegistry with no tracking.
    pub fn new() -> Self {
        Self::builder().build()
    }

    pub fn has_frequency_tracking(&self) -> bool {
        self.frequency_tracker.is_some()
    }

    pub fn touch(&self, seq_hash: SequenceHash) {
        if let Some(tracker) = &self.frequency_tracker {
            tracker.touch(seq_hash.as_u128());
        }
    }

    pub fn count(&self, seq_hash: SequenceHash) -> u32 {
        if let Some(tracker) = &self.frequency_tracker {
            tracker.count(seq_hash.as_u128())
        } else {
            0
        }
    }

    /// Check presence of sequence hashes for blocks with specific metadata type `T`.
    /// Returns `Vec<(SequenceHash, bool)>` where `bool` indicates whether a
    /// `Block<T, Registered>` is currently believed to exist somewhere in
    /// the active or inactive pool for this tier.
    ///
    /// # Consistency model
    ///
    /// This view is a **refcounted shadow** of the authoritative
    /// `BlockStore<T>` state, *not* a linearizable snapshot. The store is
    /// the single source of truth for slot state and is updated under its
    /// own mutex; the registry-side presence count is then incremented
    /// (`mark_present`) or decremented (`mark_absent`) in a separate
    /// critical section that runs *after* the store mutex has been
    /// released — see `pools/store.rs::register_completed_block`,
    /// `allocate_atomic`, `release_duplicate`, and `drain_inactive_to_mutable`.
    ///
    /// In steady state and after every operation has fully completed, the
    /// shadow count agrees with the authoritative state because the
    /// per-slot increments and decrements commute (refcounted). However,
    /// while a registration, eviction, or duplicate drop is mid-flight,
    /// `check_presence` can briefly report the pre-update value. Callers
    /// who need the exact current state must instead acquire a strong
    /// reference via `BlockManager::match_blocks` /
    /// `BlockManager::scan_matches` (which consult the store directly) or
    /// otherwise serialize against the mutating operation.
    ///
    /// Does NOT trigger frequency tracking.
    pub fn check_presence<T: crate::blocks::BlockMetadata>(
        &self,
        seq_hashes: &[SequenceHash],
    ) -> Vec<(SequenceHash, bool)> {
        seq_hashes
            .iter()
            .map(|&seq_hash| {
                let handle_result = self.match_sequence_hash(seq_hash, false);
                let present = handle_result
                    .as_ref()
                    .map(|handle| handle.has_block::<T>())
                    .unwrap_or(false);

                tracing::debug!(
                    ?seq_hash,
                    type_name = std::any::type_name::<T>(),
                    handle_found = handle_result.is_some(),
                    present,
                    "check_presence result"
                );

                (seq_hash, present)
            })
            .collect()
    }

    /// Check presence of sequence hashes for blocks with any of the specified metadata types.
    /// Returns `Vec<(SequenceHash, bool)>` where `bool` is true if a block
    /// exists for at least one of the supplied tier `TypeId`s.
    ///
    /// Same consistency caveats as [`check_presence`]: this is a
    /// refcounted shadow of authoritative store state, not a linearizable
    /// snapshot. May briefly disagree with the store mid-mutation.
    ///
    /// Does NOT trigger frequency tracking.
    pub fn check_presence_any(
        &self,
        seq_hashes: &[SequenceHash],
        type_ids: &[std::any::TypeId],
    ) -> Vec<(SequenceHash, bool)> {
        seq_hashes
            .iter()
            .map(|&seq_hash| {
                let present = self
                    .match_sequence_hash(seq_hash, false)
                    .map(|handle| handle.has_any_block(type_ids))
                    .unwrap_or(false);
                (seq_hash, present)
            })
            .collect()
    }

    /// Register a sequence hash and get a registration handle.
    /// If the sequence is already registered, returns the existing handle.
    /// Otherwise, creates a new canonical registration.
    /// This method triggers frequency tracking.
    // TODO(delegate): This is where `on_block_registered` is called. Future delegate
    // pattern should replace the direct EventsManager call here.
    #[inline]
    pub fn register_sequence_hash(&self, seq_hash: SequenceHash) -> BlockRegistrationHandle {
        let map = self.prt.prefix(&seq_hash);
        let mut weak = map.entry(seq_hash).or_default();

        if let Some(inner) = weak.upgrade() {
            return BlockRegistrationHandle::from_inner(inner);
        }

        let inner = self.create_registration(seq_hash);
        *weak = Arc::downgrade(&inner);
        let handle = BlockRegistrationHandle::from_inner(inner);

        if let Some(event_manager) = &self.event_manager
            && let Err(e) = event_manager.on_block_registered(&handle)
        {
            tracing::warn!("Failed to register block with event manager: {}", e);
        }
        self.touch(seq_hash);

        handle
    }

    /// Register a batch of sequence hashes in input order.
    pub fn register_sequence_hashes(
        &self,
        seq_hashes: impl IntoIterator<Item = SequenceHash>,
    ) -> Vec<BlockRegistrationHandle> {
        let seq_hashes: Vec<_> = seq_hashes.into_iter().collect();
        if seq_hashes.is_empty() {
            return Vec::new();
        }

        let positions_are_monotonic = seq_hashes.is_sorted_by_key(|seq_hash| seq_hash.position());
        let registered = if positions_are_monotonic {
            self.register_monotonic_sequence_hashes(&seq_hashes)
        } else {
            self.register_grouped_sequence_hashes(&seq_hashes)
        };

        registered
            .into_iter()
            .map(|(handle, is_new)| {
                if is_new {
                    if let Some(event_manager) = &self.event_manager
                        && let Err(e) = event_manager.on_block_registered(&handle)
                    {
                        tracing::warn!("Failed to register block with event manager: {}", e);
                    }
                    self.touch(handle.seq_hash());
                }
                handle
            })
            .collect()
    }

    /// Fast path for the normal sequence-registration shape: non-decreasing
    /// block positions. Equal positions are adjacent, so each position needs
    /// only one radix-prefix guard and results can be appended in input order.
    fn register_monotonic_sequence_hashes(
        &self,
        seq_hashes: &[SequenceHash],
    ) -> Vec<(BlockRegistrationHandle, bool)> {
        let mut registered = Vec::with_capacity(seq_hashes.len());
        self.register_ordered_sequence_hashes(
            seq_hashes.iter().copied().map(|seq_hash| ((), seq_hash)),
            |(), handle, is_new| registered.push((handle, is_new)),
        );
        registered
    }

    /// Fallback for callers that supply positions out of order. A flat index
    /// vector replaces the previous `BTreeMap` of per-position vectors. It is
    /// sorted by `(position, original_index)` so registration stays grouped by
    /// radix prefix without changing which duplicate occurrence is considered
    /// new. Results are restored to input order before observers run.
    fn register_grouped_sequence_hashes(
        &self,
        seq_hashes: &[SequenceHash],
    ) -> Vec<(BlockRegistrationHandle, bool)> {
        let mut ordered: Vec<_> = seq_hashes.iter().copied().enumerate().collect();
        ordered.sort_unstable_by_key(|(index, seq_hash)| (seq_hash.position(), *index));

        let mut registered = Vec::with_capacity(seq_hashes.len());
        self.register_ordered_sequence_hashes(ordered.iter().copied(), |index, handle, is_new| {
            registered.push((index, handle, is_new))
        });

        registered.sort_unstable_by_key(|(index, _, _)| *index);
        registered
            .into_iter()
            .map(|(_, handle, is_new)| (handle, is_new))
            .collect()
    }

    /// Register position-ordered entries, holding exactly one outer radix
    /// guard for each adjacent same-position group. `record` only stages the
    /// result; user-visible callbacks and frequency touches run after this
    /// method returns and releases every guard.
    fn register_ordered_sequence_hashes<K>(
        &self,
        entries: impl IntoIterator<Item = (K, SequenceHash)>,
        mut record: impl FnMut(K, BlockRegistrationHandle, bool),
    ) {
        let mut entries = entries.into_iter().peekable();
        while let Some(first) = entries.next() {
            let position = first.1.position();
            let map = self.prt.prefix(&first.1);
            let mut current = first;

            loop {
                let (key, seq_hash) = current;
                debug_assert_eq!(seq_hash.position(), position);
                let mut weak = map.entry(seq_hash).or_default();
                let (inner, is_new) = match weak.upgrade() {
                    Some(inner) => (inner, false),
                    None => {
                        let inner = self.create_registration(seq_hash);
                        *weak = Arc::downgrade(&inner);
                        (inner, true)
                    }
                };
                record(key, BlockRegistrationHandle::from_inner(inner), is_new);

                let Some(next) = entries.next_if(|(_, seq_hash)| seq_hash.position() == position)
                else {
                    break;
                };
                current = next;
            }
        }
    }

    /// Internal method for transferring block registration without triggering frequency tracking.
    /// Used when copying blocks between pools where we don't want to count the transfer as a new access.
    #[allow(dead_code)]
    pub(crate) fn transfer_registration(&self, seq_hash: SequenceHash) -> BlockRegistrationHandle {
        let map = self.prt.prefix(&seq_hash);
        let mut weak = map.entry(seq_hash).or_default();

        match weak.upgrade() {
            Some(inner) => BlockRegistrationHandle::from_inner(inner),
            None => {
                let inner = self.create_registration(seq_hash);
                *weak = Arc::downgrade(&inner);
                BlockRegistrationHandle::from_inner(inner)
            }
        }
    }

    fn create_registration(&self, seq_hash: SequenceHash) -> Arc<BlockRegistrationHandleInner> {
        Arc::new(BlockRegistrationHandleInner::new(
            seq_hash,
            Arc::downgrade(&self.prt),
        ))
    }

    /// Match a sequence hash and return a registration handle.
    /// This method triggers frequency tracking.
    #[inline]
    pub fn match_sequence_hash(
        &self,
        seq_hash: SequenceHash,
        touch: bool,
    ) -> Option<BlockRegistrationHandle> {
        let result = self
            .prt
            .prefix(&seq_hash)
            .get(&seq_hash)
            .and_then(|weak| weak.upgrade())
            .map(BlockRegistrationHandle::from_inner);

        if result.is_some() && touch {
            self.touch(seq_hash);
        }

        result
    }

    /// Check if a sequence is currently registered (has a canonical handle).
    #[inline]
    pub fn is_registered(&self, seq_hash: SequenceHash) -> bool {
        self.prt
            .prefix(&seq_hash)
            .get(&seq_hash)
            .map(|weak| weak.strong_count() > 0)
            .unwrap_or(false)
    }

    /// Get the current number of registered blocks.
    pub fn registered_count(&self) -> usize {
        self.prt.len()
    }

    /// Get the frequency tracker if frequency tracking is enabled.
    pub fn frequency_tracker(&self) -> Option<Arc<dyn FrequencyTracker<u128>>> {
        self.frequency_tracker.clone()
    }
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::new()
    }
}
