// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global registry for block deduplication via weak references and sequence hash matching.
//!
//! The registry provides:
//! - Sequence hash → block mapping using weak references
//! - Automatic cleanup when all strong references are dropped
//! - Attachment system for storing arbitrary typed data on registration handles

#![allow(dead_code)]

use crate::v2::{logical::events::EventsManager, utils::tinylfu::FrequencyTracker};

use super::{
    Block, BlockDuplicationPolicy, BlockMetadata, CompleteBlock, DuplicateBlock, MutableBlock,
    PrimaryBlock, RegisteredBlock, SequenceHash, state::Registered,
};

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

/// Type alias for registered block return function
type RegisteredReturnFn<T> = Arc<dyn Fn(Arc<Block<T, Registered>>) + Send + Sync>;

pub type PositionalRadixTree<V> = dynamo_tokens::PositionalRadixTree<V, SequenceHash>;

/// Global registry for managing block registrations.
/// Tracks canonical blocks and provides registration handles.
#[derive(Clone)]
pub struct BlockRegistry {
    prt: Arc<PositionalRadixTree<Weak<BlockRegistrationHandleInner>>>,
    frequency_tracker: Option<Arc<dyn FrequencyTracker<u128>>>,
    event_manager: Option<Arc<EventsManager>>,
}

impl BlockRegistry {
    pub fn new() -> Self {
        Self {
            frequency_tracker: None,
            event_manager: None,
            prt: Arc::new(PositionalRadixTree::new()),
        }
    }

    pub fn with_frequency_tracker(frequency_tracker: Arc<dyn FrequencyTracker<u128>>) -> Self {
        Self {
            frequency_tracker: Some(frequency_tracker),
            event_manager: None,
            prt: Arc::new(PositionalRadixTree::new()),
        }
    }

    /// Creates a new BlockRegistry with an EventsManager for distributed coordination.
    ///
    /// # Arguments
    /// * `event_manager` - Manager for emitting block registration events to the hub
    pub fn with_event_manager(event_manager: Arc<EventsManager>) -> Self {
        Self {
            frequency_tracker: None,
            event_manager: Some(event_manager),
            prt: Arc::new(PositionalRadixTree::new()),
        }
    }

    /// Creates a new BlockRegistry with both frequency tracking and event management.
    ///
    /// # Arguments
    /// * `frequency_tracker` - Tracker for block access frequency
    /// * `event_manager` - Manager for emitting block registration events to the hub
    pub fn with_frequency_and_events(
        frequency_tracker: Arc<dyn FrequencyTracker<u128>>,
        event_manager: Arc<EventsManager>,
    ) -> Self {
        Self {
            frequency_tracker: Some(frequency_tracker),
            event_manager: Some(event_manager),
            prt: Arc::new(PositionalRadixTree::new()),
        }
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

    /// Check presence of sequence hashes for blocks with specific metadata type T.
    /// Returns Vec<(SequenceHash, bool)> where bool indicates if a Block<T, Registered> exists.
    ///
    /// This checks for existence in either active or inactive pools without acquiring ownership.
    /// Does NOT trigger frequency tracking.
    ///
    /// # Example
    /// ```ignore
    /// // Check if blocks exist in G2 pool
    /// let hashes = vec![hash1, hash2, hash3];
    /// let presence = registry.check_presence::<G2>(&hashes);
    /// // presence = [(hash1, true), (hash2, false), (hash3, true)]
    /// ```
    pub fn check_presence<T: BlockMetadata>(
        &self,
        seq_hashes: &[SequenceHash],
    ) -> Vec<(SequenceHash, bool)> {
        seq_hashes
            .iter()
            .map(|&seq_hash| {
                let handle_result = self.match_sequence_hash(seq_hash, false); // touch=false, no frequency tracking
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
    /// Returns Vec<(SequenceHash, bool)> where bool is true if the block exists in ANY of the specified pools.
    ///
    /// This is more efficient than calling check_presence multiple times because it only acquires
    /// the lock once per sequence hash for all type checks.
    ///
    /// Does NOT trigger frequency tracking.
    ///
    /// # Example
    /// ```ignore
    /// use std::any::TypeId;
    ///
    /// // Check if blocks exist in G2 OR G3 pools
    /// let hashes = vec![hash1, hash2, hash3];
    /// let type_ids = [TypeId::of::<G2>(), TypeId::of::<G3>()];
    /// let presence = registry.check_presence_any(&hashes, &type_ids);
    /// // presence = [(hash1, true), (hash2, false), (hash3, true)]
    /// // true if block exists in G2 OR G3
    /// ```
    pub fn check_presence_any(
        &self,
        seq_hashes: &[SequenceHash],
        type_ids: &[TypeId],
    ) -> Vec<(SequenceHash, bool)> {
        seq_hashes
            .iter()
            .map(|&seq_hash| {
                let present = self
                    .match_sequence_hash(seq_hash, false) // touch=false, no frequency tracking
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
    #[inline]
    pub fn register_sequence_hash(&self, seq_hash: SequenceHash) -> BlockRegistrationHandle {
        let map = self.prt.prefix(&seq_hash);
        let mut weak = map.entry(seq_hash).or_default();

        if let Some(inner) = weak.upgrade() {
            return BlockRegistrationHandle { inner };
        }

        let inner = self.create_registration(seq_hash);
        *weak = Arc::downgrade(&inner);
        let handle = BlockRegistrationHandle { inner };

        if let Some(event_manager) = &self.event_manager
            && let Err(e) = event_manager.on_block_registered(&handle)
        {
            tracing::warn!("Failed to register block with event manager: {}", e);
        }
        self.touch(seq_hash);

        handle
    }

    /// Internal method for transferring block registration without triggering frequency tracking.
    /// Used when copying blocks between pools where we don't want to count the transfer as a new access.
    pub(crate) fn transfer_registration(&self, seq_hash: SequenceHash) -> BlockRegistrationHandle {
        let map = self.prt.prefix(&seq_hash);
        let mut weak = map.entry(seq_hash).or_default();

        match weak.upgrade() {
            Some(inner) => BlockRegistrationHandle { inner },
            None => {
                let inner = self.create_registration(seq_hash);
                *weak = Arc::downgrade(&inner);
                BlockRegistrationHandle { inner }
            }
        }
    }

    fn create_registration(&self, seq_hash: SequenceHash) -> Arc<BlockRegistrationHandleInner> {
        Arc::new(BlockRegistrationHandleInner {
            seq_hash,
            registry: Arc::downgrade(&self.prt),
            attachments: Mutex::new(AttachmentStore::new()),
        })
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
            .map(|inner| BlockRegistrationHandle { inner });

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

/// Error types for attachment operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachmentError {
    /// Attempted to attach a type as unique when it's already registered as multiple
    TypeAlreadyRegisteredAsMultiple(TypeId),
    /// Attempted to attach a type as multiple when it's already registered as unique
    TypeAlreadyRegisteredAsUnique(TypeId),
}

impl std::fmt::Display for AttachmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttachmentError::TypeAlreadyRegisteredAsMultiple(type_id) => {
                write!(
                    f,
                    "Type {:?} is already registered as multiple attachment",
                    type_id
                )
            }
            AttachmentError::TypeAlreadyRegisteredAsUnique(type_id) => {
                write!(
                    f,
                    "Type {:?} is already registered as unique attachment",
                    type_id
                )
            }
        }
    }
}

impl std::error::Error for AttachmentError {}

/// Tracks how a type is registered in the attachment system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttachmentMode {
    Unique,
    Multiple,
}

/// Storage for attachments on a BlockRegistrationHandle
#[derive(Debug)]
struct AttachmentStore {
    /// Unique attachments - only one value per TypeId
    unique_attachments: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Multiple attachments - multiple values per TypeId
    multiple_attachments: HashMap<TypeId, Vec<Box<dyn Any + Send + Sync>>>,
    /// Track which types are registered and how
    type_registry: HashMap<TypeId, AttachmentMode>,
    /// Storage for weak block references - separate from generic attachments, keyed by TypeId
    weak_blocks: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Explicit presence tracking for Block<T, Registered> lifecycle
    /// Key is TypeId::of::<T>() - indicates a Block<T, Registered> exists somewhere
    /// (either in active pool as Arc, or in inactive pool as owned)
    presence_markers: HashMap<TypeId, ()>,
}

impl AttachmentStore {
    fn new() -> Self {
        Self {
            unique_attachments: HashMap::new(),
            multiple_attachments: HashMap::new(),
            type_registry: HashMap::new(),
            weak_blocks: HashMap::new(),
            presence_markers: HashMap::new(),
        }
    }
}

/// Typed accessor for attachments of a specific type
pub struct TypedAttachments<'a, T> {
    handle: &'a BlockRegistrationHandle,
    _phantom: PhantomData<T>,
}

/// Handle that represents a block registration in the global registry.
/// This handle is cloneable and can be shared across pools.
#[derive(Clone, Debug)]
pub struct BlockRegistrationHandle {
    inner: Arc<BlockRegistrationHandleInner>,
}

#[derive(Debug)]
struct BlockRegistrationHandleInner {
    /// Sequence hash of the block
    seq_hash: SequenceHash,
    /// Attachments for the block
    attachments: Mutex<AttachmentStore>,
    /// Weak reference to the registry - allows us to remove the block from the registry on drop
    registry: Weak<PositionalRadixTree<Weak<BlockRegistrationHandleInner>>>,
}

impl Drop for BlockRegistrationHandleInner {
    #[inline]
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade()
            && registry
                .prefix(&self.seq_hash)
                .remove(&self.seq_hash)
                .is_none()
        {
            tracing::warn!("Failed to remove block from registry: {:?}", self.seq_hash);
        }
    }
}

impl BlockRegistrationHandle {
    pub fn seq_hash(&self) -> SequenceHash {
        self.inner.seq_hash
    }

    pub(crate) fn is_from_registry(&self, registry: &BlockRegistry) -> bool {
        self.inner
            .registry
            .upgrade()
            .map(|reg| Arc::ptr_eq(&reg, &registry.prt))
            .unwrap_or(false)
    }

    /// Mark that a Block<T, Registered> exists for this sequence hash.
    /// Called when transitioning from Complete to Registered state.
    pub(crate) fn mark_present<T: BlockMetadata>(&self) {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.inner.attachments.lock();
        attachments.presence_markers.insert(type_id, ());
    }

    /// Mark that Block<T, Registered> no longer exists for this sequence hash.
    /// Called when transitioning from Registered to Reset state.
    pub(crate) fn mark_absent<T: BlockMetadata>(&self) {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.inner.attachments.lock();
        attachments.presence_markers.remove(&type_id);
    }

    /// Check if a Block<T, Registered> currently exists for this sequence hash.
    /// Returns true if block exists in active or inactive pool, false otherwise.
    /// This check does not acquire ownership and works regardless of whether the
    /// block is Arc-wrapped (active pool) or owned (inactive pool).
    pub fn has_block<T: BlockMetadata>(&self) -> bool {
        let type_id = TypeId::of::<T>();
        let attachments = self.inner.attachments.lock();
        attachments.presence_markers.contains_key(&type_id)
    }

    /// Check if a Block exists for any of the specified metadata types.
    /// Returns true if a block exists for at least one of the types.
    /// Acquires the lock only once for efficiency.
    ///
    /// # Example
    /// ```ignore
    /// use std::any::TypeId;
    ///
    /// // Check if block exists in G2 OR G3 pools
    /// let type_ids = [TypeId::of::<G2>(), TypeId::of::<G3>()];
    /// if handle.has_any_block(&type_ids) {
    ///     println!("Block exists in at least one pool");
    /// }
    /// ```
    pub fn has_any_block(&self, type_ids: &[TypeId]) -> bool {
        let attachments = self.inner.attachments.lock();
        type_ids
            .iter()
            .any(|type_id| attachments.presence_markers.contains_key(type_id))
    }

    /// Get a typed accessor for attachments of type T
    pub fn get<T: Any + Send + Sync>(&self) -> TypedAttachments<'_, T> {
        TypedAttachments {
            handle: self,
            _phantom: PhantomData,
        }
    }

    /// Attach a unique value of type T to this handle.
    /// Only one value per type is allowed - subsequent calls will replace the previous value.
    /// Returns an error if type T is already registered as multiple attachment.
    pub fn attach_unique<T: Any + Send + Sync>(&self, value: T) -> Result<(), AttachmentError> {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.inner.attachments.lock();

        // Check if this type is already registered as multiple
        if let Some(AttachmentMode::Multiple) = attachments.type_registry.get(&type_id) {
            return Err(AttachmentError::TypeAlreadyRegisteredAsMultiple(type_id));
        }

        // Register/update as unique
        attachments
            .unique_attachments
            .insert(type_id, Box::new(value));
        attachments
            .type_registry
            .insert(type_id, AttachmentMode::Unique);

        Ok(())
    }

    /// Attach a value of type T to this handle.
    /// Multiple values per type are allowed - this will append to existing values.
    /// Returns an error if type T is already registered as unique attachment.
    pub fn attach<T: Any + Send + Sync>(&self, value: T) -> Result<(), AttachmentError> {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.inner.attachments.lock();

        // Check if this type is already registered as unique
        if let Some(AttachmentMode::Unique) = attachments.type_registry.get(&type_id) {
            return Err(AttachmentError::TypeAlreadyRegisteredAsUnique(type_id));
        }

        // Register/update as multiple
        attachments
            .multiple_attachments
            .entry(type_id)
            .or_default()
            .push(Box::new(value));
        attachments
            .type_registry
            .insert(type_id, AttachmentMode::Multiple);

        Ok(())
    }

    pub(crate) fn attach_block<T: BlockMetadata + Sync>(
        &self,
        block: PrimaryBlock<T>,
    ) -> Arc<dyn RegisteredBlock<T>> {
        let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
        let mut attachments = self.inner.attachments.lock();

        #[cfg(debug_assertions)]
        {
            if let Some(weak_any) = attachments.weak_blocks.get(&type_id)
                && let Some(weak) = weak_any.downcast_ref::<WeakBlockEntry<T>>()
            {
                debug_assert!(
                    weak.raw_block.upgrade().is_none(),
                    "Attempted to reattach block when raw block is still alive"
                );
                debug_assert!(
                    weak.primary_block.upgrade().is_none(),
                    "Attempted to reattach block when registered block is still alive"
                );
            }
        }

        let raw_block = Arc::downgrade(block.block.as_ref().unwrap());
        let reg_arc = Arc::new(block);
        let primary_block = Arc::downgrade(&reg_arc);

        attachments.weak_blocks.insert(
            type_id,
            Box::new(WeakBlockEntry {
                raw_block,
                primary_block,
            }),
        );

        reg_arc as Arc<dyn RegisteredBlock<T>>
    }

    pub(crate) fn register_block<T: BlockMetadata + Sync>(
        &self,
        mut block: CompleteBlock<T>,
        duplication_policy: BlockDuplicationPolicy,
        pool_return_fn: RegisteredReturnFn<T>,
    ) -> Arc<dyn RegisteredBlock<T>> {
        assert_eq!(
            block.sequence_hash(),
            self.seq_hash(),
            "Attempted to register block with different sequence hash"
        );

        let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
        let block_id = block.block_id();

        // Take ownership of the inner block
        let inner_block = block.block.take().unwrap();
        let reset_return_fn = block.return_fn.clone();

        // Register the block to get it in Registered state
        let registered_block = inner_block.register(self.clone());

        let mut attachments = self.inner.attachments.lock();

        // Check for existing blocks with same sequence hash
        if let Some(weak_any) = attachments.weak_blocks.get(&type_id)
            && let Some(weak_block) = weak_any.downcast_ref::<WeakBlockEntry<T>>()
        {
            // Try to get the existing primary block
            if let Some(existing_primary) = weak_block.primary_block.upgrade() {
                // Check if same block_id (shouldn't happen)
                if existing_primary.block_id() == block_id {
                    panic!("Attempted to register block with same block_id as existing");
                }

                // Handle duplicate based on policy
                match duplication_policy {
                    BlockDuplicationPolicy::Allow => {
                        // Create DuplicateBlock referencing the primary
                        let duplicate = DuplicateBlock::new(
                            registered_block,
                            existing_primary.clone(),
                            reset_return_fn,
                        );
                        return Arc::new(duplicate);
                    }
                    BlockDuplicationPolicy::Reject => {
                        // CRITICAL: Drop lock before calling reset_return_fn to avoid deadlock
                        drop(attachments);

                        // Return existing primary, discard new block
                        let reset_block = registered_block.reset();
                        let existing = existing_primary.clone();

                        reset_return_fn(reset_block);
                        return existing as Arc<dyn RegisteredBlock<T>>;
                    }
                }
            }

            // Primary couldn't be upgraded but raw block might exist
            // This is an edge case - for now, treat as creating a new primary
        }

        // No existing block or couldn't upgrade - create new primary
        let primary = PrimaryBlock::new(Arc::new(registered_block), pool_return_fn);

        // Store weak references for future lookups
        let primary_arc = Arc::new(primary);
        let raw_block = Arc::downgrade(primary_arc.block.as_ref().unwrap());
        let primary_block = Arc::downgrade(&primary_arc);

        attachments.weak_blocks.insert(
            type_id,
            Box::new(WeakBlockEntry {
                raw_block,
                primary_block,
            }),
        );

        drop(attachments); // Release lock

        primary_arc as Arc<dyn RegisteredBlock<T>>
    }

    #[inline]
    pub(crate) fn try_get_block<T: BlockMetadata + Sync>(
        &self,
        pool_return_fn: RegisteredReturnFn<T>,
    ) -> Option<Arc<dyn RegisteredBlock<T>>> {
        let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
        let attachments = self.inner.attachments.lock();

        let weak_block = attachments
            .weak_blocks
            .get(&type_id)
            .and_then(|weak_any| weak_any.downcast_ref::<WeakBlockEntry<T>>())?;

        if let Some(primary_arc) = weak_block.primary_block.upgrade() {
            drop(attachments);
            return Some(primary_arc as Arc<dyn RegisteredBlock<T>>);
        }

        if let Some(raw_arc) = weak_block.raw_block.upgrade() {
            let primary = PrimaryBlock::new(raw_arc, pool_return_fn);
            let primary_arc = Arc::new(primary);

            let new_weak = Arc::downgrade(&primary_arc);
            let weak_block_mut = WeakBlockEntry {
                raw_block: weak_block.raw_block.clone(),
                primary_block: new_weak,
            };

            drop(attachments);

            let mut attachments = self.inner.attachments.lock();
            attachments
                .weak_blocks
                .insert(type_id, Box::new(weak_block_mut));
            drop(attachments);

            return Some(primary_arc as Arc<dyn RegisteredBlock<T>>);
        }

        None
    }

    pub(crate) fn register_mutable_block<T: BlockMetadata + Sync>(
        &self,
        mutable_block: MutableBlock<T>,
        duplication_policy: BlockDuplicationPolicy,
        pool_return_fn: RegisteredReturnFn<T>,
    ) -> Arc<dyn RegisteredBlock<T>> {
        let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
        let block_id = mutable_block.block_id();

        let (inner_block, reset_return_fn) = mutable_block.into_parts();
        let registered_block = inner_block.register_with_handle(self.clone());

        let mut attachments = self.inner.attachments.lock();

        // Check for existing blocks with same sequence hash
        if let Some(weak_any) = attachments.weak_blocks.get(&type_id)
            && let Some(weak_block) = weak_any.downcast_ref::<WeakBlockEntry<T>>()
        {
            // Try to get the existing primary block
            if let Some(existing_primary) = weak_block.primary_block.upgrade() {
                // Check if same block_id (shouldn't happen)
                if existing_primary.block_id() == block_id {
                    panic!("Attempted to register block with same block_id as existing");
                }

                // Handle duplicate based on policy
                match duplication_policy {
                    BlockDuplicationPolicy::Allow => {
                        let duplicate = DuplicateBlock::new(
                            registered_block,
                            existing_primary.clone(),
                            reset_return_fn,
                        );
                        return Arc::new(duplicate);
                    }
                    BlockDuplicationPolicy::Reject => {
                        // CRITICAL: Drop lock before calling reset_return_fn to avoid deadlock
                        drop(attachments);

                        let reset_block = registered_block.reset();
                        let existing = existing_primary.clone();

                        reset_return_fn(reset_block);
                        return existing as Arc<dyn RegisteredBlock<T>>;
                    }
                }
            }
        }

        // No existing block or couldn't upgrade - create new primary
        let primary = PrimaryBlock::new(Arc::new(registered_block), pool_return_fn);

        // Store weak references for future lookups
        let primary_arc = Arc::new(primary);
        let raw_block = Arc::downgrade(primary_arc.block.as_ref().unwrap());
        let primary_block = Arc::downgrade(&primary_arc);

        attachments.weak_blocks.insert(
            type_id,
            Box::new(WeakBlockEntry {
                raw_block,
                primary_block,
            }),
        );

        drop(attachments); // Release lock

        primary_arc as Arc<dyn RegisteredBlock<T>>
    }
}

impl<'a, T: Any + Send + Sync> TypedAttachments<'a, T> {
    /// Execute a closure with immutable access to the unique attachment of type T.
    pub fn with_unique<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let type_id = TypeId::of::<T>();
        let attachments = self.handle.inner.attachments.lock();
        attachments
            .unique_attachments
            .get(&type_id)?
            .downcast_ref::<T>()
            .map(f)
    }

    /// Execute a closure with mutable access to the unique attachment of type T.
    pub fn with_unique_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.handle.inner.attachments.lock();
        attachments
            .unique_attachments
            .get_mut(&type_id)?
            .downcast_mut::<T>()
            .map(f)
    }

    /// Execute a closure with immutable access to multiple attachments of type T.
    pub fn with_multiple<R>(&self, f: impl FnOnce(&[&T]) -> R) -> R {
        let type_id = TypeId::of::<T>();
        let attachments = self.handle.inner.attachments.lock();

        let multiple_refs: Vec<&T> = attachments
            .multiple_attachments
            .get(&type_id)
            .map(|vec| vec.iter().filter_map(|v| v.downcast_ref::<T>()).collect())
            .unwrap_or_default();

        f(&multiple_refs)
    }

    /// Execute a closure with mutable access to multiple attachments of type T.
    pub fn with_multiple_mut<R>(&self, f: impl FnOnce(&mut [&mut T]) -> R) -> R {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.handle.inner.attachments.lock();

        let mut multiple_refs: Vec<&mut T> = attachments
            .multiple_attachments
            .get_mut(&type_id)
            .map(|vec| {
                vec.iter_mut()
                    .filter_map(|v| v.downcast_mut::<T>())
                    .collect()
            })
            .unwrap_or_default();

        f(&mut multiple_refs)
    }

    /// Execute a closure with immutable access to both unique and multiple attachments of type T.
    pub fn with_all<R>(&self, f: impl FnOnce(Option<&T>, &[&T]) -> R) -> R {
        let type_id = TypeId::of::<T>();
        let attachments = self.handle.inner.attachments.lock();

        let unique = attachments
            .unique_attachments
            .get(&type_id)
            .and_then(|v| v.downcast_ref::<T>());

        let multiple_refs: Vec<&T> = attachments
            .multiple_attachments
            .get(&type_id)
            .map(|vec| vec.iter().filter_map(|v| v.downcast_ref::<T>()).collect())
            .unwrap_or_default();

        f(unique, &multiple_refs)
    }

    /// Execute a closure with mutable access to both unique and multiple attachments of type T.
    pub fn with_all_mut<R>(&self, f: impl FnOnce(Option<&mut T>, &mut [&mut T]) -> R) -> R {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.handle.inner.attachments.lock();

        // Check where this type is registered to avoid double mutable borrow
        match attachments.type_registry.get(&type_id) {
            Some(AttachmentMode::Unique) => {
                // Type is registered as unique - get mutable reference to unique only
                let unique = attachments
                    .unique_attachments
                    .get_mut(&type_id)
                    .and_then(|v| v.downcast_mut::<T>());
                let mut empty_vec: Vec<&mut T> = Vec::new();
                f(unique, &mut empty_vec)
            }
            Some(AttachmentMode::Multiple) => {
                // Type is registered as multiple - get mutable references to multiple only
                let mut multiple_refs: Vec<&mut T> = attachments
                    .multiple_attachments
                    .get_mut(&type_id)
                    .map(|vec| {
                        vec.iter_mut()
                            .filter_map(|v| v.downcast_mut::<T>())
                            .collect()
                    })
                    .unwrap_or_default();
                f(None, &mut multiple_refs)
            }
            None => {
                // Type not registered at all
                let mut empty_vec: Vec<&mut T> = Vec::new();
                f(None, &mut empty_vec)
            }
        }
    }
}

struct WeakBlockEntry<T: BlockMetadata + Sync> {
    /// Weak reference to the raw block
    raw_block: Weak<Block<T, Registered>>,

    /// Weak reference to the registered block
    primary_block: Weak<PrimaryBlock<T>>,
}

#[derive(Debug)]
struct RegistryState {
    canonical_blocks: HashMap<SequenceHash, Weak<BlockRegistrationHandleInner>>,
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::{BlockId, KvbmSequenceHashProvider, utils::tinylfu::TinyLFUTracker};

    use super::*;

    // Test helper types
    #[derive(Debug, Clone, PartialEq)]
    struct TestMetadata {
        value: u32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct MetadataA {
        value: u32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct MetadataB {
        value: u32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct MetadataC {
        value: u32,
    }

    /// Helper to create a token block for testing
    pub fn create_test_token_block(tokens: &[u32]) -> dynamo_tokens::TokenBlock {
        use dynamo_tokens::TokenBlockSequence;
        let seq = TokenBlockSequence::from_slice(tokens, tokens.len() as u32, Some(1337));
        seq.blocks()[0].clone()
    }

    pub fn create_completed_block<T: BlockMetadata + std::fmt::Debug>(
        tokens: &[u32],
        block_id: BlockId,
    ) -> Block<T, crate::v2::logical::blocks::state::Complete> {
        use crate::v2::logical::blocks::{Block, state::Reset};
        let token_block = create_test_token_block(tokens);
        let block: Block<T, Reset> = Block::new(block_id, tokens.len());
        block.complete(token_block).expect("Should complete")
    }

    /// Helper to create and register a block with specific metadata type
    pub fn register_test_block<T: BlockMetadata + std::fmt::Debug>(
        registry: &BlockRegistry,
        block_id: BlockId,
        tokens: &[u32],
    ) -> crate::v2::logical::blocks::Block<T, crate::v2::logical::blocks::state::Registered> {
        let complete = create_completed_block::<T>(tokens, block_id);
        let handle = registry.register_sequence_hash(complete.sequence_hash());
        complete.register(handle)
    }

    #[test]
    fn test_type_tracking_enforcement() {
        let registry = BlockRegistry::new();
        let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
        let handle = registry.register_sequence_hash(seq_hash);

        // Test: attach unique first, then try multiple (should fail)
        handle
            .attach_unique("unique_publisher".to_string())
            .unwrap();

        let result = handle.attach("listener1".to_string());
        assert_eq!(
            result,
            Err(AttachmentError::TypeAlreadyRegisteredAsUnique(
                TypeId::of::<String>()
            ))
        );

        // Test with different types: attach multiple first, then try unique (should fail)
        handle.attach(42i32).unwrap();
        handle.attach(43i32).unwrap();

        let result = handle.attach_unique(44i32);
        assert_eq!(
            result,
            Err(AttachmentError::TypeAlreadyRegisteredAsMultiple(
                TypeId::of::<i32>()
            ))
        );
    }

    #[test]
    fn test_different_types_usage() {
        let registry = BlockRegistry::new();
        let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
        let handle = registry.register_sequence_hash(seq_hash);

        // Define some test types for better demonstration
        #[derive(Debug, Clone, PartialEq)]
        struct EventPublisher(String);

        #[derive(Debug, Clone, PartialEq)]
        struct EventListener(String);

        // Attach unique EventPublisher
        handle
            .attach_unique(EventPublisher("main_publisher".to_string()))
            .unwrap();

        // Attach multiple EventListeners
        handle
            .attach(EventListener("listener1".to_string()))
            .unwrap();
        handle
            .attach(EventListener("listener2".to_string()))
            .unwrap();

        // Retrieve by type using new API
        let publisher = handle.get::<EventPublisher>().with_unique(|p| p.clone());
        assert_eq!(
            publisher,
            Some(EventPublisher("main_publisher".to_string()))
        );

        let listeners = handle
            .get::<EventListener>()
            .with_multiple(|listeners| listeners.iter().map(|l| (*l).clone()).collect::<Vec<_>>());
        assert_eq!(listeners.len(), 2);
        assert!(listeners.contains(&EventListener("listener1".to_string())));
        assert!(listeners.contains(&EventListener("listener2".to_string())));

        // Test with_all for EventListener (should have no unique, only multiple)
        handle.get::<EventListener>().with_all(|unique, multiple| {
            assert_eq!(unique, None);
            assert_eq!(multiple.len(), 2);
        });

        // Attempting to register EventPublisher as multiple should fail
        let result = handle.attach(EventPublisher("another_publisher".to_string()));
        assert_eq!(
            result,
            Err(AttachmentError::TypeAlreadyRegisteredAsUnique(
                TypeId::of::<EventPublisher>()
            ))
        );

        // Attempting to register EventListener as unique should fail
        let result = handle.attach_unique(EventListener("unique_listener".to_string()));
        assert_eq!(
            result,
            Err(AttachmentError::TypeAlreadyRegisteredAsMultiple(
                TypeId::of::<EventListener>()
            ))
        );
    }

    #[test]
    fn test_frequency_tracking() {
        let tracker = Arc::new(TinyLFUTracker::new(100));
        let registry = BlockRegistry::with_frequency_tracker(tracker.clone());

        let block_1 = create_test_token_block(&[1, 2, 3, 4]);
        let seq_hash_1 = block_1.kvbm_sequence_hash();

        assert!(registry.has_frequency_tracking());
        assert_eq!(registry.count(seq_hash_1), 0);

        registry.touch(seq_hash_1);
        assert_eq!(registry.count(seq_hash_1), 1);

        registry.touch(seq_hash_1);
        registry.touch(seq_hash_1);
        assert_eq!(registry.count(seq_hash_1), 3);

        let block_2 = create_test_token_block(&[5, 6, 7, 8]);
        let seq_hash_2 = block_2.kvbm_sequence_hash();

        let _handle1 = registry.register_sequence_hash(seq_hash_2);
        assert_eq!(registry.count(seq_hash_2), 1);

        let _handle2 = registry.match_sequence_hash(seq_hash_2, true);
        assert_eq!(registry.count(seq_hash_2), 2);

        let _handle3 = registry.match_sequence_hash(seq_hash_2, false);
        assert_eq!(registry.count(seq_hash_2), 2);
    }

    #[test]
    fn test_transfer_registration_no_tracking() {
        let tracker = Arc::new(TinyLFUTracker::new(100));
        let registry = BlockRegistry::with_frequency_tracker(tracker.clone());

        let seq_hash_1 = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
        let seq_hash_2 = create_test_token_block(&[5, 6, 7, 8]).kvbm_sequence_hash();

        let _handle1 = registry.transfer_registration(seq_hash_1);
        assert_eq!(registry.count(seq_hash_1), 0);

        let _handle2 = registry.register_sequence_hash(seq_hash_2);
        assert_eq!(registry.count(seq_hash_2), 1);
    }

    #[test]
    fn test_presence_tracking_lifecycle() {
        let registry = BlockRegistry::new();
        let complete_block = create_completed_block::<TestMetadata>(&[1, 2, 3, 4], 42);
        let handle = registry.register_sequence_hash(complete_block.sequence_hash());

        // Initially, no block is present
        assert!(!handle.has_block::<TestMetadata>());

        // Register a block - this should mark it as present
        let registered_block = complete_block.register(handle.clone());

        // Now the block should be present
        assert!(handle.has_block::<TestMetadata>());

        // Reset the block - this should mark it as absent
        let _reset_block = registered_block.reset();

        // Now the block should not be present
        assert!(!handle.has_block::<TestMetadata>());
    }

    #[test]
    fn test_presence_tracking_different_types() {
        let registry = BlockRegistry::new();
        let complete_block = create_completed_block::<TestMetadata>(&[100, 101, 102, 103], 42);
        let handle = registry.register_sequence_hash(complete_block.sequence_hash());

        // Register a block with MetadataA
        let _registered_a = register_test_block::<MetadataA>(&registry, 42, &[100, 101, 102, 103]);

        // MetadataA should be present, but not MetadataB
        assert!(handle.has_block::<MetadataA>());
        assert!(!handle.has_block::<MetadataB>());

        // Now register a block with MetadataB (same seq_hash, different type)
        let _registered_b = register_test_block::<MetadataB>(&registry, 42, &[100, 101, 102, 103]);

        // Both should be present now
        assert!(handle.has_block::<MetadataA>());
        assert!(handle.has_block::<MetadataB>());
    }

    #[test]
    fn test_check_presence_api() {
        let registry = BlockRegistry::new();

        // Register blocks for hashes 100 and 300, but not 200
        let block_100 = register_test_block::<TestMetadata>(&registry, 100, &[0, 1, 2, 3]);
        let block_200 = create_completed_block::<TestMetadata>(&[10, 11, 12, 13], 200);
        let block_300 = register_test_block::<TestMetadata>(&registry, 300, &[20, 21, 22, 23]);

        let hashes = vec![
            block_100.sequence_hash(),
            block_200.sequence_hash(),
            block_300.sequence_hash(),
        ];

        // Check presence using the API
        let presence = registry.check_presence::<TestMetadata>(&hashes);

        assert_eq!(presence.len(), 3);
        assert_eq!(presence[0], (block_100.sequence_hash(), true)); // hash 100 is present
        assert_eq!(presence[1], (block_200.sequence_hash(), false)); // hash 200 is not present
        assert_eq!(presence[2], (block_300.sequence_hash(), true)); // hash 300 is present
    }

    #[test]
    fn test_has_any_block() {
        let registry = BlockRegistry::new();
        let complete_block = create_completed_block::<MetadataB>(&[1, 2, 3, 4], 42);
        let handle = registry.register_sequence_hash(complete_block.sequence_hash());

        // No blocks initially
        let type_ids = [TypeId::of::<MetadataA>(), TypeId::of::<MetadataB>()];
        assert!(!handle.has_any_block(&type_ids));

        // Register a block with MetadataB
        let _registered = complete_block.register(handle.clone());

        // Now should return true because MetadataB is present
        assert!(handle.has_any_block(&type_ids));

        // Check with different types (neither A nor C is present)
        let other_type_ids = [TypeId::of::<MetadataA>(), TypeId::of::<MetadataC>()];
        assert!(!handle.has_any_block(&other_type_ids));

        // Check with just MetadataB
        let b_only = [TypeId::of::<MetadataB>()];
        assert!(handle.has_any_block(&b_only));
    }

    #[test]
    fn test_check_presence_any() {
        let registry = BlockRegistry::new();

        // Create three blocks:
        // hash 100: has MetadataA
        // hash 200: has MetadataB
        // hash 300: has neither
        let block_a = register_test_block::<MetadataA>(&registry, 100, &[10, 11, 12, 13]);
        let block_b = create_completed_block::<MetadataA>(&[1, 2, 3, 4], 200);
        let block_c = register_test_block::<MetadataB>(&registry, 300, &[20, 21, 22, 23]);

        let hashes = vec![
            block_a.sequence_hash(),
            block_b.sequence_hash(),
            block_c.sequence_hash(),
        ];

        // Check presence with both types
        let type_ids = [TypeId::of::<MetadataA>(), TypeId::of::<MetadataB>()];
        let presence = registry.check_presence_any(&hashes, &type_ids);

        assert_eq!(presence.len(), 3);
        assert_eq!(presence[0], (block_a.sequence_hash(), true)); // hash 100 has MetadataA
        assert_eq!(presence[1], (block_b.sequence_hash(), false)); // hash 200 has MetadataB
        assert_eq!(presence[2], (block_c.sequence_hash(), true)); // hash 300 has neither

        // Check with only MetadataA
        let a_only = [TypeId::of::<MetadataA>()];
        let a_presence = registry.check_presence_any(&hashes, &a_only);
        assert_eq!(a_presence[0], (block_a.sequence_hash(), true)); // hash 100 has MetadataA
        assert_eq!(a_presence[1], (block_b.sequence_hash(), false)); // hash 200 doesn't have MetadataA
        assert_eq!(a_presence[2], (block_c.sequence_hash(), false)); // hash 300 doesn't have MetadataA
    }

    #[test]
    fn test_handle_drop_removes_registration() {
        let registry = BlockRegistry::new();
        let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();

        {
            let _handle = registry.register_sequence_hash(seq_hash);
            assert!(registry.is_registered(seq_hash));
            assert_eq!(registry.registered_count(), 1);
        }

        // Handle should be dropped and registration removed
        assert!(!registry.is_registered(seq_hash));
        assert_eq!(registry.registered_count(), 0);
    }

    #[test]
    fn test_multiple_handles_same_sequence() {
        let registry = BlockRegistry::new();
        let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
        let handle1 = registry.register_sequence_hash(seq_hash);
        let handle2 = handle1.clone();

        drop(handle1);

        // Sequence should still be registered because handle2 exists
        assert!(registry.is_registered(seq_hash));
        assert_eq!(registry.registered_count(), 1);

        drop(handle2);

        // Now sequence should be unregistered
        assert!(!registry.is_registered(seq_hash));
        assert_eq!(registry.registered_count(), 0);
    }

    #[test]
    fn test_mutable_access() {
        let registry = BlockRegistry::new();
        let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
        let handle = registry.register_sequence_hash(seq_hash);

        #[derive(Debug, Clone, PartialEq)]
        struct UniqueCounter(i32);

        #[derive(Debug, Clone, PartialEq)]
        struct MultipleCounter(i32);

        impl UniqueCounter {
            fn increment(&mut self) {
                self.0 += 1;
            }
        }

        impl MultipleCounter {
            fn increment(&mut self) {
                self.0 += 1;
            }
        }

        // Test unique mutable access
        handle.attach_unique(UniqueCounter(0)).unwrap();

        handle.get::<UniqueCounter>().with_unique_mut(|counter| {
            counter.increment();
            counter.increment();
        });

        // Verify the change
        let value = handle
            .get::<UniqueCounter>()
            .with_unique(|counter| counter.0);
        assert_eq!(value, Some(2));

        // Test mutable access to multiple (different type)
        handle.attach(MultipleCounter(10)).unwrap();
        handle.attach(MultipleCounter(20)).unwrap();

        handle
            .get::<MultipleCounter>()
            .with_multiple_mut(|counters| {
                for counter in counters {
                    counter.increment();
                }
            });

        // Verify multiple were modified
        let total = handle
            .get::<MultipleCounter>()
            .with_multiple(|counters| counters.iter().map(|c| c.0).sum::<i32>());
        assert_eq!(total, 32); // 11 + 21
    }

    #[test]
    fn test_with_all_mut_unique() {
        let registry = BlockRegistry::new();
        let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
        let handle = registry.register_sequence_hash(seq_hash);

        #[derive(Debug, Clone, PartialEq)]
        struct UniqueValue(i32);

        impl UniqueValue {
            fn increment(&mut self) {
                self.0 += 1;
            }
        }

        // Attach unique value
        handle.attach_unique(UniqueValue(10)).unwrap();

        // Test with_all_mut for unique type
        handle
            .get::<UniqueValue>()
            .with_all_mut(|unique, multiple| {
                assert!(unique.is_some());
                assert_eq!(multiple.len(), 0);
                if let Some(val) = unique {
                    val.increment();
                }
            });

        // Verify the change
        let value = handle.get::<UniqueValue>().with_unique(|v| v.0);
        assert_eq!(value, Some(11));
    }

    #[test]
    fn test_with_all_mut_multiple() {
        let registry = BlockRegistry::new();
        let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
        let handle = registry.register_sequence_hash(seq_hash);

        #[derive(Debug, Clone, PartialEq)]
        struct MultipleValue(i32);

        impl MultipleValue {
            fn increment(&mut self) {
                self.0 += 1;
            }
        }

        // Attach multiple values
        handle.attach(MultipleValue(1)).unwrap();
        handle.attach(MultipleValue(2)).unwrap();

        // Test with_all_mut for multiple type
        handle
            .get::<MultipleValue>()
            .with_all_mut(|unique, multiple| {
                assert!(unique.is_none());
                assert_eq!(multiple.len(), 2);
                for val in multiple {
                    val.increment();
                }
            });

        // Verify the changes
        let total = handle
            .get::<MultipleValue>()
            .with_multiple(|values| values.iter().map(|v| v.0).sum::<i32>());
        assert_eq!(total, 5); // 2 + 3
    }

    /// Tests block resurrection during the pool return transition window.
    ///
    /// # What This Tests
    ///
    /// This test validates the critical ability of `try_get_block()` to "resurrect" a block
    /// while it's transitioning back to the inactive pool. This happens when:
    /// - A `PrimaryBlock` is dropped, triggering its return function
    /// - The return function holds the `Arc<Block<T, Registered>>`
    /// - Another thread calls `try_get_block()` during this window
    /// - The weak reference to the raw block can be upgraded before the Arc is unwrapped
    ///
    /// # Why This Matters
    ///
    /// The registry maintains weak references to blocks even after the `PrimaryBlock` wrapper
    /// is dropped. This allows blocks to be found and reused while they're still in memory,
    /// even if they're transitioning to the inactive pool. This is a key optimization:
    /// - Avoids unnecessary pool insertion/removal cycles
    /// - Reduces lock contention on the pool
    /// - Enables lock-free block reuse in high-concurrency scenarios
    ///
    /// # Test Strategy
    ///
    /// The test uses two barriers to create a deterministic interleaving:
    ///
    /// 1. **Drop Thread** drops the `PrimaryBlock`, triggering a custom return function that:
    ///    - Receives the `Arc<Block<T, Registered>>`
    ///    - Signals readiness at barrier1
    ///    - Waits at barrier2 for the upgrade to complete
    ///    - Attempts to return to pool (fails if Arc was upgraded)
    ///
    /// 2. **Upgrade Thread** waits for the Arc to be held by return function, then:
    ///    - Calls `try_get_block()` which upgrades the weak reference
    ///    - Creates a new `PrimaryBlock` wrapping the same Arc
    ///    - Signals completion at barrier2
    ///
    /// # Expected Outcome
    ///
    /// - `try_get_block()` successfully upgrades and returns a new `PrimaryBlock`
    /// - The Arc refcount becomes ≥ 2 (held by both return fn and new PrimaryBlock)
    /// - `Arc::try_unwrap()` in the inactive pool's return function fails
    /// - The block never makes it into the inactive pool
    /// - The block remains accessible through the upgraded reference
    #[test]
    fn test_concurrent_try_get_block_and_drop() {
        use crate::v2::logical::pools::backends::{FifoReusePolicy, HashMapBackend};
        use crate::v2::logical::pools::*;
        use std::sync::Barrier;
        use std::thread;

        let registry = BlockRegistry::new();

        let tokens = vec![1u32, 2, 3, 4];
        let token_block = create_test_token_block(&tokens);
        let seq_hash = token_block.kvbm_sequence_hash();
        let handle = registry.register_sequence_hash(seq_hash);

        let reset_blocks: Vec<_> = (0..10)
            .map(|i| crate::v2::logical::blocks::Block::new(i, 4))
            .collect();
        let reset_pool = ResetPool::new(reset_blocks, 4);
        let reuse_policy = Box::new(FifoReusePolicy::new());
        let backend = Box::new(HashMapBackend::new(reuse_policy));
        let registered_pool = InactivePool::new(backend, &reset_pool);

        // Create barriers for synchronization
        // barrier1: signals that return function has the Arc
        // barrier2: signals that upgrade has completed
        let barrier1 = Arc::new(Barrier::new(2));
        let barrier2 = Arc::new(Barrier::new(2));
        let barrier1_clone = barrier1.clone();
        let barrier2_clone = barrier2.clone();

        // Create custom return function that holds the Arc at barriers
        let registered_pool_clone = registered_pool.clone();
        let pool_return_fn = Arc::new(
            move |block: Arc<crate::v2::logical::blocks::Block<TestMetadata, Registered>>| {
                // Signal that we have the Arc
                barrier1_clone.wait();
                // Wait for upgrade to complete
                barrier2_clone.wait();
                // Now try to return - this will fail if try_get_block upgraded the Arc
                (registered_pool_clone.return_fn())(block);
            },
        )
            as Arc<
                dyn Fn(Arc<crate::v2::logical::blocks::Block<TestMetadata, Registered>>)
                    + Send
                    + Sync,
            >;

        let complete_block = crate::v2::logical::blocks::Block::new(0, 4)
            .complete(token_block)
            .expect("Block size should match");

        let immutable_block = handle.register_block(
            CompleteBlock {
                block: Some(complete_block),
                return_fn: reset_pool.return_fn(),
            },
            BlockDuplicationPolicy::Allow,
            pool_return_fn.clone(),
        );

        let handle_clone = handle.clone();
        let real_return_fn = registered_pool.return_fn();
        let registered_pool_clone2 = registered_pool.clone();

        let upgrade_thread = thread::spawn(move || {
            // Wait for return function to receive the Arc
            barrier1.wait();
            // Try to upgrade - should succeed because Arc is held by return fn
            // Use the real return function (not the custom one) to avoid deadlock
            let result = handle_clone.try_get_block::<TestMetadata>(real_return_fn);
            // Signal that upgrade is complete
            barrier2.wait();
            result
        });

        let drop_thread = thread::spawn(move || {
            // Drop the block, which triggers the return function that waits at barriers
            drop(immutable_block);
        });

        // Get the upgraded block from try_get_block
        let upgraded_block = upgrade_thread.join().unwrap();

        drop_thread.join().unwrap();

        // Verify that try_get_block succeeded
        assert!(
            upgraded_block.is_some(),
            "Should successfully upgrade the weak reference to Arc<Block<T, Registered>>"
        );

        // Hold the block to keep Arc refcount > 1
        let _held_block = upgraded_block;

        // Verify that the block never made it to the inactive pool
        // because Arc::try_unwrap failed due to refcount >= 2
        assert_eq!(
            registered_pool_clone2.len(),
            0,
            "Block should not be in inactive pool because Arc refcount was >= 2"
        );
    }
}
