// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block registration handle and its inner implementation.

use super::attachments::{AttachmentError, AttachmentStore, TypedAttachments};
use super::{BlockRegistry, PositionalRadixTree};

use crate::blocks::{BlockMetadata, SequenceHash};

use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::sync::{Arc, Weak};

// Under `#[cfg(test)]`, swap in `tracing-mutex`'s parking_lot wrapper
// so the test suite enforces the documented `attachments → store`
// lock-acquisition ordering at runtime via a global DAG. Identical API
// in release builds; zero cost.
#[cfg(not(test))]
use parking_lot::Mutex;
#[cfg(test)]
use tracing_mutex::parkinglot::Mutex;

/// Handle that represents a block registration in the global registry.
/// This handle is cloneable and can be shared across pools.
#[derive(Clone, Debug)]
pub struct BlockRegistrationHandle {
    pub(crate) inner: Arc<BlockRegistrationHandleInner>,
}

/// Type alias for touch callback functions.
type TouchCallback = Arc<dyn Fn(SequenceHash) + Send + Sync>;

pub(crate) struct BlockRegistrationHandleInner {
    /// Sequence hash of the block
    seq_hash: SequenceHash,
    /// Attachments for the block
    pub(crate) attachments: Mutex<AttachmentStore>,
    /// Callbacks invoked when this handle is touched
    touch_callbacks: Mutex<Vec<TouchCallback>>,
    /// Weak reference to the registry - allows us to remove the block from the registry on drop
    registry: Weak<PositionalRadixTree<Weak<BlockRegistrationHandleInner>>>,
}

impl std::fmt::Debug for BlockRegistrationHandleInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockRegistrationHandleInner")
            .field("seq_hash", &self.seq_hash)
            .field("attachments", &self.attachments)
            .field(
                "touch_callbacks",
                &format!("[{} callbacks]", self.touch_callbacks.lock().len()),
            )
            .finish()
    }
}

impl BlockRegistrationHandleInner {
    pub(super) fn new(
        seq_hash: SequenceHash,
        registry: Weak<PositionalRadixTree<Weak<BlockRegistrationHandleInner>>>,
    ) -> Self {
        Self {
            seq_hash,
            attachments: Mutex::new(AttachmentStore::new()),
            touch_callbacks: Mutex::new(Vec::new()),
            registry,
        }
    }
}

impl Drop for BlockRegistrationHandleInner {
    #[inline]
    fn drop(&mut self) {
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        // The position-level write lock held by `prefix()` for the lifetime of
        // `map` serializes us against concurrent `register_sequence_hash` and
        // `transfer_registration` on this `seq_hash`. Without that lock, a
        // concurrent registration could replace the entry's `Weak` between our
        // strong-count-drop and this body running, and an unconditional remove
        // would silently delete the newer registration's entry.
        //
        // Compare the stored `Weak`'s pointer to `self`: `Weak::<T>::as_ptr()`
        // for sized `T` returns the same pointer as `&T as *const T`, and
        // during `drop_in_place` the inner allocation is still live (the
        // implicit weak from the strong refcount is released after `Drop`
        // returns). Only remove if the entry still points to us.
        let map = registry.prefix(&self.seq_hash);
        let should_remove = match map.get(&self.seq_hash) {
            Some(weak_ref) => std::ptr::eq(weak_ref.as_ptr(), self as *const Self),
            None => {
                debug_assert!(
                    false,
                    "registry entry vanished while a strong ref was alive: {:?}",
                    self.seq_hash
                );
                false
            }
        };
        if should_remove {
            map.remove(&self.seq_hash);
        }
    }
}

impl BlockRegistrationHandle {
    pub(crate) fn from_inner(inner: Arc<BlockRegistrationHandleInner>) -> Self {
        Self { inner }
    }

    pub fn seq_hash(&self) -> SequenceHash {
        self.inner.seq_hash
    }

    pub fn is_from_registry(&self, registry: &BlockRegistry) -> bool {
        self.inner
            .registry
            .upgrade()
            .map(|reg| Arc::ptr_eq(&reg, &registry.prt))
            .unwrap_or(false)
    }

    /// Increment the refcounted presence marker for tier `T`. Each
    /// presence-bearing slot transition (`Staged → Primary`,
    /// `Staged → Duplicate`) calls this exactly once.
    pub(crate) fn mark_present<T: BlockMetadata>(&self) {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.inner.attachments.lock();
        *attachments.presence_markers.entry(type_id).or_insert(0) += 1;
    }

    /// Decrement the refcounted presence marker for tier `T`. Each
    /// presence-removing slot transition (`Inactive → Mutable` via
    /// eviction, `Duplicate → Reset` via last-duplicate drop) calls this
    /// exactly once. The entry is removed on reaching zero.
    pub(crate) fn mark_absent<T: BlockMetadata>(&self) {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.inner.attachments.lock();
        match attachments.presence_markers.get_mut(&type_id) {
            Some(count) => {
                debug_assert!(*count > 0, "mark_absent on zero-count presence marker");
                *count -= 1;
                if *count == 0 {
                    attachments.presence_markers.remove(&type_id);
                }
            }
            None => debug_assert!(false, "mark_absent with no presence marker present"),
        }
    }

    /// Returns `true` if at least one `Block<T, Registered>` exists for
    /// this sequence hash (i.e., the refcount is > 0).
    ///
    /// This is a **refcounted shadow** of authoritative `BlockStore<T>`
    /// state, not a linearizable snapshot. The store is updated under
    /// its own mutex; this counter is incremented/decremented in a
    /// separate critical section that runs after the store lock is
    /// released. In steady state the shadow agrees with the store; while
    /// a registration, eviction, or duplicate drop is mid-flight it can
    /// briefly report the pre-update value. Callers who need the exact
    /// current state should go through `BlockManager::match_blocks`
    /// (which consults the store directly).
    pub fn has_block<T: BlockMetadata>(&self) -> bool {
        let type_id = TypeId::of::<T>();
        let attachments = self.inner.attachments.lock();
        attachments
            .presence_markers
            .get(&type_id)
            .copied()
            .unwrap_or(0)
            > 0
    }

    /// Returns `true` if a block exists for at least one of the
    /// specified metadata-tier `TypeId`s.
    pub fn has_any_block(&self, type_ids: &[TypeId]) -> bool {
        let attachments = self.inner.attachments.lock();
        type_ids.iter().any(|type_id| {
            attachments
                .presence_markers
                .get(type_id)
                .copied()
                .unwrap_or(0)
                > 0
        })
    }

    /// Register a callback to be invoked when this handle is touched.
    pub fn on_touch(&self, callback: Arc<dyn Fn(SequenceHash) + Send + Sync>) {
        self.inner.touch_callbacks.lock().push(callback);
    }

    /// Fire all registered touch callbacks with this handle's sequence hash.
    pub fn touch(&self) {
        let callbacks: Vec<_> = self.inner.touch_callbacks.lock().clone();
        let seq_hash = self.inner.seq_hash;
        for cb in &callbacks {
            cb(seq_hash);
        }
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

        if let Some(super::attachments::AttachmentMode::Multiple) =
            attachments.type_registry.get(&type_id)
        {
            return Err(AttachmentError::TypeAlreadyRegisteredAsMultiple(type_id));
        }

        attachments
            .unique_attachments
            .insert(type_id, Box::new(value));
        attachments
            .type_registry
            .insert(type_id, super::attachments::AttachmentMode::Unique);

        Ok(())
    }

    /// Attach a value of type T to this handle.
    /// Multiple values per type are allowed - this will append to existing values.
    /// Returns an error if type T is already registered as unique attachment.
    pub fn attach<T: Any + Send + Sync>(&self, value: T) -> Result<(), AttachmentError> {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.inner.attachments.lock();

        if let Some(super::attachments::AttachmentMode::Unique) =
            attachments.type_registry.get(&type_id)
        {
            return Err(AttachmentError::TypeAlreadyRegisteredAsUnique(type_id));
        }

        attachments
            .multiple_attachments
            .entry(type_id)
            .or_default()
            .push(Box::new(value));
        attachments
            .type_registry
            .insert(type_id, super::attachments::AttachmentMode::Multiple);

        Ok(())
    }
}
