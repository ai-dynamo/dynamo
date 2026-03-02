// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block registration handle and its inner implementation.

use super::attachments::{AttachmentError, AttachmentStore, TypedAttachments};
use super::delegate::PresenceDelegate;
use super::{BlockRegistry, PositionalRadixTree};

use crate::blocks::{BlockMetadata, SequenceHash};
use crate::BlockId;

use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

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
    /// Delegates notified on presence transitions (immutable, shared from registry)
    pub(crate) presence_delegates: Arc<[Arc<dyn PresenceDelegate>]>,
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
            .field(
                "presence_delegates",
                &format!("[{} delegates]", self.presence_delegates.len()),
            )
            .finish()
    }
}

impl BlockRegistrationHandleInner {
    pub(super) fn new(
        seq_hash: SequenceHash,
        registry: Weak<PositionalRadixTree<Weak<BlockRegistrationHandleInner>>>,
        presence_delegates: Arc<[Arc<dyn PresenceDelegate>]>,
    ) -> Self {
        Self {
            seq_hash,
            attachments: Mutex::new(AttachmentStore::new()),
            touch_callbacks: Mutex::new(Vec::new()),
            registry,
            presence_delegates,
        }
    }
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

    /// Mark that a Block<T, Registered> exists for this sequence hash.
    /// Called when transitioning from Complete to Registered state.
    /// Fires presence delegates outside the attachments lock.
    pub(crate) fn mark_present<T: BlockMetadata>(&self, block_id: BlockId) {
        let type_id = TypeId::of::<T>();
        {
            let mut attachments = self.inner.attachments.lock();
            attachments.presence_markers.insert(type_id, ());
        }
        // Fire delegates outside the lock
        if !self.inner.presence_delegates.is_empty() {
            let seq_hash = self.inner.seq_hash;
            for delegate in self.inner.presence_delegates.iter() {
                delegate.on_present(seq_hash, block_id, type_id, self);
            }
        }
    }

    /// Mark that Block<T, Registered> no longer exists for this sequence hash.
    /// Called when transitioning from Registered to Reset state.
    /// Fires presence delegates outside the attachments lock.
    pub(crate) fn mark_absent<T: BlockMetadata>(&self, block_id: BlockId) {
        let type_id = TypeId::of::<T>();
        {
            let mut attachments = self.inner.attachments.lock();
            attachments.presence_markers.remove(&type_id);
        }
        // Fire delegates outside the lock
        if !self.inner.presence_delegates.is_empty() {
            let seq_hash = self.inner.seq_hash;
            for delegate in self.inner.presence_delegates.iter() {
                delegate.on_absent(seq_hash, block_id, type_id, self);
            }
        }
    }

    /// Check if a Block<T, Registered> currently exists for this sequence hash.
    /// Returns true if block exists in active or inactive pool, false otherwise.
    pub fn has_block<T: BlockMetadata>(&self) -> bool {
        let type_id = TypeId::of::<T>();
        let attachments = self.inner.attachments.lock();
        attachments.presence_markers.contains_key(&type_id)
    }

    /// Check if a Block exists for any of the specified metadata types.
    /// Returns true if a block exists for at least one of the types.
    /// Acquires the lock only once for efficiency.
    pub fn has_any_block(&self, type_ids: &[TypeId]) -> bool {
        let attachments = self.inner.attachments.lock();
        type_ids
            .iter()
            .any(|type_id| attachments.presence_markers.contains_key(type_id))
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
