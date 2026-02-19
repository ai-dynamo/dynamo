// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attachment system for storing arbitrary typed data on registration handles.

use super::handle::BlockRegistrationHandle;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;

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
pub(super) enum AttachmentMode {
    Unique,
    Multiple,
}

/// Storage for attachments on a BlockRegistrationHandle
#[derive(Debug)]
pub(crate) struct AttachmentStore {
    /// Unique attachments - only one value per TypeId
    pub(super) unique_attachments: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Multiple attachments - multiple values per TypeId
    pub(super) multiple_attachments: HashMap<TypeId, Vec<Box<dyn Any + Send + Sync>>>,
    /// Track which types are registered and how
    pub(super) type_registry: HashMap<TypeId, AttachmentMode>,
    /// Storage for weak block references - separate from generic attachments, keyed by TypeId
    pub(crate) weak_blocks: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Explicit presence tracking for Block<T, Registered> lifecycle
    /// Key is TypeId::of::<T>() - indicates a Block<T, Registered> exists somewhere
    pub(super) presence_markers: HashMap<TypeId, ()>,
}

impl AttachmentStore {
    pub(super) fn new() -> Self {
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
    pub(super) handle: &'a BlockRegistrationHandle,
    pub(super) _phantom: PhantomData<T>,
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

        match attachments.type_registry.get(&type_id) {
            Some(AttachmentMode::Unique) => {
                let unique = attachments
                    .unique_attachments
                    .get_mut(&type_id)
                    .and_then(|v| v.downcast_mut::<T>());
                let mut empty_vec: Vec<&mut T> = Vec::new();
                f(unique, &mut empty_vec)
            }
            Some(AttachmentMode::Multiple) => {
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
                let mut empty_vec: Vec<&mut T> = Vec::new();
                f(None, &mut empty_vec)
            }
        }
    }
}
