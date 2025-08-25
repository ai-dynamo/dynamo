// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{ContentHash, OscarError, OscarResult};
use serde::{Deserialize, Serialize};
use std::time::SystemTime;
use uuid::Uuid;

/// State of a shared object in the registry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObjectState {
    /// Object is being registered by a lease.
    Registering { lease_id: Uuid },
    /// Object is available and has active references.
    Available { reference_count: u32 },
    /// Object is being deleted (no new references allowed).
    Deleting,
}

/// Information about a shared object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectInfo {
    /// Content hash of the object.
    pub hash: ContentHash,
    /// Size of the object in bytes.
    pub size: usize,
    /// Current state of the object.
    pub state: ObjectState,
    /// When the object was first registered.
    pub created_at: SystemTime,
    /// When the object state was last updated.
    pub updated_at: SystemTime,
    /// Storage backend where the object is stored.
    pub storage_backend: String,
    /// Storage-specific location/key.
    pub storage_key: String,
}

impl ObjectInfo {
    /// Create new object info for registration.
    pub fn new_registering(
        hash: ContentHash,
        size: usize,
        lease_id: Uuid,
        storage_backend: String,
        storage_key: String,
    ) -> Self {
        let now = SystemTime::now();
        Self {
            hash,
            size,
            state: ObjectState::Registering { lease_id },
            created_at: now,
            updated_at: now,
            storage_backend,
            storage_key,
        }
    }

    /// Transition to available state with initial reference.
    pub fn make_available(&mut self) -> OscarResult<()> {
        match &self.state {
            ObjectState::Registering { .. } => {
                self.state = ObjectState::Available { reference_count: 1 };
                self.updated_at = SystemTime::now();
                Ok(())
            }
            _ => Err(OscarError::InvalidOperation {
                reason: format!("Cannot make object available from state: {:?}", self.state),
            }),
        }
    }

    /// Add a reference to the object.
    pub fn add_reference(&mut self) -> OscarResult<()> {
        match &mut self.state {
            ObjectState::Available { reference_count } => {
                *reference_count = reference_count
                    .checked_add(1)
                    .ok_or_else(|| OscarError::InvalidOperation {
                        reason: "Reference count overflow".to_string(),
                    })?;
                self.updated_at = SystemTime::now();
                Ok(())
            }
            _ => Err(OscarError::InvalidOperation {
                reason: format!("Cannot add reference in state: {:?}", self.state),
            }),
        }
    }

    /// Remove a reference from the object.
    pub fn remove_reference(&mut self) -> OscarResult<bool> {
        match &mut self.state {
            ObjectState::Available { reference_count } => {
                if *reference_count == 0 {
                    return Err(OscarError::InvalidOperation {
                        reason: "Cannot remove reference: count is already zero".to_string(),
                    });
                }
                *reference_count -= 1;
                self.updated_at = SystemTime::now();
                Ok(*reference_count == 0)
            }
            _ => Err(OscarError::InvalidOperation {
                reason: format!("Cannot remove reference in state: {:?}", self.state),
            }),
        }
    }

    /// Mark object for deletion.
    pub fn mark_for_deletion(&mut self) -> OscarResult<()> {
        match &self.state {
            ObjectState::Available { reference_count: 0 } => {
                self.state = ObjectState::Deleting;
                self.updated_at = SystemTime::now();
                Ok(())
            }
            _ => Err(OscarError::InvalidOperation {
                reason: format!("Cannot delete object in state: {:?}", self.state),
            }),
        }
    }
}

/// A shared object with its data.
pub struct SharedObject {
    /// Object metadata.
    pub info: ObjectInfo,
    /// Object data.
    pub data: bytes::Bytes,
}

impl SharedObject {
    /// Create a new shared object.
    pub fn new(
        data: bytes::Bytes,
        lease_id: Uuid,
        storage_backend: String,
        storage_key: String,
    ) -> OscarResult<Self> {
        let size = data.len();
        if size > crate::MAX_OBJECT_SIZE {
            return Err(OscarError::ObjectTooLarge {
                size,
                max_size: crate::MAX_OBJECT_SIZE,
            });
        }

        let hash = crate::ObjectHasher::hash(&data);
        let info = ObjectInfo::new_registering(
            hash,
            size,
            lease_id,
            storage_backend,
            storage_key,
        );

        Ok(Self { info, data })
    }
}