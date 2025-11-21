// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identity types for the active message system.
//!
//! This module provides strongly-typed wrappers for instance and worker identifiers:
//! - [`InstanceId`]: Unique runtime instance identifier (wraps UUID)
//! - [`WorkerId`]: Deterministic 64-bit worker identifier derived from InstanceId
//!
//! # Design Principles
//!
//! 1. **Type Safety**: InstanceId cannot be confused with message IDs or other UUIDs
//! 2. **Deterministic Derivation**: WorkerId is always computed from InstanceId (xxh3_64 hash)
//! 3. **Single Source of Truth**: InstanceId is the primary identifier, WorkerId is derived
//!
//! # Example
//!
//! ```ignore
//! // InstanceIds are created internally by the runtime
//! // and obtained from ActiveMessageClient.instance_id()
//!
//! use dynamo_am::api::identity::{InstanceId, WorkerId};
//!
//! # fn get_instance_id() -> InstanceId { unimplemented!() }
//! let instance_id = get_instance_id(); // From ActiveMessageClient
//!
//! // Derive worker ID automatically
//! let worker_id: WorkerId = instance_id.worker_id();
//!
//! // WorkerId is deterministic
//! assert_eq!(worker_id, instance_id.worker_id());
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;
use xxhash_rust::xxh3::xxh3_64;

/// Unique identifier for a runtime instance.
///
/// This is a UUID-based identifier that uniquely identifies a running instance
/// of the active message runtime. It is used for:
/// - Transport-level addressing
/// - Discovery registration
/// - Routing table management
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InstanceId(Uuid);

impl InstanceId {
    /// Create a new random v4 InstanceId.
    ///
    /// This is exposed for testing and special cases. In production, use
    /// [`InstanceFactory::create()`] instead.
    pub fn new_v4() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create an InstanceId from a UUID.
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Create an InstanceId from raw bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(Uuid::from_bytes(bytes))
    }

    /// Derive the deterministic WorkerId from this InstanceId.
    ///
    /// WorkerId is computed using xxh3_64 hash of the UUID bytes.
    /// This ensures a 1:1 mapping between InstanceId and WorkerId.
    pub fn worker_id(&self) -> WorkerId {
        WorkerId::from(self)
    }

    /// Get a reference to the underlying UUID.
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }

    /// Get the underlying UUID as a u128.
    pub fn as_u128(&self) -> u128 {
        self.0.as_u128()
    }

    /// Get the underlying UUID as bytes.
    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }
}

impl fmt::Display for InstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for InstanceId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl From<InstanceId> for Uuid {
    fn from(id: InstanceId) -> Self {
        id.0
    }
}

impl AsRef<Uuid> for InstanceId {
    fn as_ref(&self) -> &Uuid {
        &self.0
    }
}

/// Deterministic 64-bit worker identifier derived from InstanceId.
///
/// WorkerId is used in:
/// - [`crate::event::EventHandle`]: Embedded in the u128 event handle (64 bits)
/// - [`crate::event::EventRoutingTable`]: Maps worker_id → instance_id for event routing
/// - Discovery systems: Lookup key for peer information
///
/// WorkerId is **always derived** from InstanceId using xxh3_64 hash.
/// This ensures consistency across the system without needing to store both values.
///
/// # Example
///
/// ```ignore
/// use dynamo_am::api::identity::{InstanceId, WorkerId};
///
/// # fn get_instance_id() -> InstanceId { unimplemented!() }
/// let instance_id = get_instance_id(); // From ActiveMessageClient
/// let worker_id = instance_id.worker_id();
///
/// // WorkerId is deterministic
/// assert_eq!(worker_id, instance_id.worker_id());
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(transparent)]
pub struct WorkerId(u64);

impl WorkerId {
    /// Create a WorkerId from a raw u64 value.
    ///
    /// This is used when decoding WorkerIds from event handles or wire formats.
    /// External users should always derive WorkerId via `instance_id.worker_id()`.
    pub fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Get the underlying u64 value.
    #[inline(always)]
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&InstanceId> for WorkerId {
    /// Derive WorkerId from InstanceId using xxh3_64 hash.
    ///
    /// This is the canonical way to compute WorkerId - it should never be
    /// constructed any other way to ensure consistency.
    fn from(id: &InstanceId) -> Self {
        Self(xxh3_64(id.as_uuid().as_bytes()))
    }
}

impl From<InstanceId> for WorkerId {
    fn from(id: InstanceId) -> Self {
        Self::from(&id)
    }
}

impl From<WorkerId> for u64 {
    fn from(id: WorkerId) -> Self {
        id.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_id_creation() {
        let id1 = InstanceId::new_v4();
        let id2 = InstanceId::new_v4();

        // Different instances have different IDs
        assert_ne!(id1, id2);

        // Can convert to/from UUID
        let uuid: Uuid = id1.into();
        let id3 = InstanceId::from(uuid);
        assert_eq!(id1, id3);
    }

    #[test]
    fn test_worker_id_deterministic() {
        let instance_id = InstanceId::new_v4();

        // WorkerId is deterministic
        let worker_id1 = instance_id.worker_id();
        let worker_id2 = instance_id.worker_id();
        assert_eq!(worker_id1, worker_id2);

        // Different instances have different worker IDs
        let other_instance = InstanceId::new_v4();
        let other_worker = other_instance.worker_id();
        assert_ne!(worker_id1, other_worker);
    }

    #[test]
    fn test_worker_id_from_conversion() {
        let instance_id = InstanceId::new_v4();

        // Both From implementations work
        let worker_id1 = WorkerId::from(&instance_id);
        let worker_id2 = WorkerId::from(instance_id);
        assert_eq!(worker_id1, worker_id2);

        // Matches .worker_id() method
        assert_eq!(worker_id1, instance_id.worker_id());
    }

    #[test]
    fn test_instance_id_display() {
        let instance_id = InstanceId::new_v4();
        let display = format!("{}", instance_id);
        let uuid_display = format!("{}", instance_id.as_uuid());
        assert_eq!(display, uuid_display);
    }

    #[test]
    fn test_worker_id_display() {
        let instance_id = InstanceId::new_v4();
        let worker_id = instance_id.worker_id();
        let display = format!("{}", worker_id);
        let u64_display = format!("{}", worker_id.as_u64());
        assert_eq!(display, u64_display);
    }

    #[test]
    fn test_instance_id_serde() {
        let instance_id = InstanceId::new_v4();

        // Serialize as JSON
        let json = serde_json::to_string(&instance_id).unwrap();

        // Should be a plain UUID string
        let uuid_json = serde_json::to_string(instance_id.as_uuid()).unwrap();
        assert_eq!(json, uuid_json);

        // Deserialize back
        let deserialized: InstanceId = serde_json::from_str(&json).unwrap();
        assert_eq!(instance_id, deserialized);
    }

    #[test]
    fn test_worker_id_serde() {
        let worker_id = InstanceId::new_v4().worker_id();

        // Serialize as JSON
        let json = serde_json::to_string(&worker_id).unwrap();

        // Should be a plain u64
        let u64_json = serde_json::to_string(&worker_id.as_u64()).unwrap();
        assert_eq!(json, u64_json);

        // Deserialize back
        let deserialized: WorkerId = serde_json::from_str(&json).unwrap();
        assert_eq!(worker_id, deserialized);
    }

    #[test]
    fn test_instance_id_as_methods() {
        let uuid = Uuid::new_v4();
        let instance_id = InstanceId::from_uuid(uuid);

        assert_eq!(instance_id.as_uuid(), &uuid);
        assert_eq!(instance_id.as_u128(), uuid.as_u128());
        assert_eq!(instance_id.as_bytes(), uuid.as_bytes());
    }

    #[test]
    fn test_instance_id_from_bytes() {
        let uuid = Uuid::new_v4();
        let bytes = *uuid.as_bytes();

        let instance_id = InstanceId::from_bytes(bytes);
        assert_eq!(instance_id.as_uuid(), &uuid);
    }

    #[test]
    fn test_worker_id_u64_conversion() {
        let instance_id = InstanceId::new_v4();
        let worker_id = instance_id.worker_id();

        let raw_u64 = worker_id.as_u64();
        let reconstructed = WorkerId::from_u64(raw_u64);

        assert_eq!(worker_id, reconstructed);
    }
}
