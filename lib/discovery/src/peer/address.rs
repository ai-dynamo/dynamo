// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Address types for peer discovery.
//!
//! This module provides types for representing worker addresses and peer information:
//! - [`WorkerAddress`]: Opaque byte representation of a peer's network address
//! - [`PeerInfo`]: Combined instance ID and worker address for a discovered peer
//!
//! These types are intentionally transport-agnostic, storing addresses as opaque bytes.
//! The interpretation of these bytes is left to the active message runtime.

use super::{InstanceId, WorkerId};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::fmt;
use xxhash_rust::xxh3::xxh3_64;

/// Opaque worker address for discovery.
///
/// This is a transport-agnostic representation of a peer's network address.
/// The bytes are opaque to discovery and are interpreted by the active message runtime.
///
/// # Checksum
///
/// WorkerAddress implements a checksum via xxh3_64 for quick comparison during
/// re-registration validation.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkerAddress(Bytes);

impl WorkerAddress {
    /// Create a new WorkerAddress from bytes.
    pub fn from_bytes(bytes: impl Into<Bytes>) -> Self {
        Self(bytes.into())
    }

    /// Get the underlying bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Get the bytes as a Bytes object.
    pub fn to_bytes(&self) -> Bytes {
        self.0.clone()
    }

    /// Compute a checksum of this address for validation.
    ///
    /// This is used to quickly check if an address has changed during re-registration.
    pub fn checksum(&self) -> u64 {
        xxh3_64(self.as_bytes())
    }
}

impl fmt::Debug for WorkerAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("WorkerAddress")
            .field(&format_args!(
                "len={}, xxh3_64=0x{:016x}",
                self.0.len(),
                self.checksum()
            ))
            .finish()
    }
}

impl fmt::Display for WorkerAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WorkerAddress(xxh3_64=0x{:016x})", self.checksum())
    }
}

/// Peer information combining instance ID and worker address.
///
/// This is the primary type returned by discovery lookups. It contains everything
/// needed to connect to and identify a peer.
///
/// # Example
///
/// ```
/// use dynamo_am_discovery::{InstanceId, WorkerAddress, PeerInfo};
/// use bytes::Bytes;
///
/// let instance_id = InstanceId::new_v4();
/// let address = WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:5555"));
/// let peer_info = PeerInfo::new(instance_id, address);
///
/// assert_eq!(peer_info.instance_id(), instance_id);
/// assert_eq!(peer_info.worker_id(), instance_id.worker_id());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerInfo {
    /// The instance ID of the peer
    pub instance_id: InstanceId,
    /// The worker address for connecting to the peer
    pub worker_address: WorkerAddress,
}

impl PeerInfo {
    /// Create a new PeerInfo.
    pub fn new(instance_id: InstanceId, worker_address: WorkerAddress) -> Self {
        Self {
            instance_id,
            worker_address,
        }
    }

    /// Get the instance ID.
    pub fn instance_id(&self) -> InstanceId {
        self.instance_id
    }

    /// Get the worker ID (derived from instance ID).
    pub fn worker_id(&self) -> WorkerId {
        self.instance_id.worker_id()
    }

    /// Get a reference to the worker address.
    pub fn worker_address(&self) -> &WorkerAddress {
        &self.worker_address
    }

    /// Get the worker address checksum for validation.
    pub fn address_checksum(&self) -> u64 {
        self.worker_address.checksum()
    }

    /// Consume self and return the worker address.
    pub fn into_address(self) -> WorkerAddress {
        self.worker_address
    }

    /// Decompose into instance ID and worker address.
    pub fn into_parts(self) -> (InstanceId, WorkerAddress) {
        (self.instance_id, self.worker_address)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_address_creation() {
        let bytes = Bytes::from_static(b"tcp://127.0.0.1:5555");
        let address = WorkerAddress::from_bytes(bytes.clone());

        assert_eq!(address.as_bytes(), bytes.as_ref());
        assert_eq!(address.to_bytes(), bytes);
    }

    #[test]
    fn test_worker_address_checksum() {
        let address1 = WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:5555"));
        let address2 = WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:5555"));
        let address3 = WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:6666"));

        // Same bytes = same checksum
        assert_eq!(address1.checksum(), address2.checksum());

        // Different bytes = (likely) different checksum
        assert_ne!(address1.checksum(), address3.checksum());
    }

    #[test]
    fn test_worker_address_equality() {
        let address1 = WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:5555"));
        let address2 = WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:5555"));
        let address3 = WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:6666"));

        assert_eq!(address1, address2);
        assert_ne!(address1, address3);
    }

    #[test]
    fn test_worker_address_debug() {
        let address = WorkerAddress::from_bytes(Bytes::from_static(b"test"));
        let debug_str = format!("{:?}", address);

        assert!(debug_str.contains("WorkerAddress"));
        assert!(debug_str.contains("len=4"));
        assert!(debug_str.contains("xxh3_64="));
    }

    #[test]
    fn test_peer_info_creation() {
        let instance_id = InstanceId::new_v4();
        let address = WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:5555"));

        let peer_info = PeerInfo::new(instance_id, address.clone());

        assert_eq!(peer_info.instance_id(), instance_id);
        assert_eq!(peer_info.worker_id(), instance_id.worker_id());
        assert_eq!(peer_info.worker_address(), &address);
    }

    #[test]
    fn test_peer_info_checksum() {
        let instance_id = InstanceId::new_v4();
        let address = WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:5555"));

        let peer_info = PeerInfo::new(instance_id, address.clone());

        assert_eq!(peer_info.address_checksum(), address.checksum());
    }

    #[test]
    fn test_peer_info_into_address() {
        let instance_id = InstanceId::new_v4();
        let address = WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:5555"));

        let peer_info = PeerInfo::new(instance_id, address.clone());
        let extracted_address = peer_info.into_address();

        assert_eq!(extracted_address, address);
    }

    #[test]
    fn test_peer_info_into_parts() {
        let instance_id = InstanceId::new_v4();
        let address = WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:5555"));

        let peer_info = PeerInfo::new(instance_id, address.clone());
        let (extracted_id, extracted_address) = peer_info.into_parts();

        assert_eq!(extracted_id, instance_id);
        assert_eq!(extracted_address, address);
    }

    #[test]
    fn test_peer_info_serde() {
        let instance_id = InstanceId::new_v4();
        let address = WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:5555"));
        let peer_info = PeerInfo::new(instance_id, address);

        // Serialize to JSON
        let json = serde_json::to_string(&peer_info).unwrap();

        // Deserialize back
        let deserialized: PeerInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.instance_id(), instance_id);
        assert_eq!(deserialized.worker_id(), instance_id.worker_id());
        assert_eq!(
            deserialized.worker_address().as_bytes(),
            b"tcp://127.0.0.1:5555"
        );
    }
}
