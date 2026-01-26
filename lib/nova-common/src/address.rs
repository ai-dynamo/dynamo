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

use crate::identity::{InstanceId, WorkerId};
use crate::transport::TransportKey;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use xxhash_rust::xxh3::xxh3_64;

/// Errors that can occur when working with WorkerAddress.
#[derive(Debug, thiserror::Error)]
pub enum WorkerAddressError {
    /// Attempted to add a key that already exists
    #[error("Key already exists: {0}")]
    KeyExists(String),

    /// Attempted to access or remove a key that doesn't exist
    #[error("Key not found: {0}")]
    KeyNotFound(String),

    /// Failed to encode the map to bytes
    #[error("Encoding error: {0}")]
    EncodingError(#[from] rmp_serde::encode::Error),

    /// Failed to decode bytes to map
    #[error("Decoding error: {0}")]
    DecodingError(#[from] rmp_serde::decode::Error),

    /// Encountered an unsupported format version
    #[error("Unsupported format version: {0}")]
    UnsupportedVersion(u8),

    /// The data format is invalid
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
}

/// Opaque worker address for discovery.
///
/// This is a transport-agnostic representation of a peer's network address.
/// The bytes are opaque to discovery and are interpreted by the active message runtime.
///
/// # Checksum
///
/// WorkerAddress implements a checksum via xxh3_64 for quick comparison during
/// re-registration validation.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct WorkerAddress(Bytes);

// Custom Serialize/Deserialize to handle Bytes
impl Serialize for WorkerAddress {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde_bytes::serialize(self.0.as_ref(), serializer)
    }
}

impl<'de> Deserialize<'de> for WorkerAddress {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: Vec<u8> = serde_bytes::deserialize(deserializer)?;
        Ok(WorkerAddress(Bytes::from(bytes)))
    }
}

impl WorkerAddress {
    /// Create a WorkerAddress from pre-encoded bytes.
    ///
    /// This is used by transport implementations to construct addresses from
    /// MessagePack-encoded map data. The bytes are assumed to be valid MessagePack.
    pub fn from_encoded(bytes: impl Into<Bytes>) -> Self {
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

    /// Get the list of available transport keys in this address.
    ///
    /// Returns the keys from the internal map as `TransportKey` for type-safe efficient
    /// storage and sharing. This allows callers to see what transport types or endpoints
    /// are available without exposing the full map.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal bytes cannot be decoded as a valid MessagePack map.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use dynamo_nova_common::{WorkerAddress, TransportKey};
    /// # let address: WorkerAddress = unimplemented!();
    /// let transports = address.available_transports().unwrap();
    /// if transports.contains(&TransportKey::from("tcp")) {
    ///     // TCP transport is available
    /// }
    /// ```
    pub fn available_transports(&self) -> Result<Vec<TransportKey>, WorkerAddressError> {
        let map = decode_to_map(self.as_bytes())?;
        Ok(map.keys().cloned().map(TransportKey::from).collect())
    }

    /// Get a single entry from the internal map.
    ///
    /// This decodes the address and extracts the entry for the given key.
    ///
    /// Accepts any type that can be converted to a string reference, including
    /// `&str`, `String`, `&String`, and `TransportKey`.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal bytes cannot be decoded as a valid MessagePack map.
    pub fn get_entry(&self, key: impl AsRef<str>) -> Result<Option<Bytes>, WorkerAddressError> {
        let map = decode_to_map(self.as_bytes())?;
        Ok(map.get(key.as_ref()).cloned())
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

// ============================================================================
// Internal Decoding Helper
// ============================================================================

/// Decode WorkerAddress bytes from MessagePack into a map.
fn decode_to_map(bytes: &[u8]) -> Result<HashMap<Arc<str>, Bytes>, WorkerAddressError> {
    if bytes.is_empty() {
        return Err(WorkerAddressError::InvalidFormat("Empty bytes".to_string()));
    }

    // Decode MessagePack
    let decoded: HashMap<String, Vec<u8>> = rmp_serde::from_slice(bytes)?;

    // Convert to HashMap<Arc<str>, Bytes>
    Ok(decoded
        .into_iter()
        .map(|(k, v)| (Arc::from(k.as_str()), Bytes::from(v)))
        .collect())
}

/// Peer information combining instance ID and worker address.
///
/// This is the primary type returned by discovery lookups. It contains everything
/// needed to connect to and identify a peer.
///
/// # Example
///
/// ```no_run
/// # // WorkerAddress is created internally, this is simplified for docs
/// use dynamo_nova_common::{InstanceId, PeerInfo};
/// # use dynamo_nova_common::WorkerAddress;
/// # let address: WorkerAddress = unimplemented!();
///
/// let instance_id = InstanceId::new_v4();
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

    // Helper to create a test address with MessagePack encoding
    fn make_test_address(entries: &[(&str, &[u8])]) -> WorkerAddress {
        let map: HashMap<String, Vec<u8>> = entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_vec()))
            .collect();
        let encoded = rmp_serde::to_vec(&map).unwrap();
        WorkerAddress::from_encoded(encoded)
    }

    #[test]
    fn test_worker_address_from_encoded() {
        let address = make_test_address(&[("endpoint", b"tcp://127.0.0.1:5555")]);

        // Verify we can get the entry back
        let entry = address.get_entry("endpoint").unwrap();
        assert_eq!(entry, Some(Bytes::from_static(b"tcp://127.0.0.1:5555")));
    }

    #[test]
    fn test_worker_address_checksum() {
        let address1 = make_test_address(&[("endpoint", b"tcp://127.0.0.1:5555")]);
        let address2 = make_test_address(&[("endpoint", b"tcp://127.0.0.1:5555")]);
        let address3 = make_test_address(&[("endpoint", b"tcp://127.0.0.1:6666")]);

        // Same content = same checksum
        assert_eq!(address1.checksum(), address2.checksum());

        // Different content = different checksum
        assert_ne!(address1.checksum(), address3.checksum());
    }

    #[test]
    fn test_worker_address_equality() {
        let address1 = make_test_address(&[("endpoint", b"tcp://127.0.0.1:5555")]);
        let address2 = make_test_address(&[("endpoint", b"tcp://127.0.0.1:5555")]);
        let address3 = make_test_address(&[("endpoint", b"tcp://127.0.0.1:6666")]);

        assert_eq!(address1, address2);
        assert_ne!(address1, address3);
    }

    #[test]
    fn test_worker_address_debug() {
        let address = make_test_address(&[("test", b"value")]);
        let debug_str = format!("{:?}", address);

        assert!(debug_str.contains("WorkerAddress"));
        assert!(debug_str.contains("len="));
        assert!(debug_str.contains("xxh3_64="));
    }

    #[test]
    fn test_available_transports() {
        let address = make_test_address(&[
            ("tcp", b"tcp://127.0.0.1:5555"),
            ("rdma", b"rdma://10.0.0.1:6666"),
            ("udp", b"udp://127.0.0.1:7777"),
        ]);

        let transports = address.available_transports().unwrap();
        assert_eq!(transports.len(), 3);
        assert!(transports.contains(&TransportKey::from("tcp")));
        assert!(transports.contains(&TransportKey::from("rdma")));
        assert!(transports.contains(&TransportKey::from("udp")));
    }

    #[test]
    fn test_available_transports_empty() {
        let address = make_test_address(&[]);
        let transports = address.available_transports().unwrap();
        assert_eq!(transports.len(), 0);
    }

    #[test]
    fn test_get_entry() {
        let address =
            make_test_address(&[("endpoint", b"tcp://127.0.0.1:5555"), ("protocol", b"tcp")]);

        // Get existing entry
        assert_eq!(
            address.get_entry("endpoint").unwrap().unwrap(),
            Bytes::from_static(b"tcp://127.0.0.1:5555")
        );

        // Get nonexistent entry
        assert!(address.get_entry("nonexistent").unwrap().is_none());
    }

    #[test]
    fn test_get_entry_with_transport_key() {
        let address = make_test_address(&[
            ("tcp", b"tcp://127.0.0.1:5555"),
            ("rdma", b"rdma://10.0.0.1:6666"),
        ]);

        // Test get_entry with TransportKey
        let tcp_key = TransportKey::from("tcp");
        let result = address.get_entry(tcp_key).unwrap();
        assert_eq!(result, Some(Bytes::from_static(b"tcp://127.0.0.1:5555")));

        // Test get_entry with String
        let result = address.get_entry(String::from("rdma")).unwrap();
        assert_eq!(result, Some(Bytes::from_static(b"rdma://10.0.0.1:6666")));
    }

    #[test]
    fn test_peer_info_creation() {
        let instance_id = InstanceId::new_v4();
        let address = make_test_address(&[("endpoint", b"tcp://127.0.0.1:5555")]);

        let peer_info = PeerInfo::new(instance_id, address.clone());

        assert_eq!(peer_info.instance_id(), instance_id);
        assert_eq!(peer_info.worker_id(), instance_id.worker_id());
        assert_eq!(peer_info.worker_address(), &address);
    }

    #[test]
    fn test_peer_info_checksum() {
        let instance_id = InstanceId::new_v4();
        let address = make_test_address(&[("endpoint", b"tcp://127.0.0.1:5555")]);

        let peer_info = PeerInfo::new(instance_id, address.clone());

        assert_eq!(peer_info.address_checksum(), address.checksum());
    }

    #[test]
    fn test_peer_info_into_address() {
        let instance_id = InstanceId::new_v4();
        let address = make_test_address(&[("endpoint", b"tcp://127.0.0.1:5555")]);

        let peer_info = PeerInfo::new(instance_id, address.clone());
        let extracted_address = peer_info.into_address();

        assert_eq!(extracted_address, address);
    }

    #[test]
    fn test_peer_info_into_parts() {
        let instance_id = InstanceId::new_v4();
        let address = make_test_address(&[("endpoint", b"tcp://127.0.0.1:5555")]);

        let peer_info = PeerInfo::new(instance_id, address.clone());
        let (extracted_id, extracted_address) = peer_info.into_parts();

        assert_eq!(extracted_id, instance_id);
        assert_eq!(extracted_address, address);
    }

    #[test]
    fn test_peer_info_serde() {
        let instance_id = InstanceId::new_v4();
        let address = make_test_address(&[("endpoint", b"tcp://127.0.0.1:5555")]);
        let peer_info = PeerInfo::new(instance_id, address);

        // Serialize to JSON
        let json = serde_json::to_string(&peer_info).unwrap();

        // Deserialize back
        let deserialized: PeerInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.instance_id(), instance_id);
        assert_eq!(deserialized.worker_id(), instance_id.worker_id());

        // Verify the entry is preserved
        let entry = deserialized.worker_address().get_entry("endpoint").unwrap();
        assert_eq!(entry, Some(Bytes::from_static(b"tcp://127.0.0.1:5555")));
    }
}
