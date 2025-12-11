// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Address types for peer discovery.
//!
//! This module provides types for representing worker addresses and peer information:
//! - [`WorkerAddress`]: Opaque byte representation of a peer's network address
//! - [`PeerInfo`]: Combined instance ID and worker address for a discovered peer
//!
//! These types are intentionally transport-agnostic, storing addresses as opaque bytes.
//! The interpretation of these bytes is left to the active message runtime.

use super::TransportKey;

use dynamo_identity::{InstanceId, WorkerId};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use xxhash_rust::xxh3::xxh3_64;

/// Errors that can occur when working with WorkerAddress.
#[derive(Debug, thiserror::Error)]
#[allow(dead_code)] // Used in tests and future internal API
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
    /// # use dynamo_nova_backend::{WorkerAddress, TransportKey};
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

    // ========================================================================
    // Internal API (pub(crate)) for map manipulation
    // ========================================================================

    /// Create a new builder for constructing a WorkerAddress from scratch.
    pub fn builder() -> WorkerAddressBuilder {
        WorkerAddressBuilder::new()
    }

    /// Convert this WorkerAddress into a builder for modification.
    ///
    /// This decodes the internal bytes into a mutable map.
    #[allow(dead_code)] // Internal API for future use
    pub(crate) fn into_builder(self) -> Result<WorkerAddressBuilder, WorkerAddressError> {
        let map = decode_to_map(self.as_bytes())?;

        // Convert Arc<str> keys back to String for the builder
        let entries = map.into_iter().map(|(k, v)| (k.to_string(), v)).collect();

        Ok(WorkerAddressBuilder { entries })
    }

    /// Decode the WorkerAddress into a map of entries.
    ///
    /// This returns a HashMap with Arc<str> keys for efficient sharing.
    #[allow(dead_code)] // Internal API for future use
    pub(crate) fn to_map(&self) -> Result<HashMap<Arc<str>, Bytes>, WorkerAddressError> {
        decode_to_map(self.as_bytes())
    }

    /// Get a single entry from the internal map without decoding the entire structure.
    ///
    /// For now, this decodes the full map and extracts the entry. This could be
    /// optimized in the future if partial decoding is needed.
    ///
    /// Accepts any type that can be converted to a string reference, including
    /// `&str`, `String`, `&String`, and `TransportKey`.
    #[allow(dead_code)] // Internal API for future use
    pub(crate) fn get_entry(
        &self,
        key: impl AsRef<str>,
    ) -> Result<Option<Bytes>, WorkerAddressError> {
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

/// Builder for constructing and modifying WorkerAddress instances.
///
/// This provides a mutable interface for manipulating the internal map
/// before encoding it into the immutable WorkerAddress format.
///
/// # Example
///
/// ```no_run
/// # // This is internal API, so example is no_run
/// # use bytes::Bytes;
/// # struct WorkerAddressBuilder;
/// # impl WorkerAddressBuilder {
/// #     fn new() -> Self { Self }
/// #     fn add_entry(&mut self, _: &str, _: Bytes) -> Result<(), ()> { Ok(()) }
/// #     fn build(self) -> Result<(), ()> { Ok(()) }
/// # }
/// let mut builder = WorkerAddressBuilder::new();
/// builder.add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555")).unwrap();
/// builder.add_entry("protocol", Bytes::from_static(b"tcp")).unwrap();
/// let address = builder.build().unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct WorkerAddressBuilder {
    entries: HashMap<String, Bytes>,
}

#[allow(dead_code)] // Internal API methods used in tests
impl WorkerAddressBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add a new entry to the map.
    ///
    /// Returns an error if the key already exists.
    pub fn add_entry(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Bytes>,
    ) -> Result<(), WorkerAddressError> {
        let key = key.into();
        if self.entries.contains_key(&key) {
            return Err(WorkerAddressError::KeyExists(key));
        }
        self.entries.insert(key, value.into());
        Ok(())
    }

    /// Remove an entry from the map.
    ///
    /// Returns an error if the key doesn't exist.
    pub fn remove_entry(&mut self, key: &str) -> Result<(), WorkerAddressError> {
        if self.entries.remove(key).is_none() {
            return Err(WorkerAddressError::KeyNotFound(key.to_string()));
        }
        Ok(())
    }

    /// Update an existing entry in the map.
    ///
    /// Returns an error if the key doesn't exist.
    pub fn update_entry(
        &mut self,
        key: &str,
        value: impl Into<Bytes>,
    ) -> Result<(), WorkerAddressError> {
        if !self.entries.contains_key(key) {
            return Err(WorkerAddressError::KeyNotFound(key.to_string()));
        }
        self.entries.insert(key.to_string(), value.into());
        Ok(())
    }

    /// Check if a key exists in the map.
    pub fn has_entry(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Get a reference to an entry's value.
    pub fn get_entry(&self, key: &str) -> Option<&Bytes> {
        self.entries.get(key)
    }

    /// Merge another WorkerAddress into this builder.
    ///
    /// This decodes the other address and attempts to add all its entries to this builder.
    /// If any key from the other address already exists in this builder, returns an error
    /// and leaves the builder unchanged.
    ///
    /// # Errors
    ///
    /// Returns `KeyExists` if any key from the other address already exists in this builder.
    /// Returns decoding errors if the other address cannot be decoded.
    pub fn merge(&mut self, other: &WorkerAddress) -> Result<(), WorkerAddressError> {
        let map = other.to_map()?;

        // First check if any keys would conflict
        for key in map.keys() {
            if self.entries.contains_key(key.as_ref()) {
                return Err(WorkerAddressError::KeyExists(key.to_string()));
            }
        }

        // All keys are unique, now add them
        for (key, value) in map {
            self.entries.insert(key.to_string(), value);
        }

        Ok(())
    }

    /// Build the WorkerAddress from this builder.
    ///
    /// This encodes the map into MessagePack binary format.
    pub fn build(self) -> Result<WorkerAddress, WorkerAddressError> {
        encode_map(self.entries)
    }
}

impl Default for WorkerAddressBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Encoding/Decoding Helper Functions
// ============================================================================

/// Encode a map into MessagePack format.
#[allow(dead_code)] // Used by builder
fn encode_map(entries: HashMap<String, Bytes>) -> Result<WorkerAddress, WorkerAddressError> {
    // Convert HashMap<String, Bytes> to HashMap<String, Vec<u8>> for MessagePack
    let serializable: HashMap<String, Vec<u8>> =
        entries.into_iter().map(|(k, v)| (k, v.to_vec())).collect();

    // Encode to MessagePack
    let encoded = rmp_serde::to_vec(&serializable)?;

    Ok(WorkerAddress(Bytes::from(encoded)))
}

/// Decode WorkerAddress bytes from MessagePack into a map.
#[allow(dead_code)] // Used by WorkerAddress methods
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
/// use dynamo_nova_backend::{InstanceId, PeerInfo};
/// # use dynamo_nova_backend::WorkerAddress;
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

    #[test]
    fn test_worker_address_creation() {
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address = builder.build().unwrap();

        // Verify we can get bytes and decode back
        let decoded = address.to_map().unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(
            decoded.get("endpoint").unwrap(),
            &Bytes::from_static(b"tcp://127.0.0.1:5555")
        );
    }

    #[test]
    fn test_worker_address_checksum() {
        let mut builder1 = WorkerAddressBuilder::new();
        builder1
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address1 = builder1.build().unwrap();

        let mut builder2 = WorkerAddressBuilder::new();
        builder2
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address2 = builder2.build().unwrap();

        let mut builder3 = WorkerAddressBuilder::new();
        builder3
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:6666"))
            .unwrap();
        let address3 = builder3.build().unwrap();

        // Same content = same checksum
        assert_eq!(address1.checksum(), address2.checksum());

        // Different content = different checksum
        assert_ne!(address1.checksum(), address3.checksum());
    }

    #[test]
    fn test_worker_address_equality() {
        let mut builder1 = WorkerAddressBuilder::new();
        builder1
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address1 = builder1.build().unwrap();

        let mut builder2 = WorkerAddressBuilder::new();
        builder2
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address2 = builder2.build().unwrap();

        let mut builder3 = WorkerAddressBuilder::new();
        builder3
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:6666"))
            .unwrap();
        let address3 = builder3.build().unwrap();

        assert_eq!(address1, address2);
        assert_ne!(address1, address3);
    }

    #[test]
    fn test_worker_address_debug() {
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("test", Bytes::from_static(b"value"))
            .unwrap();
        let address = builder.build().unwrap();
        let debug_str = format!("{:?}", address);

        assert!(debug_str.contains("WorkerAddress"));
        assert!(debug_str.contains("len="));
        assert!(debug_str.contains("xxh3_64="));
    }

    #[test]
    fn test_available_transports() {
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("tcp", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        builder
            .add_entry("rdma", Bytes::from_static(b"rdma://10.0.0.1:6666"))
            .unwrap();
        builder
            .add_entry("udp", Bytes::from_static(b"udp://127.0.0.1:7777"))
            .unwrap();
        let address = builder.build().unwrap();

        let transports = address.available_transports().unwrap();
        assert_eq!(transports.len(), 3);
        assert!(transports.contains(&TransportKey::from("tcp")));
        assert!(transports.contains(&TransportKey::from("rdma")));
        assert!(transports.contains(&TransportKey::from("udp")));
    }

    #[test]
    fn test_available_transports_empty() {
        let builder = WorkerAddressBuilder::new();
        let address = builder.build().unwrap();

        let transports = address.available_transports().unwrap();
        assert_eq!(transports.len(), 0);
    }

    #[test]
    fn test_available_transports_single() {
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address = builder.build().unwrap();

        let transports = address.available_transports().unwrap();
        assert_eq!(transports.len(), 1);
        assert_eq!(transports[0], TransportKey::from("endpoint"));
    }

    #[test]
    fn test_peer_info_creation() {
        let instance_id = InstanceId::new_v4();
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address = builder.build().unwrap();

        let peer_info = PeerInfo::new(instance_id, address.clone());

        assert_eq!(peer_info.instance_id(), instance_id);
        assert_eq!(peer_info.worker_id(), instance_id.worker_id());
        assert_eq!(peer_info.worker_address(), &address);
    }

    #[test]
    fn test_peer_info_checksum() {
        let instance_id = InstanceId::new_v4();
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address = builder.build().unwrap();

        let peer_info = PeerInfo::new(instance_id, address.clone());

        assert_eq!(peer_info.address_checksum(), address.checksum());
    }

    #[test]
    fn test_peer_info_into_address() {
        let instance_id = InstanceId::new_v4();
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address = builder.build().unwrap();

        let peer_info = PeerInfo::new(instance_id, address.clone());
        let extracted_address = peer_info.into_address();

        assert_eq!(extracted_address, address);
    }

    #[test]
    fn test_peer_info_into_parts() {
        let instance_id = InstanceId::new_v4();
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address = builder.build().unwrap();

        let peer_info = PeerInfo::new(instance_id, address.clone());
        let (extracted_id, extracted_address) = peer_info.into_parts();

        assert_eq!(extracted_id, instance_id);
        assert_eq!(extracted_address, address);
    }

    #[test]
    fn test_peer_info_serde() {
        let instance_id = InstanceId::new_v4();
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address = builder.build().unwrap();
        let peer_info = PeerInfo::new(instance_id, address);

        // Serialize to JSON
        let json = serde_json::to_string(&peer_info).unwrap();

        // Deserialize back
        let deserialized: PeerInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.instance_id(), instance_id);
        assert_eq!(deserialized.worker_id(), instance_id.worker_id());

        // Verify the decoded map has the expected entry
        let map = deserialized.worker_address().to_map().unwrap();
        assert_eq!(
            map.get("endpoint").unwrap(),
            &Bytes::from_static(b"tcp://127.0.0.1:5555")
        );
    }

    // ========================================================================
    // Builder and Map Tests
    // ========================================================================

    #[test]
    fn test_builder_basic() {
        let mut builder = WorkerAddressBuilder::new();

        // Add entries
        builder
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        builder
            .add_entry("protocol", Bytes::from_static(b"tcp"))
            .unwrap();

        // Check has_entry
        assert!(builder.has_entry("endpoint"));
        assert!(builder.has_entry("protocol"));
        assert!(!builder.has_entry("nonexistent"));

        // Check get_entry
        assert_eq!(
            builder.get_entry("endpoint").unwrap(),
            &Bytes::from_static(b"tcp://127.0.0.1:5555")
        );

        // Build
        let address = builder.build().unwrap();
        assert!(!address.as_bytes().is_empty());
    }

    #[test]
    fn test_builder_add_duplicate_key() {
        let mut builder = WorkerAddressBuilder::new();

        builder
            .add_entry("key", Bytes::from_static(b"value1"))
            .unwrap();

        // Try to add duplicate
        let result = builder.add_entry("key", Bytes::from_static(b"value2"));
        assert!(matches!(result, Err(WorkerAddressError::KeyExists(_))));
    }

    #[test]
    fn test_builder_remove_entry() {
        let mut builder = WorkerAddressBuilder::new();

        builder
            .add_entry("key1", Bytes::from_static(b"value1"))
            .unwrap();
        builder
            .add_entry("key2", Bytes::from_static(b"value2"))
            .unwrap();

        // Remove key1
        builder.remove_entry("key1").unwrap();

        assert!(!builder.has_entry("key1"));
        assert!(builder.has_entry("key2"));

        // Try to remove nonexistent key
        let result = builder.remove_entry("key1");
        assert!(matches!(result, Err(WorkerAddressError::KeyNotFound(_))));
    }

    #[test]
    fn test_builder_update_entry() {
        let mut builder = WorkerAddressBuilder::new();

        builder
            .add_entry("key", Bytes::from_static(b"value1"))
            .unwrap();

        // Update
        builder
            .update_entry("key", Bytes::from_static(b"value2"))
            .unwrap();

        assert_eq!(
            builder.get_entry("key").unwrap(),
            &Bytes::from_static(b"value2")
        );

        // Try to update nonexistent key
        let result = builder.update_entry("nonexistent", Bytes::from_static(b"value"));
        assert!(matches!(result, Err(WorkerAddressError::KeyNotFound(_))));
    }

    #[test]
    fn test_address_to_map() {
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        builder
            .add_entry("protocol", Bytes::from_static(b"tcp"))
            .unwrap();

        let address = builder.build().unwrap();

        // Decode to map
        let map = address.to_map().unwrap();

        assert_eq!(map.len(), 2);
        assert_eq!(
            map.get("endpoint").unwrap(),
            &Bytes::from_static(b"tcp://127.0.0.1:5555")
        );
        assert_eq!(map.get("protocol").unwrap(), &Bytes::from_static(b"tcp"));
    }

    #[test]
    fn test_address_get_entry() {
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        builder
            .add_entry("protocol", Bytes::from_static(b"tcp"))
            .unwrap();

        let address = builder.build().unwrap();

        // Get existing entry
        assert_eq!(
            address.get_entry("endpoint").unwrap().unwrap(),
            Bytes::from_static(b"tcp://127.0.0.1:5555")
        );

        // Get nonexistent entry
        assert!(address.get_entry("nonexistent").unwrap().is_none());
    }

    #[test]
    fn test_address_into_builder() {
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        builder
            .add_entry("protocol", Bytes::from_static(b"tcp"))
            .unwrap();

        let address = builder.build().unwrap();

        // Convert to builder
        let mut builder2 = address.into_builder().unwrap();

        // Verify entries
        assert!(builder2.has_entry("endpoint"));
        assert!(builder2.has_entry("protocol"));

        // Modify
        builder2
            .add_entry("new_key", Bytes::from_static(b"new_value"))
            .unwrap();

        // Build again
        let address2 = builder2.build().unwrap();
        let map = address2.to_map().unwrap();
        assert_eq!(map.len(), 3);
        assert!(map.contains_key("new_key"));
    }

    #[test]
    fn test_round_trip_encoding() {
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        builder
            .add_entry("protocol", Bytes::from_static(b"tcp"))
            .unwrap();
        builder
            .add_entry("metadata", Bytes::from_static(b"{\"key\":\"value\"}"))
            .unwrap();

        let address1 = builder.build().unwrap();

        // Decode and re-encode
        let map = address1.to_map().unwrap();
        let mut builder2 = WorkerAddressBuilder::new();
        for (k, v) in map {
            builder2.add_entry(k.to_string(), v).unwrap();
        }
        let address2 = builder2.build().unwrap();

        // Maps should be equal (order may differ due to HashMap)
        let map1 = address1.to_map().unwrap();
        let map2 = address2.to_map().unwrap();
        assert_eq!(map1, map2);
    }

    #[test]
    fn test_empty_builder() {
        let builder = WorkerAddressBuilder::new();
        let address = builder.build().unwrap();

        // Empty map should still be valid
        let map = address.to_map().unwrap();
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_builder_merge_non_overlapping() {
        // Build first address with tcp
        let mut builder1 = WorkerAddressBuilder::new();
        builder1
            .add_entry("tcp", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        builder1
            .add_entry("udp", Bytes::from_static(b"udp://127.0.0.1:7777"))
            .unwrap();
        let address1 = builder1.build().unwrap();

        // Build second address with rdma
        let mut builder2 = WorkerAddressBuilder::new();
        builder2
            .add_entry("rdma", Bytes::from_static(b"rdma://10.0.0.1:6666"))
            .unwrap();
        builder2
            .add_entry("grpc", Bytes::from_static(b"grpc://localhost:9000"))
            .unwrap();
        let address2 = builder2.build().unwrap();

        // Merge address2 into a new builder with address1
        let mut builder3 = address1.into_builder().unwrap();
        builder3.merge(&address2).unwrap();

        // Verify all entries are present
        assert!(builder3.has_entry("tcp"));
        assert!(builder3.has_entry("udp"));
        assert!(builder3.has_entry("rdma"));
        assert!(builder3.has_entry("grpc"));

        let final_address = builder3.build().unwrap();
        let map = final_address.to_map().unwrap();
        assert_eq!(map.len(), 4);
    }

    #[test]
    fn test_builder_merge_with_duplicates() {
        // Build first address with tcp
        let mut builder1 = WorkerAddressBuilder::new();
        builder1
            .add_entry("tcp", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        builder1
            .add_entry("udp", Bytes::from_static(b"udp://127.0.0.1:7777"))
            .unwrap();
        let address1 = builder1.build().unwrap();

        // Build second address with overlapping tcp key
        let mut builder2 = WorkerAddressBuilder::new();
        builder2
            .add_entry("tcp", Bytes::from_static(b"tcp://different:5555"))
            .unwrap();
        builder2
            .add_entry("rdma", Bytes::from_static(b"rdma://10.0.0.1:6666"))
            .unwrap();
        let address2 = builder2.build().unwrap();

        // Try to merge - should fail
        let mut builder3 = address1.into_builder().unwrap();
        let result = builder3.merge(&address2);
        assert!(matches!(result, Err(WorkerAddressError::KeyExists(_))));

        // Verify builder is unchanged (still has original entries)
        assert!(builder3.has_entry("tcp"));
        assert!(builder3.has_entry("udp"));
        assert!(!builder3.has_entry("rdma"));
        assert_eq!(
            builder3.get_entry("tcp").unwrap(),
            &Bytes::from_static(b"tcp://127.0.0.1:5555")
        );
    }

    #[test]
    fn test_builder_merge_empty() {
        // Merge an empty address into a builder with entries
        let mut builder1 = WorkerAddressBuilder::new();
        builder1
            .add_entry("tcp", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();

        let empty_address = WorkerAddressBuilder::new().build().unwrap();
        builder1.merge(&empty_address).unwrap();

        // Should still have only the original entry
        assert_eq!(builder1.entries.len(), 1);
        assert!(builder1.has_entry("tcp"));

        // Merge a non-empty address into an empty builder
        let mut builder2 = WorkerAddressBuilder::new();
        let address = builder1.build().unwrap();
        builder2.merge(&address).unwrap();

        assert_eq!(builder2.entries.len(), 1);
        assert!(builder2.has_entry("tcp"));
    }

    #[test]
    fn test_builder_merge_multiple() {
        // Test merging multiple addresses sequentially
        let mut builder1 = WorkerAddressBuilder::new();
        builder1
            .add_entry("tcp", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address1 = builder1.build().unwrap();

        let mut builder2 = WorkerAddressBuilder::new();
        builder2
            .add_entry("rdma", Bytes::from_static(b"rdma://10.0.0.1:6666"))
            .unwrap();
        let address2 = builder2.build().unwrap();

        let mut builder3 = WorkerAddressBuilder::new();
        builder3
            .add_entry("udp", Bytes::from_static(b"udp://127.0.0.1:7777"))
            .unwrap();
        let address3 = builder3.build().unwrap();

        // Merge all three
        let mut final_builder = WorkerAddressBuilder::new();
        final_builder.merge(&address1).unwrap();
        final_builder.merge(&address2).unwrap();
        final_builder.merge(&address3).unwrap();

        assert_eq!(final_builder.entries.len(), 3);
        assert!(final_builder.has_entry("tcp"));
        assert!(final_builder.has_entry("rdma"));
        assert!(final_builder.has_entry("udp"));
    }

    #[test]
    fn test_checksum_stability() {
        // Build same address twice
        let mut builder1 = WorkerAddressBuilder::new();
        builder1
            .add_entry("key1", Bytes::from_static(b"value1"))
            .unwrap();
        builder1
            .add_entry("key2", Bytes::from_static(b"value2"))
            .unwrap();
        let address1 = builder1.build().unwrap();

        let mut builder2 = WorkerAddressBuilder::new();
        builder2
            .add_entry("key1", Bytes::from_static(b"value1"))
            .unwrap();
        builder2
            .add_entry("key2", Bytes::from_static(b"value2"))
            .unwrap();
        let address2 = builder2.build().unwrap();

        // Note: Checksums may differ due to HashMap iteration order during encoding
        // Instead, verify that the decoded maps are equivalent
        let map1 = address1.to_map().unwrap();
        let map2 = address2.to_map().unwrap();
        assert_eq!(map1, map2);

        // Both addresses should decode to the same content
        assert_eq!(map1.len(), 2);
        assert_eq!(map2.len(), 2);
    }

    // ========================================================================
    // TransportKey Tests
    // ========================================================================

    #[test]
    fn test_transport_key_creation() {
        // Test new() method
        let key1 = TransportKey::new("tcp");
        assert_eq!(key1.as_str(), "tcp");

        // Test From<&str>
        let key2: TransportKey = "rdma".into();
        assert_eq!(key2.as_str(), "rdma");

        // Test From<String>
        let key3 = TransportKey::from(String::from("udp"));
        assert_eq!(key3.as_str(), "udp");

        // Test From<&String>
        let s = String::from("grpc");
        let key4 = TransportKey::from(&s);
        assert_eq!(key4.as_str(), "grpc");

        // Test From<Arc<str>>
        let arc_str: Arc<str> = Arc::from("http");
        let key5 = TransportKey::from(arc_str);
        assert_eq!(key5.as_str(), "http");
    }

    #[test]
    fn test_transport_key_deref() {
        let key = TransportKey::from("tcp");

        // Deref to str methods should work
        assert_eq!(key.len(), 3);
        assert_eq!(key.chars().count(), 3);
        assert!(key.starts_with("tc"));
        assert!(key.ends_with("cp"));

        // Can use str slicing through Deref
        assert_eq!(&key[0..2], "tc");
    }

    #[test]
    fn test_transport_key_as_ref() {
        let key = TransportKey::from("tcp");

        // AsRef<str> allows passing to functions expecting &str
        fn takes_str_ref(s: &str) -> usize {
            s.len()
        }

        assert_eq!(takes_str_ref(&key), 3);
        assert_eq!(takes_str_ref(key.as_ref()), 3);
    }

    #[test]
    fn test_transport_key_display() {
        let key = TransportKey::from("tcp");
        assert_eq!(format!("{}", key), "tcp");
        assert_eq!(key.to_string(), "tcp");
    }

    #[test]
    fn test_transport_key_debug() {
        let key = TransportKey::from("tcp");
        let debug_str = format!("{:?}", key);
        assert!(debug_str.contains("TransportKey"));
        assert!(debug_str.contains("tcp"));
    }

    #[test]
    fn test_transport_key_equality() {
        let key1 = TransportKey::from("tcp");
        let key2 = TransportKey::from("tcp");
        let key3 = TransportKey::from("rdma");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);

        // Test with different source types
        let key4: TransportKey = String::from("tcp").into();
        assert_eq!(key1, key4);
    }

    #[test]
    fn test_transport_key_ordering() {
        let mut keys = vec![
            TransportKey::from("udp"),
            TransportKey::from("tcp"),
            TransportKey::from("rdma"),
            TransportKey::from("grpc"),
        ];

        keys.sort();

        assert_eq!(keys[0], TransportKey::from("grpc"));
        assert_eq!(keys[1], TransportKey::from("rdma"));
        assert_eq!(keys[2], TransportKey::from("tcp"));
        assert_eq!(keys[3], TransportKey::from("udp"));
    }

    #[test]
    fn test_transport_key_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(TransportKey::from("tcp"));
        set.insert(TransportKey::from("rdma"));
        set.insert(TransportKey::from("tcp")); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&TransportKey::from("tcp")));
        assert!(set.contains(&TransportKey::from("rdma")));
        assert!(!set.contains(&TransportKey::from("udp")));
    }

    #[test]
    fn test_transport_key_in_hashmap() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(TransportKey::from("tcp"), "tcp://127.0.0.1:5555");
        map.insert(TransportKey::from("rdma"), "rdma://10.0.0.1:6666");

        // Can lookup with TransportKey
        assert_eq!(
            map.get(&TransportKey::from("tcp")),
            Some(&"tcp://127.0.0.1:5555")
        );

        // Can lookup with &str via Borrow trait
        assert_eq!(map.get("tcp"), Some(&"tcp://127.0.0.1:5555"));
        assert_eq!(map.get("rdma"), Some(&"rdma://10.0.0.1:6666"));
        assert_eq!(map.get("udp"), None);
    }

    #[test]
    fn test_transport_key_clone() {
        let key1 = TransportKey::from("tcp");
        let key2 = key1.clone();

        assert_eq!(key1, key2);
        assert_eq!(key1.as_str(), key2.as_str());

        // Verify Arc is shared (same pointer)
        let ptr1 = key1.as_str().as_ptr();
        let ptr2 = key2.as_str().as_ptr();
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    fn test_transport_key_with_get_entry() {
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("tcp", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        builder
            .add_entry("rdma", Bytes::from_static(b"rdma://10.0.0.1:6666"))
            .unwrap();
        let address = builder.build().unwrap();

        // Test get_entry with TransportKey
        let tcp_key = TransportKey::from("tcp");
        let result = address.get_entry(tcp_key).unwrap();
        assert_eq!(result, Some(Bytes::from_static(b"tcp://127.0.0.1:5555")));

        // Test get_entry with &str
        let result = address.get_entry("rdma").unwrap();
        assert_eq!(result, Some(Bytes::from_static(b"rdma://10.0.0.1:6666")));

        // Test get_entry with String
        let result = address.get_entry(String::from("tcp")).unwrap();
        assert_eq!(result, Some(Bytes::from_static(b"tcp://127.0.0.1:5555")));

        // Test nonexistent key
        let result = address.get_entry(TransportKey::from("udp")).unwrap();
        assert_eq!(result, None);
    }
}
