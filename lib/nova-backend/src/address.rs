// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal address builder for constructing WorkerAddress instances.
//!
//! This module provides the builder pattern for creating WorkerAddress instances
//! from transport-specific endpoint data. It is internal to nova-backend.

use bytes::Bytes;
use dynamo_nova_common::{WorkerAddress, WorkerAddressError};
use std::collections::HashMap;
use std::sync::Arc;

/// Builder for constructing WorkerAddress instances.
///
/// This provides a mutable interface for collecting transport endpoints
/// before encoding them into the immutable WorkerAddress format.
#[derive(Debug, Clone, Default)]
pub(crate) struct WorkerAddressBuilder {
    entries: HashMap<String, Bytes>,
}

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

    /// Check if a key exists in the map.
    #[allow(dead_code)]
    pub fn has_entry(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Get a reference to an entry's value.
    #[allow(dead_code)]
    pub fn get_entry(&self, key: &str) -> Option<&Bytes> {
        self.entries.get(key)
    }

    /// Merge another WorkerAddress into this builder.
    ///
    /// This decodes the other address and attempts to add all its entries to this builder.
    /// If any key from the other address already exists in this builder, returns an error
    /// and leaves the builder unchanged.
    pub fn merge(&mut self, other: &WorkerAddress) -> Result<(), WorkerAddressError> {
        let map = decode_to_map(other.as_bytes())?;

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
        // Convert HashMap<String, Bytes> to HashMap<String, Vec<u8>> for MessagePack
        let serializable: HashMap<String, Vec<u8>> = self
            .entries
            .into_iter()
            .map(|(k, v)| (k, v.to_vec()))
            .collect();

        // Encode to MessagePack
        let encoded = rmp_serde::to_vec(&serializable)?;

        Ok(WorkerAddress::from_encoded(encoded))
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let mut builder = WorkerAddressBuilder::new();

        builder
            .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        builder
            .add_entry("protocol", Bytes::from_static(b"tcp"))
            .unwrap();

        assert!(builder.has_entry("endpoint"));
        assert!(builder.has_entry("protocol"));
        assert!(!builder.has_entry("nonexistent"));

        let address = builder.build().unwrap();
        assert!(!address.as_bytes().is_empty());

        // Verify we can read the entries back
        let entry = address.get_entry("endpoint").unwrap();
        assert_eq!(entry, Some(Bytes::from_static(b"tcp://127.0.0.1:5555")));
    }

    #[test]
    fn test_builder_add_duplicate_key() {
        let mut builder = WorkerAddressBuilder::new();

        builder
            .add_entry("key", Bytes::from_static(b"value1"))
            .unwrap();

        let result = builder.add_entry("key", Bytes::from_static(b"value2"));
        assert!(matches!(result, Err(WorkerAddressError::KeyExists(_))));
    }

    #[test]
    fn test_builder_merge() {
        // Build first address
        let mut builder1 = WorkerAddressBuilder::new();
        builder1
            .add_entry("tcp", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address1 = builder1.build().unwrap();

        // Build second address
        let mut builder2 = WorkerAddressBuilder::new();
        builder2
            .add_entry("rdma", Bytes::from_static(b"rdma://10.0.0.1:6666"))
            .unwrap();
        let address2 = builder2.build().unwrap();

        // Merge both into a new builder
        let mut builder3 = WorkerAddressBuilder::new();
        builder3.merge(&address1).unwrap();
        builder3.merge(&address2).unwrap();

        let final_address = builder3.build().unwrap();

        // Verify both entries are present
        assert_eq!(
            final_address.get_entry("tcp").unwrap(),
            Some(Bytes::from_static(b"tcp://127.0.0.1:5555"))
        );
        assert_eq!(
            final_address.get_entry("rdma").unwrap(),
            Some(Bytes::from_static(b"rdma://10.0.0.1:6666"))
        );
    }

    #[test]
    fn test_builder_merge_with_conflict() {
        let mut builder1 = WorkerAddressBuilder::new();
        builder1
            .add_entry("tcp", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address1 = builder1.build().unwrap();

        let mut builder2 = WorkerAddressBuilder::new();
        builder2
            .add_entry("tcp", Bytes::from_static(b"tcp://different:5555"))
            .unwrap();
        let address2 = builder2.build().unwrap();

        // Merge first address
        let mut builder3 = WorkerAddressBuilder::new();
        builder3.merge(&address1).unwrap();

        // Try to merge conflicting address - should fail
        let result = builder3.merge(&address2);
        assert!(matches!(result, Err(WorkerAddressError::KeyExists(_))));

        // Builder should be unchanged
        assert!(builder3.has_entry("tcp"));
        assert_eq!(
            builder3.get_entry("tcp").unwrap(),
            &Bytes::from_static(b"tcp://127.0.0.1:5555")
        );
    }

    #[test]
    fn test_empty_builder() {
        let builder = WorkerAddressBuilder::new();
        let address = builder.build().unwrap();

        // Empty address should still be valid
        let transports = address.available_transports().unwrap();
        assert_eq!(transports.len(), 0);
    }

    // ========================================================================
    // Integration tests: Verify WorkerAddressBuilder (nova-backend) produces
    // addresses that WorkerAddress (nova-common) can correctly decode.
    // These tests ensure the two crates stay in sync on the wire format.
    // ========================================================================

    #[test]
    fn test_builder_address_integration_get_entry() {
        // Build an address with multiple entries
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("tcp", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        builder
            .add_entry("rdma", Bytes::from_static(b"rdma://10.0.0.1:6666"))
            .unwrap();
        builder
            .add_entry("binary_data", Bytes::from_static(&[0x00, 0x01, 0x02, 0xFF]))
            .unwrap();
        let address = builder.build().unwrap();

        // Verify WorkerAddress::get_entry() correctly decodes each entry
        assert_eq!(
            address.get_entry("tcp").unwrap(),
            Some(Bytes::from_static(b"tcp://127.0.0.1:5555"))
        );
        assert_eq!(
            address.get_entry("rdma").unwrap(),
            Some(Bytes::from_static(b"rdma://10.0.0.1:6666"))
        );
        assert_eq!(
            address.get_entry("binary_data").unwrap(),
            Some(Bytes::from_static(&[0x00, 0x01, 0x02, 0xFF]))
        );
        assert_eq!(address.get_entry("nonexistent").unwrap(), None);
    }

    #[test]
    fn test_builder_address_integration_available_transports() {
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("tcp", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        builder
            .add_entry("rdma", Bytes::from_static(b"rdma://10.0.0.1:6666"))
            .unwrap();
        builder
            .add_entry("grpc", Bytes::from_static(b"grpc://localhost:9000"))
            .unwrap();
        let address = builder.build().unwrap();

        // Verify WorkerAddress::available_transports() returns all keys
        let transports = address.available_transports().unwrap();
        assert_eq!(transports.len(), 3);
        assert!(transports.contains(&dynamo_nova_common::TransportKey::from("tcp")));
        assert!(transports.contains(&dynamo_nova_common::TransportKey::from("rdma")));
        assert!(transports.contains(&dynamo_nova_common::TransportKey::from("grpc")));
    }

    #[test]
    fn test_builder_address_integration_checksum_stability() {
        // Build same address twice - checksums should match
        let mut builder1 = WorkerAddressBuilder::new();
        builder1
            .add_entry("key", Bytes::from_static(b"value"))
            .unwrap();
        let address1 = builder1.build().unwrap();

        let mut builder2 = WorkerAddressBuilder::new();
        builder2
            .add_entry("key", Bytes::from_static(b"value"))
            .unwrap();
        let address2 = builder2.build().unwrap();

        // Same content should produce same checksum
        assert_eq!(address1.checksum(), address2.checksum());

        // Different content should produce different checksum
        let mut builder3 = WorkerAddressBuilder::new();
        builder3
            .add_entry("key", Bytes::from_static(b"different"))
            .unwrap();
        let address3 = builder3.build().unwrap();
        assert_ne!(address1.checksum(), address3.checksum());
    }

    #[test]
    fn test_builder_address_integration_bytes_roundtrip() {
        // Build an address
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("endpoint", Bytes::from_static(b"test://value"))
            .unwrap();
        let address = builder.build().unwrap();

        // Get raw bytes and create new address via from_encoded
        let raw_bytes = address.to_bytes();
        let address2 = WorkerAddress::from_encoded(raw_bytes);

        // Both should be equal and decode the same
        assert_eq!(address, address2);
        assert_eq!(address.checksum(), address2.checksum());
        assert_eq!(
            address.get_entry("endpoint").unwrap(),
            address2.get_entry("endpoint").unwrap()
        );
    }

    #[test]
    fn test_builder_address_integration_serde_roundtrip() {
        // Build an address
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("tcp", Bytes::from_static(b"tcp://127.0.0.1:5555"))
            .unwrap();
        let address = builder.build().unwrap();

        // Serialize to JSON and back
        let json = serde_json::to_string(&address).unwrap();
        let deserialized: WorkerAddress = serde_json::from_str(&json).unwrap();

        // Should be equal and decode correctly
        assert_eq!(address, deserialized);
        assert_eq!(
            address.get_entry("tcp").unwrap(),
            deserialized.get_entry("tcp").unwrap()
        );
    }
}
