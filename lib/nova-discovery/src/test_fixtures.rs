// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test fixtures and helpers for discovery tests.
//!
//! This module provides common test utilities for constructing WorkerAddress
//! instances for testing purposes.

use std::collections::HashMap;

use crate::peer::WorkerAddress;

/// Create a WorkerAddress from a map of entries.
///
/// This encodes the entries using MessagePack format, matching the format
/// used by nova-backend's WorkerAddressBuilder.
fn make_address_from_entries(entries: &[(&str, &[u8])]) -> WorkerAddress {
    let map: HashMap<String, Vec<u8>> = entries
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_vec()))
        .collect();
    let encoded = rmp_serde::to_vec(&map).expect("Failed to encode test address");
    WorkerAddress::from_encoded(encoded)
}

/// Create a simple test address with a default TCP endpoint.
///
/// This is the most common test address used throughout the discovery tests.
pub fn make_test_address() -> WorkerAddress {
    make_address_from_entries(&[("endpoint", b"tcp://127.0.0.1:5555")])
}

/// Create a test address with a custom endpoint.
///
/// # Arguments
/// * `endpoint` - The endpoint bytes to use
///
/// # Example
/// ```ignore
/// let address = make_test_address_with_endpoint(b"tcp://127.0.0.1:8080");
/// ```
pub fn make_test_address_with_endpoint(endpoint: &[u8]) -> WorkerAddress {
    make_address_from_entries(&[("endpoint", endpoint)])
}

/// Create a test address with a custom port.
///
/// This is useful for tests that need multiple distinct addresses with different checksums.
///
/// # Arguments
/// * `port` - The port number to use
///
/// # Example
/// ```ignore
/// let address1 = make_test_address_with_port(5555);
/// let address2 = make_test_address_with_port(6666);
/// assert_ne!(address1.checksum(), address2.checksum());
/// ```
pub fn make_test_address_with_port(port: u16) -> WorkerAddress {
    let endpoint = format!("tcp://127.0.0.1:{}", port);
    make_test_address_with_endpoint(endpoint.as_bytes())
}

/// Create a test address with multiple transports.
///
/// This demonstrates the multi-transport capabilities of WorkerAddress.
///
/// # Example
/// ```ignore
/// let address = make_multiaddr_test_address();
/// let transports = address.available_transports().unwrap();
/// assert!(transports.contains(&TransportKey::from("tcp")));
/// assert!(transports.contains(&TransportKey::from("rdma")));
/// ```
#[allow(dead_code)]
pub fn make_multiaddr_test_address() -> WorkerAddress {
    make_address_from_entries(&[
        ("tcp", b"tcp://127.0.0.1:5555"),
        ("rdma", b"rdma://10.0.0.1:6666"),
    ])
}
