// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test fixtures and helpers for discovery tests.
//!
//! This module provides common test utilities for constructing WorkerAddress
//! instances using the builder pattern.

use bytes::Bytes;

use crate::peer::WorkerAddress;

/// Create a simple test address with a default TCP endpoint.
///
/// This is the most common test address used throughout the discovery tests.
pub fn make_test_address() -> WorkerAddress {
    let mut builder = WorkerAddress::builder();
    builder
        .add_entry("endpoint", Bytes::from_static(b"tcp://127.0.0.1:5555"))
        .unwrap();
    builder.build().unwrap()
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
    let mut builder = WorkerAddress::builder();
    builder
        .add_entry("endpoint", Bytes::copy_from_slice(endpoint))
        .unwrap();
    builder.build().unwrap()
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
/// This demonstrates the multi-transport capabilities of the new WorkerAddress.
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
    let mut builder = WorkerAddress::builder();
    builder
        .add_entry("tcp", Bytes::from_static(b"tcp://127.0.0.1:5555"))
        .unwrap();
    builder
        .add_entry("rdma", Bytes::from_static(b"rdma://10.0.0.1:6666"))
        .unwrap();
    builder.build().unwrap()
}
