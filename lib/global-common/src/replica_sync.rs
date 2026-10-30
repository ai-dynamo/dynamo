// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The gRPC contract Global Router replicas sync bookings and session
//! bindings over (DEP #17639). The schema is `proto/replica_sync.proto`;
//! [`proto`] is its generated code, present with the `grpc` feature.
//!
//! The conversions between the wire types and the router core's own event
//! types live in `dynamo-global-router`, which depends on the core; this
//! crate holds only the contract.

/// The contract version a subscriber sends in `SubscribeRequest.contract_marker`.
///
/// Additive changes to `dynamo.globalrouter.replica.v1` keep this value. The
/// server must reject a subscriber whose marker differs with
/// `FAILED_PRECONDITION`; the check itself lands with the stream (C2b).
pub const CONTRACT_MARKER: u32 = 1;

/// Generated code for package `dynamo.globalrouter.replica.v1`.
#[cfg(feature = "grpc")]
pub mod proto {
    #![allow(clippy::all)]
    tonic::include_proto!("dynamo.globalrouter.replica.v1");

    /// The serialized file descriptor set, for gRPC reflection.
    pub const FILE_DESCRIPTOR_SET: &[u8] =
        tonic::include_file_descriptor_set!("replica_sync_descriptor");
}
