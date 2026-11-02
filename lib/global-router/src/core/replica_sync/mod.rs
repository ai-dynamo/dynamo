// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The gRPC replica-sync transport (DEP #17639): the contract, its
//! conversions to and from the router core's session bindings, and the
//! stream that carries them, with each replica's requests in flight,
//! between Global Frontend replicas.
//!
//! The contract is `dynamo_global_common::replica_sync`: the proto under
//! `lib/global-common/proto/replica_sync.proto` and its generated code,
//! re-exported here as [`proto`]. [`wire`] converts between the generated
//! messages and the core's [`AffinityBindingEvent`].
//!
//! [`ReplicaHub`] is the transport's state in one replica: it publishes the
//! core's binding changes and this replica's in-flight counts as frames and
//! hands inbound frames to the core and to the frontends' counts. [`server`]
//! serves the frames to subscribing peers as routes on the replica's API
//! router, and [`dialer`] subscribes to each peer and applies what it
//! sends. [`discovery`] keeps the hub's peers matching a set published
//! from outside, such as the Global Frontend's poll of Global Discovery.
//!
//! [`AffinityBindingEvent`]: dynamo_kv_router::services::selection::AffinityBindingEvent

pub mod dialer;
pub mod discovery;
mod selector;
pub mod server;
pub mod transport;
pub mod wire;

pub use discovery::follow_peer_set;
pub use dynamo_global_common::replica_sync::{CONTRACT_MARKER, proto};
pub(in crate::core) use selector::GrpcTransport;
pub use transport::{GrpcReplicaSync, ReplicaHub, ReplicaSyncStats, peer_url};
pub use wire::{WireError, from_wire_binding, to_wire_binding};
