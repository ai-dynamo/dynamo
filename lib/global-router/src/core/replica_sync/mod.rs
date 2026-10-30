// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The gRPC replica-sync transport (DEP #17639): the contract, its
//! conversions to and from the router core's event types, and the stream
//! that carries them between Global Frontend replicas.
//!
//! The contract is `dynamo_global_common::replica_sync`: the proto under
//! `lib/global-common/proto/replica_sync.proto` and its generated code,
//! re-exported here as [`proto`]. [`wire`] converts between the generated
//! messages and the core's [`ActiveSequenceEvent`] and
//! [`AffinityBindingEvent`], which the core's host-mode replica channels
//! carry per partition.
//!
//! [`ReplicaHub`] is the transport's state in one replica: it takes the
//! core's outbound events through the host replica channels, batches them
//! into frames, and hands inbound frames back to the core. [`server`]
//! serves the frames to subscribing peers as routes on the replica's API
//! router, and [`dialer`] subscribes to each peer and applies what it
//! sends. The built-in ZMQ mesh (`--replica-sync-port`) is a separate
//! transport the core runs itself; a replica uses one or the other.
//!
//! [`ActiveSequenceEvent`]: dynamo_kv_router::protocols::ActiveSequenceEvent
//! [`AffinityBindingEvent`]: dynamo_kv_router::services::selection::AffinityBindingEvent

pub mod dialer;
pub mod server;
pub mod transport;
pub mod wire;

pub use dynamo_global_common::replica_sync::{CONTRACT_MARKER, proto};
pub use transport::{GrpcReplicaSync, ReplicaHub, ReplicaSyncStats, peer_url};
pub use wire::{WireError, from_wire_binding, from_wire_event, to_wire_binding, to_wire_event};
