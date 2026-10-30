// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The gRPC replica-sync contract and its conversions to and from the
//! router core's event types (DEP #17639).
//!
//! The contract is `dynamo_global_common::replica_sync`: the proto under
//! `lib/global-common/proto/replica_sync.proto` and its generated code,
//! re-exported here as [`proto`]. [`wire`] converts between the generated
//! messages and the core's [`ActiveSequenceEvent`] and
//! [`AffinityBindingEvent`], which the core's host-mode replica channels
//! carry per partition. Nothing here serves or dials the stream yet; the
//! transport lands with the next step of the stack.
//!
//! [`ActiveSequenceEvent`]: dynamo_kv_router::protocols::ActiveSequenceEvent
//! [`AffinityBindingEvent`]: dynamo_kv_router::services::selection::AffinityBindingEvent

pub mod wire;

pub use dynamo_global_common::replica_sync::{CONTRACT_MARKER, proto};
pub use wire::{WireError, from_wire_binding, from_wire_event, to_wire_binding, to_wire_event};
