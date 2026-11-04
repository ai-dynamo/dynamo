// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The gRPC replica-sync contract (DEP #17639) as the Global Router uses
//! it: the proto's generated code and its conversions to and from the
//! router core's session bindings.
//!
//! The contract is `dynamo_global_common::replica_sync`: the proto under
//! `lib/global-common/proto/replica_sync.proto` and its generated code,
//! re-exported here as [`proto`]. [`wire`] converts between the generated
//! messages and the core's [`AffinityBindingEvent`]. The stream that
//! carries them between Global Frontend replicas lands separately.
//!
//! [`AffinityBindingEvent`]: dynamo_kv_router::services::selection::AffinityBindingEvent

pub mod wire;

pub use dynamo_global_common::replica_sync::{CONTRACT_MARKER, proto};
pub use wire::{WireError, from_wire_binding, to_wire_binding};
