// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared identities for routing partitions, logical KV indexers, and cluster-local producer pools.
//!
//! Hashed indexer identity material is resolved on control paths. Mutation and query paths carry
//! only those fixed-size values or physical lane indices.

pub use dynamo_router_protocol::identity::{
    DEFAULT_ROUTING_GROUP, RoutingPartitionId, RoutingPartitionRef,
};

/// The routing group a request or worker record belongs to when it names none.
pub fn default_routing_group() -> String {
    DEFAULT_ROUTING_GROUP.to_string()
}
