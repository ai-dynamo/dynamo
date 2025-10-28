// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

mod mock;
pub use mock::{MockDiscoveryClient, SharedMockRegistry};

mod key_value;
pub use key_value::KeyValueDiscoveryClient;

/// Query key to refer to discovery objects
/// Only Endpoints can be queried and discovered for now
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DiscoveryKey {
    /// Query key for endpoint instances
    Endpoint {
        namespace: String,
        component: String,
        endpoint: String,
    },
}

/// Objects in the discovery plane
/// Only endpoints can be discovered for now
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum DiscoveryInstance {
    Endpoint {
        namespace: String,
        component: String,
        endpoint: String,
        instance_id: u64,
    },
}

/// Events emitted by the discovery client watch stream
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryEvent {
    /// A new instance was added
    Added(DiscoveryInstance),
    /// An instance was removed (identified by instance_id)
    Removed(u64),
}

/// Stream type for discovery events
pub type DiscoveryStream = Pin<Box<dyn Stream<Item = Result<DiscoveryEvent>> + Send>>;

/// Discovery client trait for service discovery across different backends
#[async_trait]
pub trait DiscoveryClient: Send + Sync {
    /// Returns a unique identifier for this worker (e.g lease id if using etcd or generated id for memory store)
    /// Discovery objects created by this worker will be associated with this id.
    fn instance_id(&self) -> u64;

    /// Registers an object in the discovery plane with the instance id
    async fn serve(&self, key: DiscoveryKey) -> Result<DiscoveryInstance>;

    /// Returns a stream of discovery events (Added/Removed) for the given discovery key
    async fn list_and_watch(&self, key: DiscoveryKey) -> Result<DiscoveryStream>;
}
