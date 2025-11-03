// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::component::TransportType;
use crate::Result;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

mod mock;
pub use mock::{MockDiscoveryClient, SharedMockRegistry};

/// Query key for prefix-based discovery queries
/// Supports hierarchical queries from all endpoints down to specific endpoints
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DiscoveryKey {
    /// Query all endpoints in the system
    AllEndpoints,
    /// Query all endpoints in a specific namespace
    NamespacedEndpoints { namespace: String },
    /// Query all endpoints in a namespace/component
    ComponentEndpoints {
        namespace: String,
        component: String,
    },
    /// Query a specific endpoint
    Endpoint {
        namespace: String,
        component: String,
        endpoint: String,
    },
    // TODO: Extend to support ModelCard queries:
    // - AllModels
    // - NamespacedModels { namespace }
    // - ComponentModels { namespace, component }
    // - Model { namespace, component, model_name }
}

/// Specification for registering objects in the discovery plane
/// Represents the input to the register() operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoverySpec {
    /// Endpoint specification for registration
    Endpoint {
        namespace: String,
        component: String,
        endpoint: String,
        /// Transport type and routing information
        transport: TransportType,
    },
    // TODO: Add ModelCard variant:
    // - ModelCard { namespace, component, model_name, card: ModelDeploymentCard }
}

impl DiscoverySpec {
    /// Attaches an instance ID to create a DiscoveryInstance
    pub fn with_instance_id(self, instance_id: u64) -> DiscoveryInstance {
        match self {
            Self::Endpoint {
                namespace,
                component,
                endpoint,
                transport,
            } => DiscoveryInstance::Endpoint(crate::component::Instance {
                namespace,
                component,
                endpoint,
                instance_id,
                transport,
            }),
        }
    }
}

/// Registered instances in the discovery plane
/// Represents objects that have been successfully registered with an instance ID
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum DiscoveryInstance {
    /// Registered endpoint instance - wraps the component::Instance directly
    Endpoint(crate::component::Instance),
    // TODO: Add ModelCard variant:
    // - ModelCard(ModelDeploymentCard) or similar
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
    async fn register(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance>;

    /// Returns a list of currently registered instances for the given discovery key
    /// This is a one-time snapshot without watching for changes
    async fn list(&self, key: DiscoveryKey) -> Result<Vec<DiscoveryInstance>>;

    /// Returns a stream of discovery events (Added/Removed) for the given discovery key
    async fn list_and_watch(&self, key: DiscoveryKey) -> Result<DiscoveryStream>;
}
