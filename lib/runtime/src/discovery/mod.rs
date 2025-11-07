// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::component::TransportType;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

mod mock;
pub use mock::{MockDiscoveryClient, SharedMockRegistry};

pub mod utils;
pub use utils::watch_and_extract_field;

/// Query key for prefix-based discovery queries
/// Supports hierarchical queries from all endpoints down to specific endpoints
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DiscoveryKey {
    /// Query all endpoints in the system
    AllEndpoints,
    /// Query all endpoints in a specific namespace
    NamespacedEndpoints {
        namespace: String,
    },
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
    AllModelCards,
    NamespacedModelCards {
        namespace: String,
    },
    ComponentModelCards {
        namespace: String,
        component: String,
    },
    EndpointModelCards {
        namespace: String,
        component: String,
        endpoint: String,
    },
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
    ModelCard {
        namespace: String,
        component: String,
        endpoint: String,
        /// ModelDeploymentCard serialized as JSON
        /// This allows lib/runtime to remain independent of lib/llm types
        /// DiscoverySpec.from_model_card() and DiscoveryInstance.deserialize_model_card() are ergonomic helpers to create and deserialize the model card.
        card_json: serde_json::Value,
    },
}

impl DiscoverySpec {
    /// Creates a ModelCard discovery spec from a serializable type
    /// The card will be serialized to JSON to avoid cross-crate dependencies
    pub fn from_model_card<T>(
        namespace: String,
        component: String,
        endpoint: String,
        card: &T,
    ) -> crate::Result<Self>
    where
        T: Serialize,
    {
        let card_json = serde_json::to_value(card)?;
        Ok(Self::ModelCard {
            namespace,
            component,
            endpoint,
            card_json,
        })
    }

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
            Self::ModelCard {
                namespace,
                component,
                endpoint,
                card_json,
            } => DiscoveryInstance::ModelCard {
                namespace,
                component,
                endpoint,
                instance_id,
                card_json,
            },
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
    ModelCard {
        namespace: String,
        component: String,
        endpoint: String,
        instance_id: u64,
        /// ModelDeploymentCard serialized as JSON
        /// This allows lib/runtime to remain independent of lib/llm types
        card_json: serde_json::Value,
    },
}

impl DiscoveryInstance {
    /// Returns the instance ID for this discovery instance
    pub fn instance_id(&self) -> u64 {
        match self {
            Self::Endpoint(inst) => inst.instance_id,
            Self::ModelCard { instance_id, .. } => *instance_id,
        }
    }

    /// Deserializes the model card JSON into the specified type T
    /// Returns an error if this is not a ModelCard instance or if deserialization fails
    pub fn deserialize_model_card<T>(&self) -> crate::Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        match self {
            Self::ModelCard { card_json, .. } => Ok(serde_json::from_value(card_json.clone())?),
            Self::Endpoint(_) => {
                crate::raise!("Cannot deserialize model card from Endpoint instance")
            }
        }
    }
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
