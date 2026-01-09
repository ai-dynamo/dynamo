// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tokio_util::sync::CancellationToken;

mod metadata;
pub use metadata::{DiscoveryMetadata, MetadataSnapshot};

mod mock;
pub use mock::{MockDiscovery, SharedMockRegistry};
mod kv_store;
pub use kv_store::KVStoreDiscovery;

mod kube;
pub use kube::{KubeDiscoveryClient, hash_pod_name};

pub mod utils;
use crate::component::TransportType;
pub use utils::watch_and_extract_field;

/// Query key for prefix-based discovery queries
/// Supports hierarchical queries from all endpoints down to specific endpoints
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DiscoveryQuery {
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
    AllModels,
    NamespacedModels {
        namespace: String,
    },
    ComponentModels {
        namespace: String,
        component: String,
    },
    EndpointModels {
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
    Model {
        namespace: String,
        component: String,
        endpoint: String,
        /// ModelDeploymentCard serialized as JSON
        /// This allows lib/runtime to remain independent of lib/llm types
        /// DiscoverySpec.from_model() and DiscoveryInstance.deserialize_model() are ergonomic helpers to create and deserialize the model card.
        card_json: serde_json::Value,
        /// Optional suffix appended after instance_id in the key path (e.g., for LoRA adapters)
        /// Key format: {namespace}/{component}/{endpoint}/{instance_id}[/{model_suffix}]
        model_suffix: Option<String>,
    },
}

impl DiscoverySpec {
    /// Creates a Model discovery spec from a serializable type
    /// The card will be serialized to JSON to avoid cross-crate dependencies
    pub fn from_model<T>(
        namespace: String,
        component: String,
        endpoint: String,
        card: &T,
    ) -> Result<Self>
    where
        T: Serialize,
    {
        Self::from_model_with_suffix(namespace, component, endpoint, card, None)
    }

    /// Creates a Model discovery spec with an optional suffix (e.g., for LoRA adapters)
    /// The suffix is appended after the instance_id in the key path
    pub fn from_model_with_suffix<T>(
        namespace: String,
        component: String,
        endpoint: String,
        card: &T,
        model_suffix: Option<String>,
    ) -> Result<Self>
    where
        T: Serialize,
    {
        let card_json = serde_json::to_value(card)?;
        Ok(Self::Model {
            namespace,
            component,
            endpoint,
            card_json,
            model_suffix,
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
            Self::Model {
                namespace,
                component,
                endpoint,
                card_json,
                model_suffix,
            } => DiscoveryInstance::Model {
                namespace,
                component,
                endpoint,
                instance_id,
                card_json,
                model_suffix,
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
    Model {
        namespace: String,
        component: String,
        endpoint: String,
        instance_id: u64,
        /// ModelDeploymentCard serialized as JSON
        /// This allows lib/runtime to remain independent of lib/llm types
        card_json: serde_json::Value,
        /// Optional suffix appended after instance_id in the key path (e.g., for LoRA adapters)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model_suffix: Option<String>,
    },
}

impl DiscoveryInstance {
    /// Returns the instance ID for this discovery instance
    pub fn instance_id(&self) -> u64 {
        match self {
            Self::Endpoint(inst) => inst.instance_id,
            Self::Model { instance_id, .. } => *instance_id,
        }
    }

    /// Deserializes the model JSON into the specified type T
    /// Returns an error if this is not a Model instance or if deserialization fails
    pub fn deserialize_model<T>(&self) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        match self {
            Self::Model { card_json, .. } => Ok(serde_json::from_value(card_json.clone())?),
            Self::Endpoint(_) => {
                anyhow::bail!("Cannot deserialize model from Endpoint instance")
            }
        }
    }

    /// Extracts the unique identifier for this discovery instance
    /// Used for tracking, diffing, and removal events
    pub fn id(&self) -> DiscoveryInstanceId {
        match self {
            Self::Endpoint(inst) => DiscoveryInstanceId::Endpoint(EndpointInstanceId {
                namespace: inst.namespace.clone(),
                component: inst.component.clone(),
                endpoint: inst.endpoint.clone(),
                instance_id: inst.instance_id,
            }),
            Self::Model {
                namespace,
                component,
                endpoint,
                instance_id,
                model_suffix,
                ..
            } => DiscoveryInstanceId::Model(ModelCardInstanceId {
                namespace: namespace.clone(),
                component: component.clone(),
                endpoint: endpoint.clone(),
                instance_id: *instance_id,
                model_suffix: model_suffix.clone(),
            }),
        }
    }
}

/// Unique identifier for an endpoint instance
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EndpointInstanceId {
    pub namespace: String,
    pub component: String,
    pub endpoint: String,
    pub instance_id: u64,
}

impl EndpointInstanceId {
    /// Converts to a path string: `{namespace}/{component}/{endpoint}/{instance_id:x}`
    pub fn to_path(&self) -> String {
        format!(
            "{}/{}/{}/{:x}",
            self.namespace, self.component, self.endpoint, self.instance_id
        )
    }

    /// Parses from a path string: `{namespace}/{component}/{endpoint}/{instance_id:x}`
    pub fn from_path(path: &str) -> Result<Self> {
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() != 4 {
            anyhow::bail!(
                "Invalid EndpointInstanceId path: expected 4 parts, got {}",
                parts.len()
            );
        }
        Ok(Self {
            namespace: parts[0].to_string(),
            component: parts[1].to_string(),
            endpoint: parts[2].to_string(),
            instance_id: u64::from_str_radix(parts[3], 16)
                .map_err(|e| anyhow::anyhow!("Invalid instance_id hex: {}", e))?,
        })
    }
}

/// Unique identifier for a model card instance
/// The combination of (namespace, component, endpoint, instance_id, model_suffix) uniquely identifies a model card
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelCardInstanceId {
    pub namespace: String,
    pub component: String,
    pub endpoint: String,
    pub instance_id: u64,
    /// None for base models, Some(slug) for LoRA adapters
    pub model_suffix: Option<String>,
}

impl ModelCardInstanceId {
    /// Converts to a path string: `{namespace}/{component}/{endpoint}/{instance_id:x}[/{model_suffix}]`
    pub fn to_path(&self) -> String {
        match &self.model_suffix {
            Some(suffix) => format!(
                "{}/{}/{}/{:x}/{}",
                self.namespace, self.component, self.endpoint, self.instance_id, suffix
            ),
            None => format!(
                "{}/{}/{}/{:x}",
                self.namespace, self.component, self.endpoint, self.instance_id
            ),
        }
    }

    /// Parses from a path string: `{namespace}/{component}/{endpoint}/{instance_id:x}[/{model_suffix}]`
    pub fn from_path(path: &str) -> Result<Self> {
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() < 4 || parts.len() > 5 {
            anyhow::bail!(
                "Invalid ModelCardInstanceId path: expected 4 or 5 parts, got {}",
                parts.len()
            );
        }
        Ok(Self {
            namespace: parts[0].to_string(),
            component: parts[1].to_string(),
            endpoint: parts[2].to_string(),
            instance_id: u64::from_str_radix(parts[3], 16)
                .map_err(|e| anyhow::anyhow!("Invalid instance_id hex: {}", e))?,
            model_suffix: parts.get(4).map(|s| s.to_string()),
        })
    }
}

/// Union of instance identifiers for different discovery object types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiscoveryInstanceId {
    Endpoint(EndpointInstanceId),
    Model(ModelCardInstanceId),
}

impl DiscoveryInstanceId {
    /// Returns the raw instance_id regardless of variant type
    pub fn instance_id(&self) -> u64 {
        match self {
            Self::Endpoint(eid) => eid.instance_id,
            Self::Model(mid) => mid.instance_id,
        }
    }

    /// Extracts the EndpointInstanceId, returning an error if this is a Model variant
    pub fn extract_endpoint_id(&self) -> Result<&EndpointInstanceId> {
        match self {
            Self::Endpoint(eid) => Ok(eid),
            Self::Model(_) => anyhow::bail!("Expected Endpoint variant, got Model"),
        }
    }

    /// Extracts the ModelCardInstanceId, returning an error if this is an Endpoint variant
    pub fn extract_model_id(&self) -> Result<&ModelCardInstanceId> {
        match self {
            Self::Model(mid) => Ok(mid),
            Self::Endpoint(_) => anyhow::bail!("Expected Model variant, got Endpoint"),
        }
    }
}

/// Events emitted by the discovery watch stream
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryEvent {
    /// A new instance was added
    Added(DiscoveryInstance),
    /// An instance was removed (identified by its unique ID)
    Removed(DiscoveryInstanceId),
}

/// Stream type for discovery events
pub type DiscoveryStream = Pin<Box<dyn Stream<Item = Result<DiscoveryEvent>> + Send>>;

/// Discovery trait for service discovery across different backends
#[async_trait]
pub trait Discovery: Send + Sync {
    /// Returns a unique identifier for this worker (e.g lease id if using etcd or generated id for memory store)
    /// Discovery objects created by this worker will be associated with this id.
    fn instance_id(&self) -> u64;

    /// Registers an object in the discovery plane with the instance id
    async fn register(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance>;

    /// Unregisters an instance from the discovery plane
    async fn unregister(&self, instance: DiscoveryInstance) -> Result<()>;

    /// Returns a list of currently registered instances for the given discovery query
    /// This is a one-time snapshot without watching for changes
    async fn list(&self, query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>>;

    /// Returns a stream of discovery events (Added/Removed) for the given discovery query
    /// The optional cancellation token can be used to stop the watch stream
    async fn list_and_watch(
        &self,
        query: DiscoveryQuery,
        cancel_token: Option<CancellationToken>,
    ) -> Result<DiscoveryStream>;
}
