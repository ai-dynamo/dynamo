// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::protocols::EndpointId;
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
use crate::component::{DeviceType, TransportType};
pub use utils::watch_and_extract_field;

/// Transport kind for event plane - used for configuration and env var selection.
///
/// This enum represents the *type* of transport without connection details.
/// Use `EventTransport` when you need the full transport configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EventTransportKind {
    /// NATS Core pub/sub
    Nats,
    /// ZMQ pub/sub
    #[default]
    Zmq,
}

impl EventTransportKind {
    /// Parse from environment variable `DYN_EVENT_PLANE`.
    ///
    /// Returns `Zmq` if the variable is not set or is empty: ZMQ is the default
    /// event plane for all backends. NATS remains available as an explicit opt-in
    /// (`DYN_EVENT_PLANE=nats`). When you have access to a runtime, prefer
    /// [`DistributedRuntime::default_event_transport_kind`], which resolves the same
    /// default through the configured discovery backend.
    ///
    /// Returns an error for unrecognised values.
    pub fn from_env() -> Result<Self> {
        match std::env::var(crate::config::environment_names::event_plane::DYN_EVENT_PLANE)
            .as_deref()
        {
            Ok("nats") => Ok(Self::Nats),
            Ok("zmq") | Ok("") | Err(_) => Ok(Self::Zmq),
            Ok(other) => anyhow::bail!(
                "Invalid DYN_EVENT_PLANE value '{}'. Valid values: 'nats', 'zmq'",
                other
            ),
        }
    }

    /// Logs a warning if an invalid value is encountered.
    pub fn from_env_or_default() -> Self {
        Self::from_env().unwrap_or_else(|e| {
            tracing::warn!("{e}, defaulting to ZMQ");
            Self::Zmq
        })
    }

    /// Get the default codec for this transport kind.
    /// NATS defaults to JSON, ZMQ defaults to MsgPack.
    pub fn default_codec(&self) -> EventCodecKind {
        match self {
            Self::Nats => EventCodecKind::Json,
            Self::Zmq => EventCodecKind::Msgpack,
        }
    }
}

/// Codec kind for event plane serialization.
///
/// This enum represents the serialization format for event envelopes and payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventCodecKind {
    /// JSON codec - human-readable, good for debugging
    Json,
    /// MessagePack codec - compact binary format
    Msgpack,
}

impl EventCodecKind {
    /// Parse from environment variable `DYN_EVENT_PLANE_CODEC`.
    /// Returns None if not set, allowing transport to select default.
    /// Returns error for invalid values.
    pub fn from_env() -> Result<Option<Self>> {
        match std::env::var(crate::config::environment_names::event_plane::DYN_EVENT_PLANE_CODEC)
            .as_deref()
        {
            Err(_) => Ok(None), // Not set
            Ok("") => Ok(None), // Empty
            Ok("json") => Ok(Some(Self::Json)),
            Ok("msgpack") => Ok(Some(Self::Msgpack)),
            Ok(other) => anyhow::bail!(
                "Invalid DYN_EVENT_PLANE_CODEC value '{}'. Valid values: 'json', 'msgpack'",
                other
            ),
        }
    }

    /// Parse from environment variable with transport-specific default.
    /// Logs a warning if an invalid value is encountered.
    pub fn from_env_or_transport_default(transport: EventTransportKind) -> Self {
        Self::from_env()
            .unwrap_or_else(|e| {
                tracing::warn!(
                    "{}, defaulting to {:?} for {:?}",
                    e,
                    transport.default_codec(),
                    transport
                );
                None
            })
            .unwrap_or_else(|| transport.default_codec())
    }
}

/// Transport configuration for event plane channels.
///
/// This enum carries both the transport kind and its connection configuration.
/// Kept separate from `TransportType` (request plane) to distinguish event semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "config")]
pub enum EventTransport {
    /// NATS Core pub/sub - subject prefix for the channel
    Nats {
        /// Subject prefix (e.g., "namespace.dynamo.component.backend")
        subject_prefix: String,
    },
    /// ZMQ pub/sub - endpoint address (direct mode)
    Zmq {
        /// ZMQ endpoint (e.g., "tcp://host:port")
        endpoint: String,
    },
    /// ZMQ broker endpoints (broker mode) - for discovery of brokers
    ZmqBroker {
        /// XSUB endpoints (publishers connect here)
        xsub_endpoints: Vec<String>,
        /// XPUB endpoints (subscribers connect here)
        xpub_endpoints: Vec<String>,
    },
}

impl EventTransport {
    /// Get the transport kind
    pub fn kind(&self) -> EventTransportKind {
        match self {
            Self::Nats { .. } => EventTransportKind::Nats,
            Self::Zmq { .. } | Self::ZmqBroker { .. } => EventTransportKind::Zmq,
        }
    }

    /// Create a NATS transport with the given subject prefix
    pub fn nats(subject_prefix: impl Into<String>) -> Self {
        Self::Nats {
            subject_prefix: subject_prefix.into(),
        }
    }

    /// Create a ZMQ transport with the given endpoint
    pub fn zmq(endpoint: impl Into<String>) -> Self {
        Self::Zmq {
            endpoint: endpoint.into(),
        }
    }

    /// Get the subject prefix (NATS) or endpoint (ZMQ)
    /// For ZmqBroker, returns the first XSUB endpoint
    pub fn address(&self) -> &str {
        match self {
            Self::Nats { subject_prefix } => subject_prefix,
            Self::Zmq { endpoint } => endpoint,
            Self::ZmqBroker { xsub_endpoints, .. } => {
                xsub_endpoints.first().map(|s| s.as_str()).unwrap_or("")
            }
        }
    }
}

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
    /// Unified event channel query with optional scope filters
    EventChannels(EventChannelQuery),
}

/// Scope of an event channel.
///
/// Event scopes are exact ownership domains. A namespace-scoped query does not
/// match component- or endpoint-scoped publishers in that namespace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventScope {
    Namespace {
        name: String,
    },
    Component {
        namespace: String,
        component: String,
    },
    Endpoint {
        endpoint: EndpointId,
    },
}

impl EventScope {
    pub fn namespace(&self) -> &str {
        match self {
            Self::Namespace { name } => name,
            Self::Component { namespace, .. } => namespace,
            Self::Endpoint { endpoint } => &endpoint.namespace,
        }
    }

    pub fn component(&self) -> Option<&str> {
        match self {
            Self::Namespace { .. } => None,
            Self::Component { component, .. } => Some(component),
            Self::Endpoint { endpoint } => Some(&endpoint.component),
        }
    }

    pub fn endpoint(&self) -> Option<&EndpointId> {
        match self {
            Self::Endpoint { endpoint } => Some(endpoint),
            Self::Namespace { .. } | Self::Component { .. } => None,
        }
    }

    /// Canonical NATS/ZMQ-broker subject prefix for this scope.
    pub fn subject_prefix(&self) -> String {
        match self {
            Self::Namespace { name } => {
                format!("namespace.{}", encode_event_segment(name))
            }
            Self::Component {
                namespace,
                component,
            } => format!(
                "namespace.{}.component.{}",
                encode_event_segment(namespace),
                encode_event_segment(component)
            ),
            Self::Endpoint { endpoint } => format!(
                "namespace.{}.component.{}.endpoint.{}",
                encode_event_segment(&endpoint.namespace),
                encode_event_segment(&endpoint.component),
                encode_event_segment(&endpoint.name)
            ),
        }
    }

    /// Canonical subject/routing key for a topic in this exact scope.
    pub fn subject(&self, topic: &str) -> String {
        format!("{}.{}", self.subject_prefix(), encode_event_segment(topic))
    }

    pub(crate) fn path_prefix(&self) -> String {
        match self {
            Self::Namespace { name } => {
                format!("namespace/{}", encode_event_segment(name))
            }
            Self::Component {
                namespace,
                component,
            } => format!(
                "namespace/{}/component/{}",
                encode_event_segment(namespace),
                encode_event_segment(component)
            ),
            Self::Endpoint { endpoint } => format!(
                "namespace/{}/component/{}/endpoint/{}",
                encode_event_segment(&endpoint.namespace),
                encode_event_segment(&endpoint.component),
                encode_event_segment(&endpoint.name)
            ),
        }
    }
}

/// Percent-encode a subject/path segment while keeping common identifier
/// characters readable. The encoding is reversible and prevents NATS wildcard
/// or discovery path delimiters from changing the channel identity.
pub(crate) fn encode_event_segment(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_') {
            encoded.push(char::from(byte));
        } else {
            use std::fmt::Write as _;
            write!(encoded, "%{byte:02X}").expect("writing to String cannot fail");
        }
    }
    encoded
}

fn decode_event_segment(value: &str) -> Result<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }
        if index + 2 >= bytes.len() {
            anyhow::bail!("invalid percent-encoded event segment: {value}");
        }
        let hex = std::str::from_utf8(&bytes[index + 1..index + 3])?;
        decoded
            .push(u8::from_str_radix(hex, 16).map_err(|error| {
                anyhow::anyhow!("invalid event segment escape %{hex}: {error}")
            })?);
        index += 3;
    }
    String::from_utf8(decoded)
        .map_err(|error| anyhow::anyhow!("event segment is not valid UTF-8: {error}"))
}

/// Unified query for event channels with an optional exact scope and topic.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EventChannelQuery {
    /// Exact scope. `None` is reserved for administrative all-channel queries.
    pub scope: Option<EventScope>,
    pub topic: Option<String>,
}

impl EventChannelQuery {
    /// Query all event channels (no filters)
    pub fn all() -> Self {
        Self {
            scope: None,
            topic: None,
        }
    }

    /// Query event channels in a specific namespace
    pub fn namespace(namespace: impl Into<String>) -> Self {
        Self {
            scope: Some(EventScope::Namespace {
                name: namespace.into(),
            }),
            topic: None,
        }
    }

    pub fn namespace_topic(namespace: impl Into<String>, topic: impl Into<String>) -> Self {
        Self {
            scope: Some(EventScope::Namespace {
                name: namespace.into(),
            }),
            topic: Some(topic.into()),
        }
    }

    /// Query event channels for a specific component
    pub fn component(namespace: impl Into<String>, component: impl Into<String>) -> Self {
        Self {
            scope: Some(EventScope::Component {
                namespace: namespace.into(),
                component: component.into(),
            }),
            topic: None,
        }
    }

    /// Query event channels for a specific topic
    pub fn topic(
        namespace: impl Into<String>,
        component: impl Into<String>,
        topic: impl Into<String>,
    ) -> Self {
        Self {
            scope: Some(EventScope::Component {
                namespace: namespace.into(),
                component: component.into(),
            }),
            topic: Some(topic.into()),
        }
    }

    pub fn endpoint(endpoint: EndpointId) -> Self {
        Self {
            scope: Some(EventScope::Endpoint { endpoint }),
            topic: None,
        }
    }

    pub fn endpoint_topic(endpoint: EndpointId, topic: impl Into<String>) -> Self {
        Self {
            scope: Some(EventScope::Endpoint { endpoint }),
            topic: Some(topic.into()),
        }
    }

    /// Get the query specificity (0=all, 1=scope, 2=scope+topic).
    pub fn scope_level(&self) -> u8 {
        if self.topic.is_some() {
            2
        } else if self.scope.is_some() {
            1
        } else {
            0
        }
    }
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
        /// Optional execution device for this endpoint instance.
        /// Used by hetero routing to distinguish CPU and CUDA workers.
        device_type: Option<DeviceType>,
        /// Serving endpoint whose worker-local state this endpoint exposes.
        /// Regular serving endpoints leave this unset.
        source_endpoint: Option<EndpointId>,
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
    /// Event plane channel specification
    /// Used for registering event publishers/subscribers for discovery
    EventChannel {
        scope: EventScope,
        /// Topic name for this channel (e.g., "kv-events", "kv-metrics")
        topic: String,
        /// Unique identity of this publisher incarnation.
        ///
        /// A process can host multiple publishers for the same topic, so event
        /// channels cannot use the process-level discovery instance ID.
        publisher_id: u64,
        /// Event transport type (NATS subject prefix or ZMQ endpoint)
        transport: EventTransport,
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

    /// Converts this registration spec into a discovery instance.
    ///
    /// Endpoint and model specs use `default_instance_id`, normally the
    /// discovery client's process-level ID. Event channel specs already carry
    /// a publisher-level ID, so they use that instead.
    pub fn into_instance(self, default_instance_id: u64) -> DiscoveryInstance {
        match self {
            Self::Endpoint {
                namespace,
                component,
                endpoint,
                transport,
                device_type,
                source_endpoint,
            } => DiscoveryInstance::Endpoint(crate::component::Instance {
                namespace,
                component,
                endpoint,
                instance_id: default_instance_id,
                transport,
                device_type,
                source_endpoint,
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
                instance_id: default_instance_id,
                card_json,
                model_suffix,
            },
            Self::EventChannel {
                scope,
                topic,
                publisher_id,
                transport,
            } => DiscoveryInstance::EventChannel {
                scope,
                topic,
                instance_id: publisher_id,
                transport,
            },
        }
    }

    /// Compatibility alias for [`DiscoverySpec::into_instance`].
    pub fn with_instance_id(self, default_instance_id: u64) -> DiscoveryInstance {
        self.into_instance(default_instance_id)
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
    /// Registered event channel instance for event plane pub/sub
    EventChannel {
        scope: EventScope,
        /// Topic name for this channel (e.g., "kv-events", "kv-metrics")
        topic: String,
        instance_id: u64,
        /// Event transport type (NATS subject prefix or ZMQ endpoint)
        transport: EventTransport,
    },
}

impl DiscoveryInstance {
    /// Returns the instance ID for this discovery instance
    pub fn instance_id(&self) -> u64 {
        match self {
            Self::Endpoint(inst) => inst.instance_id,
            Self::Model { instance_id, .. } => *instance_id,
            Self::EventChannel { instance_id, .. } => *instance_id,
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
            Self::EventChannel { .. } => {
                anyhow::bail!("Cannot deserialize model from EventChannel instance")
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
                source_endpoint: inst.source_endpoint.clone(),
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
            Self::EventChannel {
                scope,
                topic,
                instance_id,
                ..
            } => DiscoveryInstanceId::EventChannel(EventChannelInstanceId {
                scope: scope.clone(),
                topic: topic.clone(),
                instance_id: *instance_id,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_endpoint: Option<EndpointId>,
}

impl EndpointInstanceId {
    /// Converts to a path string.
    ///
    /// Regular endpoints retain the legacy
    /// `{namespace}/{component}/{endpoint}/{instance_id:x}` form. Attributed
    /// endpoints append `/source/{namespace}/{component}/{endpoint}` so two
    /// otherwise identical auxiliary endpoints can coexist for distinct
    /// serving pools.
    pub fn to_path(&self) -> String {
        let base = format!(
            "{}/{}/{}/{:x}",
            self.namespace, self.component, self.endpoint, self.instance_id
        );
        match &self.source_endpoint {
            Some(source) => format!(
                "{base}/source/{}/{}/{}",
                source.namespace, source.component, source.name
            ),
            None => base,
        }
    }

    /// Parses either the legacy path or an attributed endpoint path.
    pub fn from_path(path: &str) -> Result<Self> {
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() != 4 && (parts.len() != 8 || parts[4] != "source") {
            anyhow::bail!(
                "Invalid EndpointInstanceId path: expected 4 legacy parts or 8 attributed parts, got {}",
                parts.len()
            );
        }
        Ok(Self {
            namespace: parts[0].to_string(),
            component: parts[1].to_string(),
            endpoint: parts[2].to_string(),
            instance_id: u64::from_str_radix(parts[3], 16)
                .map_err(|e| anyhow::anyhow!("Invalid instance_id hex: {}", e))?,
            source_endpoint: (parts.len() == 8).then(|| EndpointId {
                namespace: parts[5].to_string(),
                component: parts[6].to_string(),
                name: parts[7].to_string(),
            }),
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

/// Unique identifier for an event channel instance
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventChannelInstanceId {
    pub scope: EventScope,
    /// Topic name for this channel (e.g., "kv-events", "kv-metrics")
    pub topic: String,
    pub instance_id: u64,
}

impl EventChannelInstanceId {
    /// Converts to a delimiter-safe path containing the exact channel scope.
    pub fn to_path(&self) -> String {
        format!(
            "{}/topic/{}/{:x}",
            self.scope.path_prefix(),
            encode_event_segment(&self.topic),
            self.instance_id
        )
    }

    /// Parses a path produced by [`Self::to_path`].
    pub fn from_path(path: &str) -> Result<Self> {
        let parts: Vec<&str> = path.split('/').collect();
        let (scope, topic_index, instance_index) = match parts.as_slice() {
            ["namespace", namespace, "topic", _, _] => (
                EventScope::Namespace {
                    name: decode_event_segment(namespace)?,
                },
                3,
                4,
            ),
            [
                "namespace",
                namespace,
                "component",
                component,
                "topic",
                _,
                _,
            ] => (
                EventScope::Component {
                    namespace: decode_event_segment(namespace)?,
                    component: decode_event_segment(component)?,
                },
                5,
                6,
            ),
            [
                "namespace",
                namespace,
                "component",
                component,
                "endpoint",
                endpoint,
                "topic",
                _,
                _,
            ] => (
                EventScope::Endpoint {
                    endpoint: EndpointId {
                        namespace: decode_event_segment(namespace)?,
                        component: decode_event_segment(component)?,
                        name: decode_event_segment(endpoint)?,
                    },
                },
                7,
                8,
            ),
            _ => anyhow::bail!("invalid EventChannelInstanceId path: {path}"),
        };
        Ok(Self {
            scope,
            topic: decode_event_segment(parts[topic_index])?,
            instance_id: u64::from_str_radix(parts[instance_index], 16)
                .map_err(|e| anyhow::anyhow!("Invalid instance_id hex: {}", e))?,
        })
    }
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
    EventChannel(EventChannelInstanceId),
}

impl DiscoveryInstanceId {
    /// Returns the raw instance_id regardless of variant type
    pub fn instance_id(&self) -> u64 {
        match self {
            Self::Endpoint(eid) => eid.instance_id,
            Self::Model(mid) => mid.instance_id,
            Self::EventChannel(ecid) => ecid.instance_id,
        }
    }

    /// Extracts the EndpointInstanceId, returning an error if this is a Model or EventChannel variant
    pub fn extract_endpoint_id(&self) -> Result<&EndpointInstanceId> {
        match self {
            Self::Endpoint(eid) => Ok(eid),
            Self::Model(_) => anyhow::bail!("Expected Endpoint variant, got Model"),
            Self::EventChannel(_) => anyhow::bail!("Expected Endpoint variant, got EventChannel"),
        }
    }

    /// Extracts the ModelCardInstanceId, returning an error if this is an Endpoint or EventChannel variant
    pub fn extract_model_id(&self) -> Result<&ModelCardInstanceId> {
        match self {
            Self::Model(mid) => Ok(mid),
            Self::Endpoint(_) => anyhow::bail!("Expected Model variant, got Endpoint"),
            Self::EventChannel(_) => anyhow::bail!("Expected Model variant, got EventChannel"),
        }
    }

    /// Extracts the EventChannelInstanceId, returning an error if this is an Endpoint or Model variant
    pub fn extract_event_channel_id(&self) -> Result<&EventChannelInstanceId> {
        match self {
            Self::EventChannel(ecid) => Ok(ecid),
            Self::Endpoint(_) => anyhow::bail!("Expected EventChannel variant, got Endpoint"),
            Self::Model(_) => anyhow::bail!("Expected EventChannel variant, got Model"),
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct ModelRegistrationIdentity {
    display_name: String,
    source_path: Option<String>,
}

impl ModelRegistrationIdentity {
    fn base_identity(&self) -> &str {
        self.source_path.as_deref().unwrap_or(&self.display_name)
    }

    fn is_compatible_with(&self, other: &Self) -> bool {
        self.base_identity() == other.base_identity()
    }
}

fn extract_model_registration_identity(
    card_json: &serde_json::Value,
    _model_suffix: Option<&str>,
) -> Result<ModelRegistrationIdentity> {
    let display_name = card_json
        .get("display_name")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            anyhow::anyhow!("failed to deserialize model display_name from card_json")
        })?;
    let source_path = card_json
        .get("source_path")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    Ok(ModelRegistrationIdentity {
        display_name,
        source_path,
    })
}

fn find_conflicting_model_name(
    instances: &[DiscoveryInstance],
    requested_identity: &ModelRegistrationIdentity,
) -> Result<Option<String>> {
    for instance in instances {
        if let DiscoveryInstance::Model {
            card_json,
            model_suffix,
            ..
        } = instance
        {
            let existing_identity =
                extract_model_registration_identity(card_json, model_suffix.as_deref())?;
            if !requested_identity.is_compatible_with(&existing_identity) {
                return Ok(Some(existing_identity.display_name));
            }
        }
    }

    Ok(None)
}

/// Discovery trait for service discovery across different backends
#[async_trait]
pub trait Discovery: Send + Sync {
    /// Returns a unique identifier for this worker (e.g lease id if using etcd or generated id for memory store)
    /// Endpoint and model objects created by this worker use this ID. Event
    /// channels use a publisher-level ID because a worker can own more than one
    /// publisher for the same topic.
    fn instance_id(&self) -> u64;

    /// Registers an object in the discovery plane with the instance id
    async fn register(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
        let (namespace, component, endpoint, requested_identity) = match &spec {
            DiscoverySpec::Model {
                namespace,
                component,
                endpoint,
                card_json,
                model_suffix,
                ..
            } => (
                namespace.clone(),
                component.clone(),
                endpoint.clone(),
                extract_model_registration_identity(card_json, model_suffix.as_deref())?,
            ),
            _ => return self.register_internal(spec).await,
        };

        let query = DiscoveryQuery::EndpointModels {
            namespace: namespace.clone(),
            component: component.clone(),
            endpoint: endpoint.clone(),
        };

        if let Some(conflicting_name) =
            find_conflicting_model_name(&self.list(query.clone()).await?, &requested_identity)?
        {
            let requested_name = &requested_identity.display_name;
            anyhow::bail!(
                "Cannot register model '{requested_name}' on endpoint '{namespace}/{component}/{endpoint}': a different model '{conflicting_name}' is already registered there"
            );
        }

        let instance = self.register_internal(spec).await?;

        if let Some(conflicting_name) =
            find_conflicting_model_name(&self.list(query).await?, &requested_identity)?
        {
            let requested_name = &requested_identity.display_name;
            if let Err(unregister_err) = self.unregister(instance.clone()).await {
                return Err(anyhow::anyhow!(
                    "Cannot register model '{requested_name}' on endpoint '{namespace}/{component}/{endpoint}': a different model '{conflicting_name}' is already registered there"
                ))
                .context(format!(
                    "failed to roll back conflicting model registration for instance {instance_id}: {unregister_err}",
                    instance_id = instance.instance_id()
                ));
            }

            anyhow::bail!(
                "Cannot register model '{requested_name}' on endpoint '{namespace}/{component}/{endpoint}': a different model '{conflicting_name}' is already registered there"
            );
        }

        Ok(instance)
    }

    /// Backend-specific raw registration implementation.
    async fn register_internal(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance>;

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

    /// Clean up resources held by this discovery backend.
    /// For KV store backends, this deletes owned registrations immediately rather than
    /// waiting for TTL expiry. Default is a no-op for backends that don't need cleanup.
    fn shutdown(&self) {}
}

#[cfg(test)]
mod event_channel_scope_tests {
    use super::*;

    #[test]
    fn endpoint_instance_id_paths_preserve_legacy_and_source_identity() {
        let legacy = EndpointInstanceId {
            namespace: "ns".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
            instance_id: 0xfeed,
            source_endpoint: None,
        };
        assert_eq!(legacy.to_path(), "ns/backend/generate/feed");
        assert_eq!(
            EndpointInstanceId::from_path(&legacy.to_path()).unwrap(),
            legacy
        );

        let attributed = EndpointInstanceId {
            endpoint: "worker_kv_indexer_query_dp0".to_string(),
            source_endpoint: Some(EndpointId {
                namespace: "workers".to_string(),
                component: "backend".to_string(),
                name: "generate".to_string(),
            }),
            ..legacy
        };
        assert_eq!(
            attributed.to_path(),
            "ns/backend/worker_kv_indexer_query_dp0/feed/source/workers/backend/generate"
        );
        assert_eq!(
            EndpointInstanceId::from_path(&attributed.to_path()).unwrap(),
            attributed
        );
    }

    #[test]
    fn endpoint_channel_id_path_round_trips_reserved_segments() {
        let id = EventChannelInstanceId {
            scope: EventScope::Endpoint {
                endpoint: EndpointId {
                    namespace: "ns.with/slash".to_string(),
                    component: "component.*".to_string(),
                    name: "endpoint.>/%".to_string(),
                },
            },
            topic: "kv.events/>".to_string(),
            instance_id: 0xfeed,
        };

        let path = id.to_path();
        assert!(!path.contains("ns.with/slash"));
        assert_eq!(EventChannelInstanceId::from_path(&path).unwrap(), id);
    }

    #[test]
    fn endpoint_subject_prefix_escapes_nats_delimiters_and_wildcards() {
        let scope = EventScope::Endpoint {
            endpoint: EndpointId {
                namespace: "ns.one".to_string(),
                component: "worker/*".to_string(),
                name: "generate.>".to_string(),
            },
        };

        assert_eq!(
            scope.subject("kv.events/*"),
            "namespace.ns%2Eone.component.worker%2F%2A.endpoint.generate%2E%3E.kv%2Eevents%2F%2A"
        );
    }

    #[test]
    fn nats_and_broker_routing_keys_isolate_sibling_endpoints() {
        let endpoint_a = EventScope::Endpoint {
            endpoint: EndpointId {
                namespace: "serving".to_string(),
                component: "worker".to_string(),
                name: "decode-a".to_string(),
            },
        };
        let endpoint_b = EventScope::Endpoint {
            endpoint: EndpointId {
                namespace: "serving".to_string(),
                component: "worker".to_string(),
                name: "decode-b".to_string(),
            },
        };

        assert_eq!(
            endpoint_a.subject("kv-events"),
            "namespace.serving.component.worker.endpoint.decode-a.kv-events"
        );
        assert_eq!(
            endpoint_b.subject("kv-events"),
            "namespace.serving.component.worker.endpoint.decode-b.kv-events"
        );
        assert_ne!(
            endpoint_a.subject("kv-events"),
            endpoint_b.subject("kv-events")
        );
    }
}
