// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic Event Plane for transport-agnostic pub/sub communication.
//!
//! The event plane provides a unified interface for publishing and subscribing
//! to events across different transport backends (NATS, ZMQ - to be implemented in next PRs).
//!
//! # Usage
//!
//! ```ignore
//! // Create an EventPlane scoped to a namespace
//! let event_plane = EventPlane::for_namespace(&namespace);
//! event_plane.publish("kv-events", &my_event).await?;
//!
//! // Create an EventPlane scoped to a component
//! let event_plane = EventPlane::for_component(&component);
//! let mut stream = event_plane.subscribe("kv-events").await?;
//! ```

pub(crate) mod nats;
mod traits;

pub use nats::NatsEnvelope;
pub use traits::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::DistributedRuntime;
use crate::component::{Component, Namespace};
use crate::traits::DistributedRuntimeProvider;

/// Scope of the event plane - determines the subject prefix for pub/sub.
#[derive(Debug, Clone)]
pub enum EventScope {
    /// Namespace-level scope: `namespace.{name}`
    Namespace { name: String },
    /// Component-level scope: `namespace.{namespace}.component.{component}`
    Component {
        namespace: String,
        component: String,
    },
}

impl EventScope {
    /// Returns the subject prefix for this scope.
    pub fn subject_prefix(&self) -> String {
        match self {
            EventScope::Namespace { name } => format!("namespace.{}", name),
            EventScope::Component {
                namespace,
                component,
            } => {
                format!("namespace.{}.component.{}", namespace, component)
            }
        }
    }
}

/// Event plane for transport-agnostic pub/sub communication.
///
/// The EventPlane wraps different transport backends (NATS, ZMQ - to be implemented in next PRs) and provides
/// a unified interface for publishing and subscribing to events with envelope
/// metadata for sequencing and gap detection.
pub struct EventPlane {
    /// The scope determines the subject prefix
    scope: EventScope,
    /// Unique identifier for this publisher (from discovery instance_id)
    publisher_id: u64,
    /// Monotonically increasing sequence number
    sequence: AtomicU64,
    /// Reference to the distributed runtime for NATS access
    drt: Arc<DistributedRuntime>,
}

impl EventPlane {
    /// Create an EventPlane scoped to a Namespace.
    ///
    /// Events published through this EventPlane will have subjects prefixed
    /// with `namespace.{name}`.
    ///
    /// # Example
    /// ```ignore
    /// let event_plane = EventPlane::for_namespace(&namespace);
    /// event_plane.publish("kv-metrics", &metrics_event).await?;
    /// // Published to: namespace.{name}.kv-metrics
    /// ```
    pub fn for_namespace(ns: &Namespace) -> Self {
        Self {
            scope: EventScope::Namespace { name: ns.name() },
            publisher_id: ns.drt().discovery().instance_id(),
            sequence: AtomicU64::new(0),
            drt: Arc::new(ns.drt().clone()),
        }
    }

    /// Create an EventPlane scoped to a Component.
    ///
    /// Events published through this EventPlane will have subjects prefixed
    /// with `namespace.{namespace}.component.{component}`.
    ///
    /// # Example
    /// ```ignore
    /// let event_plane = EventPlane::for_component(&component);
    /// event_plane.publish("kv-events", &kv_event).await?;
    /// // Published to: namespace.{ns}.component.{comp}.kv-events
    /// ```
    pub fn for_component(comp: &Component) -> Self {
        Self {
            scope: EventScope::Component {
                namespace: comp.namespace().name(),
                component: comp.name().to_string(),
            },
            publisher_id: comp.drt().discovery().instance_id(),
            sequence: AtomicU64::new(0),
            drt: Arc::new(comp.drt().clone()),
        }
    }

    /// Get the scope of this EventPlane.
    pub fn scope(&self) -> &EventScope {
        &self.scope
    }

    /// Get the subject prefix for this EventPlane.
    pub fn subject_prefix(&self) -> String {
        self.scope.subject_prefix()
    }

    /// Build the full subject for a topic.
    fn full_subject(&self, topic: &str) -> String {
        format!("{}.{}", self.subject_prefix(), topic)
    }

    /// Get the current timestamp in milliseconds since Unix epoch.
    fn current_timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Get the next sequence number.
    fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::SeqCst)
    }
}

#[async_trait]
impl GenericEventPublisher for EventPlane {
    async fn publish<T: Serialize + Send + Sync>(&self, topic: &str, event: &T) -> Result<()> {
        let payload = serde_json::to_vec(event)?;
        self.publish_bytes(topic, payload).await
    }

    async fn publish_bytes(&self, topic: &str, bytes: Vec<u8>) -> Result<()> {
        let envelope = nats::NatsEnvelope {
            publisher_id: self.publisher_id,
            sequence: self.next_sequence(),
            published_at: Self::current_timestamp_ms(),
            payload: bytes,
        };

        let envelope_bytes = serde_json::to_vec(&envelope)?;
        let subject = self.full_subject(topic);

        self.drt
            .kv_router_nats_publish(subject, Bytes::from(envelope_bytes))
            .await
    }

    fn publisher_id(&self) -> u64 {
        self.publisher_id
    }
}

#[async_trait]
impl GenericEventSubscriber for EventPlane {
    async fn subscribe(&self, topic: &str) -> Result<EventStream> {
        let subject = self.full_subject(topic);
        nats::subscribe_to_nats(&self.drt, subject).await
    }

    async fn subscribe_typed<T: DeserializeOwned + Send + 'static>(
        &self,
        topic: &str,
    ) -> Result<TypedEventStream<T>> {
        let subject = self.full_subject(topic);
        nats::subscribe_typed_to_nats(&self.drt, subject).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_scope_subject_prefix() {
        let ns_scope = EventScope::Namespace {
            name: "test-ns".to_string(),
        };
        assert_eq!(ns_scope.subject_prefix(), "namespace.test-ns");

        let comp_scope = EventScope::Component {
            namespace: "test-ns".to_string(),
            component: "test-comp".to_string(),
        };
        assert_eq!(
            comp_scope.subject_prefix(),
            "namespace.test-ns.component.test-comp"
        );
    }

    #[test]
    fn test_nats_envelope_serialization_roundtrip() {
        let envelope = NatsEnvelope {
            publisher_id: 12345,
            sequence: 42,
            published_at: 1700000000000,
            payload: b"test event data".to_vec(),
        };

        // Serialize to JSON
        let json = serde_json::to_string(&envelope).expect("serialize");

        // Deserialize back
        let deserialized: NatsEnvelope = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.publisher_id, 12345);
        assert_eq!(deserialized.sequence, 42);
        assert_eq!(deserialized.published_at, 1700000000000);
        assert_eq!(deserialized.payload, b"test event data".to_vec());
    }

    #[test]
    fn test_nats_envelope_with_json_payload() {
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct TestEvent {
            worker_id: u64,
            message: String,
        }

        let event = TestEvent {
            worker_id: 123,
            message: "hello".to_string(),
        };

        // Serialize the event to JSON bytes
        let event_bytes = serde_json::to_vec(&event).expect("serialize event");

        // Wrap in envelope
        let envelope = NatsEnvelope {
            publisher_id: 999,
            sequence: 1,
            published_at: 1700000000000,
            payload: event_bytes.clone(),
        };

        // Serialize envelope
        let envelope_json = serde_json::to_string(&envelope).expect("serialize envelope");

        // Deserialize envelope
        let deserialized_envelope: NatsEnvelope =
            serde_json::from_str(&envelope_json).expect("deserialize envelope");

        // Deserialize the payload
        let deserialized_event: TestEvent =
            serde_json::from_slice(&deserialized_envelope.payload).expect("deserialize payload");

        assert_eq!(deserialized_event, event);
    }

    #[test]
    fn test_timestamp_generation() {
        // Just verify that current_timestamp_ms returns a reasonable value
        let ts = EventPlane::current_timestamp_ms();

        // Should be after Jan 1, 2020 (1577836800000) and before Jan 1, 2100 (4102444800000)
        assert!(ts > 1577836800000, "Timestamp should be after 2020");
        assert!(ts < 4102444800000, "Timestamp should be before 2100");
    }

    #[test]
    fn test_event_envelope_serde() {
        let envelope = EventEnvelope {
            topic: "test.topic".to_string(),
            publisher_id: 42,
            sequence: 10,
            published_at: 1700000000000,
            payload: Bytes::from("test data"),
        };

        let json = serde_json::to_string(&envelope).expect("serialize");
        let deserialized: EventEnvelope = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.topic, "test.topic");
        assert_eq!(deserialized.publisher_id, 42);
        assert_eq!(deserialized.sequence, 10);
        assert_eq!(deserialized.published_at, 1700000000000);
        assert_eq!(deserialized.payload, Bytes::from("test data"));
    }
}
