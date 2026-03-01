// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event bus infrastructure for registry events.
//!
//! Provides a pub/sub event system for broadcasting cache events to workers
//! and interested components. Events are fire-and-forget, enabling real-time
//! coordination across the cluster.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         DISTRIBUTED EVENT BUS                                │
//! │                                                                              │
//! │   Pluggable Backends:  InProcess │ ZMQ PUB/SUB │ NATS │ Redis │ etc.        │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!          ▲              ▲              ▲              ▲              ▲
//!          │              │              │              │              │
//!     ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
//!     │Worker 1 │    │Worker 2 │    │Worker 3 │    │Registry │    │ Metrics │
//!     │ emit()  │    │ emit()  │    │ emit()  │    │   Hub   │    │ Service │
//!     │on_event │    │on_event │    │on_event │    │on_event │    │on_event │
//!     └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

/// Topic/channel for event routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventTopic {
    /// Cache hit/miss events (for eviction decisions)
    CacheAccess,
    /// Block lifecycle events (offload, evict, invalidate)
    BlockLifecycle,
    /// Cluster membership changes
    ClusterMembership,
    /// Metrics and telemetry
    Metrics,
}

/// Storage tier for events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageTier {
    /// G1 - GPU memory
    Device = 0,
    /// G2 - CPU pinned memory
    Host = 1,
    /// G3 - Local disk
    Disk = 2,
    /// G4 - Remote storage (object/shared disk)
    Remote = 3,
}

/// Remote storage type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageType {
    Object,
    Disk,
}

/// Reason for eviction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionReason {
    /// Hit capacity limit
    Capacity,
    /// Least recently used
    Lru,
    /// Least frequently used
    Lfu,
    /// Explicit eviction request
    Manual,
    /// Worker shutting down
    Shutdown,
}

/// Core event types - extensible with #[non_exhaustive].
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegistryEvent {
    // ═══════════════════════════════════════════════════════════════════
    // CACHE ACCESS EVENTS (Topic: CacheAccess)
    // ═══════════════════════════════════════════════════════════════════
    /// Cache hit - block was found and used
    CacheHit {
        sequence_hashes: Vec<u64>,
        tier: StorageTier,
        worker_id: u64,
        timestamp_ms: u64,
    },

    /// Cache miss - block was not found
    CacheMiss {
        sequence_hashes: Vec<u64>,
        tier: StorageTier,
        worker_id: u64,
    },

    // ═══════════════════════════════════════════════════════════════════
    // BLOCK LIFECYCLE EVENTS (Topic: BlockLifecycle)
    // ═══════════════════════════════════════════════════════════════════
    /// Blocks offloaded to remote storage
    BlocksOffloaded {
        sequence_hashes: Vec<u64>,
        storage_type: StorageType,
        location: String,
        worker_id: u64,
    },

    /// Blocks onboarded from remote storage
    BlocksOnboarded {
        sequence_hashes: Vec<u64>,
        storage_type: StorageType,
        location: String,
        worker_id: u64,
    },

    /// Blocks evicted from a tier
    BlocksEvicted {
        sequence_hashes: Vec<u64>,
        tier: StorageTier,
        worker_id: u64,
        reason: EvictionReason,
    },

    /// Blocks invalidated (e.g., NoSuchKey error)
    BlocksInvalidated {
        sequence_hashes: Vec<u64>,
        reason: String,
        worker_id: u64,
    },

    // ═══════════════════════════════════════════════════════════════════
    // CLUSTER EVENTS (Topic: ClusterMembership)
    // ═══════════════════════════════════════════════════════════════════
    /// Worker joined the cluster
    WorkerJoined { worker_id: u64 },

    /// Worker left the cluster
    WorkerLeft { worker_id: u64, graceful: bool },
}

impl RegistryEvent {
    /// Get the topic for this event.
    pub fn topic(&self) -> EventTopic {
        match self {
            RegistryEvent::CacheHit { .. } | RegistryEvent::CacheMiss { .. } => {
                EventTopic::CacheAccess
            }
            RegistryEvent::BlocksOffloaded { .. }
            | RegistryEvent::BlocksOnboarded { .. }
            | RegistryEvent::BlocksEvicted { .. }
            | RegistryEvent::BlocksInvalidated { .. } => EventTopic::BlockLifecycle,
            RegistryEvent::WorkerJoined { .. } | RegistryEvent::WorkerLeft { .. } => {
                EventTopic::ClusterMembership
            }
        }
    }

    /// Get current timestamp in milliseconds.
    pub fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// Trait for event bus implementations.
#[async_trait]
pub trait EventBus: Send + Sync {
    /// Publish an event (fire-and-forget).
    async fn publish(&self, event: RegistryEvent) -> Result<()>;

    /// Subscribe to events on a topic.
    fn subscribe(&self, topic: EventTopic) -> EventReceiver;

    /// Batch publish for efficiency.
    async fn publish_batch(&self, events: Vec<RegistryEvent>) -> Result<()> {
        for event in events {
            self.publish(event).await?;
        }
        Ok(())
    }
}

/// Event receiver for subscriptions.
pub type EventReceiver = broadcast::Receiver<RegistryEvent>;

/// In-process event bus implementation.
///
/// Uses tokio broadcast channels for zero-copy event distribution.
/// Suitable for single-node deployments or testing.
pub struct InProcessEventBus {
    channels: RwLock<HashMap<EventTopic, broadcast::Sender<RegistryEvent>>>,
    channel_capacity: usize,
}

impl InProcessEventBus {
    /// Create a new in-process event bus.
    pub fn new(channel_capacity: usize) -> Self {
        Self {
            channels: RwLock::new(HashMap::new()),
            channel_capacity,
        }
    }

    fn get_or_create_channel(&self, topic: EventTopic) -> broadcast::Sender<RegistryEvent> {
        {
            let channels = self.channels.read();
            if let Some(sender) = channels.get(&topic) {
                return sender.clone();
            }
        }

        let mut channels = self.channels.write();
        channels
            .entry(topic)
            .or_insert_with(|| broadcast::channel(self.channel_capacity).0)
            .clone()
    }
}

impl Default for InProcessEventBus {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[async_trait]
impl EventBus for InProcessEventBus {
    async fn publish(&self, event: RegistryEvent) -> Result<()> {
        let topic = event.topic();
        let sender = self.get_or_create_channel(topic);
        // Ignore send errors (no subscribers is fine for fire-and-forget)
        let _ = sender.send(event);
        Ok(())
    }

    fn subscribe(&self, topic: EventTopic) -> EventReceiver {
        let sender = self.get_or_create_channel(topic);
        sender.subscribe()
    }
}

/// Event handler trait for processing received events.
#[async_trait]
pub trait EventHandler: Send + Sync {
    /// Handle a received event.
    async fn on_event(&self, event: RegistryEvent) -> Result<()>;
}

/// Spawn a background task to process events from a subscription.
pub fn spawn_event_handler<H: EventHandler + 'static>(
    mut receiver: EventReceiver,
    handler: Arc<H>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match receiver.recv().await {
                Ok(event) => {
                    if let Err(e) = handler.on_event(event).await {
                        tracing::warn!(error = %e, "Event handler error");
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(skipped = n, "Event handler lagged, skipped events");
                }
                Err(broadcast::error::RecvError::Closed) => {
                    tracing::debug!("Event channel closed, handler shutting down");
                    break;
                }
            }
        }
    })
}

/// Error type for event bus configuration.
#[derive(Debug, Clone)]
pub struct EventBusConfigError {
    /// The unsupported backend name
    pub backend: String,
    /// Error message
    pub message: String,
}

impl std::fmt::Display for EventBusConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Unsupported event bus backend '{}': {}",
            self.backend, self.message
        )
    }
}

impl std::error::Error for EventBusConfigError {}

/// Configuration for the event bus.
#[derive(Debug, Clone)]
pub struct EventBusConfig {
    /// Backend type: "in_process", "zmq", "nats", "redis"
    pub backend: String,
    /// Channel capacity for in-process bus
    pub channel_capacity: usize,
    /// Connection URL (for network backends)
    pub url: Option<String>,
}

impl Default for EventBusConfig {
    fn default() -> Self {
        Self {
            backend: "in_process".to_string(),
            channel_capacity: 1024,
            url: None,
        }
    }
}

impl EventBusConfig {
    /// Create from environment variables.
    ///
    /// Environment variables:
    /// - `DYN_EVENT_BUS_BACKEND`: "in_process", "zmq", "nats", "redis"
    /// - `DYN_EVENT_BUS_CAPACITY`: Channel capacity (default: 1024)
    /// - `DYN_EVENT_BUS_URL`: Connection URL for network backends
    pub fn from_env() -> Self {
        let backend =
            std::env::var("DYN_EVENT_BUS_BACKEND").unwrap_or_else(|_| "in_process".to_string());

        let channel_capacity = std::env::var("DYN_EVENT_BUS_CAPACITY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);

        let url = std::env::var("DYN_EVENT_BUS_URL").ok();

        Self {
            backend,
            channel_capacity,
            url,
        }
    }

    /// Build an event bus from this configuration.
    ///
    /// # Errors
    ///
    /// Returns `EventBusConfigError` if the specified backend is not supported.
    /// Currently only "in_process" is implemented; "zmq", "nats", and "redis"
    /// are reserved for future implementation.
    pub fn build(&self) -> std::result::Result<Box<dyn EventBus>, EventBusConfigError> {
        match self.backend.as_str() {
            "in_process" => Ok(Box::new(InProcessEventBus::new(self.channel_capacity))),
            "zmq" => Err(EventBusConfigError {
                backend: self.backend.clone(),
                message: "ZMQ event bus backend is not yet implemented".to_string(),
            }),
            "nats" => Err(EventBusConfigError {
                backend: self.backend.clone(),
                message: "NATS event bus backend is not yet implemented".to_string(),
            }),
            "redis" => Err(EventBusConfigError {
                backend: self.backend.clone(),
                message: "Redis event bus backend is not yet implemented".to_string(),
            }),
            other => Err(EventBusConfigError {
                backend: other.to_string(),
                message:
                    "Unknown backend. Supported: 'in_process'. Reserved: 'zmq', 'nats', 'redis'"
                        .to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[tokio::test]
    async fn test_in_process_event_bus() {
        let bus = InProcessEventBus::new(100);

        // Subscribe before publishing
        let mut receiver = bus.subscribe(EventTopic::CacheAccess);

        // Publish event
        let event = RegistryEvent::CacheHit {
            sequence_hashes: vec![1, 2, 3],
            tier: StorageTier::Host,
            worker_id: 0,
            timestamp_ms: RegistryEvent::now_ms(),
        };
        bus.publish(event.clone()).await.unwrap();

        // Receive event
        let received = receiver.recv().await.unwrap();
        match received {
            RegistryEvent::CacheHit {
                sequence_hashes, ..
            } => {
                assert_eq!(sequence_hashes, vec![1, 2, 3]);
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[tokio::test]
    async fn test_event_handler() {
        struct TestHandler {
            count: AtomicU64,
        }

        #[async_trait]
        impl EventHandler for TestHandler {
            async fn on_event(&self, _event: RegistryEvent) -> Result<()> {
                self.count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
        }

        let bus = InProcessEventBus::new(100);
        let handler = Arc::new(TestHandler {
            count: AtomicU64::new(0),
        });

        let receiver = bus.subscribe(EventTopic::CacheAccess);
        let _task = spawn_event_handler(receiver, handler.clone());

        // Publish events
        for i in 0..5 {
            bus.publish(RegistryEvent::CacheHit {
                sequence_hashes: vec![i],
                tier: StorageTier::Host,
                worker_id: 0,
                timestamp_ms: RegistryEvent::now_ms(),
            })
            .await
            .unwrap();
        }

        // Give handler time to process
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        assert_eq!(handler.count.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_event_bus_config_in_process() {
        let config = EventBusConfig::default();
        assert_eq!(config.backend, "in_process");

        let result = config.build();
        assert!(result.is_ok(), "in_process backend should succeed");
    }

    #[test]
    fn test_event_bus_config_unsupported_backends() {
        // ZMQ - reserved but not implemented
        let config = EventBusConfig {
            backend: "zmq".to_string(),
            ..Default::default()
        };
        let Err(err) = config.build() else {
            panic!("ZMQ should return error");
        };
        assert_eq!(err.backend, "zmq");
        assert!(err.message.contains("not yet implemented"));

        // NATS - reserved but not implemented
        let config = EventBusConfig {
            backend: "nats".to_string(),
            ..Default::default()
        };
        let Err(err) = config.build() else {
            panic!("NATS should return error");
        };
        assert_eq!(err.backend, "nats");

        // Redis - reserved but not implemented
        let config = EventBusConfig {
            backend: "redis".to_string(),
            ..Default::default()
        };
        let Err(err) = config.build() else {
            panic!("Redis should return error");
        };
        assert_eq!(err.backend, "redis");

        // Unknown backend
        let config = EventBusConfig {
            backend: "unknown_backend".to_string(),
            ..Default::default()
        };
        let Err(err) = config.build() else {
            panic!("Unknown backend should return error");
        };
        assert_eq!(err.backend, "unknown_backend");
        assert!(err.message.contains("Unknown backend"));
    }
}
