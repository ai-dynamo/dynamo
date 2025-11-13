// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bytes::Bytes;
use dynamo_identity::InstanceId;
use futures::future::BoxFuture;

use super::{PeerInfo, WorkerAddress};

use std::{fmt, sync::Arc, time::Duration};

#[derive(thiserror::Error, Debug)]
pub enum TransportError {
    #[error("No endpoint found for transport")]
    NoEndpoint,

    #[error("Invalid endpoint format")]
    InvalidEndpoint,

    #[error("Peer not registered: {0}")]
    PeerNotRegistered(InstanceId),

    #[error("Transport not started")]
    NotStarted,

    #[error("No responders for peer")]
    NoResponders,
}

/// Error type specific to health check operations
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum HealthCheckError {
    #[error("Peer not registered with transport")]
    PeerNotRegistered,

    #[error("Transport not started")]
    TransportNotStarted,

    #[error("Connection never established to peer")]
    NeverConnected,

    #[error("Connection failed or peer unreachable")]
    ConnectionFailed,

    #[error("Health check timed out")]
    Timeout,
}

pub trait Transport: Send + Sync {
    fn key(&self) -> TransportKey;
    fn address(&self) -> WorkerAddress;
    fn register(&self, peer_info: PeerInfo) -> Result<(), TransportError>;

    /// Sends an active message to the remote instance
    fn send_message(
        &self,
        instance_id: InstanceId,
        header: Vec<u8>,
        payload: Vec<u8>,
        message_type: MessageType,
        on_error: Arc<dyn TransportErrorHandler>,
    );

    fn start(
        &self,
        instance_id: InstanceId,
        channels: TransportAdapter,
        rt: tokio::runtime::Handle,
    ) -> BoxFuture<'_, anyhow::Result<()>>;

    fn shutdown(&self);

    /// Check if a registered peer is reachable and healthy
    ///
    /// Returns Ok(()) if peer responds to health check within timeout.
    /// Different transports implement this differently:
    /// - NATS: request/reply to health subject
    /// - TCP: check existing connection or attempt new connection
    /// - HTTP: HEAD request to health endpoint
    /// - UCX: endpoint status check
    ///
    /// # Errors
    /// - `PeerNotRegistered`: Peer was never registered with this transport
    /// - `TransportNotStarted`: Transport hasn't been started yet
    /// - `NeverConnected`: Peer is registered but no connection has been established
    /// - `ConnectionFailed`: Connection exists/existed but is currently unhealthy or unreachable
    /// - `Timeout`: Health check took longer than the specified timeout
    fn check_health(
        &self,
        instance_id: InstanceId,
        timeout: Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), HealthCheckError>> + Send + '_>,
    >;
}

pub trait TransportErrorHandler: Send + Sync {
    fn on_error(&self, header: Bytes, payload: Bytes, error: String);
}

/// Message type discriminator for routing frames to appropriate streams
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Message = 0,
    Response = 1,
    Ack = 2,
    Event = 3,
}

impl MessageType {
    /// Try to convert a u8 to a MessageType
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(MessageType::Message),
            1 => Some(MessageType::Response),
            2 => Some(MessageType::Ack),
            3 => Some(MessageType::Event),
            _ => None,
        }
    }

    /// Convert MessageType to u8
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Clone)]
pub struct TransportAdapter {
    pub message_stream: flume::Sender<(Bytes, Bytes)>,
    pub response_stream: flume::Sender<(Bytes, Bytes)>,
    pub event_stream: flume::Sender<(Bytes, Bytes)>,
}

pub struct DataStreams {
    pub message_stream: flume::Receiver<(Bytes, Bytes)>,
    pub response_stream: flume::Receiver<(Bytes, Bytes)>,
    pub event_stream: flume::Receiver<(Bytes, Bytes)>,
}

type DataStreamTuple = (
    flume::Receiver<(Bytes, Bytes)>,
    flume::Receiver<(Bytes, Bytes)>,
    flume::Receiver<(Bytes, Bytes)>,
);

impl DataStreams {
    pub fn into_parts(self) -> DataStreamTuple {
        (self.message_stream, self.response_stream, self.event_stream)
    }
}

pub fn make_channels() -> (TransportAdapter, DataStreams) {
    let (message_tx, message_rx) = flume::unbounded();
    let (response_tx, response_rx) = flume::unbounded();
    let (event_tx, event_rx) = flume::unbounded();
    (
        TransportAdapter {
            message_stream: message_tx,
            response_stream: response_tx,
            event_stream: event_tx,
        },
        DataStreams {
            message_stream: message_rx,
            response_stream: response_rx,
            event_stream: event_rx,
        },
    )
}

/// A type-safe wrapper around transport keys for WorkerAddress.
///
/// This provides a zero-cost abstraction over `Arc<str>` with type safety
/// to prevent accidentally mixing transport keys with other string types.
///
/// # Examples
///
/// ```
/// use dynamo_nova_backend::TransportKey;
///
/// let key = TransportKey::new("tcp");
/// assert_eq!(key.as_str(), "tcp");
///
/// // Ergonomic conversions
/// let key2: TransportKey = "rdma".into();
/// let key3 = TransportKey::from("udp");
///
/// // Use in collections
/// use std::collections::HashMap;
/// let mut transports = HashMap::new();
/// transports.insert(TransportKey::from("tcp"), "tcp://127.0.0.1:5555");
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TransportKey(Arc<str>);

impl TransportKey {
    /// Create a new TransportKey from any type that can be converted into Arc<str>.
    pub fn new(key: impl Into<Arc<str>>) -> Self {
        Self(key.into())
    }

    /// Get the key as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// Deref to str for ergonomic usage
impl std::ops::Deref for TransportKey {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// AsRef for flexible parameter types
impl AsRef<str> for TransportKey {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// From conversions for ergonomic construction
impl From<&str> for TransportKey {
    fn from(s: &str) -> Self {
        Self(Arc::from(s))
    }
}

impl From<String> for TransportKey {
    fn from(s: String) -> Self {
        Self(Arc::from(s))
    }
}

impl From<Arc<str>> for TransportKey {
    fn from(s: Arc<str>) -> Self {
        Self(s)
    }
}

impl From<&String> for TransportKey {
    fn from(s: &String) -> Self {
        Self(Arc::from(s.as_str()))
    }
}

impl From<TransportKey> for String {
    fn from(val: TransportKey) -> Self {
        val.0.to_string()
    }
}

// Borrow trait for HashMap lookups with &str
impl std::borrow::Borrow<str> for TransportKey {
    fn borrow(&self) -> &str {
        &self.0
    }
}

// Display for printing
impl fmt::Display for TransportKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[allow(dead_code)]
pub struct GenericTransport<T: Transport + Send + Sync + 'static> {
    transport: T,
}
