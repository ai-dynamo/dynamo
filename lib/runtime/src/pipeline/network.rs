// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Network layer for distributed communication
//!
//! Provides request distribution across multiple transport protocols:
//! - HTTP/2 for standard deployments
//! - TCP with length-prefixed protocol for high-performance scenarios
//! - NATS for legacy/messaging-based deployments

pub mod codec;
pub mod egress;
pub mod ingress;
pub mod manager;
pub mod tcp;

use crate::SystemHealth;
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use codec::{TwoPartCodec, TwoPartMessage, TwoPartMessageType};
use derive_builder::Builder;
use futures::StreamExt;
// io::Cursor, TryStreamExt
use super::{AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, ResponseStream};
use serde::{Deserialize, Serialize};

use super::{
    AsyncTransportEngine, Context, Data, Error, ManyOut, PipelineError, PipelineIO, SegmentSource,
    ServiceBackend, ServiceEngine, SingleIn, Source, context,
};
use ingress::push_handler::WorkHandlerMetrics;

// Define stream error message constant
pub const STREAM_ERR_MSG: &str = "Stream ended before generation completed";

// Add Prometheus metrics types
use crate::metrics::MetricsHierarchy;
use prometheus::{CounterVec, Histogram, IntCounter, IntCounterVec, IntGauge};

pub trait Codable: PipelineIO + Serialize + for<'de> Deserialize<'de> {}
impl<T: PipelineIO + Serialize + for<'de> Deserialize<'de>> Codable for T {}

/// `WorkQueueConsumer` is a generic interface for a work queue that can be used to send and receive
#[async_trait]
pub trait WorkQueueConsumer {
    async fn dequeue(&self) -> Result<Bytes, String>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StreamType {
    Request,
    Response,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControlMessage {
    Stop,
    Kill,
    Sentinel,
}

/// This is the first message in a `ResponseStream`. This is not a message that gets process
/// by the general pipeline, but is a control message that is awaited before the
/// [`AsyncEngine::generate`] method is allowed to return.
///
/// If an error is present, the [`AsyncEngine::generate`] method will return the error instead
/// of returning the `ResponseStream`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResponseStreamPrologue {
    error: Option<String>,
}

pub type StreamProvider<T> = tokio::sync::oneshot::Receiver<Result<T, String>>;

/// The [`RegisteredStream`] object is acquired from a [`StreamProvider`] and is used to provide
/// an awaitable receiver which will the `T` which is either a stream writer for a request stream
/// or a stream reader for a response stream.
///
/// make this an raii object linked to some stream provider
/// if the object has not been awaited an the type T unwrapped, the registered stream
/// on the stream provider will be informed and can clean up a stream that will never
/// be connected.
#[derive(Debug)]
pub struct RegisteredStream<T> {
    pub connection_info: ConnectionInfo,
    pub stream_provider: StreamProvider<T>,
}

impl<T> RegisteredStream<T> {
    pub fn into_parts(self) -> (ConnectionInfo, StreamProvider<T>) {
        (self.connection_info, self.stream_provider)
    }
}

/// After registering a stream, the [`PendingConnections`] object is returned to the caller. This
/// object can be used to await the connection to be established.
pub struct PendingConnections {
    pub send_stream: Option<RegisteredStream<StreamSender>>,
    pub recv_stream: Option<RegisteredStream<StreamReceiver>>,
}

impl PendingConnections {
    pub fn into_parts(
        self,
    ) -> (
        Option<RegisteredStream<StreamSender>>,
        Option<RegisteredStream<StreamReceiver>>,
    ) {
        (self.send_stream, self.recv_stream)
    }
}

/// A [`ResponseService`] implements a services in which a context a specific subject with will
/// be associated with a stream of responses.
#[async_trait::async_trait]
pub trait ResponseService {
    async fn register(&self, options: StreamOptions) -> PendingConnections;
}

// #[derive(Debug, Clone, Serialize, Deserialize)]
// struct Handshake {
//     request_id: String,
//     worker_id: Option<String>,
//     error: Option<String>,
// }

// impl Handshake {
//     pub fn validate(&self) -> Result<(), String> {
//         if let Some(e) = &self.error {
//             return Err(e.clone());
//         }
//         Ok(())
//     }
// }

// this probably needs to be come a ResponseStreamSender
// since the prologue in this scenario sender telling the receiver
// that all is good and it's ready to send
//
// in the RequestStreamSender, the prologue would be coming from the
// receiver, so the sender would have to await the prologue which if
// was not an error, would indicate the RequestStreamReceiver is read
// to receive data.
pub struct StreamSender {
    tx: tokio::sync::mpsc::Sender<TwoPartMessage>,
    prologue: Option<ResponseStreamPrologue>,
}

impl StreamSender {
    pub async fn send(&self, data: Bytes) -> Result<()> {
        Ok(self.tx.send(TwoPartMessage::from_data(data)).await?)
    }

    pub async fn send_control(&self, control: ControlMessage) -> Result<()> {
        let bytes = serde_json::to_vec(&control)?;
        Ok(self
            .tx
            .send(TwoPartMessage::from_header(bytes.into()))
            .await?)
    }

    #[allow(clippy::needless_update)]
    pub async fn send_prologue(&mut self, error: Option<String>) -> Result<(), String> {
        if let Some(prologue) = self.prologue.take() {
            let prologue = ResponseStreamPrologue { error, ..prologue };
            let header_bytes: Bytes = match serde_json::to_vec(&prologue) {
                Ok(b) => b.into(),
                Err(err) => {
                    tracing::error!(%err, "send_prologue: ResponseStreamPrologue did not serialize to a JSON array");
                    return Err("Invalid prologue".to_string());
                }
            };
            self.tx
                .send(TwoPartMessage::from_header(header_bytes))
                .await
                .map_err(|e| e.to_string())?;
        } else {
            panic!("Prologue already sent; or not set; logic error");
        }
        Ok(())
    }
}

pub struct StreamReceiver {
    rx: tokio::sync::mpsc::Receiver<Bytes>,
}

/// Connection Info is encoded as JSON and then again serialized has part of the Transport
/// Layer. The double serialization is not performance critical as it is only done once per
/// connection. The primary reason storing the ConnecitonInfo has a JSON string is for type
/// erasure. The Transport Layer will check the [`ConnectionInfo::transport`] type and then
/// route it to the appropriate instance of the Transport, which will then deserialize the
/// [`ConnectionInfo::info`] field to its internal connection info object.
///
/// Optionally, this object could become strongly typed for which all possible combinations
/// of transport and connection info would need to be enumerated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    pub transport: String,
    pub info: String,
}

/// When registering a new TransportStream on the server, the caller specifies if the
/// stream is a sender, receiver or both.
///
/// Senders and Receivers are with share a Context, but result in separate tcp socket
/// connections to the server. Internally, we may use bcast channels to coordinate the
/// internal control messages between the sender and receiver socket connections.
#[derive(Clone, Builder)]
pub struct StreamOptions {
    /// Context
    pub context: Arc<dyn AsyncEngineContext>,

    /// Register with the server that this connection will have a server-side Sender
    /// that can be picked up by the Request/Forward pipeline
    ///
    /// TODO - note, this option is currently not implemented and will cause a panic
    pub enable_request_stream: bool,

    /// Register with the server that this connection will have a server-side Receiver
    /// that can be picked up by the Response/Reverse pipeline
    pub enable_response_stream: bool,

    /// The number of messages to buffer before blocking
    #[builder(default = "8")]
    pub send_buffer_count: usize,

    /// The number of messages to buffer before blocking
    #[builder(default = "8")]
    pub recv_buffer_count: usize,
}

impl StreamOptions {
    pub fn builder() -> StreamOptionsBuilder {
        StreamOptionsBuilder::default()
    }
}

pub struct Egress<Req: PipelineIO, Resp: PipelineIO> {
    transport_engine: Arc<dyn AsyncTransportEngine<Req, Resp>>,
}

#[async_trait]
impl<T: Data, U: Data> AsyncEngine<SingleIn<T>, ManyOut<U>, Error>
    for Egress<SingleIn<T>, ManyOut<U>>
where
    T: Data + Serialize,
    U: for<'de> Deserialize<'de> + Data,
{
    async fn generate(&self, request: SingleIn<T>) -> Result<ManyOut<U>, Error> {
        self.transport_engine.generate(request).await
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RequestType {
    SingleIn,
    ManyIn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResponseType {
    SingleOut,
    ManyOut,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RequestControlMessage {
    id: String,
    request_type: RequestType,
    response_type: ResponseType,
    connection_info: ConnectionInfo,
}

pub struct Ingress<Req: PipelineIO, Resp: PipelineIO> {
    segment: OnceLock<Arc<SegmentSource<Req, Resp>>>,
    metrics: OnceLock<Arc<WorkHandlerMetrics>>,
    /// Endpoint-specific notifier for health check timer resets
    endpoint_health_check_notifier: OnceLock<Arc<tokio::sync::Notify>>,
}

impl<Req: PipelineIO + Sync, Resp: PipelineIO> Ingress<Req, Resp> {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            segment: OnceLock::new(),
            metrics: OnceLock::new(),
            endpoint_health_check_notifier: OnceLock::new(),
        })
    }

    pub fn attach(&self, segment: Arc<SegmentSource<Req, Resp>>) -> Result<()> {
        self.segment
            .set(segment)
            .map_err(|_| anyhow::anyhow!("Segment already set"))
    }

    pub fn add_metrics(
        &self,
        endpoint: &crate::component::Endpoint,
        metrics_labels: Option<&[(&str, &str)]>,
    ) -> Result<()> {
        let metrics = WorkHandlerMetrics::from_endpoint(endpoint, metrics_labels)
            .map_err(|e| anyhow::anyhow!("Failed to create work handler metrics: {}", e))?;

        self.metrics
            .set(Arc::new(metrics))
            .map_err(|_| anyhow::anyhow!("Metrics already set"))
    }

    pub fn link(segment: Arc<SegmentSource<Req, Resp>>) -> Result<Arc<Self>> {
        let ingress = Ingress::new();
        ingress.attach(segment)?;
        Ok(ingress)
    }

    pub fn for_pipeline(segment: Arc<SegmentSource<Req, Resp>>) -> Result<Arc<Self>> {
        let ingress = Ingress::new();
        ingress.attach(segment)?;
        Ok(ingress)
    }

    pub fn for_engine(engine: ServiceEngine<Req, Resp>) -> Result<Arc<Self>> {
        let frontend = SegmentSource::<Req, Resp>::new();
        let backend = ServiceBackend::from_engine(engine);

        // create the pipeline
        let pipeline = frontend.link(backend)?.link(frontend)?;

        let ingress = Ingress::new();
        ingress.attach(pipeline)?;

        Ok(ingress)
    }

    /// Helper method to access metrics if available
    fn metrics(&self) -> Option<&Arc<WorkHandlerMetrics>> {
        self.metrics.get()
    }
}

#[async_trait]
pub trait PushWorkHandler: Send + Sync {
    async fn handle_payload(&self, payload: Bytes) -> Result<(), PipelineError>;

    /// Add metrics to the handler
    fn add_metrics(
        &self,
        endpoint: &crate::component::Endpoint,
        metrics_labels: Option<&[(&str, &str)]>,
    ) -> Result<()>;

    /// Set the endpoint-specific notifier for health check timer resets
    fn set_endpoint_health_check_notifier(
        &self,
        _notifier: Arc<tokio::sync::Notify>,
    ) -> Result<()> {
        // Default implementation for backwards compatibility
        Ok(())
    }
}

/*
/// `NetworkStreamWrapper` is a simple wrapper used to detect proper stream termination
/// in network communication between ingress and egress components.
///
/// **Purpose**: This wrapper solves the problem of detecting whether a stream ended
/// gracefully or was cut off prematurely (e.g., due to network issues).
///
/// **Design Rationale**:
/// - Cannot use `Annotated` directly because the generic type `U` varies:
///   - Sometimes `U = Annotated<...>`
///   - Sometimes `U = LLMEngineOutput<...>`
/// - Using `Annotated` would require double-wrapping like `Annotated<Annotated<...>>`
/// - A simple wrapper is cleaner and more straightforward
///
/// **Stream Flow**:
/// ```
/// At AsyncEngine:
///   response 1 -> response 2 -> response 3 -> <end>
///
/// Between ingress/egress:
///   response 1 <end=false> -> response 2 <end=false> -> response 3 <end=false> -> (null) <end=true>
///
/// At client:
///   response 1 -> response 2 -> response 3 -> <end>
/// ```
///
/// **Error Handling**:
/// If the stream is cut off before proper termination, the egress is responsible for
/// injecting an error response to communicate the incomplete stream to the client:
/// ```
/// At AsyncEngine:
///   response 1 -> ... <without end flag>
///
/// At egress:
///   response 1 <end=false> -> <stream ended without end flag -> convert to error>
///
/// At client:
///   response 1 -> error response
/// ```
///
/// The detection must be done at egress level because premature stream termination
/// can be due to network issues that only the egress component can detect.
*/
/// TODO: Detect end-of-stream using Server-Sent Events (SSE). This will be removed.
#[derive(Serialize, Deserialize, Debug)]
pub struct NetworkStreamWrapper<U> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<U>,
    pub complete_final: bool,
}
