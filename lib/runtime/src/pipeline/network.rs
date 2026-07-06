// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use super::{
    AsyncTransportEngine, Context, Data, Error, ManyIn, ManyOut, PipelineError, PipelineIO,
    SegmentSource, ServiceBackend, ServiceEngine, SingleIn, Source, context,
};
use crate::metrics::MetricsHierarchy;
use crate::metrics::prometheus_names::work_handler;
use crate::protocols::maybe_error::MaybeError;
use ingress::push_handler::WorkHandlerMetrics;
use prometheus::{CounterVec, Histogram, IntCounter, IntCounterVec, IntGauge};

/// Shared default maximum TCP message size across request-plane components.
pub(crate) const DEFAULT_TCP_MAX_MESSAGE_SIZE: usize = 32 * 1024 * 1024;

static TCP_MAX_MESSAGE_SIZE: OnceLock<usize> = OnceLock::new();
static REQUEST_PLANE_PAYLOAD_CODEC: OnceLock<RequestPlanePayloadCodec> = OnceLock::new();

/// Read the configured TCP max message size once and share it across client,
/// server, and zero-copy decoder code paths.
pub(crate) fn get_tcp_max_message_size() -> usize {
    *TCP_MAX_MESSAGE_SIZE.get_or_init(|| {
        std::env::var("DYN_TCP_MAX_MESSAGE_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_TCP_MAX_MESSAGE_SIZE)
    })
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RequestPlanePayloadCodec {
    /// The serde default deliberately remains JSON for wire compatibility with
    /// control messages produced before the payload codec field existed.
    #[default]
    Json,
    Msgpack,
}

impl RequestPlanePayloadCodec {
    pub fn configured() -> Self {
        *REQUEST_PLANE_PAYLOAD_CODEC.get_or_init(Self::from_env)
    }

    fn from_env() -> Self {
        let value =
            std::env::var(crate::config::environment_names::request_plane::DYN_REQUEST_PLANE_CODEC)
                .ok();
        Self::from_config_value(value.as_deref())
    }

    fn from_config_value(value: Option<&str>) -> Self {
        match value {
            None | Some("") | Some("msgpack") => Self::Msgpack,
            Some("json") => Self::Json,
            Some(other) => {
                tracing::warn!(
                    env_var =
                        crate::config::environment_names::request_plane::DYN_REQUEST_PLANE_CODEC,
                    value = other,
                    "invalid request plane payload codec, defaulting to msgpack"
                );
                Self::Msgpack
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Msgpack => "msgpack",
        }
    }

    pub fn encode<T: Serialize>(&self, value: &T) -> Result<Vec<u8>> {
        match self {
            Self::Json => Ok(serde_json::to_vec(value)?),
            Self::Msgpack => Ok(rmp_serde::to_vec_named(value)?),
        }
    }

    pub fn decode<T: DeserializeOwned>(&self, bytes: &[u8]) -> Result<T> {
        match self {
            Self::Json => Ok(serde_json::from_slice(bytes)?),
            Self::Msgpack => Ok(rmp_serde::from_slice(bytes)?),
        }
    }
}

pub trait Codable: PipelineIO + Serialize + for<'de> Deserialize<'de> {}
impl<T: PipelineIO + Serialize + for<'de> Deserialize<'de>> Codable for T {}

/// `WorkQueueConsumer` is a generic interface for a work queue that can be used to send and receive
#[async_trait]
pub trait WorkQueueConsumer {
    async fn dequeue(&self) -> Result<Bytes, String>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum StreamType {
    Request,
    Response,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RequestType {
    SingleIn,
    ManyIn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ResponseType {
    SingleOut,
    ManyOut,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RequestControlMessage {
    pub(crate) id: String,
    pub(crate) request_type: RequestType,
    pub(crate) response_type: ResponseType,
    #[serde(default)]
    pub(crate) payload_codec: RequestPlanePayloadCodec,
    pub(crate) connection_info: ConnectionInfo,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub(crate) metadata: std::collections::BTreeMap<String, String>,
    /// Wall-clock send timestamp (nanos since UNIX epoch) for transport latency breakdown.
    /// Uses `SystemTime` so accuracy depends on NTP sync between frontend and backend hosts.
    /// Reliable for single-machine profiling; treat cross-host values as approximate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) frontend_send_ts_ns: Option<u64>,
    /// For bidirectional dispatch (`request_type == ManyIn`): connection info the
    /// worker dials back to in order to receive subsequent request frames. `None`
    /// for the unary path, which is the wire-compatible default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) request_stream_connection_info: Option<ConnectionInfo>,
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

/// Owning `Drop` here (rather than on `RegisteredStream`) lets `into_parts()`
/// move the public fields out by plain destructure.
struct Cleanup(Option<Box<dyn FnOnce() + Send + 'static>>);

impl Drop for Cleanup {
    fn drop(&mut self) {
        if let Some(f) = self.0.take() {
            f();
        }
    }
}

/// Awaitable handle for a stream sender or receiver. Drop without calling
/// [`into_parts()`] runs the optional cleanup closure, removing the
/// registration from the stream server's maps.
pub struct RegisteredStream<T> {
    pub connection_info: ConnectionInfo,
    pub stream_provider: StreamProvider<T>,
    cleanup: Cleanup,
}

impl<T> std::fmt::Debug for RegisteredStream<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredStream")
            .field("connection_info", &self.connection_info)
            .finish_non_exhaustive()
    }
}

impl<T> RegisteredStream<T> {
    pub(crate) fn new(connection_info: ConnectionInfo, stream_provider: StreamProvider<T>) -> Self {
        Self {
            connection_info,
            stream_provider,
            cleanup: Cleanup(None),
        }
    }

    pub(crate) fn with_cleanup<F>(mut self, cleanup: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        self.cleanup.0 = Some(Box::new(cleanup));
        self
    }

    /// Consume the registration, disarming the RAII cleanup. Caller takes
    /// responsibility for cleanup if the stream provider is never awaited.
    pub fn into_parts(self) -> (ConnectionInfo, StreamProvider<T>) {
        let Self {
            connection_info,
            stream_provider,
            mut cleanup,
        } = self;
        cleanup.0.take();
        (connection_info, stream_provider)
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

#[cfg(test)]
mod registered_stream_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn dummy_conn_info() -> ConnectionInfo {
        ConnectionInfo {
            transport: "test".to_string(),
            info: "{}".to_string(),
        }
    }

    /// Drop without `into_parts()` must run the cleanup closure.
    #[test]
    fn drop_runs_cleanup() {
        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = flag.clone();

        let (_tx, rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
        let stream = RegisteredStream::new(dummy_conn_info(), rx).with_cleanup(move || {
            flag_clone.store(true, Ordering::SeqCst);
        });

        drop(stream);
        assert!(
            flag.load(Ordering::SeqCst),
            "cleanup must fire when RegisteredStream is dropped"
        );
    }

    /// `into_parts()` must disarm the cleanup. After the call, dropping the
    /// returned halves must NOT trigger the closure -- the caller has taken
    /// ownership of cleanup responsibility.
    #[test]
    fn into_parts_disarms_cleanup() {
        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = flag.clone();

        let (_tx, rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
        let stream = RegisteredStream::new(dummy_conn_info(), rx).with_cleanup(move || {
            flag_clone.store(true, Ordering::SeqCst);
        });

        let (conn, provider) = stream.into_parts();
        drop(conn);
        drop(provider);

        assert!(
            !flag.load(Ordering::SeqCst),
            "into_parts() must disarm the cleanup closure"
        );
    }

    /// `RegisteredStream` with no cleanup configured must drop cleanly.
    #[test]
    fn drop_without_cleanup_is_a_noop() {
        let (_tx, rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
        let stream: RegisteredStream<()> = RegisteredStream::new(dummy_conn_info(), rx);
        drop(stream); // must not panic; nothing observable to assert beyond that
    }
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
        // leaving the original logic in place for now
        // error overrides the dissolved prologue, but the only field on `ResponseStreamPrologue` is `error`
        // so the second argument can never be used, and the value of error passed by the caller would always be used
        if let Some(_prologue) = self.prologue.take() {
            // let prologue = ResponseStreamPrologue { error, ..prologue };
            let prologue = ResponseStreamPrologue { error };
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
    /// that can be picked up by the Request/Forward pipeline. The downstream side
    /// dials in via [`crate::pipeline::network::tcp::client::TcpClient::create_request_stream`]
    /// to receive the frames the server pushes.
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

#[cfg(test)]
mod tests {
    use super::{
        NetworkStreamWrapper, RequestControlMessage, RequestPlanePayloadCodec, RequestType,
        ResponseType,
    };
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct TestPayload {
        id: u64,
        text: String,
        tokens: Vec<u32>,
    }

    #[test]
    fn request_control_message_defaults_missing_metadata() {
        let json = r#"{
            "id": "request-123",
            "request_type": "single_in",
            "response_type": "many_out",
            "connection_info": {
                "transport": "tcp",
                "info": "{}"
            }
        }"#;

        let message: RequestControlMessage =
            serde_json::from_str(json).expect("control message should deserialize");

        assert_eq!(message.id, "request-123");
        assert!(matches!(message.request_type, RequestType::SingleIn));
        assert!(matches!(message.response_type, ResponseType::ManyOut));
        assert_eq!(message.payload_codec, RequestPlanePayloadCodec::Json);
        assert_eq!(message.connection_info.transport, "tcp");
        assert_eq!(message.connection_info.info, "{}");
        assert!(message.metadata.is_empty());
        assert!(message.frontend_send_ts_ns.is_none());
    }

    #[test]
    fn request_control_message_decodes_msgpack_payload_codec() {
        let json = r#"{
            "id": "request-123",
            "request_type": "single_in",
            "response_type": "many_out",
            "payload_codec": "msgpack",
            "connection_info": {
                "transport": "tcp",
                "info": "{}"
            }
        }"#;

        let message: RequestControlMessage =
            serde_json::from_str(json).expect("control message should deserialize");

        assert_eq!(message.payload_codec, RequestPlanePayloadCodec::Msgpack);
    }

    #[test]
    fn request_plane_payload_codec_configuration_defaults_to_msgpack() {
        assert_eq!(
            RequestPlanePayloadCodec::from_config_value(None),
            RequestPlanePayloadCodec::Msgpack
        );
        assert_eq!(
            RequestPlanePayloadCodec::from_config_value(Some("")),
            RequestPlanePayloadCodec::Msgpack
        );
        assert_eq!(
            RequestPlanePayloadCodec::from_config_value(Some("invalid")),
            RequestPlanePayloadCodec::Msgpack
        );
    }

    #[test]
    fn request_plane_payload_codec_configuration_honors_explicit_overrides() {
        assert_eq!(
            RequestPlanePayloadCodec::from_config_value(Some("json")),
            RequestPlanePayloadCodec::Json
        );
        assert_eq!(
            RequestPlanePayloadCodec::from_config_value(Some("msgpack")),
            RequestPlanePayloadCodec::Msgpack
        );
    }

    #[test]
    fn request_plane_payload_codec_round_trips_response_wrapper_json_and_msgpack() {
        let wrapper = NetworkStreamWrapper {
            data: Some(TestPayload {
                id: 42,
                text: "line\nquote\"slash\\unicode 中".to_string(),
                tokens: vec![1, 2, 3, 65535],
            }),
            complete_final: false,
        };

        for codec in [
            RequestPlanePayloadCodec::Json,
            RequestPlanePayloadCodec::Msgpack,
        ] {
            let encoded = codec.encode(&wrapper).expect("wrapper should encode");
            let decoded: NetworkStreamWrapper<TestPayload> =
                codec.decode(&encoded).expect("wrapper should decode");
            assert_eq!(decoded, wrapper);
        }
    }
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

/// Result of encoding one response item for the request plane.
pub struct EncodedResponseFrame {
    pub bytes: Bytes,
    pub is_error: bool,
    /// Stop consuming the engine stream after publishing this frame. The
    /// normal complete-final frame is still sent.
    pub stop_stream: bool,
}

/// Converts request-plane bytes into the item consumed by an ingress engine.
pub trait IngressRequestDecoder<T>: Send + Sync + 'static
where
    T: Data,
{
    fn decode_request(
        &self,
        payload_codec: RequestPlanePayloadCodec,
        bytes: Bytes,
    ) -> impl std::future::Future<Output = std::result::Result<T, PipelineError>> + Send;
}

/// Converts an ingress engine response into its complete on-wire frame.
pub trait IngressResponseEncoder<U>: Send + Sync + 'static
where
    U: Data,
{
    fn encode_response(
        &self,
        payload_codec: RequestPlanePayloadCodec,
        response: Option<U>,
        complete_final: bool,
    ) -> impl std::future::Future<Output = std::result::Result<EncodedResponseFrame, PipelineError>> + Send;
}

/// Complete request/response payload adapter for an ingress engine.
pub trait IngressPayloadAdapter<T, U>:
    IngressRequestDecoder<T> + IngressResponseEncoder<U>
where
    T: Data,
    U: Data,
{
}

impl<T, U, Adapter> IngressPayloadAdapter<T, U> for Adapter
where
    T: Data,
    U: Data,
    Adapter: IngressRequestDecoder<T> + IngressResponseEncoder<U>,
{
}

/// Default adapter for ordinary Rust request and response types.
#[derive(Debug, Default)]
pub struct SerdeIngressPayloadAdapter;

impl<T> IngressRequestDecoder<T> for SerdeIngressPayloadAdapter
where
    T: Data + DeserializeOwned,
{
    #[inline]
    fn decode_request(
        &self,
        payload_codec: RequestPlanePayloadCodec,
        bytes: Bytes,
    ) -> impl std::future::Future<Output = std::result::Result<T, PipelineError>> + Send {
        let decoded = payload_codec.decode(&bytes).map_err(|err| {
            PipelineError::DeserializationError(format!(
                "Failed deserializing {} request payload: {}",
                payload_codec.name(),
                err
            ))
        });
        std::future::ready(decoded)
    }
}

impl<U> IngressResponseEncoder<U> for SerdeIngressPayloadAdapter
where
    U: Data + Serialize + MaybeError,
{
    #[inline]
    fn encode_response(
        &self,
        payload_codec: RequestPlanePayloadCodec,
        response: Option<U>,
        complete_final: bool,
    ) -> impl std::future::Future<Output = std::result::Result<EncodedResponseFrame, PipelineError>> + Send
    {
        let is_error = response
            .as_ref()
            .is_some_and(|response| response.err().is_some());
        let wrapper = NetworkStreamWrapper {
            data: response,
            complete_final,
        };
        let encoded = payload_codec.encode(&wrapper).map_err(|err| {
            PipelineError::SerializationError(format!(
                "Failed serializing {} request-plane response: {}",
                payload_codec.name(),
                err
            ))
        });
        std::future::ready(encoded.map(|bytes| EncodedResponseFrame {
            bytes: bytes.into(),
            is_error,
            stop_stream: false,
        }))
    }
}

pub struct Ingress<Req: PipelineIO, Resp: PipelineIO, Adapter = SerdeIngressPayloadAdapter> {
    segment: OnceLock<Arc<SegmentSource<Req, Resp>>>,
    metrics: OnceLock<Arc<WorkHandlerMetrics>>,
    /// Endpoint-specific notifier for health check timer resets
    endpoint_health_check_notifier: OnceLock<Arc<tokio::sync::Notify>>,
    payload_adapter: Arc<Adapter>,
}

impl<Req: PipelineIO + Sync, Resp: PipelineIO> Ingress<Req, Resp> {
    pub fn new() -> Arc<Self> {
        Ingress::new_with_adapter(SerdeIngressPayloadAdapter)
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
        Self::for_engine_with_adapter(engine, SerdeIngressPayloadAdapter)
    }
}

impl<Req, Resp, Adapter> Ingress<Req, Resp, Adapter>
where
    Req: PipelineIO + Sync,
    Resp: PipelineIO,
    Adapter: Send + Sync + 'static,
{
    pub fn new_with_adapter(payload_adapter: Adapter) -> Arc<Self> {
        Arc::new(Self {
            segment: OnceLock::new(),
            metrics: OnceLock::new(),
            endpoint_health_check_notifier: OnceLock::new(),
            payload_adapter: Arc::new(payload_adapter),
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

        // Register global transport breakdown metrics (idempotent)
        crate::metrics::work_handler_perf::ensure_work_handler_perf_metrics_registered(
            endpoint.get_metrics_registry(),
        );

        // Register worker-pool saturation metrics (idempotent). These are
        // process-global and shared across all endpoints attached to the
        // same shared TCP server.
        crate::metrics::work_handler_pool::ensure_work_handler_pool_metrics_registered(
            endpoint.get_metrics_registry(),
        );

        self.metrics
            .set(Arc::new(metrics))
            .map_err(|_| anyhow::anyhow!("Metrics already set"))
    }

    pub fn for_engine_with_adapter(
        engine: ServiceEngine<Req, Resp>,
        payload_adapter: Adapter,
    ) -> Result<Arc<Self>> {
        let frontend = SegmentSource::<Req, Resp>::new();
        let backend = ServiceBackend::from_engine(engine);

        // create the pipeline
        let pipeline = frontend.link(backend)?.link(frontend)?;

        let ingress = Ingress::new_with_adapter(payload_adapter);
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
    async fn handle_payload(
        &self,
        payload: Bytes,
        request_id: Option<String>,
    ) -> Result<(), PipelineError>;

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
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct NetworkStreamWrapper<U> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<U>,
    pub complete_final: bool,
}
