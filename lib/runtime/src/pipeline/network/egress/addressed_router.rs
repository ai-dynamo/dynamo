// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use super::unified_client::RequestPlaneClient;
use super::*;
use crate::component::Instance;
use crate::discovery::EndpointInstanceId;
use crate::dynamo_nvtx_range;
use crate::engine::{AsyncEngine, AsyncEngineContextProvider, Data};
use crate::error::{DynamoError, ErrorType};
use crate::logging::inject_trace_headers_into_map;
use crate::metrics::frontend_perf::STAGE_DURATION_SECONDS;
use crate::metrics::request_plane::{
    REQUEST_PLANE_INFLIGHT, REQUEST_PLANE_QUEUE_SECONDS, REQUEST_PLANE_ROUNDTRIP_TTFT_SECONDS,
    REQUEST_PLANE_SEND_SECONDS,
};
use crate::pipeline::network::ConnectionInfo;
use crate::pipeline::network::NetworkStreamWrapper;
use crate::pipeline::network::PendingConnections;
use crate::pipeline::network::RegisteredStream;
use crate::pipeline::network::RequestControlMessage;
use crate::pipeline::network::RequestType;
use crate::pipeline::network::ResponseType;
use crate::pipeline::network::StreamOptions;
use crate::pipeline::network::StreamProvider;
use crate::pipeline::network::StreamReceiver;
use crate::pipeline::network::StreamSender;
use crate::pipeline::network::TwoPartCodec;
use crate::pipeline::network::codec::TwoPartMessage;
use crate::pipeline::network::tcp;
use crate::pipeline::{ManyIn, ManyOut, PipelineError, ResponseStream, SingleIn};
use crate::protocols::maybe_error::MaybeError;
use crate::traits::DistributedRuntimeProvider;

use anyhow::{Error, Result};
use futures::stream::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio_stream::{StreamExt, StreamNotifyClose, wrappers::ReceiverStream};
use tracing::Instrument;

/// Stream transformation helper that:
/// - decodes a response byte stream from network into the fully-shaped `ManyOut<U>`
/// - emits TTFT and transport-roundtrip metrics on first response
/// - hands off the `InflightGuard` to a stream-lifetime `InflightDecStream` so
///   the inflight gauge stays accurate for the whole response lifetime.
fn decode_response_stream<U>(
    response_rx: tokio::sync::mpsc::Receiver<bytes::Bytes>,
    engine_ctx: Arc<dyn crate::engine::AsyncEngineContext>,
    queue_start: Instant,
    tx_start: Instant,
    inflight_guard: InflightGuard,
) -> ManyOut<U>
where
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    let engine_ctx_for_stream = engine_ctx.clone();
    let mut is_complete_final = false;
    let mut first_response = true;
    let stream = StreamNotifyClose::new(ReceiverStream::new(response_rx)).filter_map(move |res| {
        if let Some(res_bytes) = res {
            if first_response {
                first_response = false;
                REQUEST_PLANE_ROUNDTRIP_TTFT_SECONDS.observe(tx_start.elapsed().as_secs_f64());
                STAGE_DURATION_SECONDS
                    .with_label_values(&["transport_roundtrip"])
                    .observe(queue_start.elapsed().as_secs_f64());
            }
            if is_complete_final {
                let err = DynamoError::msg(
                    "Response received after generation ended - this should never happen",
                );
                return Some(U::from_err(err));
            }
            match serde_json::from_slice::<NetworkStreamWrapper<U>>(&res_bytes) {
                Ok(item) => {
                    is_complete_final = item.complete_final;
                    if let Some(data) = item.data {
                        Some(data)
                    } else if is_complete_final {
                        None
                    } else {
                        let err =
                            DynamoError::msg("Empty response received - this should never happen");
                        Some(U::from_err(err))
                    }
                }
                Err(err) => {
                    let json_str = String::from_utf8_lossy(&res_bytes);
                    tracing::warn!(%err, %json_str, "Failed deserializing JSON to response");
                    Some(U::from_err(DynamoError::msg(err.to_string())))
                }
            }
        } else if is_complete_final {
            None
        } else if engine_ctx_for_stream.is_stopped() {
            tracing::debug!("Request cancelled and then trying to read a response");
            None
        } else {
            let err = DynamoError::builder()
                .error_type(ErrorType::Disconnected)
                .message("Stream ended before generation completed")
                .build();
            tracing::debug!("{err}");
            Some(U::from_err(err))
        }
    });

    inflight_guard.disarm();
    let stream = InflightDecStream { inner: stream };
    ResponseStream::new(Box::pin(stream), engine_ctx)
}

const CONTROL_MESSAGE_MAX_BYTES: usize = 128 * 1024;

fn serialize_control_message(control_message: &RequestControlMessage) -> Result<Vec<u8>, Error> {
    let ctrl = serde_json::to_vec(control_message)?;
    if ctrl.len() > CONTROL_MESSAGE_MAX_BYTES {
        return Err(PipelineError::Generic(format!(
            "request control message too large: {} bytes exceeds limit {}",
            ctrl.len(),
            CONTROL_MESSAGE_MAX_BYTES
        ))
        .into());
    }
    Ok(ctrl)
}

/// Build the request control message, and serialize for transfer.
///
/// `request` provides the optional unary request payload. Should set for
/// SingleIn generation.
/// `send_conn_info` provides the connection info for the request stream.
/// Should set for ManyIn generation.
fn build_request_envelope<T>(
    context: &context::Context<()>,
    recv_conn_info: ConnectionInfo,
    send_conn_info: Option<ConnectionInfo>,
    request: Option<&T>,
) -> Result<bytes::Bytes, Error>
where
    T: serde::Serialize + ?Sized,
{
    let request_id = context.id();
    let request_type = if send_conn_info.is_some() {
        RequestType::ManyIn
    } else {
        RequestType::SingleIn
    };
    let control_message = RequestControlMessage {
        id: request_id.to_string(),
        request_type,
        response_type: ResponseType::ManyOut,
        connection_info: recv_conn_info,
        metadata: context.metadata().clone(),
        frontend_send_ts_ns: None,
        request_stream_connection_info: send_conn_info,
    };

    let ctrl = serialize_control_message(&control_message)?;
    let data: Option<Vec<u8>> = match request {
        Some(req) => Some(serde_json::to_vec(req)?),
        None => None,
    };

    let msg = match data {
        Some(d) => {
            tracing::trace!(
                request_id,
                "packaging two-part message; ctrl: {} bytes, data: {} bytes",
                ctrl.len(),
                d.len(),
            );
            TwoPartMessage::from_parts(ctrl.into(), d.into())
        }
        None => {
            tracing::trace!(
                request_id,
                "packaging bidirectional header-only envelope; ctrl: {} bytes",
                ctrl.len(),
            );
            TwoPartMessage::from_header(ctrl.into())
        }
    };

    let codec = TwoPartCodec::default();
    let buffer = codec.encode_message(msg)?;
    Ok(buffer)
}

/// Await the network request-stream dial-in (if `request_stream_provider` is `Some`)
/// and spawn a detached task that forwards every item from `input_stream` onto
/// the request stream. Returns once the forwarder is spawned; `Err` if request-stream
/// dial-in fails.
async fn spawn_request_stream_forwarder<T>(
    request_stream_provider: Option<StreamProvider<StreamSender>>,
    mut input_stream: crate::engine::DataStream<T>,
    engine_ctx: Arc<dyn crate::engine::AsyncEngineContext>,
) -> Result<(), Error>
where
    T: serde::Serialize + Send + 'static,
{
    let Some(provider) = request_stream_provider else {
        return Ok(());
    };

    let request_sender = match provider.await {
        Ok(Ok(sender)) => sender,
        Ok(Err(e)) => {
            return Err(anyhow::anyhow!(
                DynamoError::builder()
                    .error_type(ErrorType::CannotConnect)
                    .message(format!("Worker dial-in failed for request stream: {e}"))
                    .build()
            ));
        }
        Err(_) => {
            return Err(anyhow::anyhow!(
                DynamoError::builder()
                    .error_type(ErrorType::Disconnected)
                    .message("Worker disconnected before request stream was established")
                    .build()
            ));
        }
    };

    // The task exits on stream end, context kill/stop, send error (worker
    // dropped its receiver), or local serialize failure. On any exit
    // `request_sender` drops and triggers transport shutdown (see server.rs for details)
    // which closes the upstream mpsc, triggering the server-side handler to emit
    // `Sentinel`, which signals the worker's reader to end cleanly.
    tokio::spawn(async move {
        loop {
            let item = tokio::select! {
                biased;
                _ = engine_ctx.killed() => break,
                _ = engine_ctx.stopped() => break,
                item = input_stream.next() => match item {
                    Some(item) => item,
                    None => break,
                },
            };
            let bytes = match serde_json::to_vec(&item) {
                Ok(b) => b,
                Err(e) => {
                    // Stream-side framing failure: the engine sees a
                    // partial input, so kill the context to abort both
                    // directions consistently rather than silently
                    // dropping frames.
                    tracing::error!(
                        error = %e,
                        "failed to serialize bidirectional request frame; killing context"
                    );
                    engine_ctx.kill();
                    break;
                }
            };
            if request_sender.send(bytes.into()).await.is_err() {
                tracing::debug!("worker request-stream receiver dropped; forwarder exiting");
                break;
            }
        }
    });

    Ok(())
}

/// RAII guard that decrements REQUEST_PLANE_INFLIGHT on drop unless disarmed.
/// Protects against gauge leaks when `?` operators cause early returns between
/// the increment and `InflightDecStream` construction.
struct InflightGuard {
    armed: bool,
}

impl InflightGuard {
    fn new() -> Self {
        Self { armed: true }
    }

    /// Consume the guard without decrementing. Call this when `InflightDecStream`
    /// takes over responsibility for the decrement.
    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if self.armed {
            REQUEST_PLANE_INFLIGHT.dec();
        }
    }
}

/// Wrapper that decrements request-plane inflight gauge when the stream is dropped.
struct InflightDecStream<S> {
    inner: S,
}

impl<S, T> Stream for InflightDecStream<S>
where
    S: Stream<Item = T> + Unpin,
{
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<S> Drop for InflightDecStream<S> {
    fn drop(&mut self) {
        REQUEST_PLANE_INFLIGHT.dec();
    }
}

/// Extract the TCP stream subject from a [`ConnectionInfo`], if it carries a
/// well-formed [`tcp::TcpStreamConnectionInfo`]. Used for the pre-dispatch
/// tombstone check.
fn subject_of(conn_info: &ConnectionInfo) -> Option<String> {
    serde_json::from_str::<tcp::TcpStreamConnectionInfo>(&conn_info.info)
        .ok()
        .map(|ci| ci.subject)
}

pub struct AddressedRequest<T> {
    request: T,
    address: String,
    /// Carries endpoint name + instance_id so cancellation is scoped to the
    /// exact (endpoint, instance) pair, not all endpoints on the same runtime.
    instance: Option<Instance>,
}

impl<T> AddressedRequest<T> {
    pub fn new(request: T, address: String) -> Self {
        Self {
            request,
            address,
            instance: None,
        }
    }

    pub fn with_instance(request: T, address: String, instance: Instance) -> Self {
        Self {
            request,
            address,
            instance: Some(instance),
        }
    }

    pub fn for_instance(request: T, instance: Instance) -> Self {
        let address = instance.transport.address().to_string();
        Self::with_instance(request, address, instance)
    }

    pub(crate) fn into_parts(self) -> (T, String, Option<Instance>) {
        (self.request, self.address, self.instance)
    }
}

pub struct AddressedPushRouter {
    // Request transport (unified trait object - works with all transports)
    req_client: Arc<dyn RequestPlaneClient>,

    // Response transport (TCP streaming - unchanged)
    resp_transport: Arc<tcp::server::TcpStreamServer>,
}

impl AddressedPushRouter {
    /// Create a new router with a request plane client
    ///
    /// This is the unified constructor that works with any transport type.
    /// The client is provided as a trait object, hiding the specific implementation.
    pub fn new(
        req_client: Arc<dyn RequestPlaneClient>,
        resp_transport: Arc<tcp::server::TcpStreamServer>,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            req_client,
            resp_transport,
        }))
    }

    pub async fn from_runtime_provider(
        provider: &impl DistributedRuntimeProvider,
    ) -> Result<Arc<Self>> {
        let manager = provider.drt().network_manager();
        let req_client = manager.create_client()?;
        let resp_transport = provider.drt().tcp_server().await?;

        tracing::debug!(
            transport = req_client.transport_name(),
            "Creating AddressedPushRouter with request plane client"
        );

        Self::new(req_client, resp_transport)
    }

    /// Cancel all pending response-stream registrations for an instance.
    pub async fn cancel_instance_streams(&self, instance_id: &EndpointInstanceId) -> usize {
        self.resp_transport
            .cancel_instance_streams(instance_id)
            .await
    }

    /// Clear the tombstone after an instance reappears in discovery.
    pub async fn clear_instance_tombstone(&self, instance_id: &EndpointInstanceId) {
        self.resp_transport
            .clear_instance_tombstone(instance_id)
            .await
    }

    /// Bidirectional generation. Note that it doesn't implement the AsyncEngine trait directly
    /// because there is no trivial way to wrap (instance and address) into ManyIn style.
    /// May wrap as SingleIn<AddressedStreamRequest<T>> and unwrap here but really just syntax
    /// sugar, so we just do it inline here. Will consider only if we do want to call this from
    /// typed erased AsyncEngine impls.
    pub async fn generate_bidirectional<T, U>(
        &self,
        instance: Instance,
        address: String,
        input: ManyIn<T>,
    ) -> Result<ManyOut<U>, Error>
    where
        T: Data + Serialize,
        U: Data + for<'de> Deserialize<'de> + MaybeError,
    {
        let (request_stream, context) = input.into_parts();
        let input_stream = request_stream.take().ok_or_else(|| {
            anyhow::anyhow!("RequestStream::take called twice on bidirectional dispatch input")
        })?;

        self.dispatch_and_finalize::<T, U>(
            &context,
            address,
            Some(&instance),
            None,
            Some(input_stream),
        )
        .await
    }

    /// Shared dispatch core for both unary and bidirectional requests. Wire
    /// shape is inferred from the inputs:
    ///   - `input_stream = Some(_)` + `request = None` → bidirectional,
    ///     header-only envelope. The worker dials back for both halves and
    ///     pulls request frames off the spawned forwarder.
    ///   - `input_stream = None` + `request = Some(_)` → unary, two-part
    ///     `[ctrl, data]` envelope. The payload travels in the data part.
    async fn dispatch_and_finalize<T, U>(
        &self,
        context: &context::Context<()>,
        address: String,
        instance: Option<&Instance>,
        request: Option<&T>,
        input_stream: Option<crate::engine::DataStream<T>>,
    ) -> Result<ManyOut<U>, Error>
    where
        T: Data + Serialize,
        U: Data + for<'de> Deserialize<'de> + MaybeError,
    {
        let engine_ctx = context.context();

        let queue_start = Instant::now();
        REQUEST_PLANE_INFLIGHT.inc();
        let inflight_guard = InflightGuard::new();

        let enable_request_stream = input_stream.is_some();

        // Hold the `RegisteredStream` as their RAII cleanup stays armed while held,
        // which simplifies the cancellation of registration on error. Each side is
        // disarmed by `into_parts()` on awaiting stream provider: past that point the
        // subject is reaped by the worker's dial-in (instance healthy) or the discovery
        // watcher (instance dropped), so no cleanup is owed.
        let (send_registered, recv_registered) = self
            .register_streams(engine_ctx.clone(), enable_request_stream, true)
            .await?;
        let recv_registered = recv_registered.ok_or_else(|| {
            anyhow::anyhow!("response stream registration missing despite enable_response_stream")
        })?;

        // Tombstone check: if discovery already removed the worker, fail fast
        // with a migratable error rather than writing to the request plane.
        // Dropping the held registrations on this return runs their cleanup.
        let recv_subject = subject_of(&recv_registered.connection_info);
        let send_subject = send_registered
            .as_ref()
            .and_then(|r| subject_of(&r.connection_info));
        if let (Some(subject), Some(inst)) = (&recv_subject, instance)
            && !self
                .resp_transport
                .associate_instance(
                    subject,
                    send_subject.as_deref(),
                    &inst.endpoint_instance_id(),
                )
                .await
        {
            return Err(anyhow::anyhow!(
                DynamoError::builder()
                    .error_type(ErrorType::Disconnected)
                    .message("Worker removed before request could be sent (tombstoned instance)")
                    .build()
            ));
        }

        let buffer = build_request_envelope(
            context,
            recv_registered.connection_info.clone(),
            send_registered.as_ref().map(|r| r.connection_info.clone()),
            request,
        )?;
        REQUEST_PLANE_QUEUE_SECONDS.observe(queue_start.elapsed().as_secs_f64());

        let tx_start = Instant::now();
        self.dispatch_buffer(address, buffer, context.id()).await?;
        REQUEST_PLANE_SEND_SECONDS.observe(tx_start.elapsed().as_secs_f64());

        // Spawn the forwarder before awaiting the response prologue so request
        // frames pre-load into the worker's input buffer while the engine
        // initialises in parallel. The response provider only resolves after
        // `engine.generate()` returns; awaiting it second avoids stalling the
        // request-side handshake on engine setup latency.
        if let Some(stream) = input_stream {
            let request_stream_provider = send_registered.map(|r| {
                let (_conn_info, provider) = r.into_parts();
                provider
            });
            spawn_request_stream_forwarder(request_stream_provider, stream, engine_ctx.clone())
                .await?;
        }

        let _nvtx_wait = dynamo_nvtx_range!("transport.tcp.wait_backend");
        tracing::trace!(request_id = context.id(), "awaiting transport handshake");

        // Disarms the recv-side cleanup; see the holding rationale above.
        let (_recv_conn_info, response_stream_provider) = recv_registered.into_parts();

        // RecvError → migratable Disconnected (watcher cancelled the subject
        // or the worker died before establishing the response stream).
        let response_stream = match response_stream_provider.await {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) => {
                // generate() failed before any response bytes; migrate via
                // CannotConnect since the dominant cause is a worker-local
                // setup/version issue. The wire prologue carries only an
                // opaque string today, so app-level rejections also retry
                // -- safe because no side effects are visible yet. Follow-up:
                // structured prologue error type for finer routing.
                return Err(anyhow::anyhow!(
                    DynamoError::builder()
                        .error_type(ErrorType::CannotConnect)
                        .message(format!(
                            "Worker generate() failed before response stream: {e}"
                        ))
                        .build()
                ));
            }
            Err(_recv_err) => {
                // oneshot dropped: either the discovery watcher cancelled
                // this subject or the worker died mid-handshake.
                return Err(anyhow::anyhow!(
                    DynamoError::builder()
                        .error_type(ErrorType::Disconnected)
                        .message("Worker disconnected before response stream was established")
                        .build()
                ));
            }
        };
        drop(_nvtx_wait);

        Ok(decode_response_stream(
            response_stream.rx,
            engine_ctx,
            queue_start,
            tx_start,
            inflight_guard,
        ))
    }

    /// Register the requested halves of a data-plane stream with the response
    /// transport. Returns `(send_stream, recv_stream)` mirroring the
    /// `PendingConnections::into_parts` shape — either side is `None` when not
    /// requested. Asserts post-registration that the transport produced
    /// exactly the requested shape; a mismatch is a transport-layer bug, not
    /// a runtime error path.
    async fn register_streams(
        &self,
        engine_ctx: Arc<dyn crate::engine::AsyncEngineContext>,
        enable_request_stream: bool,
        enable_response_stream: bool,
    ) -> Result<
        (
            Option<RegisteredStream<StreamSender>>,
            Option<RegisteredStream<StreamReceiver>>,
        ),
        Error,
    > {
        let options = StreamOptions::builder()
            .context(engine_ctx)
            .enable_request_stream(enable_request_stream)
            .enable_response_stream(enable_response_stream)
            .build()?;

        let pending: PendingConnections = self.resp_transport.register(options).await;
        let (send_stream, recv_stream) = pending.into_parts();

        // Transport-layer invariant: the data plane produces exactly the halves
        // we requested. A mismatch is a bug in the transport, not a runtime
        // error path, so assert only in debug builds rather than panicking prod.
        debug_assert_eq!(
            send_stream.is_some(),
            enable_request_stream,
            "data-plane registration: request-stream presence does not match request"
        );
        debug_assert_eq!(
            recv_stream.is_some(),
            enable_response_stream,
            "data-plane registration: response-stream presence does not match request"
        );

        Ok((send_stream, recv_stream))
    }

    /// Build standard request-plane headers (trace propagation, request-id,
    /// frontend send-timestamp) and write the encoded buffer through the
    /// request-plane client.
    async fn dispatch_buffer(
        &self,
        address: String,
        buffer: bytes::Bytes,
        request_id: &str,
    ) -> Result<(), Error> {
        let mut headers = std::collections::HashMap::new();
        inject_trace_headers_into_map(&mut headers);
        headers.insert("request-id".to_string(), request_id.to_string());
        let send_ts_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        headers.insert("x-frontend-send-ts-ns".to_string(), send_ts_ns.to_string());

        let _nvtx_send = dynamo_nvtx_range!("transport.tcp.send");
        self.req_client
            .send_request(address, buffer, headers)
            .await?;
        drop(_nvtx_send);
        Ok(())
    }
}

#[async_trait::async_trait]
impl<T, U> AsyncEngine<SingleIn<AddressedRequest<T>>, ManyOut<U>, Error> for AddressedPushRouter
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(&self, request: SingleIn<AddressedRequest<T>>) -> Result<ManyOut<U>, Error> {
        let (addressed_request, context) = request.transfer(());
        let (request, address, instance_info) = addressed_request.into_parts();

        self.dispatch_and_finalize::<T, U>(
            &context,
            address,
            instance_info.as_ref(),
            Some(&request),
            None,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CONTROL_MESSAGE_MAX_BYTES, ConnectionInfo, RequestControlMessage, RequestType,
        ResponseType, serialize_control_message,
    };
    use std::collections::BTreeMap;

    fn base_control_message(metadata: BTreeMap<String, String>) -> RequestControlMessage {
        RequestControlMessage {
            id: "request-123".to_string(),
            request_type: RequestType::SingleIn,
            response_type: ResponseType::ManyOut,
            connection_info: ConnectionInfo {
                transport: "tcp".to_string(),
                info: "{}".to_string(),
            },
            metadata,
            frontend_send_ts_ns: None,
            request_stream_connection_info: None,
        }
    }

    #[test]
    fn serialize_control_message_succeeds_under_limit() {
        let mut metadata = BTreeMap::new();
        metadata.insert("x-tiny-blob".to_string(), "alpha".to_string());

        let ctrl = serialize_control_message(&base_control_message(metadata))
            .expect("control message should serialize under the limit");
        assert!(ctrl.len() <= CONTROL_MESSAGE_MAX_BYTES);
    }

    #[test]
    fn serialize_control_message_errors_over_limit() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "x-large-blob".to_string(),
            "x".repeat(CONTROL_MESSAGE_MAX_BYTES),
        );

        let err = serialize_control_message(&base_control_message(metadata))
            .expect_err("oversized control message should fail")
            .to_string();
        assert!(err.contains("request control message too large"));
        assert!(err.contains(&CONTROL_MESSAGE_MAX_BYTES.to_string()));
    }
}
