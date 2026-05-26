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

/// Wrap a response-stream `mpsc::Receiver<Bytes>` into the fully-shaped
/// `ManyOut<U>` returned by both the unary `AsyncEngine::generate` impl and
/// the bidirectional `generate_bidirectional` method: deserialize each frame
/// as `NetworkStreamWrapper<U>`, emit per-stream TTFT + transport-roundtrip
/// metrics on first response, and bridge the inflight-gauge from the
/// caller-owned `InflightGuard` into a stream-lifetime `InflightDecStream`.
fn finalize_response_stream<U>(
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

/// Build the request control envelope, optionally serialize the unary
/// data payload, and encode the whole thing into a wire buffer. Returns
/// the encoded `Bytes` and the NVTX span label that the caller should
/// wrap the wire write with.
///
/// Wire shape is inferred from the inputs:
///   - `send_conn_info = Some(_)` + `request = None` → header-only
///     envelope, `RequestType::ManyIn`. The worker dials back via the
///     attached connection info for inbound frames.
///   - `send_conn_info = None` + `request = Some(_)` → two-part
///     `[ctrl, data]` envelope, `RequestType::SingleIn`. The payload
///     travels in the data part.
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

    let msg = match &data {
        Some(d) => {
            tracing::trace!(
                request_id,
                "packaging two-part message; ctrl: {} bytes, data: {} bytes",
                ctrl.len(),
                d.len(),
            );
            TwoPartMessage::from_parts(ctrl.into(), d.clone().into())
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

/// RAII guard that cancels the recv and (optionally) send halves of a
/// registration on drop unless [`Self::disarm`] is called first. Drop is
/// sync; the async `cancel_*_stream` calls fire as a detached
/// `tokio::spawn` task. Cancellation is idempotent so the detached
/// completion order does not need to be observed by the caller.
struct CancelGuard {
    armed: bool,
    transport: Arc<tcp::server::TcpStreamServer>,
    recv_subject: Option<String>,
    send_subject: Option<String>,
}

impl CancelGuard {
    fn arm(
        transport: Arc<tcp::server::TcpStreamServer>,
        recv_subject: Option<String>,
        send_subject: Option<String>,
    ) -> Self {
        Self {
            armed: true,
            transport,
            recv_subject,
            send_subject,
        }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let transport = self.transport.clone();
        let recv = self.recv_subject.take();
        let send = self.send_subject.take();
        tokio::spawn(async move {
            if let Some(s) = recv {
                transport.cancel_recv_stream(&s).await;
            }
            if let Some(s) = send {
                transport.cancel_send_stream(&s).await;
            }
        });
    }
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

    /// Bidirectional sibling of the `AsyncEngine<SingleIn<AddressedRequest<T>>, ManyOut<U>>`
    /// impl: dispatch a `ManyIn<T>` to a specific `(instance, address)` pair.
    /// All input frames flow on the request-stream half of the call-home TCP
    /// transport; the initial envelope is header-only. The caller (typically
    /// `PushRouter`'s bidirectional impl) has already resolved the
    /// `(instance, address)` pair.
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
        let queue_start = Instant::now();
        REQUEST_PLANE_INFLIGHT.inc();
        let inflight_guard = InflightGuard::new();

        let engine_ctx = input.context();
        let engine_ctx_for_forwarder = engine_ctx.clone();
        let (request_stream, ctx_unit) = input.into_parts();
        let mut input_stream = request_stream
            .take()
            .expect("RequestStream::take called twice on bidirectional dispatch input");

        // Different from response (always streaming), the request stream is optional.
        let enable_request_stream = true;

        // Register both halves on the response transport: a `send_stream`
        // (upstream → worker, carrying subsequent request frames) and a
        // `recv_stream` (worker → upstream, carrying response chunks).
        let (pending_send_stream, pending_recv_stream) = self
            .register_streams(engine_ctx.clone(), enable_request_stream, true)
            .await;
        let (resp_stream_conn_info, response_stream_provider) =
            pending_recv_stream.unwrap().into_parts();
        let (req_stream_conn_info, request_stream_provider) = if enable_request_stream {
            let (connection_info, request_stream_provider) =
                pending_send_stream.unwrap().into_parts();
            (Some(connection_info), Some(request_stream_provider))
        } else {
            (None, None)
        };

        let (recv_subject, send_subject) = self
            .resolve_subjects(
                &resp_stream_conn_info,
                req_stream_conn_info.as_ref(),
                Some(&instance),
            )
            .await?;
        let cancel_guard =
            CancelGuard::arm(self.resp_transport.clone(), recv_subject, send_subject);

        // Bidirectional envelope is header-only: every request frame
        // (including the first) flows on the request-stream socket. The
        // worker decodes the control message, dials back for both streams,
        // and pulls frames off the request-stream as the engine asks for
        // them.
        let buffer = build_request_envelope::<()>(
            &ctx_unit,
            resp_stream_conn_info,
            req_stream_conn_info,
            None,
        )?;
        REQUEST_PLANE_QUEUE_SECONDS.observe(queue_start.elapsed().as_secs_f64());

        let tx_start = Instant::now();
        self.dispatch_buffer(address, buffer, ctx_unit.id()).await?;
        REQUEST_PLANE_SEND_SECONDS.observe(tx_start.elapsed().as_secs_f64());

        // Resolve the request-stream dial-back first and spawn the forwarder
        // immediately so frames start pre-loading into the worker's input
        // buffer while the engine initialises in parallel. The response side
        // carries the engine prologue and only resolves after
        // `engine.generate()` returns — awaiting it second avoids stalling
        // the request-side handshake on engine setup latency.
        if let Some(request_stream_provider) = request_stream_provider {
            let request_sender = match request_stream_provider.await {
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

            // Forwarder: pump every frame in `input` to the worker via the
            // request-stream send half. Exit on stream end, context kill/stop,
            // send error (worker dropped its receiver), or local serialize
            // failure. Dropping `request_sender` on exit closes the upstream
            // mpsc → server-side handler sends Sentinel → wire closes cleanly.
            tokio::spawn(async move {
                loop {
                    let item = tokio::select! {
                        biased;
                        _ = engine_ctx_for_forwarder.killed() => break,
                        _ = engine_ctx_for_forwarder.stopped() => break,
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
                            engine_ctx_for_forwarder.kill();
                            break;
                        }
                    };
                    if request_sender.send(bytes.into()).await.is_err() {
                        tracing::debug!(
                            "worker request-stream receiver dropped; forwarder exiting"
                        );
                        break;
                    }
                }
            });
        }

        let response_stream = match response_stream_provider.await {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) => {
                return Err(anyhow::anyhow!(
                    DynamoError::builder()
                        .error_type(ErrorType::CannotConnect)
                        .message(format!(
                            "Worker generate() failed before response stream: {e}"
                        ))
                        .build()
                ));
            }
            Err(_) => {
                return Err(anyhow::anyhow!(
                    DynamoError::builder()
                        .error_type(ErrorType::Disconnected)
                        .message("Worker disconnected before response stream was established")
                        .build()
                ));
            }
        };

        cancel_guard.disarm();
        Ok(finalize_response_stream(
            response_stream.rx,
            engine_ctx,
            queue_start,
            tx_start,
            inflight_guard,
        ))
    }

    /// Cancel both halves of a registration on the data plane. Used by
    /// [`Self::resolve_subjects`] on the tombstone-detected branch where
    /// the per-call [`CancelGuard`] hasn't been armed yet; all other
    /// cleanup paths go through that guard's `Drop`.
    async fn cancel_both(&self, recv_subject: &Option<String>, send_subject: &Option<String>) {
        if let Some(s) = recv_subject {
            self.resp_transport.cancel_recv_stream(s).await;
        }
        if let Some(s) = send_subject {
            self.resp_transport.cancel_send_stream(s).await;
        }
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
    ) -> (
        Option<RegisteredStream<StreamSender>>,
        Option<RegisteredStream<StreamReceiver>>,
    ) {
        let options = StreamOptions::builder()
            .context(engine_ctx)
            .enable_request_stream(enable_request_stream)
            .enable_response_stream(enable_response_stream)
            .build()
            .unwrap();

        let pending: PendingConnections = self.resp_transport.register(options).await;
        let (send_stream, recv_stream) = pending.into_parts();

        assert_eq!(
            send_stream.is_some(),
            enable_request_stream,
            "data-plane registration: request-stream presence does not match request"
        );
        assert_eq!(
            recv_stream.is_some(),
            enable_response_stream,
            "data-plane registration: response-stream presence does not match request"
        );

        (send_stream, recv_stream)
    }

    /// Resolve the TCP subjects from the recv-side (always present on TCP)
    /// and the optional send-side connection-info, then run the tombstone
    /// check via `resp_transport.associate_instance`. On tombstone, defensively
    /// invokes `cancel_both` (idempotent with the transport's internal cleanup)
    /// before returning a migratable `Disconnected` error.
    ///
    /// `send_conn_info` is `None` for the unary path; bidirectional dispatch
    /// passes `Some` when the send half was registered. `instance` is `None`
    /// for non-addressed unary callers; bidirectional always passes `Some`.
    async fn resolve_subjects(
        &self,
        recv_conn_info: &ConnectionInfo,
        send_conn_info: Option<&ConnectionInfo>,
        instance: Option<&Instance>,
    ) -> Result<(Option<String>, Option<String>), Error> {
        let recv_subject: Option<String> =
            serde_json::from_str::<tcp::TcpStreamConnectionInfo>(&recv_conn_info.info)
                .ok()
                .map(|ci| ci.subject);
        let send_subject: Option<String> = send_conn_info.and_then(|ci| {
            serde_json::from_str::<tcp::TcpStreamConnectionInfo>(&ci.info)
                .ok()
                .map(|ci| ci.subject)
        });

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
            self.cancel_both(&recv_subject, &send_subject).await;
            return Err(anyhow::anyhow!(
                DynamoError::builder()
                    .error_type(ErrorType::Disconnected)
                    .message("Worker removed before request could be sent (tombstoned instance)")
                    .build()
            ));
        }

        Ok((recv_subject, send_subject))
    }

    /// Build the standard request-plane headers (trace propagation +
    /// request-id + frontend send-timestamp) and dispatch the encoded
    /// buffer through the request-plane client inside an `nvtx_label`
    /// range. The wire-write step of a request-plane dispatch; the
    /// envelope is built upstream by [`build_request_envelope`].
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
        let queue_start = Instant::now();
        REQUEST_PLANE_INFLIGHT.inc();
        let inflight_guard = InflightGuard::new();

        let request_id = request.context().id().to_string();
        let (addressed_request, context) = request.transfer(());
        let (request, address, instance_info) = addressed_request.into_parts();
        let engine_ctx = context.context();

        let request_type = RequestType::SingleIn;
        let response_type = ResponseType::ManyOut;

        // Register only the recv half on the data plane for a single-in / many-out.
        // Request will be passed as part of the control message to the worker.
        let (_, pending_response_stream) = self
            .register_streams(
                engine_ctx.clone(),
                request_type == RequestType::ManyIn,
                response_type == ResponseType::ManyOut,
            )
            .await;

        // separate out the connection info and the stream provider from the registered stream
        let (connection_info, response_stream_provider) =
            pending_response_stream.unwrap().into_parts();

        let (recv_subject, send_subject) = self
            .resolve_subjects(&connection_info, None, instance_info.as_ref())
            .await?;
        let cancel_guard =
            CancelGuard::arm(self.resp_transport.clone(), recv_subject, send_subject);

        let buffer = build_request_envelope(&context, connection_info, None, Some(&request))?;
        REQUEST_PLANE_QUEUE_SECONDS.observe(queue_start.elapsed().as_secs_f64());

        let tx_start = Instant::now();
        self.dispatch_buffer(address, buffer, context.id()).await?;
        REQUEST_PLANE_SEND_SECONDS.observe(tx_start.elapsed().as_secs_f64());

        let _nvtx_wait = dynamo_nvtx_range!("transport.tcp.wait_backend");
        tracing::trace!(request_id, "awaiting transport handshake");

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

        cancel_guard.disarm();
        Ok(finalize_response_stream(
            response_stream.rx,
            engine_ctx,
            queue_start,
            tx_start,
            inflight_guard,
        ))
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
