// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
use crate::pipeline::network::StreamOptions;
use crate::pipeline::network::TwoPartCodec;
use crate::pipeline::network::codec::TwoPartMessage;
use crate::pipeline::network::tcp;
use crate::pipeline::{ManyOut, PipelineError, ResponseStream, SingleIn};
use crate::protocols::maybe_error::MaybeError;
use crate::traits::DistributedRuntimeProvider;

use anyhow::{Error, Result};
use futures::stream::Stream;
use serde::Deserialize;
use serde::Serialize;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio_stream::{StreamExt, StreamNotifyClose, wrappers::ReceiverStream};
use tracing::Instrument;

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
    /// Wall-clock send timestamp (nanos since UNIX epoch) for transport latency breakdown.
    /// Uses `SystemTime` so accuracy depends on NTP sync between frontend and backend hosts.
    /// Reliable for single-machine profiling; treat cross-host values as approximate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frontend_send_ts_ns: Option<u64>,
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
        let engine_ctx_ = engine_ctx.clone();

        // registration options for the data plane in a singe in / many out configuration
        let options = StreamOptions::builder()
            .context(engine_ctx.clone())
            .enable_request_stream(false)
            .enable_response_stream(true)
            .build()
            .unwrap();

        // register our needs with the data plane
        // todo - generalize this with a generic data plane object which hides the specific transports
        let pending_connections: PendingConnections = self.resp_transport.register(options).await;

        // validate and unwrap the RegisteredStream object
        let pending_response_stream = match pending_connections.into_parts() {
            (None, Some(recv_stream)) => recv_stream,
            _ => {
                panic!("Invalid data plane registration for a SingleIn/ManyOut transport");
            }
        };

        // separate out the connection info and the stream provider from the registered stream
        let (connection_info, response_stream_provider) = pending_response_stream.into_parts();

        // Snapshot subject before connection_info is moved; used for cleanup.
        let recv_subject: Option<String> =
            serde_json::from_str::<tcp::TcpStreamConnectionInfo>(&connection_info.info)
                .ok()
                .map(|ci| ci.subject);

        // If the instance is already tombstoned, fail fast with a migratable
        // error instead of writing to the request plane.
        if let (Some(subject), Some(inst)) = (&recv_subject, &instance_info) {
            let endpoint_instance_id = inst.endpoint_instance_id();
            if !self
                .resp_transport
                .associate_instance(subject, &endpoint_instance_id)
                .await
            {
                return Err(anyhow::anyhow!(
                    DynamoError::builder()
                        .error_type(ErrorType::Disconnected)
                        .message(
                            "Worker removed before request could be sent (tombstoned instance)"
                        )
                        .build()
                ));
            }
        }

        // package up the connection info as part of the "header" component of the two part message
        // used to issue the request on the
        // todo -- this object should be automatically created by the register call, and achieved by to the two into_parts()
        // calls. all the information here is provided by the [`StreamOptions`] object and/or the dataplane object
        let control_message = RequestControlMessage {
            id: engine_ctx.id().to_string(),
            request_type: RequestType::SingleIn,
            response_type: ResponseType::ManyOut,
            connection_info,
            frontend_send_ts_ns: None,
        };

        // next build the two part message where we package the connection info and the request into
        // a single Vec<u8> that can be sent over the wire.
        // --- package this up in the WorkQueuePublisher ---
        let ctrl = match serde_json::to_vec(&control_message) {
            Ok(v) => v,
            Err(e) => {
                if let Some(subject) = &recv_subject {
                    self.resp_transport.cancel_recv_stream(subject).await;
                }
                return Err(e.into());
            }
        };
        let data = match serde_json::to_vec(&request) {
            Ok(v) => v,
            Err(e) => {
                if let Some(subject) = &recv_subject {
                    self.resp_transport.cancel_recv_stream(subject).await;
                }
                return Err(e.into());
            }
        };

        tracing::trace!(
            request_id,
            "packaging two-part message; ctrl: {} bytes, data: {} bytes",
            ctrl.len(),
            data.len()
        );

        let msg = TwoPartMessage::from_parts(ctrl.into(), data.into());

        // the request plane / work queue should provide a two part message codec that can be used
        // or it should take a two part message directly
        // todo - update this
        let codec = TwoPartCodec::default();
        let buffer = match codec.encode_message(msg) {
            Ok(v) => v,
            Err(e) => {
                if let Some(subject) = &recv_subject {
                    self.resp_transport.cancel_recv_stream(subject).await;
                }
                return Err(e.into());
            }
        };

        REQUEST_PLANE_QUEUE_SECONDS.observe(queue_start.elapsed().as_secs_f64());
        let tx_start = Instant::now();

        // TRANSPORT ABSTRACT REQUIRED - END HERE

        // Send request using unified client interface
        tracing::trace!(
            request_id,
            transport = self.req_client.transport_name(),
            address = %address,
            "Sending request via request plane client"
        );

        // Prepare trace headers using shared helper
        let mut headers = std::collections::HashMap::new();
        inject_trace_headers_into_map(&mut headers);
        headers.insert("request-id".to_string(), request_id.clone());

        // Stamp send time right before the transport write so the network
        // transit metric excludes serialization/encoding overhead.
        let send_ts_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        headers.insert("x-frontend-send-ts-ns".to_string(), send_ts_ns.to_string());

        // Phase A: Frontend → Backend (network + queue + ack)
        let _nvtx_send = dynamo_nvtx_range!("transport.tcp.send");
        let send_result = self.req_client.send_request(address, buffer, headers).await;
        drop(_nvtx_send);

        if let Err(e) = send_result {
            if let Some(subject) = &recv_subject {
                self.resp_transport.cancel_recv_stream(subject).await;
            }
            return Err(e);
        }
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
                if let Some(subject) = &recv_subject {
                    self.resp_transport.cancel_recv_stream(subject).await;
                }
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
                if let Some(subject) = &recv_subject {
                    self.resp_transport.cancel_recv_stream(subject).await;
                }
                return Err(anyhow::anyhow!(
                    DynamoError::builder()
                        .error_type(ErrorType::Disconnected)
                        .message("Worker disconnected before response stream was established")
                        .build()
                ));
            }
        };
        drop(_nvtx_wait);

        // TODO: Detect end-of-stream using Server-Sent Events (SSE)
        let mut is_complete_final = false;
        let mut first_response = true;
        let stream = tokio_stream::StreamNotifyClose::new(
            tokio_stream::wrappers::ReceiverStream::new(response_stream.rx),
        )
        .filter_map(move |res| {
            if let Some(res_bytes) = res {
                if first_response {
                    first_response = false;
                    let roundtrip_ttft = tx_start.elapsed().as_secs_f64();
                    REQUEST_PLANE_ROUNDTRIP_TTFT_SECONDS.observe(roundtrip_ttft);
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
                            let err = DynamoError::msg(
                                "Empty response received - this should never happen",
                            );
                            Some(U::from_err(err))
                        }
                    }
                    Err(err) => {
                        // legacy log print
                        let json_str = String::from_utf8_lossy(&res_bytes);
                        tracing::warn!(%err, %json_str, "Failed deserializing JSON to response");

                        Some(U::from_err(DynamoError::msg(err.to_string())))
                    }
                }
            } else if is_complete_final {
                // end of stream
                None
            } else if engine_ctx_.is_stopped() {
                // Gracefully end the stream if 'stop_generating()' was called. Do NOT check for
                // 'is_killed()' here because it implies the stream ended abnormally which should be
                // handled by the error branch below.
                tracing::debug!("Request cancelled and then trying to read a response");
                None
            } else {
                // stream ended unexpectedly
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
        Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
    }
}
