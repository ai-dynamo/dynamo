// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use crate::engine::AsyncEngineContext;
use crate::metrics::prometheus_names::work_handler;
use crate::metrics::work_handler_perf::{
    WORK_HANDLER_NETWORK_TRANSIT_SECONDS, WORK_HANDLER_TIME_TO_FIRST_RESPONSE_SECONDS,
};
use crate::pipeline::{ManyIn, RequestStream};
use crate::protocols::maybe_error::MaybeError;
use futures::StreamExt;
use prometheus::{Histogram, IntCounter, IntCounterVec, IntGauge};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::Instrument;
use tracing::info_span;

/// Metrics configuration for profiling work handlers
#[derive(Clone, Debug)]
pub struct WorkHandlerMetrics {
    pub request_counter: IntCounter,
    pub request_duration: Histogram,
    pub inflight_requests: IntGauge,
    pub request_bytes: IntCounter,
    pub response_bytes: IntCounter,
    pub error_counter: IntCounterVec,
    pub cancellation_total: IntCounter,
}

impl WorkHandlerMetrics {
    pub fn new(
        request_counter: IntCounter,
        request_duration: Histogram,
        inflight_requests: IntGauge,
        request_bytes: IntCounter,
        response_bytes: IntCounter,
        error_counter: IntCounterVec,
        cancellation_total: IntCounter,
    ) -> Self {
        Self {
            request_counter,
            request_duration,
            inflight_requests,
            request_bytes,
            response_bytes,
            error_counter,
            cancellation_total,
        }
    }

    /// Create WorkHandlerMetrics from an endpoint using its built-in labeling
    pub fn from_endpoint(
        endpoint: &crate::component::Endpoint,
        metrics_labels: Option<&[(&str, &str)]>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let metrics_labels = metrics_labels.unwrap_or(&[]);
        let metrics = endpoint.metrics();
        let request_counter = metrics.create_intcounter(
            work_handler::REQUESTS_TOTAL,
            "Total number of requests processed by work handler",
            metrics_labels,
        )?;

        // Custom buckets for inference workloads: retain sub-second resolution for
        // fast operations, extend well beyond the default 10s ceiling to capture
        // long-running generation requests that can last minutes.
        let request_duration_buckets = vec![
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0,
            300.0, 600.0,
        ];
        let request_duration = metrics.create_histogram(
            work_handler::REQUEST_DURATION_SECONDS,
            "Time spent processing requests by work handler",
            metrics_labels,
            Some(request_duration_buckets),
        )?;

        let inflight_requests = metrics.create_intgauge(
            work_handler::INFLIGHT_REQUESTS,
            "Number of requests currently being processed by work handler",
            metrics_labels,
        )?;

        let request_bytes = metrics.create_intcounter(
            work_handler::REQUEST_BYTES_TOTAL,
            "Total number of bytes received in requests by work handler",
            metrics_labels,
        )?;

        let response_bytes = metrics.create_intcounter(
            work_handler::RESPONSE_BYTES_TOTAL,
            "Total number of bytes sent in responses by work handler",
            metrics_labels,
        )?;

        let error_counter = metrics.create_intcountervec(
            work_handler::ERRORS_TOTAL,
            "Total number of errors in work handler processing",
            &[work_handler::ERROR_TYPE_LABEL],
            metrics_labels,
        )?;

        let cancellation_total = metrics.create_intcounter(
            work_handler::CANCELLATION_TOTAL,
            "Total number of requests cancelled by work handler",
            metrics_labels,
        )?;

        Ok(Self::new(
            request_counter,
            request_duration,
            inflight_requests,
            request_bytes,
            response_bytes,
            error_counter,
            cancellation_total,
        ))
    }
}

// RAII guard to ensure inflight gauge is decremented, request duration is observed,
// and lifecycle logs are emitted on all code paths.
struct RequestMetricsGuard {
    inflight_requests: prometheus::IntGauge,
    request_duration: prometheus::Histogram,
    start_time: Instant,
    request_id: Option<String>,
}

impl Drop for RequestMetricsGuard {
    fn drop(&mut self) {
        self.inflight_requests.dec();
        self.request_duration
            .observe(self.start_time.elapsed().as_secs_f64());
        if let Some(request_id) = &self.request_id {
            tracing::info!(request_id = %request_id, "request completed");
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PublishFailureDisposition {
    Cancelled,
    Error,
}

fn classify_publish_failure(context: &dyn AsyncEngineContext) -> PublishFailureDisposition {
    if context.is_stopped() || context.is_killed() {
        PublishFailureDisposition::Cancelled
    } else {
        PublishFailureDisposition::Error
    }
}

impl<Req: PipelineIO + Sync, Resp: PipelineIO> Ingress<Req, Resp> {
    /// Pump every chunk from the engine's response stream out to the
    /// upstream-side `StreamSender`, plus the terminal complete-final
    /// frame. Captures the per-frame metrics, the publish-failure error
    /// classification (client-side disconnect vs. real failure), and the
    /// health-check notifier policy (notify only on non-error chunks and
    /// at clean stream end).
    async fn pump_response_stream<U>(
        &self,
        mut stream: ManyOut<U>,
        publisher: &StreamSender,
        payload_codec: RequestPlanePayloadCodec,
    ) where
        U: Data + Serialize + MaybeError + std::fmt::Debug,
    {
        let context = stream.context();

        // TODO: Detect end-of-stream using Server-Sent Events (SSE)
        let mut send_complete_final = true;
        let mut saw_error_response = false;
        while let Some(resp) = stream.next().await {
            tracing::trace!("Sending response: {:?}", resp);
            let is_error = resp.err().is_some();
            if is_error {
                saw_error_response = true;
            }
            let resp_wrapper = NetworkStreamWrapper {
                data: Some(resp),
                complete_final: false,
            };
            let resp_bytes = payload_codec
                .encode(&resp_wrapper)
                .expect("fatal error: invalid request-plane response object");
            if let Some(m) = self.metrics() {
                m.response_bytes.inc_by(resp_bytes.len() as u64);
            }
            if (publisher.send(resp_bytes.into()).await).is_err() {
                send_complete_final = false;
                if classify_publish_failure(context.as_ref())
                    == PublishFailureDisposition::Cancelled
                {
                    // Say there are 2 threads accessing `context`, the sequence can be either:
                    // 1. context.stop_generating (other) -> publisher.send failure (this)
                    //    -> context.is_stopped (this)
                    // 2. publisher.send failure (this) -> context.stop_generating (other)
                    //    -> context.is_stopped (this)
                    // Case 1 can happen when client closed the connection after receiving the
                    // complete response from frontend. Hence, send failure can be expected in this
                    // case.
                    tracing::warn!("Failed to publish response for stream {}", context.id());
                } else {
                    // Otherwise, this is an error.
                    tracing::error!("Failed to publish response for stream {}", context.id());
                    context.stop_generating();
                }
                // Account errors in all cases, including cancellation. Therefore this metric can be
                // inflated.
                if let Some(m) = self.metrics() {
                    m.error_counter
                        .with_label_values(&[work_handler::error_types::PUBLISH_RESPONSE])
                        .inc();
                }
                break;
            } else if !is_error {
                // Only notify on non-error chunks — error responses don't prove
                // the engine is healthy and should not reset the canary timer.
                if let Some(notifier) = self.endpoint_health_check_notifier.get() {
                    notifier.notify_one();
                }
            }
        }
        if send_complete_final {
            let resp_wrapper = NetworkStreamWrapper::<U> {
                data: None,
                complete_final: true,
            };
            let resp_bytes = payload_codec
                .encode(&resp_wrapper)
                .expect("fatal error: invalid request-plane response final object");
            if let Some(m) = self.metrics() {
                m.response_bytes.inc_by(resp_bytes.len() as u64);
            }
            if (publisher.send(resp_bytes.into()).await).is_err() {
                match classify_publish_failure(context.as_ref()) {
                    PublishFailureDisposition::Cancelled => tracing::warn!(
                        "Failed to publish complete final for stream {}",
                        context.id()
                    ),
                    PublishFailureDisposition::Error => tracing::error!(
                        "Failed to publish complete final for stream {}",
                        context.id()
                    ),
                }
                if let Some(m) = self.metrics() {
                    m.error_counter
                        .with_label_values(&[work_handler::error_types::PUBLISH_FINAL])
                        .inc();
                }
            }
            // Only notify on stream completion if no error responses were seen
            if let (false, Some(notifier)) = (
                saw_error_response,
                self.endpoint_health_check_notifier.get(),
            ) {
                notifier.notify_one();
            }
        }
    }

    /// Decode the wire envelope into its [`RequestControlMessage`] and the
    /// optional data payload, shared by every [`IngressDispatch`] shape:
    ///   - `HeaderAndData` → `(control, Some(data))` — the unary wire shape,
    ///     where the request body travels in the data half.
    ///   - `HeaderOnly` → `(control, None)` — the bidirectional wire shape,
    ///     where request frames flow on the request-stream socket instead.
    ///
    /// The caller decides whether its path expects the data payload. The
    /// deserialization and invalid-message error counters are incremented
    /// here so every shape reports them consistently.
    fn decode_control_message(
        &self,
        payload: Bytes,
    ) -> Result<(RequestControlMessage, Option<Bytes>), PipelineError> {
        let msg = TwoPartCodec::default()
            .decode_message(payload)?
            .into_message_type();

        let (header, data) = match msg {
            TwoPartMessageType::HeaderAndData(header, data) => (header, Some(data)),
            TwoPartMessageType::HeaderOnly(header) => (header, None),
            _ => {
                if let Some(m) = self.metrics() {
                    m.error_counter
                        .with_label_values(&[work_handler::error_types::INVALID_MESSAGE])
                        .inc();
                }
                return Err(PipelineError::Generic(String::from(
                    "Unexpected message from work queue; expected a header-only or header-and-data TwoPartMessage",
                )));
            }
        };

        let control_msg: RequestControlMessage =
            serde_json::from_slice(&header).map_err(|err| {
                if let Some(m) = self.metrics() {
                    m.error_counter
                        .with_label_values(&[work_handler::error_types::DESERIALIZATION])
                        .inc();
                }
                let json_str = String::from_utf8_lossy(&header);
                PipelineError::DeserializationError(format!(
                    "Failed deserializing to RequestControlMessage. err={err}, json_str={json_str}, header_len={}",
                    header.len(),
                ))
            })?;

        Ok((control_msg, data))
    }
}
/// The output of [`IngressDispatch::parse_and_build_request`]: the typed
/// request the engine consumes, plus the bits of the on-wire control
/// message the shared handler needs after parsing (the response-stream
/// connection info and the frontend send timestamp).
struct ParsedRequest<Req> {
    request: Req,
    response_connection_info: ConnectionInfo,
    frontend_send_ts_ns: Option<u64>,
    payload_codec: RequestPlanePayloadCodec,
}

/// Per-shape strategy for turning a raw payload into a typed engine
/// request. Captures the wire-shape divergence between the unary
/// (`HeaderAndData`) and bidirectional (`HeaderOnly` + dial-in for the
/// request stream) paths; everything else — metrics-guard, response stream
/// open, `segment.generate`, prologue, pump — lives in
/// [`Ingress::handle_payload_shared`] below.
#[async_trait]
trait IngressDispatch: Send + Sync {
    type Request: PipelineIO;

    async fn parse_and_build_request(
        &self,
        payload: Bytes,
    ) -> Result<ParsedRequest<Self::Request>, PipelineError>;
}

#[async_trait]
impl<T, U> IngressDispatch for Ingress<SingleIn<T>, ManyOut<U>>
where
    T: Data + for<'de> Deserialize<'de> + std::fmt::Debug,
    U: Data + Serialize + MaybeError + std::fmt::Debug,
{
    type Request = SingleIn<T>;

    async fn parse_and_build_request(
        &self,
        payload: Bytes,
    ) -> Result<ParsedRequest<SingleIn<T>>, PipelineError> {
        let (control_msg, data) = self.decode_control_message(payload)?;

        // The unary path carries the request body in the data half; a
        // header-only envelope means the sender used the bidirectional shape.
        let data = data.ok_or_else(|| {
            if let Some(m) = self.metrics() {
                m.error_counter
                    .with_label_values(&[work_handler::error_types::INVALID_MESSAGE])
                    .inc();
            }
            PipelineError::Generic(String::from(
                "unary engine received a header-only envelope; expected a request payload",
            ))
        })?;
        let payload_codec = control_msg.payload_codec;
        let request_t: T = payload_codec.decode(&data).map_err(|err| {
            if let Some(m) = self.metrics() {
                m.error_counter
                    .with_label_values(&[work_handler::error_types::DESERIALIZATION])
                    .inc();
            }
            PipelineError::DeserializationError(format!(
                "Failed deserializing {} request payload: {}",
                payload_codec.name(),
                err
            ))
        })?;

        tracing::trace!(
            request_id = %control_msg.id,
            metadata_entries = control_msg.metadata.len(),
            "received control message"
        );
        tracing::trace!("received request: {:?}", request_t);

        let request: context::Context<T> =
            Context::with_id_and_metadata(request_t, control_msg.id, control_msg.metadata);

        Ok(ParsedRequest {
            request,
            response_connection_info: control_msg.connection_info,
            frontend_send_ts_ns: control_msg.frontend_send_ts_ns,
            payload_codec,
        })
    }
}

#[async_trait]
impl<T, U> IngressDispatch for Ingress<ManyIn<T>, ManyOut<U>>
where
    T: Data + for<'de> Deserialize<'de> + std::fmt::Debug,
    U: Data + Serialize + MaybeError + std::fmt::Debug,
{
    type Request = ManyIn<T>;

    async fn parse_and_build_request(
        &self,
        payload: Bytes,
    ) -> Result<ParsedRequest<ManyIn<T>>, PipelineError> {
        let (control_msg, data) = self.decode_control_message(payload)?;

        // Bidirectional envelopes are header-only — all request frames
        // (including the first) flow on the request-stream socket once it's
        // dialed in. A data payload means the sender used the unary wire
        // shape; reject it.
        if data.is_some() {
            if let Some(m) = self.metrics() {
                m.error_counter
                    .with_label_values(&[work_handler::error_types::INVALID_MESSAGE])
                    .inc();
            }
            return Err(PipelineError::Generic(String::from(
                "bidirectional engine received a non-header-only envelope",
            )));
        }

        if !matches!(control_msg.request_type, RequestType::ManyIn) {
            if let Some(m) = self.metrics() {
                m.error_counter
                    .with_label_values(&[work_handler::error_types::INVALID_MESSAGE])
                    .inc();
            }
            return Err(PipelineError::Generic(String::from(
                "bidirectional engine received a non-ManyIn request envelope",
            )));
        }

        let req_stream_conn_info = control_msg
            .request_stream_connection_info
            .clone()
            .ok_or_else(|| {
                PipelineError::Generic(String::from(
                    "bidirectional control message missing request_stream_connection_info",
                ))
            })?;

        let request_context: context::Context<()> = context::Context::with_id_and_metadata(
            (),
            control_msg.id.clone(),
            control_msg.metadata.clone(),
        );
        let payload_codec = control_msg.payload_codec;
        let context_arc: Arc<dyn AsyncEngineContext> = request_context.context();

        // Open the request stream (upstream → worker) up front. The shared
        // handler opens the response stream uniformly after we return. If
        // response-stream open subsequently fails, the forwarder task
        // spawned below exits cleanly when `frame_tx.send` observes the
        // dropped `frame_rx`.
        let request_stream_recv = tcp::client::TcpClient::create_request_stream(
            context_arc.clone(),
            req_stream_conn_info,
            None,
        )
        .await
        .map_err(|e| {
            if let Some(m) = self.metrics() {
                m.error_counter
                    .with_label_values(&[work_handler::error_types::RESPONSE_STREAM])
                    .inc();
            }
            PipelineError::Generic(format!("Failed to create request stream: {e}"))
        })?;

        // Forwarder: deserialize raw bytes off the request socket into `T`
        // and feed the engine's `ManyIn<T>` input. Every request frame
        // (including the first) flows over this socket — the envelope is
        // header-only.
        let (frame_tx, frame_rx) = tokio::sync::mpsc::channel::<T>(8);
        let forwarder_ctx = context_arc.clone();
        tokio::spawn(async move {
            let mut rx = request_stream_recv.rx;
            while let Some(bytes) = rx.recv().await {
                // Stop forwarding on either kill or soft-stop, matching the
                // send-side `spawn_request_stream_forwarder`. Without the
                // `stopped()` check, a `stop_generating()` would leave this
                // task pumping frames into a channel the engine has abandoned.
                if forwarder_ctx.is_killed() || forwarder_ctx.is_stopped() {
                    break;
                }
                match payload_codec.decode::<T>(&bytes) {
                    Ok(item) => {
                        if frame_tx.send(item).await.is_err() {
                            tracing::debug!(
                                "engine consumer dropped; bidirectional input forwarder exiting"
                            );
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            error = %e,
                            codec = payload_codec.name(),
                            "failed to deserialize bidirectional request frame; killing context"
                        );
                        forwarder_ctx.kill();
                        break;
                    }
                }
            }
        });

        let input_stream: crate::engine::DataStream<T> =
            Box::pin(tokio_stream::wrappers::ReceiverStream::new(frame_rx));
        let request: ManyIn<T> = request_context.map(|_| RequestStream::new(input_stream));

        Ok(ParsedRequest {
            request,
            response_connection_info: control_msg.connection_info,
            frontend_send_ts_ns: control_msg.frontend_send_ts_ns,
            payload_codec,
        })
    }
}

impl<Req: PipelineIO + Sync, U> Ingress<Req, ManyOut<U>>
where
    U: Data + Serialize + MaybeError + std::fmt::Debug,
{
    /// Shared body of `PushWorkHandler::handle_payload` for every
    /// `Ingress<Req, ManyOut<U>>` shape that has an [`IngressDispatch`]
    /// impl. Sets up the inflight metrics guard, calls
    /// `parse_and_build_request` for the wire-shape-specific request
    /// building, opens the response stream uniformly, dispatches via
    /// the engine, sends the prologue, and pumps the response through
    /// [`Self::pump_response_stream`].
    async fn handle_payload_shared(
        &self,
        payload: Bytes,
        request_id: Option<String>,
    ) -> Result<(), PipelineError>
    where
        Self: IngressDispatch<Request = Req>,
    {
        let t2_wallclock_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let start_time = std::time::Instant::now();

        // Increment inflight and ensure it's decremented on all exits via RAII guard
        let _inflight_guard = self.metrics().map(|m| {
            m.request_counter.inc();
            m.inflight_requests.inc();
            m.request_bytes.inc_by(payload.len() as u64);
            if let Some(rid) = &request_id {
                tracing::info!(request_id = %rid, "request received");
            }
            RequestMetricsGuard {
                inflight_requests: m.inflight_requests.clone(),
                request_duration: m.request_duration.clone(),
                start_time,
                request_id: request_id.clone(),
            }
        });

        let ParsedRequest {
            request,
            response_connection_info,
            frontend_send_ts_ns,
            payload_codec,
        } = self.parse_and_build_request(payload).await?;

        // Compute network transit time (T2 - T1) using cross-process wall-clock timestamps
        if let Some(t1_ns) = frontend_send_ts_ns {
            let transit_ns = t2_wallclock_ns.saturating_sub(t1_ns);
            WORK_HANDLER_NETWORK_TRANSIT_SECONDS.observe(transit_ns as f64 / 1_000_000_000.0);
        }

        // todo - eventually have a handler class which will returned an abstracted object, but for now,
        // we only support tcp here, so we can just unwrap the connection info
        tracing::trace!("creating tcp response stream");
        let mut publisher = tcp::client::TcpClient::create_response_stream(
            request.context(),
            response_connection_info,
            self.metrics().map(|m| m.cancellation_total.clone()),
        )
        .await
        .map_err(|e| {
            if let Some(m) = self.metrics() {
                m.error_counter
                    .with_label_values(&[work_handler::error_types::RESPONSE_STREAM])
                    .inc();
            }
            PipelineError::Generic(format!("Failed to create response stream: {e}"))
        })?;

        tracing::trace!("calling generate");
        let stream = self
            .segment
            .get()
            .expect("segment not set")
            .generate(request)
            .await
            .map_err(|e| {
                if let Some(m) = self.metrics() {
                    m.error_counter
                        .with_label_values(&[work_handler::error_types::GENERATE])
                        .inc();
                }
                PipelineError::GenerateError(e)
            });

        // the prolouge is sent to the client to indicate that the stream is ready to receive data
        // or if the generate call failed, the error is sent to the client
        let stream = match stream {
            Ok(stream) => {
                tracing::trace!("Successfully generated response stream; sending prologue");
                let _result = publisher.send_prologue(None).await;
                WORK_HANDLER_TIME_TO_FIRST_RESPONSE_SECONDS
                    .observe(start_time.elapsed().as_secs_f64());
                stream
            }
            Err(e) => {
                let error_string = e.to_string();

                #[cfg(debug_assertions)]
                {
                    tracing::debug!(
                        "Failed to generate response stream (with debug backtrace): {:?}",
                        e
                    );
                }
                #[cfg(not(debug_assertions))]
                {
                    tracing::error!("Failed to generate response stream: {error_string}");
                }

                let _result = publisher.send_prologue(Some(error_string)).await;
                Err(e)?
            }
        };

        self.pump_response_stream(stream, &publisher, payload_codec)
            .await;

        // Ensure the metrics guard is not dropped until the end of the function.
        // Drop fires "request completed" log via RAII.
        drop(_inflight_guard);

        Ok(())
    }
}

#[async_trait]
impl<T, U> PushWorkHandler for Ingress<SingleIn<T>, ManyOut<U>>
where
    T: Data + for<'de> Deserialize<'de> + std::fmt::Debug,
    U: Data + Serialize + MaybeError + std::fmt::Debug,
{
    fn add_metrics(
        &self,
        endpoint: &crate::component::Endpoint,
        metrics_labels: Option<&[(&str, &str)]>,
    ) -> Result<()> {
        // Call the inherent `Ingress::add_metrics`, not this trait method.
        Ingress::add_metrics(self, endpoint, metrics_labels)
    }

    fn set_endpoint_health_check_notifier(&self, notifier: Arc<tokio::sync::Notify>) -> Result<()> {
        self.endpoint_health_check_notifier
            .set(notifier)
            .map_err(|_| anyhow::anyhow!("Endpoint health check notifier already set"))?;
        Ok(())
    }

    async fn handle_payload(
        &self,
        payload: Bytes,
        request_id: Option<String>,
    ) -> Result<(), PipelineError> {
        self.handle_payload_shared(payload, request_id).await
    }
}

#[async_trait]
impl<T, U> PushWorkHandler for Ingress<ManyIn<T>, ManyOut<U>>
where
    T: Data + for<'de> Deserialize<'de> + std::fmt::Debug,
    U: Data + Serialize + MaybeError + std::fmt::Debug,
{
    fn add_metrics(
        &self,
        endpoint: &crate::component::Endpoint,
        metrics_labels: Option<&[(&str, &str)]>,
    ) -> Result<()> {
        // Call the inherent `Ingress::add_metrics`, not this trait method.
        Ingress::add_metrics(self, endpoint, metrics_labels)
    }

    fn set_endpoint_health_check_notifier(&self, notifier: Arc<tokio::sync::Notify>) -> Result<()> {
        self.endpoint_health_check_notifier
            .set(notifier)
            .map_err(|_| anyhow::anyhow!("Endpoint health check notifier already set"))?;
        Ok(())
    }

    async fn handle_payload(
        &self,
        payload: Bytes,
        request_id: Option<String>,
    ) -> Result<(), PipelineError> {
        self.handle_payload_shared(payload, request_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::context::Controller;

    #[test]
    fn publish_failure_after_stop_or_kill_is_cancellation() {
        let live = Controller::new("live".to_string());
        assert_eq!(
            classify_publish_failure(&live),
            PublishFailureDisposition::Error
        );

        let stopped = Controller::new("stopped".to_string());
        stopped.stop();
        assert_eq!(
            classify_publish_failure(&stopped),
            PublishFailureDisposition::Cancelled
        );

        let killed = Controller::new("killed".to_string());
        killed.kill();
        assert_eq!(
            classify_publish_failure(&killed),
            PublishFailureDisposition::Cancelled
        );
    }
}
