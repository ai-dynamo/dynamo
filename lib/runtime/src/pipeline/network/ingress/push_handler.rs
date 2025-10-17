// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::metrics::prometheus_names::work_handler;
use crate::protocols::maybe_error::MaybeError;
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
}

impl WorkHandlerMetrics {
    pub fn new(
        request_counter: IntCounter,
        request_duration: Histogram,
        inflight_requests: IntGauge,
        request_bytes: IntCounter,
        response_bytes: IntCounter,
        error_counter: IntCounterVec,
    ) -> Self {
        Self {
            request_counter,
            request_duration,
            inflight_requests,
            request_bytes,
            response_bytes,
            error_counter,
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

        let request_duration = metrics.create_histogram(
            work_handler::REQUEST_DURATION_SECONDS,
            "Time spent processing requests by work handler",
            metrics_labels,
            None,
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

        Ok(Self::new(
            request_counter,
            request_duration,
            inflight_requests,
            request_bytes,
            response_bytes,
            error_counter,
        ))
    }
}

// RAII guard to ensure inflight gauge is decremented and request duration is observed on all code paths.
struct RequestMetricsGuard {
    inflight_requests: prometheus::IntGauge,
    request_duration: prometheus::Histogram,
    start_time: Instant,
}
impl Drop for RequestMetricsGuard {
    fn drop(&mut self) {
        self.inflight_requests.dec();
        self.request_duration
            .observe(self.start_time.elapsed().as_secs_f64());
    }
}

#[async_trait]
impl<T: Data, U: Data> PushWorkHandler for Ingress<SingleIn<T>, ManyOut<U>>
where
    T: Data + for<'de> Deserialize<'de> + std::fmt::Debug,
    U: Data + Serialize + MaybeError + std::fmt::Debug,
{
    fn add_metrics(
        &self,
        endpoint: &crate::component::Endpoint,
        metrics_labels: Option<&[(&str, &str)]>,
    ) -> Result<()> {
        // Call the Ingress-specific add_metrics implementation
        use crate::pipeline::network::Ingress;
        Ingress::add_metrics(self, endpoint, metrics_labels)
    }

    fn set_endpoint_health_check_notifier(&self, notifier: Arc<tokio::sync::Notify>) -> Result<()> {
        use crate::pipeline::network::Ingress;
        self.endpoint_health_check_notifier
            .set(notifier)
            .map_err(|_| anyhow::anyhow!("Endpoint health check notifier already set"))?;
        Ok(())
    }

    async fn handle_payload(&self, payload: Bytes) -> Result<(), PipelineError> {
        let start_time = std::time::Instant::now();

        // Increment inflight and ensure it's decremented on all exits via RAII guard
        let _inflight_guard = self.metrics().map(|m| {
            m.request_counter.inc();
            m.inflight_requests.inc();
            m.request_bytes.inc_by(payload.len() as u64);
            RequestMetricsGuard {
                inflight_requests: m.inflight_requests.clone(),
                request_duration: m.request_duration.clone(),
                start_time,
            }
        });

        // decode the control message and the request
        let msg = TwoPartCodec::default()
            .decode_message(payload)?
            .into_message_type();

        // we must have a header and a body
        // it will be held by this closure as a Some(permit)
        let (control_msg, request) = match msg {
            TwoPartMessageType::HeaderAndData(header, data) => {
                tracing::trace!(
                    "received two part message with ctrl: {} bytes, data: {} bytes",
                    header.len(),
                    data.len()
                );
                let control_msg: RequestControlMessage = match serde_json::from_slice(&header) {
                    Ok(cm) => cm,
                    Err(err) => {
                        let json_str = String::from_utf8_lossy(&header);
                        if let Some(m) = self.metrics() {
                            m.error_counter
                                .with_label_values(&[work_handler::error_types::DESERIALIZATION])
                                .inc();
                        }
                        return Err(PipelineError::DeserializationError(format!(
                            "Failed deserializing to RequestControlMessage. err={err}, json_str={json_str}"
                        )));
                    }
                };
                let request: T = serde_json::from_slice(&data)?;
                (control_msg, request)
            }
            _ => {
                if let Some(m) = self.metrics() {
                    m.error_counter
                        .with_label_values(&[work_handler::error_types::INVALID_MESSAGE])
                        .inc();
                }
                return Err(PipelineError::Generic(String::from(
                    "Unexpected message from work queue; unable extract a TwoPartMessage with a header and data",
                )));
            }
        };

        // extend request with context
        tracing::trace!("received control message: {:?}", control_msg);
        tracing::trace!("received request: {:?}", request);
        let request: context::Context<T> = Context::with_id(request, control_msg.id);

        // todo - eventually have a handler class which will returned an abstracted object, but for now,
        // we only support tcp here, so we can just unwrap the connection info
        tracing::trace!("creating tcp response stream");
        let mut publisher = tcp::client::TcpClient::create_response_stream(
            request.context(),
            control_msg.connection_info,
        )
        .await
        .map_err(|e| {
            if let Some(m) = self.metrics() {
                m.error_counter
                    .with_label_values(&[work_handler::error_types::RESPONSE_STREAM])
                    .inc();
            }
            PipelineError::Generic(format!("Failed to create response stream: {:?}", e,))
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
        let mut stream = match stream {
            Ok(stream) => {
                tracing::trace!("Successfully generated response stream; sending prologue");
                let _result = publisher.send_prologue(None).await;
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
                    tracing::error!("Failed to generate response stream: {}", error_string);
                }

                let _result = publisher.send_prologue(Some(error_string)).await;
                Err(e)?
            }
        };

        let context = stream.context();

        // TODO: Detect end-of-stream using Server-Sent Events (SSE)
        let mut send_complete_final = true;
        while let Some(resp) = stream.next().await {
            tracing::trace!("Sending response: {:?}", resp);
            if let Some(err) = resp.err()
                && format!("{:?}", err) == STREAM_ERR_MSG
            {
                tracing::warn!(STREAM_ERR_MSG);
                send_complete_final = false;
                break;
            }
            let resp_wrapper = NetworkStreamWrapper {
                data: Some(resp),
                complete_final: false,
            };
            let resp_bytes = serde_json::to_vec(&resp_wrapper)
                .expect("fatal error: invalid response object - this should never happen");
            if let Some(m) = self.metrics() {
                m.response_bytes.inc_by(resp_bytes.len() as u64);
            }
            if (publisher.send(resp_bytes.into()).await).is_err() {
                tracing::error!("Failed to publish response for stream {}", context.id());
                context.stop_generating();
                send_complete_final = false;
                if let Some(m) = self.metrics() {
                    m.error_counter
                        .with_label_values(&[work_handler::error_types::PUBLISH_RESPONSE])
                        .inc();
                }
                break;
            }
        }
        if send_complete_final {
            let resp_wrapper = NetworkStreamWrapper::<U> {
                data: None,
                complete_final: true,
            };
            let resp_bytes = serde_json::to_vec(&resp_wrapper)
                .expect("fatal error: invalid response object - this should never happen");
            if let Some(m) = self.metrics() {
                m.response_bytes.inc_by(resp_bytes.len() as u64);
            }
            if (publisher.send(resp_bytes.into()).await).is_err() {
                tracing::error!(
                    "Failed to publish complete final for stream {}",
                    context.id()
                );
                if let Some(m) = self.metrics() {
                    m.error_counter
                        .with_label_values(&[work_handler::error_types::PUBLISH_FINAL])
                        .inc();
                }
            }
            // Notify the health check manager that the stream has finished.
            // This resets the timer, delaying the next canary health check.
            if let Some(notifier) = self.endpoint_health_check_notifier.get() {
                notifier.notify_one();
            }
        }

        // Ensure the metrics guard is not dropped until the end of the function.
        drop(_inflight_guard);

        Ok(())
    }
}
