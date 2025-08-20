// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;
use crate::protocols::maybe_error::MaybeError;
use prometheus::{Histogram, IntCounter, IntCounterVec, IntGauge};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info_span;
use tracing::Instrument;

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
        let request_counter = endpoint.create_intcounter(
            "requests_total",
            "Total number of requests processed by work handler",
            metrics_labels,
        )?;

        let request_duration = endpoint.create_histogram(
            "request_duration_seconds",
            "Time spent processing requests by work handler",
            metrics_labels,
            None,
        )?;

        let inflight_requests = endpoint.create_intgauge(
            "inflight_requests",
            "Number of requests currently being processed by work handler",
            metrics_labels,
        )?;

        let request_bytes = endpoint.create_intcounter(
            "request_bytes_total",
            "Total number of bytes received in requests by work handler",
            metrics_labels,
        )?;

        let response_bytes = endpoint.create_intcounter(
            "response_bytes_total",
            "Total number of bytes sent in responses by work handler",
            metrics_labels,
        )?;

        let error_counter = endpoint.create_intcountervec(
            "errors_total",
            "Total number of errors in work handler processing",
            &["error_type"],
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

    async fn handle_payload(&self, payload: Bytes) -> Result<(), PipelineError> {
        let start_time = std::time::Instant::now();

        if let Some(m) = self.metrics() {
            m.request_counter.inc();
            m.inflight_requests.inc();
            m.request_bytes.inc_by(payload.len() as u64);
        }

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
                                .with_label_values(&["deserialization"])
                                .inc();
                        }
                        return Err(PipelineError::DeserializationError(
                            format!("Failed deserializing to RequestControlMessage. err={err}, json_str={json_str}"),
                        ));
                    }
                };
                let request: T = serde_json::from_slice(&data)?;
                (control_msg, request)
            }
            _ => {
                if let Some(m) = self.metrics() {
                    m.error_counter
                        .with_label_values(&["invalid_message"])
                        .inc();
                }
                return Err(PipelineError::Generic(String::from("Unexpected message from work queue; unable extract a TwoPartMessage with a header and data")));
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
                    .with_label_values(&["response_stream"])
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
                    m.error_counter.with_label_values(&["generate"]).inc();
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
            if let Some(err) = resp.err() {
                const STREAM_ERR_MSG: &str = "Stream ended before generation completed";
                if format!("{:?}", err) == STREAM_ERR_MSG {
                    tracing::warn!(STREAM_ERR_MSG);
                    send_complete_final = false;
                    break;
                }
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
                        .with_label_values(&["publish_response"])
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
                    m.error_counter.with_label_values(&["publish_final"]).inc();
                }
            }
        }

        if let Some(m) = self.metrics() {
            let duration = start_time.elapsed();
            m.request_duration.observe(duration.as_secs_f64());
            m.inflight_requests.dec();
        }

        Ok(())
    }
}
