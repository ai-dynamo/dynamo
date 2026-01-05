// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod metrics;
mod request;
mod response;

use metrics::RequestMetricsGuard;
pub use metrics::WorkHandlerMetrics;

use super::*;
use crate::metrics::prometheus_names::work_handler;
use crate::protocols::maybe_error::MaybeError;
use request::decode_payload;
use response::{create_response_publisher, send_prologue_and_get_stream, stream_responses};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
            RequestMetricsGuard::new(m, start_time)
        });

        // decode the control message and the request
        let decoded = decode_payload::<T>(payload).map_err(|e| {
            if let Some(m) = self.metrics() {
                let error_type = match &e {
                    PipelineError::DeserializationError(_) => {
                        work_handler::error_types::DESERIALIZATION
                    }
                    _ => work_handler::error_types::INVALID_MESSAGE,
                };
                m.error_counter.with_label_values(&[error_type]).inc();
            }
            e
        })?;
        let (control_msg, request) = (decoded.control_msg, decoded.request);

        // extend request with context
        tracing::trace!("received control message: {:?}", control_msg);
        tracing::trace!("received request: {:?}", request);
        let request: context::Context<T> = Context::with_id(request, control_msg.id);

        // Create response channel
        // TODO: eventually have a handler class which will return an abstracted object,
        // but for now, we only support tcp here
        let mut publisher = create_response_publisher(
            request.context(),
            control_msg.connection_info,
            self.metrics(),
        )
        .await?;

        // Call generate (business logic)
        tracing::trace!("calling generate");
        let generate_result = self
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

        // Send prologue and get stream (or propagate error after sending error prologue)
        let mut stream = send_prologue_and_get_stream(&mut publisher, generate_result).await?;

        // Stream responses to client
        stream_responses(
            &mut stream,
            &mut publisher,
            self.metrics(),
            self.endpoint_health_check_notifier.get(),
        )
        .await?;

        // Ensure the metrics guard is not dropped until the end of the function.
        drop(_inflight_guard);

        Ok(())
    }
}
