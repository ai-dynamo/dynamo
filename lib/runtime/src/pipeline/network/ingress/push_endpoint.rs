// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::config::HealthStatus;
use crate::logging::TraceParent;
use crate::protocols::LeaseId;
use crate::{RequestTracker, SystemHealth};
use anyhow::Result;
use async_nats::service::endpoint::Endpoint;
use derive_builder::Builder;
use std::collections::HashMap;
use std::sync::Mutex;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

#[derive(Builder)]
pub struct PushEndpoint {
    pub service_handler: Arc<dyn PushWorkHandler>,
    pub cancellation_token: CancellationToken,
    #[builder(default = "true")]
    pub graceful_shutdown: bool,
    #[builder(setter(strip_option))]
    pub request_tracker: Option<RequestTracker>,
    /// Main runtime cancellation token to wait for complete shutdown
    #[builder(setter(strip_option))]
    pub runtime_token: Option<CancellationToken>,
}

/// version of crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

impl PushEndpoint {
    pub fn builder() -> PushEndpointBuilder {
        PushEndpointBuilder::default()
    }

    pub async fn start(
        self,
        endpoint: Endpoint,
        namespace: String,
        component_name: String,
        endpoint_name: String,
        instance_id: i64,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        let mut endpoint = endpoint;

        // Use external tracker if provided, otherwise create local tracking
        let (inflight, notify, use_external_tracker) = if self.request_tracker.is_some() {
            // External tracker handles everything
            (Arc::new(AtomicU64::new(0)), Arc::new(Notify::new()), true)
        } else {
            // Local tracking for backward compatibility
            (Arc::new(AtomicU64::new(0)), Arc::new(Notify::new()), false)
        };

        let component_name_local: Arc<String> = Arc::from(component_name);
        let endpoint_name_local: Arc<String> = Arc::from(endpoint_name);
        let namespace_local: Arc<String> = Arc::from(namespace);

        system_health
            .lock()
            .unwrap()
            .set_endpoint_health_status(endpoint_name_local.as_str(), HealthStatus::Ready);

        loop {
            let req = tokio::select! {
                biased;

                // await on service request
                req = endpoint.next() => {
                    req
                }

                // process shutdown
                _ = self.cancellation_token.cancelled() => {
                    tracing::info!("Shutting down service");
                    if let Err(e) = endpoint.stop().await {
                        tracing::warn!("Failed to stop NATS service: {:?}", e);
                    }
                    break;
                }
            };

            if let Some(req) = req {
                let response = "".to_string();
                if let Err(e) = req.respond(Ok(response.into())).await {
                    tracing::warn!(
                        "Failed to respond to request; this may indicate the request has shutdown: {:?}",
                        e
                    );
                }

                let ingress = self.service_handler.clone();
                let endpoint_name: Arc<String> = Arc::clone(&endpoint_name_local);
                let component_name: Arc<String> = Arc::clone(&component_name_local);
                let namespace: Arc<String> = Arc::clone(&namespace_local);

                // Track the request - either with external tracker or local counter
                let request_guard = if use_external_tracker {
                    self.request_tracker.as_ref().map(|t| t.track_request())
                } else {
                    inflight.fetch_add(1, Ordering::SeqCst);
                    None
                };
                let inflight_clone = inflight.clone();
                let notify_clone = notify.clone();

                // Handle headers here for tracing

                let mut traceparent = TraceParent::default();

                if let Some(headers) = req.message.headers.as_ref() {
                    traceparent = TraceParent::from_headers(headers);
                }

                tokio::spawn(async move {
                    // Keep the guard alive for the duration of the request
                    let _guard = request_guard;

                    tracing::trace!(instance_id, "handling new request");
                    let result = ingress
                        .handle_payload(req.message.payload)
                        .instrument(
                            // Create span with trace ids as set
                            // in headers.
                            tracing::info_span!(
                                "handle_payload",
                                component = component_name.as_ref(),
                                endpoint = endpoint_name.as_ref(),
                                namespace = namespace.as_ref(),
                                instance_id = instance_id,
                                trace_id = traceparent.trace_id,
                                parent_id = traceparent.parent_id,
                                x_request_id = traceparent.x_request_id,
                                x_dynamo_request_id = traceparent.x_dynamo_request_id,
                                tracestate = traceparent.tracestate
                            ),
                        )
                        .await;
                    match result {
                        Ok(_) => {
                            tracing::trace!(instance_id, "request handled successfully");
                        }
                        Err(e) => {
                            tracing::warn!("Failed to handle request: {}", e.to_string());
                        }
                    }

                    // For local tracking, decrease the counter
                    if _guard.is_none() {
                        inflight_clone.fetch_sub(1, Ordering::SeqCst);
                        notify_clone.notify_one();
                    }
                });
            } else {
                break;
            }
        }

        system_health
            .lock()
            .unwrap()
            .set_endpoint_health_status(endpoint_name_local.as_str(), HealthStatus::NotReady);

        // await for all inflight requests to complete if graceful shutdown
        if self.graceful_shutdown {
            if use_external_tracker {
                // When using external tracker, we need to wait for the runtime's
                // graceful shutdown to complete before returning, otherwise Python
                // will clean up resources (like the vLLM engine) prematurely
                tracing::info!("Endpoint stopped accepting requests, waiting for runtime graceful shutdown");
                
                // First wait for all requests to complete
                if let Some(tracker) = &self.request_tracker {
                    tracker.wait_for_completion().await;
                    tracing::info!("All inflight requests completed");
                }
                
                // Then wait for the runtime to fully shutdown
                // This ensures Python engines do not clean up resources prematurely 
                if let Some(runtime_token) = &self.runtime_token {
                    tracing::info!("Waiting for runtime infrastructure shutdown");
                    runtime_token.cancelled().await;
                    tracing::info!("Runtime shutdown complete");
                }
            } else {
                // Local tracking - wait here
                tracing::info!(
                    "Waiting for {} inflight requests to complete",
                    inflight.load(Ordering::SeqCst)
                );
                while inflight.load(Ordering::SeqCst) > 0 {
                    notify.notified().await;
                }
                tracing::info!("All inflight requests completed");
            }
        } else {
            tracing::info!("Skipping graceful shutdown, not waiting for inflight requests");
        }

        Ok(())
    }
}
