// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::SystemHealth;
use crate::config::HealthStatus;
use crate::logging::make_handle_payload_span;
use crate::protocols::LeaseId;
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

        let inflight = Arc::new(AtomicU64::new(0));
        let notify = Arc::new(Notify::new());
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
                    tracing::info!("PushEndpoint received cancellation signal, shutting down service");
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

                // increment the inflight counter
                inflight.fetch_add(1, Ordering::SeqCst);
                let inflight_clone = inflight.clone();
                let notify_clone = notify.clone();

                // Handle headers here for tracing
                let span = if let Some(headers) = req.message.headers.as_ref() {
                    make_handle_payload_span(
                        headers,
                        component_name.as_ref(),
                        endpoint_name.as_ref(),
                        namespace.as_ref(),
                        instance_id,
                    )
                } else {
                    tracing::info_span!("handle_payload")
                };

                tokio::spawn(async move {
                    tracing::trace!(instance_id, "handling new request");
                    let result = ingress
                        .handle_payload(req.message.payload)
                        .instrument(span)
                        .await;
                    match result {
                        Ok(_) => {
                            tracing::trace!(instance_id, "request handled successfully");
                        }
                        Err(e) => {
                            tracing::warn!("Failed to handle request: {}", e.to_string());
                        }
                    }

                    // decrease the inflight counter
                    inflight_clone.fetch_sub(1, Ordering::SeqCst);
                    notify_clone.notify_one();
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
            tracing::info!(
                "Waiting for {} inflight requests to complete",
                inflight.load(Ordering::SeqCst)
            );
            while inflight.load(Ordering::SeqCst) > 0 {
                notify.notified().await;
            }
            tracing::info!("All inflight requests completed");
        } else {
            tracing::info!("Skipping graceful shutdown, not waiting for inflight requests");
        }

        Ok(())
    }
}
