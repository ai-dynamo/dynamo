// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    future::Future,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use super::*;
use crate::SystemHealth;
use crate::config::HealthStatus;
use crate::logging::make_handle_payload_span;
use crate::protocols::LeaseId;
use anyhow::Result;
use async_nats::service::endpoint::Endpoint;
use derive_builder::Builder;
use parking_lot::Mutex;
use std::collections::HashMap;
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
        instance_id: u64,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        let mut endpoint = endpoint;
        let mut stop_service_after_drain = false;

        let inflight = Arc::new(AtomicU64::new(0));
        let notify = Arc::new(Notify::new());
        let component_name_local: Arc<String> = Arc::from(component_name);
        let endpoint_name_local: Arc<String> = Arc::from(endpoint_name);
        let namespace_local: Arc<String> = Arc::from(namespace);

        system_health
            .lock()
            .set_endpoint_registered(endpoint_name_local.as_str());

        loop {
            let req = tokio::select! {
                biased;

                // await on service request
                req = endpoint.next() => {
                    req
                }

                // process shutdown
                _ = self.cancellation_token.cancelled() => {
                    tracing::info!(
                        "PushEndpoint received cancellation signal, stopping service after inflight requests drain"
                    );
                    stop_service_after_drain = true;
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
                    tracing::info_span!(target: "request_span", "handle_payload")
                };

                // Extract request_id from headers before passing payload
                let request_id = req
                    .message
                    .headers
                    .as_ref()
                    .and_then(|h| h.get("request-id").map(|v| v.to_string()))
                    .or_else(|| {
                        req.message
                            .headers
                            .as_ref()
                            .and_then(|h| h.get("x-dynamo-request-id").map(|v| v.to_string()))
                    });

                tokio::spawn(async move {
                    tracing::trace!(instance_id, "handling new request");
                    let result = ingress
                        .handle_payload(req.message.payload, request_id)
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
            .set_endpoint_health_status(endpoint_name_local.as_str(), HealthStatus::NotReady);

        if stop_service_after_drain {
            if self.graceful_shutdown {
                let timeout = crate::runtime::runtime_graceful_shutdown_timeout();
                finish_shutdown_after_inflight_drain(
                    &inflight,
                    &notify,
                    Some(timeout),
                    move || async move { endpoint.stop().await.map_err(anyhow::Error::from) },
                )
                .await;
            } else if let Err(e) = endpoint.stop().await {
                tracing::warn!("Failed to stop NATS service: {:?}", e);
            }
        } else if self.graceful_shutdown {
            let timeout = crate::runtime::runtime_graceful_shutdown_timeout();
            wait_for_inflight_requests(&inflight, &notify, Some(timeout)).await;
        } else {
            tracing::info!(
                endpoint_name = endpoint_name_local.as_str(),
                "Skipping graceful shutdown, not waiting for inflight requests"
            );
        }

        Ok(())
    }
}

pub(crate) async fn wait_for_inflight_requests(
    inflight: &AtomicU64,
    notify: &Notify,
    timeout: Option<Duration>,
) {
    let inflight_count = inflight.load(Ordering::SeqCst);
    if inflight_count > 0 {
        tracing::info!(
            inflight_count = inflight_count,
            timeout_secs = timeout.map(|d| d.as_secs()),
            "Waiting for inflight NATS requests to complete"
        );
    }

    let wait = async {
        loop {
            let notified = notify.notified();
            if inflight.load(Ordering::SeqCst) == 0 {
                break;
            }
            notified.await;
        }
    };

    match timeout {
        Some(timeout) => {
            if tokio::time::timeout(timeout, wait).await.is_err() {
                tracing::warn!(
                    inflight_count = inflight.load(Ordering::SeqCst),
                    timeout_secs = timeout.as_secs(),
                    "Timed out waiting for inflight NATS requests; continuing shutdown"
                );
            }
        }
        None => wait.await,
    }

    if inflight_count > 0 && inflight.load(Ordering::SeqCst) == 0 {
        tracing::info!("All inflight NATS requests completed");
    }
}

async fn finish_shutdown_after_inflight_drain<StopService, StopServiceFuture>(
    inflight: &AtomicU64,
    notify: &Notify,
    timeout: Option<Duration>,
    stop_service: StopService,
) where
    StopService: FnOnce() -> StopServiceFuture,
    StopServiceFuture: Future<Output = Result<()>>,
{
    wait_for_inflight_requests(inflight, notify, timeout).await;
    if let Err(e) = stop_service().await {
        tracing::warn!("Failed to stop NATS service: {:?}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    #[tokio::test]
    async fn graceful_shutdown_stops_service_after_inflight_requests_drain() {
        let inflight = Arc::new(AtomicU64::new(1));
        let notify = Arc::new(Notify::new());
        let stop_called = Arc::new(AtomicBool::new(false));

        let inflight_for_shutdown = inflight.clone();
        let inflight_for_stop = inflight.clone();
        let notify_for_shutdown = notify.clone();
        let stop_called_for_shutdown = stop_called.clone();

        let shutdown_task = tokio::spawn(async move {
            finish_shutdown_after_inflight_drain(
                inflight_for_shutdown.as_ref(),
                notify_for_shutdown.as_ref(),
                None,
                move || {
                    let inflight_for_stop = inflight_for_stop.clone();
                    let stop_called_for_shutdown = stop_called_for_shutdown.clone();
                    async move {
                        assert_eq!(
                            inflight_for_stop.load(Ordering::SeqCst),
                            0,
                            "service stop should happen only after inflight requests drain"
                        );
                        stop_called_for_shutdown.store(true, Ordering::SeqCst);
                        Ok(())
                    }
                },
            )
            .await;
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            !stop_called.load(Ordering::SeqCst),
            "service stop should not happen while requests are still inflight"
        );

        inflight.fetch_sub(1, Ordering::SeqCst);
        notify.notify_one();

        shutdown_task.await.unwrap();

        assert!(
            stop_called.load(Ordering::SeqCst),
            "service stop should run after inflight requests are drained"
        );
    }

    #[tokio::test]
    async fn graceful_shutdown_stop_runs_after_inflight_timeout() {
        let inflight = Arc::new(AtomicU64::new(1));
        let notify = Arc::new(Notify::new());
        let stop_called = Arc::new(AtomicBool::new(false));
        let stop_called_for_shutdown = stop_called.clone();

        finish_shutdown_after_inflight_drain(
            inflight.as_ref(),
            notify.as_ref(),
            Some(Duration::from_millis(20)),
            move || async move {
                stop_called_for_shutdown.store(true, Ordering::SeqCst);
                Ok(())
            },
        )
        .await;

        assert!(
            stop_called.load(Ordering::SeqCst),
            "service stop should run once the inflight drain timeout expires"
        );
    }
}
