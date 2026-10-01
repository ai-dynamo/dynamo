// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

                // Stop admission even when the subscription backlog stays ready.
                _ = self.cancellation_token.cancelled() => {
                    tracing::info!("PushEndpoint received cancellation signal, shutting down service");
                    if let Err(e) = endpoint.stop().await {
                        tracing::warn!("Failed to stop NATS service: {:?}", e);
                    }
                    break;
                }

                // await on service request
                req = endpoint.next() => {
                    req
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
                    tracing::info_span!(
                        target: "request_span",
                        "handle_payload",
                        otel.kind = "server"
                    )
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

        // await for all inflight requests to complete if graceful shutdown
        if self.graceful_shutdown {
            super::drain_inflight(
                inflight,
                notify,
                endpoint_name_local.as_str(),
                crate::runtime::graceful_shutdown_timeout(),
            )
            .await;
        } else {
            tracing::info!(
                endpoint_name = endpoint_name_local.as_str(),
                "Skipping graceful shutdown, not waiting for inflight requests"
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::error::PipelineError;
    use async_nats::service::ServiceExt;
    use bytes::Bytes;
    use std::time::Duration;
    use tokio::sync::Semaphore;

    struct CountingHandler {
        calls: Arc<AtomicU64>,
        cancel_after_call: Option<CancellationToken>,
        finish: Arc<Semaphore>,
    }

    #[async_trait::async_trait]
    impl PushWorkHandler for CountingHandler {
        async fn handle_payload(&self, _: Bytes, _: Option<String>) -> Result<(), PipelineError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some(cancel) = &self.cancel_after_call {
                cancel.cancel();
            }
            self.finish.acquire().await.unwrap().forget();
            Ok(())
        }
        fn add_metrics(
            &self,
            _: &crate::component::Endpoint,
            _: Option<&[(&str, &str)]>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    async fn buffered_nats_peer(
        backlog: usize,
    ) -> (
        std::net::SocketAddr,
        tokio_util::task::AbortOnDropHandle<()>,
    ) {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let peer = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let (reader, mut writer) = socket.into_split();
            writer
                .write_all(
                    concat!(
                        "INFO {\"server_id\":\"cancel-test\",\"version\":\"2.10.0\",",
                        "\"proto\":1,\"max_payload\":1048576}\r\n"
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            let mut lines = BufReader::new(reader).lines();
            let mut subscription = None;
            let mut marker = None;
            let mut sent = false;
            while let Some(line) = lines.next_line().await.unwrap() {
                let fields: Vec<_> = line.split_whitespace().collect();
                if fields.first() == Some(&"SUB") && fields.get(1) == Some(&"cancel_probe.work") {
                    subscription = fields.last().map(|sid| sid.to_string());
                }
                if fields.first() == Some(&"SUB") && fields.get(1) == Some(&"cancel_probe.ready") {
                    marker = fields.last().map(|sid| sid.to_string());
                }
                if let (Some(sid), Some(marker_sid)) = (&subscription, &marker)
                    && !sent
                {
                    let frame =
                        format!("MSG cancel_probe.work {sid} cancel_probe.reply 4\r\nwork\r\n");
                    for _ in 0..backlog {
                        writer.write_all(frame.as_bytes()).await.unwrap();
                    }
                    writer
                        .write_all(
                            format!("MSG cancel_probe.ready {marker_sid} 0\r\n\r\n").as_bytes(),
                        )
                        .await
                        .unwrap();
                    sent = true;
                }
                if line == "PING" {
                    writer.write_all(b"PONG\r\n").await.unwrap();
                }
            }
        }));
        (address, peer)
    }

    async fn run_buffered_endpoint(pre_cancelled: bool, backlog: usize) -> u64 {
        let (address, _peer) = buffered_nats_peer(backlog).await;
        let client = async_nats::connect(address.to_string()).await.unwrap();
        let service = client
            .service_builder()
            .start("cancel_probe", "1.0.0")
            .await
            .unwrap();
        let endpoint = service.endpoint("cancel_probe.work").await.unwrap();
        let mut marker = client.subscribe("cancel_probe.ready").await.unwrap();
        // The peer routes every request before this marker on the same connection.
        // Waiting for its delivery establishes that Endpoint has buffered work.
        tokio::time::timeout(Duration::from_secs(5), marker.next())
            .await
            .unwrap()
            .unwrap();
        let cancel = CancellationToken::new();
        if pre_cancelled {
            cancel.cancel();
        }
        let calls = Arc::new(AtomicU64::new(0));
        let finish = Arc::new(Semaphore::new(if pre_cancelled { backlog } else { 0 }));
        let health = Arc::new(Mutex::new(SystemHealth::new(
            crate::HealthStatus::Ready,
            vec![],
            false,
            "/health".into(),
            "/live".into(),
        )));
        let push = PushEndpoint::builder()
            .service_handler(Arc::new(CountingHandler {
                calls: calls.clone(),
                cancel_after_call: (!pre_cancelled).then(|| cancel.clone()),
                finish: finish.clone(),
            }) as Arc<dyn PushWorkHandler>)
            .cancellation_token(cancel.clone())
            .graceful_shutdown(true)
            .build()
            .unwrap();
        let started = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(push.start(
            endpoint,
            "test".into(),
            "component".into(),
            "generate".into(),
            1,
            health.clone(),
        )));
        let mut returned_before_completion = false;
        if !pre_cancelled {
            tokio::time::timeout(Duration::from_secs(5), async {
                cancel.cancelled().await;
                while health.lock().get_endpoint_health_status("generate")
                    != Some(crate::HealthStatus::NotReady)
                {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            returned_before_completion = started.is_finished();
            finish.add_permits(backlog);
        }
        tokio::time::timeout(Duration::from_secs(5), started)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(
            !returned_before_completion,
            "accepted handlers must drain before shutdown returns"
        );
        calls.load(Ordering::SeqCst)
    }

    #[tokio::test]
    async fn cancelled_endpoint_does_not_dispatch_buffered_nats_requests() {
        assert_eq!(run_buffered_endpoint(true, 8).await, 0);
    }

    #[tokio::test]
    async fn cancellation_stops_backlog_and_drains_accepted_requests() {
        let calls = run_buffered_endpoint(false, 4096).await;
        assert!(calls > 0);
        assert!(
            calls < 4096,
            "cancellation did not stop buffered request admission: {calls}"
        );
    }
    #[tokio::test]
    async fn cancellation_drains_accepted_handler() {
        assert_eq!(run_buffered_endpoint(false, 1).await, 1);
    }
}
