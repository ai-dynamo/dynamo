// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Requires a NATS server at NATS_SERVER (default: nats://127.0.0.1:4222).
#![cfg(feature = "integration")]

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

use async_nats::service::ServiceExt;
use dynamo_runtime::pipeline::{
    PipelineError,
    network::{PushWorkHandler, ingress::push_endpoint::PushEndpoint},
};
use dynamo_runtime::{SystemHealth, config::HealthStatus};
use futures::StreamExt;
use parking_lot::Mutex;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

#[derive(Default)]
struct Handler {
    calls: AtomicU64,
    started: Notify,
    release: Notify,
}

#[async_trait::async_trait]
impl PushWorkHandler for Handler {
    async fn handle_payload(
        &self,
        _: bytes::Bytes,
        _: Option<String>,
    ) -> Result<(), PipelineError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();
        self.release.notified().await;
        Ok(())
    }

    fn add_metrics(
        &self,
        _: &dynamo_runtime::component::Endpoint,
        _: Option<&[(&str, &str)]>,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn cancellation_rejects_queued_requests_and_drains_accepted_work() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let server = std::env::var(dynamo_runtime::config::environment_names::nats::NATS_SERVER)
            .unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let client = async_nats::connect(server).await.unwrap();
        let service = client.service_builder()
            .start("shutdown_test", "0.0.1").await.unwrap();
        for accepted in [false, true] {
            let subject = format!("shutdown.{}", uuid::Uuid::new_v4().simple());
            let endpoint = service.endpoint(&subject).await.unwrap();
            let handler = Arc::new(Handler::default());
            let cancellation = CancellationToken::new();
            let health = Arc::new(Mutex::new(SystemHealth::new(
                HealthStatus::Ready, vec![subject.clone()], false,
                "/health".into(), "/live".into(),
            )));
            let push = PushEndpoint::builder()
                .service_handler(handler.clone())
                .cancellation_token(cancellation.clone())
                .build().unwrap();
            let run = push.start(
                endpoint, "test".into(), "worker".into(), subject.clone(), 1, health.clone(),
            );
            // Receiving a later message on the same connection ensures the
            // endpoint request has been delivered before we poll its receive loop.
            let barrier_subject = client.new_inbox();
            let mut barrier = client.subscribe(barrier_subject.clone()).await.unwrap();
            client.publish_with_reply(subject.clone(), client.new_inbox(), "request".into())
                .await.unwrap();
            client.publish(barrier_subject, "barrier".into()).await.unwrap();
            barrier.next().await.unwrap();
            if !accepted {
                cancellation.cancel();
            }
            tokio::pin!(run);
            if accepted {
                tokio::select! {
                    _ = handler.started.notified() => {},
                    result = &mut run => panic!("endpoint stopped before accepting work: {result:?}"),
                }
                cancellation.cancel();
                // Poll shutdown until it marks the endpoint not-ready and waits
                // for the already accepted handler. Unsubscribing must not stop it.
                tokio::select! {
                    _ = async {
                        while health.lock().get_endpoint_health_status(&subject) != Some(HealthStatus::NotReady) {
                            tokio::task::yield_now().await;
                        }
                    } => {},
                    result = &mut run => panic!("endpoint stopped before draining: {result:?}"),
                }
            }
            handler.release.notify_one();
            run.await.unwrap();
            assert_eq!(handler.calls.load(Ordering::SeqCst), u64::from(accepted));
        }
    }).await.expect("NATS shutdown did not complete");
}
