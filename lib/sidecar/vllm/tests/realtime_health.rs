// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercise canary health through the real bidirectional Worker and TCP path.

#[path = "common/realtime.rs"]
mod common;

use std::{
    sync::{Arc, atomic::Ordering},
    time::Duration,
};

use base64::{Engine, prelude::BASE64_STANDARD};
use dynamo_backend_common::{
    BidirectionalEngine, DynamoError, EngineConfig, GenerateContext, HEALTH_CHECK_KEY, ModelInput,
    Worker, WorkerConfig,
};
use dynamo_realtime::{Client, Limits, Protocol};
use dynamo_runtime::{
    config::HealthStatus,
    engine::AsyncEngine,
    health_check::{HealthCheckConfig, start_health_check_manager},
    pipeline::{Context, RequestStream},
};
use futures::{StreamExt, stream::BoxStream};
use serde_json::{Value, json};
use tokio_util::{sync::CancellationToken, task::AbortOnDropHandle};

// The regression is in Worker's activity wiring; keep the test engine limited
// to the real client's session/probe operations rather than expose private APIs.
struct AsrEngine(Client);

#[async_trait::async_trait]
impl BidirectionalEngine for AsrEngine {
    async fn start(&self, _: u64) -> Result<EngineConfig, DynamoError> {
        Ok(EngineConfig {
            model: common::MODEL.into(),
            ..EngineConfig::default()
        })
    }

    async fn generate(
        &self,
        mut requests: BoxStream<'static, Value>,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<Value, DynamoError>>, DynamoError> {
        let first = requests.next().await.expect("session or canary input");
        if first[HEALTH_CHECK_KEY] == true {
            self.0
                .realtime_health(Protocol::VllmTranscription)
                .await
                .map_err(|error| dynamo_sidecar_common::engine_shutdown(error.to_string()))?;
            return Ok(futures::stream::once(async { Ok(json!({"healthy": true})) }).boxed());
        }
        let input = futures::stream::once(async { first })
            .chain(requests)
            .boxed();
        Ok(self
            .0
            .session_with_cancellation(Protocol::VllmTranscription, input, async move {
                tokio::select! { _ = ctx.stopped() => {}, _ = ctx.killed() => {} }
            })
            .map(Ok)
            .boxed())
    }

    async fn health_check_payload(&self) -> Result<Option<Value>, DynamoError> {
        Ok(Some(json!({HEALTH_CHECK_KEY: true})))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.0.shutdown().await;
        Ok(())
    }
}

#[test]
fn realtime_health_ignores_local_session_traffic() {
    common::configure_process();
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(outage_and_recovery());
}

async fn outage_and_recovery() {
    let upstream = common::start_upstream().await;
    let discovery = tempfile::tempdir().unwrap();
    let (runtime, drt) = common::runtime(discovery.path()).await;
    start_health_check_manager(
        drt.clone(),
        Some(HealthCheckConfig {
            canary_wait_time: Duration::from_millis(100),
            request_timeout: Duration::from_secs(1),
        }),
    )
    .await
    .unwrap();
    let engine = Arc::new(AsrEngine(
        Client::new(
            upstream.url.parse().unwrap(),
            common::MODEL.into(),
            Limits::for_protocol(Protocol::VllmTranscription),
            None,
        )
        .unwrap(),
    ));
    let config = WorkerConfig {
        namespace: "media-test".into(),
        component: "asr".into(),
        endpoint_types: "realtime".into(),
        model_input: ModelInput::Text,
        enable_kv_routing: false,
        enable_local_indexer: false,
        ..WorkerConfig::default()
    };
    let shutdown = CancellationToken::new();
    let worker = AbortOnDropHandle::new(tokio::spawn(
        Worker::new_bidirectional(engine, config).run_with_drt(drt.clone(), shutdown.clone()),
    ));
    let endpoint = common::endpoint(&drt);
    let router = common::router(&endpoint).await;
    let health = drt.system_health();
    let wait_health = |expected: HealthStatus, minimum_probes: usize| {
        let health = health.clone();
        let probes = upstream.probes.clone();
        let endpoint = endpoint.clone();
        async move {
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if probes.load(Ordering::SeqCst) >= minimum_probes
                        && health.lock().get_endpoint_health_status(endpoint.name())
                            == Some(expected.clone())
                    {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .expect("canary must update endpoint health");
        }
    };
    wait_health(HealthStatus::Ready, 1).await;
    upstream.healthy.store(false, Ordering::SeqCst);
    wait_health(HealthStatus::NotReady, 2).await;
    let before_traffic = upstream.probes.load(Ordering::SeqCst);
    let mut ticks = tokio::time::interval(Duration::from_millis(20));
    for _ in 0..20 {
        ticks.tick().await;
        let input = futures::stream::iter([
            common::session_update(),
            json!({"type": "input_audio_buffer.append", "audio": BASE64_STANDARD.encode([0u8; 960])}),
            json!({"type": "input_audio_buffer.commit"}),
        ]);
        let events: Vec<_> = tokio::time::timeout(Duration::from_secs(5), async {
            router
                .generate(Context::new(RequestStream::new(input.boxed())))
                .await
                .unwrap()
                .collect()
                .await
        })
        .await
        .unwrap();
        assert!(events.iter().all(|event| !event.is_error()));
        assert!(
            events
                .iter()
                .any(|event| event.data.as_ref().unwrap()["type"] == "session.updated")
        );
        assert!(
            events
                .iter()
                .any(|event| event.data.as_ref().unwrap()["type"]
                    == "conversation.item.input_audio_transcription.failed")
        );
    }
    assert_eq!(
        health.lock().get_endpoint_health_status(endpoint.name()),
        Some(HealthStatus::NotReady),
        "local acknowledgements and failures must not mark the upstream healthy"
    );
    assert!(
        upstream.probes.load(Ordering::SeqCst) >= before_traffic + 2,
        "failed session traffic must not suppress canaries"
    );
    upstream.healthy.store(true, Ordering::SeqCst);
    wait_health(
        HealthStatus::Ready,
        upstream.probes.load(Ordering::SeqCst) + 1,
    )
    .await;
    shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(5), worker)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    runtime.shutdown();
}
