// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_backend_common::{EngineRecovery, KvEventSource};
use dynamo_llm::kv_router::publisher::{KvStreamStatus, ZmqRecoveryConfig};
use tokio::time::{Duration, timeout};

async fn started(server: &FakeServer) -> (Arc<VllmSidecarEngine>, Vec<ZmqRecoveryConfig>) {
    let engine = Arc::new(engine(
        &server.endpoint,
        DisaggregationMode::Aggregated,
        1,
        model_info(),
    ));
    engine.start(1).await.unwrap();
    let controls = engine
        .kv_event_sources()
        .await
        .unwrap()
        .into_iter()
        .map(|source| {
            let KvEventSource::Zmq {
                recovery: Some(recovery),
                ..
            } = source
            else {
                panic!("recovery source")
            };
            recovery
        })
        .collect();
    (engine, controls)
}

#[tokio::test]
async fn rank_that_loses_continuity_while_another_recovers_cannot_register() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let (engine, controls) = started(&server).await;
    controls[0].status.send_replace(KvStreamStatus::Success);
    let gate = engine.wait_for_startup();
    tokio::pin!(gate);
    assert!(futures::poll!(&mut gate).is_pending());
    controls[0].status.send_replace(KvStreamStatus::Recovering);
    controls[1].status.send_replace(KvStreamStatus::Success);
    assert!(
        timeout(Duration::from_secs(2), &mut gate)
            .await
            .unwrap()
            .is_err()
    );
    controls[0].status.send_replace(KvStreamStatus::Success);
    engine.wait_for_startup().await.unwrap();
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn missing_history_waits_for_a_replacement_engine() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let (engine, controls) = started(&server).await;
    controls[0]
        .status
        .send_replace(KvStreamStatus::MissingHistory {
            expected: 0,
            got: 10,
        });
    controls[1].status.send_replace(KvStreamStatus::Success);
    assert!(engine.wait_for_startup().await.is_err());
    let recovered = tokio::spawn({
        let engine = engine.clone();
        async move { engine.recover_startup().await }
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!recovered.is_finished());
    server
        .service
        .server_info_override
        .lock()
        .await
        .get_or_insert_with(server_info)
        .instance_id = "replacement-instance".into();
    let config = timeout(Duration::from_secs(5), recovered)
        .await
        .unwrap()
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(config.model, "model-source");
    assert!(server.service.control_calls.lock().await.is_empty());
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn changed_instance_fails_startup_and_recovers_as_replacement() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let (engine, controls) = started(&server).await;
    for control in &controls {
        control.status.send_replace(KvStreamStatus::Success);
    }
    server
        .service
        .server_info_override
        .lock()
        .await
        .get_or_insert_with(server_info)
        .instance_id = "replacement-instance".into();
    assert!(engine.wait_for_startup().await.is_err());
    let config = timeout(Duration::from_secs(5), engine.recover_startup())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(config.model, "model-source");
    assert!(server.service.control_calls.lock().await.is_empty());
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn paused_engine_keeps_in_flight_requests_and_kv_ingestion() {
    let service = FakeVllm::default();
    service
        .hold_before_first_token
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let (engine, mut controls) = started(&server).await;
    for control in &controls {
        control.status.send_replace(KvStreamStatus::Success);
    }
    engine.wait_for_startup().await.unwrap();
    let mut stream = engine
        .generate(
            minimal_request(),
            GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
        )
        .await
        .unwrap();
    first_token_pending(&server).await;
    server
        .health
        .set_service_status(INFERENCE_SERVICE, HealthServingStatus::NotServing)
        .await;
    timeout(Duration::from_secs(2), engine.wait_for_unavailable())
        .await
        .unwrap()
        .unwrap();
    assert!(
        engine
            .generate(
                minimal_request(),
                GenerateContext::new(dynamo_backend_common::testing::mock_context(), None)
            )
            .await
            .is_err()
    );
    controls[0].status.send_replace(KvStreamStatus::Recovering);
    let recovered = tokio::spawn({
        let engine = engine.clone();
        async move { engine.recover_serving().await }
    });
    server.service.release_first_token.notify_one();
    finish(&mut stream).await;
    server
        .health
        .set_service_status(INFERENCE_SERVICE, HealthServingStatus::Serving)
        .await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!recovered.is_finished());
    controls[0].status.send_replace(KvStreamStatus::Success);
    assert!(matches!(
        timeout(Duration::from_secs(2), recovered)
            .await
            .unwrap()
            .unwrap()
            .unwrap(),
        EngineRecovery::SameInstance
    ));
    // Neither Suspend nor a reconnect interrupted KV ingestion.
    for control in &mut controls {
        assert!(control.commands.try_recv().is_err());
    }
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn kv_gap_rejects_new_requests_without_cancelling_in_flight_streams() {
    let service = FakeVllm::default();
    service
        .hold_before_first_token
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let (engine, mut controls) = started(&server).await;
    for control in &controls {
        control.status.send_replace(KvStreamStatus::Success);
    }
    engine.wait_for_startup().await.unwrap();
    let mut stream = engine
        .generate(
            minimal_request(),
            GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
        )
        .await
        .unwrap();
    first_token_pending(&server).await;
    controls[0].status.send_replace(KvStreamStatus::Recovering);
    timeout(Duration::from_secs(2), engine.wait_for_unavailable())
        .await
        .unwrap()
        .unwrap();
    assert!(
        engine
            .generate(
                minimal_request(),
                GenerateContext::new(dynamo_backend_common::testing::mock_context(), None)
            )
            .await
            .is_err()
    );
    server.service.release_first_token.notify_one();
    finish(&mut stream).await;
    controls[0].status.send_replace(KvStreamStatus::Success);
    assert!(matches!(
        engine.recover_serving().await.unwrap(),
        EngineRecovery::SameInstance
    ));
    for control in &mut controls {
        assert!(control.commands.try_recv().is_err());
    }
    engine.cleanup().await.unwrap();
}

async fn first_token_pending(server: &FakeServer) {
    timeout(Duration::from_secs(2), async {
        while !server
            .service
            .first_token_pending
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

async fn finish(
    stream: &mut futures::stream::BoxStream<
        'static,
        Result<dynamo_backend_common::LLMEngineOutput, dynamo_backend_common::DynamoError>,
    >,
) {
    let mut finish_reason = None;
    while let Some(output) = timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
    {
        finish_reason = output.unwrap().finish_reason.or(finish_reason);
    }
    // Recovery must not cancel a request the engine is still serving.
    assert!(matches!(finish_reason, Some(reason) if reason != FinishReason::Cancelled));
}
