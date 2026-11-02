// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_backend_common::{EngineRecovery, KvEventSource};
use dynamo_llm::kv_router::publisher::{KvStreamCommand, KvStreamStatus, ZmqRecoveryConfig};
use tokio::time::{Duration, Instant, timeout};

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

/// Ends both Health.Watch streams, as a lost engine connection would, or restores serving.
async fn set_health(server: &mut FakeServer, is_serving: bool) {
    for service in [CONTROL_SERVICE, INFERENCE_SERVICE] {
        if !is_serving {
            server.health.clear_service_status(service).await;
        }
        let status = if is_serving {
            HealthServingStatus::Serving
        } else {
            HealthServingStatus::NotServing
        };
        server.health.set_service_status(service, status).await;
    }
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
async fn missing_history_requests_shutdown_and_accepts_replacement_without_acknowledgement() {
    let service = FakeVllm::default();
    *service.shutdown_replacement.lock().await = Some("replacement-instance".into());
    let server = FakeServer::start(service).await;
    let (engine, controls) = started(&server).await;
    controls[0]
        .status
        .send_replace(KvStreamStatus::MissingHistory {
            expected: 0,
            got: 10,
        });
    controls[1].status.send_replace(KvStreamStatus::Success);
    assert!(engine.wait_for_startup().await.is_err());
    let Some(EngineRecovery::Replacement(config)) =
        timeout(Duration::from_secs(5), engine.recover_startup())
            .await
            .unwrap()
            .unwrap()
    else {
        panic!("expected a replacement engine")
    };
    assert_eq!(config.model, "model-source");
    assert_eq!(
        server
            .service
            .control_calls
            .lock()
            .await
            .iter()
            .filter(|(name, _)| name == "shutdown")
            .count(),
        1
    );
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn shutdown_reaches_only_the_verified_instance() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let endpoint = GrpcEndpoint::parse(&server.endpoint, "--grpc-endpoint").unwrap();
    assert!(
        !crate::client::shutdown_instance(&endpoint, "replaced-instance")
            .await
            .unwrap()
    );
    assert!(server.service.control_calls.lock().await.is_empty());
    let status = crate::client::shutdown_instance(&endpoint, "test-instance")
        .await
        .unwrap_err();
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert_eq!(
        server
            .service
            .control_calls
            .lock()
            .await
            .iter()
            .filter(|(name, _)| name == "shutdown")
            .count(),
        1
    );
}

#[tokio::test]
async fn failed_recovery_does_not_shutdown_and_unmanaged_engine_failure_is_explicit() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let (engine, controls) = started(&server).await;
    controls[0].status.send_replace(KvStreamStatus::Failed {
        reason: "replay timed out".into(),
    });
    assert!(engine.wait_for_startup().await.is_err());
    assert!(engine.recover_startup().await.is_err());
    assert!(server.service.control_calls.lock().await.is_empty());
    let sources = engine.kv_event_sources().await.unwrap();
    for source in sources {
        let KvEventSource::Zmq {
            recovery: Some(recovery),
            ..
        } = source
        else {
            panic!()
        };
        recovery
            .status
            .send_replace(KvStreamStatus::MissingHistory {
                expected: 0,
                got: 9,
            });
    }
    let error = match engine.recover_startup().await {
        Ok(_) => panic!("unmanaged engine"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("fake frontend does not own engine")
    );
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn health_loss_keeps_kv_recovery_running_for_the_same_engine() {
    let mut server = FakeServer::start(FakeVllm::default()).await;
    let (engine, mut controls) = started(&server).await;
    set_health(&mut server, false).await;
    assert!(engine.wait_for_startup().await.is_err());
    set_health(&mut server, true).await;
    for control in &controls {
        control.status.send_replace(KvStreamStatus::Success);
    }
    assert!(matches!(
        timeout(Duration::from_secs(10), engine.recover_startup())
            .await
            .unwrap()
            .unwrap(),
        Some(EngineRecovery::SameInstance)
    ));
    set_health(&mut server, false).await;
    timeout(Duration::from_secs(2), engine.wait_for_unavailable())
        .await
        .unwrap()
        .unwrap();
    set_health(&mut server, true).await;
    assert!(matches!(
        timeout(Duration::from_secs(10), engine.recover_serving())
            .await
            .unwrap()
            .unwrap(),
        EngineRecovery::SameInstance
    ));
    // KV listeners were neither paused nor reconnected.
    for control in &mut controls {
        assert!(control.commands.try_recv().is_err());
    }
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn failed_replay_restarts_only_after_the_same_engine_is_verified() {
    let failed = || KvStreamStatus::Failed {
        reason: "KV recovery round exhausted".into(),
    };
    // With health verified, failed replay is final.
    let server = FakeServer::start(FakeVllm::default()).await;
    let (engine, mut controls) = started(&server).await;
    controls[0].status.send_replace(failed());
    assert!(engine.wait_for_startup().await.is_err());
    assert!(engine.recover_startup().await.is_err());
    assert!(controls[0].commands.try_recv().is_err());
    engine.cleanup().await.unwrap();
    // While health is lost it waits; the same engine then restarts replay, unless the listener refuses.
    for is_refused in [false, true] {
        let mut server = FakeServer::start(FakeVllm::default()).await;
        let (engine, mut controls) = started(&server).await;
        set_health(&mut server, false).await;
        assert!(engine.wait_for_startup().await.is_err());
        controls[0].status.send_replace(failed());
        let recovered = tokio::spawn({
            let engine = engine.clone();
            async move { engine.recover_startup().await }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!recovered.is_finished());
        set_health(&mut server, true).await;
        let Some(KvStreamCommand::Restart { deadline, accepted }) =
            timeout(Duration::from_secs(10), controls[0].commands.recv())
                .await
                .unwrap()
        else {
            panic!("expected a restart")
        };
        assert!(deadline > Instant::now());
        assert!(controls[1].commands.try_recv().is_err());
        if is_refused {
            drop(accepted);
            assert!(
                timeout(Duration::from_secs(5), recovered)
                    .await
                    .unwrap()
                    .unwrap()
                    .is_err()
            );
        } else {
            for control in &controls {
                control.status.send_replace(KvStreamStatus::Success);
            }
            accepted.send(()).unwrap();
            assert!(matches!(
                timeout(Duration::from_secs(5), recovered)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap(),
                Some(EngineRecovery::SameInstance)
            ));
        }
        assert!(server.service.control_calls.lock().await.is_empty());
        engine.cleanup().await.unwrap();
    }
}

#[tokio::test]
async fn changed_instance_fails_startup_and_recovers_as_replacement() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let (engine, controls) = started(&server).await;
    // Replay that ran out against the old engine must not block the replacement.
    controls[0].status.send_replace(KvStreamStatus::Failed {
        reason: "KV recovery round exhausted".into(),
    });
    controls[1].status.send_replace(KvStreamStatus::Success);
    server
        .service
        .server_info_override
        .lock()
        .await
        .get_or_insert_with(server_info)
        .instance_id = "replacement-instance".into();
    assert!(engine.wait_for_startup().await.is_err());
    let Some(EngineRecovery::Replacement(config)) =
        timeout(Duration::from_secs(5), engine.recover_startup())
            .await
            .unwrap()
            .unwrap()
    else {
        panic!("expected a replacement engine")
    };
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
    // No command paused or reconnected KV ingestion.
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
