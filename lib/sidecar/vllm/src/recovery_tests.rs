// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_backend_common::EngineRecovery;
use tokio::time::{Duration, timeout};

async fn started(server: &FakeServer) -> Arc<VllmSidecarEngine> {
    let engine = Arc::new(engine(
        &server.endpoint,
        DisaggregationMode::Aggregated,
        1,
        model_info(),
    ));
    engine.start(1).await.unwrap();
    engine
}

#[tokio::test]
async fn changed_instance_fails_startup_and_recovers_as_replacement() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let engine = started(&server).await;
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
    engine.wait_for_startup().await.unwrap();
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn paused_engine_keeps_in_flight_requests_and_recovers_on_the_same_connection() {
    let service = FakeVllm::default();
    service
        .hold_before_first_token
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = started(&server).await;
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
    let recovered = tokio::spawn({
        let engine = engine.clone();
        async move { engine.recover_serving().await }
    });
    server.service.release_first_token.notify_one();
    finish(&mut stream).await;
    assert!(!recovered.is_finished());
    server
        .health
        .set_service_status(INFERENCE_SERVICE, HealthServingStatus::Serving)
        .await;
    assert!(matches!(
        timeout(Duration::from_secs(2), recovered)
            .await
            .unwrap()
            .unwrap()
            .unwrap(),
        EngineRecovery::SameInstance
    ));
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
