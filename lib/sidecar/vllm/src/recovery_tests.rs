// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_backend_common::{EngineRecovery, KvEventSource};
use dynamo_llm::kv_router::publisher::{KvStreamCommand, KvStreamStatus, ZmqRecoveryControl};
use tokio::time::{Duration, timeout};

async fn started(server: &FakeServer) -> (Arc<VllmSidecarEngine>, Vec<ZmqRecoveryControl>) {
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
                bootstrap: Some(mut bootstrap),
                ..
            } = source
            else {
                panic!("recovery source")
            };
            bootstrap.recovery.take().unwrap()
        })
        .collect();
    (engine, controls)
}

#[tokio::test]
async fn empty_replay_probes_each_rank_but_inference_success_does_not_open_readiness() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let (engine, mut controls) = started(&server).await;
    let waiter = tokio::spawn({
        let engine = engine.clone();
        async move { engine.wait_for_startup().await }
    });
    for control in &controls {
        control.status.send_replace(KvStreamStatus::ProbeRequired);
    }
    for control in &mut controls {
        let Some(KvStreamCommand::RetryReplay(accepted)) =
            timeout(Duration::from_secs(2), control.commands.recv())
                .await
                .unwrap()
        else {
            panic!("replay retry")
        };
        control.status.send_replace(KvStreamStatus::Recovering);
        accepted.send(()).unwrap();
    }
    assert!(!waiter.is_finished());
    let requests = server.service.requests.lock().await;
    assert_eq!(requests.len(), 2);
    for request in requests.iter() {
        assert!(
            matches!(&request.prompt, Some(pb::generate_request::Prompt::TokenIds(tokens)) if tokens.ids.len() > 16)
        );
        assert_eq!(request.stopping.as_ref().unwrap().max_new_tokens, 1);
        assert!(request.kv.as_ref().unwrap().kv_transfer_params.is_none());
        assert!(!request.kv.as_ref().unwrap().cache_salt.is_empty());
    }
    drop(requests);
    let mut ranks = server
        .service
        .data_parallel_rank_metadata
        .lock()
        .await
        .clone();
    ranks.sort();
    assert_eq!(ranks, vec![Some("0".into()), Some("1".into())]);
    controls[0].status.send_replace(KvStreamStatus::Ready);
    assert!(!waiter.is_finished());
    controls[1].status.send_replace(KvStreamStatus::Ready);
    timeout(Duration::from_secs(2), waiter)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(server.service.control_calls.lock().await.is_empty());
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn rank_that_loses_continuity_while_another_bootstraps_cannot_register() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let (engine, controls) = started(&server).await;
    controls[0].status.send_replace(KvStreamStatus::Ready);
    let gate = engine.wait_for_startup();
    tokio::pin!(gate);
    assert!(futures::poll!(&mut gate).is_pending());
    controls[0].status.send_replace(KvStreamStatus::Recovering);
    controls[1].status.send_replace(KvStreamStatus::Ready);
    assert!(
        timeout(Duration::from_secs(2), &mut gate)
            .await
            .unwrap()
            .is_err()
    );
    controls[0].status.send_replace(KvStreamStatus::Ready);
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
    controls[1].status.send_replace(KvStreamStatus::Ready);
    assert!(engine.wait_for_startup().await.is_err());
    let config = timeout(Duration::from_secs(5), engine.recover_startup())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
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
async fn uncertainty_does_not_shutdown_and_unmanaged_engine_failure_is_explicit() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let (engine, controls) = started(&server).await;
    controls[0].status.send_replace(KvStreamStatus::Uncertain {
        reason: "empty replay timed out".into(),
        is_terminal: false,
    });
    assert!(engine.wait_for_startup().await.is_err());
    assert!(engine.recover_startup().await.unwrap().is_some());
    assert!(server.service.control_calls.lock().await.is_empty());
    let sources = engine.kv_event_sources().await.unwrap();
    for source in sources {
        let KvEventSource::Zmq {
            bootstrap: Some(mut bootstrap),
            ..
        } = source
        else {
            panic!()
        };
        bootstrap
            .recovery
            .take()
            .unwrap()
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
async fn health_watch_withdraws_and_same_instance_recovery_waits_for_replay() {
    let server = FakeServer::start(FakeVllm::default()).await;
    let (engine, mut controls) = started(&server).await;
    for control in &controls {
        control.status.send_replace(KvStreamStatus::Ready);
    }
    engine.wait_for_startup().await.unwrap();
    let unavailable = tokio::spawn({
        let engine = engine.clone();
        async move { engine.wait_for_unavailable().await }
    });
    assert!(
        timeout(Duration::from_millis(30), async {
            while !unavailable.is_finished() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .is_err()
    );
    server
        .health
        .set_service_status(INFERENCE_SERVICE, HealthServingStatus::NotServing)
        .await;
    timeout(Duration::from_secs(2), unavailable)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    for control in &mut controls {
        assert!(matches!(
            control.commands.recv().await,
            Some(KvStreamCommand::Suspend)
        ));
        control.status.send_replace(KvStreamStatus::Recovering);
    }
    server
        .health
        .set_service_status(INFERENCE_SERVICE, HealthServingStatus::Serving)
        .await;
    let recovered = tokio::spawn({
        let engine = engine.clone();
        async move { engine.recover_serving().await }
    });
    for control in &mut controls {
        let Some(KvStreamCommand::ReconnectSameInstance(accepted)) =
            timeout(Duration::from_secs(2), control.commands.recv())
                .await
                .unwrap()
        else {
            panic!("reconnect")
        };
        accepted.send(()).unwrap();
    }
    assert!(!recovered.is_finished());
    assert!(
        engine
            .generate(
                minimal_request(),
                GenerateContext::new(dynamo_backend_common::testing::mock_context(), None)
            )
            .await
            .is_err()
    );
    assert!(server.service.requests.lock().await.is_empty());
    for control in &controls {
        control.status.send_replace(KvStreamStatus::Ready);
    }
    assert!(matches!(
        timeout(Duration::from_secs(2), recovered)
            .await
            .unwrap()
            .unwrap()
            .unwrap(),
        EngineRecovery::SameInstance
    ));
    engine.cleanup().await.unwrap();
    assert!(server.service.control_calls.lock().await.is_empty());
}
