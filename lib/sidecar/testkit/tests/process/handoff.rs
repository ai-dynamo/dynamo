// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_backend_common::{DisaggregationMode, FinishReason};
use dynamo_llm::discovery::{LoadThresholdHandle, ModelManager};
use dynamo_llm::kv_router::PrefillRouter;
use dynamo_llm::session_affinity::SessionAffinityMode;
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Operator, RouterMode};
use dynamo_sidecar_testkit::{
    assert::terminal,
    bounded,
    control::{Controller, Event, OpenAction, RequestPlan},
};

use crate::process::{Environment, outputs};
use crate::support::{FixtureConfig, SidecarFixture, vllm};

async fn ready(router: &PrefillRouter) {
    bounded("real PrefillRouter availability", async {
        loop {
            if let Ok(reservation) = router
                .reserve_prefill_worker(
                    "ready",
                    &[11, 22, 33, 44],
                    None,
                    None,
                    None,
                    0.0,
                    0,
                    None,
                    None,
                    Default::default(),
                )
                .await
            {
                reservation.release().await.unwrap();
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await;
}

pub async fn vllm_prefill_router_preserves_handoff_failure_and_cancellation() {
    let env = Environment::new().await;
    let prefill_control = Controller::default();
    let decode_control = Controller::default();
    let mut prefill_peer = vllm::Fixture::start(
        prefill_control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            disaggregation_mode: DisaggregationMode::Prefill,
            ..Default::default()
        },
    )
    .await;
    let mut decode_peer = vllm::Fixture::start(
        decode_control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            disaggregation_mode: DisaggregationMode::Decode,
            ..Default::default()
        },
    )
    .await;
    let mut prefill = env.spawn::<vllm::Fixture>(
        &prefill_peer.server.endpoint(),
        DisaggregationMode::Prefill,
        5,
    );
    let mut decode = env.spawn::<vllm::Fixture>(
        &decode_peer.server.endpoint(),
        DisaggregationMode::Decode,
        5,
    );
    let prefill_router = env.ready("prefill").await;
    let decode_router = env.ready("backend").await;
    let cards = env.cards().await;
    assert_eq!(cards.len(), 2);
    for role in [WorkerType::Prefill, WorkerType::Decode] {
        let card = cards
            .iter()
            .find(|card| card.worker_type == Some(role))
            .unwrap();
        assert_eq!(card.name(), env.model);
        assert_eq!(card.kv_cache_block_size, 4);
    }
    let (activation, activated) = tokio::sync::oneshot::channel();
    let router = PrefillRouter::new(
        activated,
        Arc::new(ModelManager::new()),
        RouterMode::RoundRobin,
        4,
        None,
        None,
        None,
        SessionAffinityMode::Hard,
        env.model.clone(),
        env.namespace.clone(),
        LoadThresholdHandle::new(Default::default()),
        env.runtime.primary_token(),
    );
    activation.send(env.endpoint("prefill")).unwrap();
    ready(&router).await;

    for id in ["handoff", "handoff-repeat"] {
        let observed_prefill = prefill_control.request(id, RequestPlan::default());
        let observed_decode = decode_control.request(id, RequestPlan::default());
        let stream = bounded(
            "prefill/decode handoff",
            router.generate(env.request(id, 3), decode_router.clone()),
        )
        .await
        .unwrap();
        terminal(
            outputs(stream).await,
            &observed_decode.tokens(),
            4,
            FinishReason::Length,
        );
        let prefill_wire = observed_prefill.native_request().unwrap();
        let decode_wire = observed_decode.native_request().unwrap();
        assert_eq!(prefill_wire.request_id, id);
        assert_eq!(decode_wire.request_id, id);
        assert_eq!(prefill_wire.stopping.unwrap().max_new_tokens, 1);
        assert_eq!(decode_wire.stopping.unwrap().max_new_tokens, 3);
        let mut native_handoff = observed_prefill
            .native_responses()
            .into_iter()
            .find_map(|response| response.outputs?.finish_info?.kv_transfer_params)
            .expect("native prefill handoff");
        let mut forwarded = decode_wire.kv.unwrap().kv_transfer_params.unwrap();
        assert_eq!(
            forwarded.fields["mocker_request_id"].kind,
            Some(prost_types_v14::value::Kind::StringValue(id.to_string()))
        );
        assert_eq!(
            native_handoff.fields.remove("remote_port").unwrap().kind,
            Some(prost_types_v14::value::Kind::NumberValue(0.0))
        );
        assert_eq!(
            forwarded.fields.remove("remote_port").unwrap().kind,
            Some(prost_types_v14::value::Kind::StringValue("0".to_string()))
        );
        assert_eq!(forwarded, native_handoff);
    }

    let failed = prefill_control.request(
        "failed-prefill",
        RequestPlan {
            open: OpenAction::Fail,
            stream: None,
        },
    );
    let unsubmitted = decode_control.request("failed-prefill", RequestPlan::default());
    let result = bounded(
        "failed prefill",
        router.generate(env.request("failed-prefill", 3), decode_router.clone()),
    )
    .await;
    assert!(
        result.is_err(),
        "failed prefill must not produce a successful handoff"
    );
    assert!(!unsubmitted.reached(Event::Received));
    bounded("failed prefill native release", failed.wait(Event::Dropped)).await;
    ready(&router).await;

    let prefill_observation = prefill_control.request("cancel-handoff", RequestPlan::default());
    let decode_observation = decode_control.request(
        "cancel-handoff",
        RequestPlan {
            open: OpenAction::Hold,
            stream: None,
        },
    );
    let request = env.request("cancel-handoff", 3);
    let context = request.context();
    let generation = router.generate(request, decode_router.clone());
    tokio::pin!(generation);
    let stream = bounded("decode request accepted during handoff", async {
        tokio::select! {
            result = &mut generation => {
                let stream = result.unwrap();
                decode_observation.wait(Event::Received).await;
                Some(stream)
            }
            _ = decode_observation.wait(Event::Received) => None,
        }
    })
    .await;
    context.stop_generating();
    assert!(prefill_observation.reached(Event::Received));
    decode_observation.release();
    bounded("handoff cancellation completion", async {
        match match stream {
            Some(stream) => Ok(stream),
            None => generation.await,
        } {
            Ok(stream) => assert!(
                outputs(stream)
                    .await
                    .iter()
                    .filter_map(|item| item.as_ref().ok())
                    .all(|output| !matches!(
                        output.finish_reason,
                        Some(FinishReason::Stop | FinishReason::Length)
                    ))
            ),
            Err(error) => assert!(
                dynamo_runtime::error::match_error_chain(
                    error.as_ref(),
                    &[
                        dynamo_backend_common::ErrorType::Cancelled,
                        dynamo_backend_common::ErrorType::Backend(
                            dynamo_backend_common::BackendError::Cancelled
                        )
                    ],
                    &[],
                ),
                "{error}"
            ),
        }
    })
    .await;
    bounded("prefill release", prefill_observation.wait(Event::Dropped)).await;
    bounded("decode release", decode_observation.wait(Event::Dropped)).await;
    prefill_peer.scheduler_idle().await;
    decode_peer.scheduler_idle().await;
    let p = prefill_control.request("after-handoff-cancel", RequestPlan::default());
    let d = decode_control.request("after-handoff-cancel", RequestPlan::default());
    let stream = router
        .generate(
            env.request("after-handoff-cancel", 3),
            decode_router.clone(),
        )
        .await
        .unwrap();
    terminal(outputs(stream).await, &d.tokens(), 4, FinishReason::Length);
    assert!(p.reached(Event::Received));
    prefill.shutdown().await;
    decode.shutdown().await;
    env.withdrawn("backend", &decode_router).await;
    env.withdrawn("prefill", &prefill_router).await;
    prefill_peer.shutdown().await;
    decode_peer.shutdown().await;
}
