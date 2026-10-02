// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::{BackendError, DisaggregationMode, FinishReason, LLMEngine};
use dynamo_llm::discovery::{LoadThresholdHandle, ModelManager};
use dynamo_llm::kv_router::PrefillRouter;
use dynamo_llm::model_card::TokenizerKind;
use dynamo_llm::session_affinity::SessionAffinityMode;
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::pipeline::{AsyncEngine, AsyncEngineContextProvider, Operator, RouterMode};
use dynamo_sidecar_testkit::{
    assert::{failure, terminal},
    bounded,
    control::{
        Controller, Event, OpenAction, RequestHandle, RequestPlan, StreamAction, StreamFault,
        StreamPoint,
    },
};
use futures::StreamExt;
use serde_json::{Value, json};

#[path = "support/process.rs"]
mod process;
#[allow(dead_code)]
mod support;

use process::{Environment, Gate, outputs};
use support::{FixtureConfig, HandoffFixture, ProcessFixture, SidecarFixture, sglang, vllm};

async fn healthy<F: ProcessFixture>(
    env: &Environment,
    router: &crate::process::Router,
    control: &Controller<F::Protocol>,
    id: &str,
) -> RequestHandle<F::Protocol> {
    healthy_with_model::<F>(env, router, control, id, &env.model).await
}

async fn healthy_with_model<F: ProcessFixture>(
    env: &Environment,
    router: &crate::process::Router,
    control: &Controller<F::Protocol>,
    id: &str,
    model: &str,
) -> RequestHandle<F::Protocol> {
    bounded("router fault recovery", async {
        while router.selectable_worker_ids().is_err() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await;
    let handle = control.request(id, RequestPlan::default());
    let mut request = env.request(id, 3);
    request.model = model.into();
    F::configure_request(&mut request);
    let expected = request.content().clone();
    let stream = bounded("Dynamo request ingress", router.generate(request))
        .await
        .unwrap();
    let result = outputs(stream).await;
    assert_eq!(handle.tokens().len(), 3);
    F::assert_stream(&handle, &expected, &result);
    terminal(result, &handle.tokens(), 4, FinishReason::Length);
    bounded(
        "native healthy request release",
        handle.wait(Event::Dropped),
    )
    .await;
    handle
}

#[tokio::test]
async fn vllm_registration_and_errors_recover_through_worker_ingress() {
    tokio::time::timeout(
        Duration::from_secs(60),
        registration_and_errors_recover_through_worker_ingress::<vllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

async fn registration_and_errors_recover_through_worker_ingress<F: ProcessFixture>() {
    let env = Environment::new().await;
    let control = Controller::default();
    let mut peer = F::start(
        control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            ..Default::default()
        },
    )
    .await;
    let mut child = env.spawn::<F>(&peer.endpoint(), DisaggregationMode::Aggregated, 5);
    let router = env.ready("backend").await;
    let cards = env.cards().await;
    assert_eq!(cards.len(), 1);
    let card = &cards[0];
    assert_eq!(card.name(), env.model);
    assert_eq!(card.worker_type, Some(WorkerType::Aggregated));
    assert!(card.tokenizer.is_some());
    assert!(card.prompt_formatter.is_some());
    F::assert_registration(card);
    healthy::<F>(&env, &router, &control, "registered").await;

    let rejected = control.request("unsupported", RequestPlan::default());
    let mut request = env.request("unsupported", 3);
    request.sampling_options.n = Some(2);
    let error = match bounded("unsupported request ingress", router.generate(request)).await {
        Err(error) => error
            .downcast::<dynamo_backend_common::DynamoError>()
            .unwrap(),
        Ok(_) => panic!("unsupported request must fail before a response stream"),
    };
    assert!(
        dynamo_runtime::error::match_error_chain(
            &error,
            &[dynamo_backend_common::ErrorType::Backend(
                BackendError::InvalidArgument
            )],
            &[],
        ),
        "{error}"
    );
    assert!(!rejected.reached(Event::Received));
    healthy::<F>(&env, &router, &control, "after-unsupported").await;

    for (id, plan, kind) in [
        (
            "open-error",
            RequestPlan {
                open: OpenAction::Fail,
                stream: None,
            },
            BackendError::CannotConnect,
        ),
        (
            "read-error",
            RequestPlan {
                open: OpenAction::Continue,
                stream: Some(StreamFault {
                    at: StreamPoint::TokenResponse(1),
                    action: StreamAction::Fail,
                    pause: false,
                }),
            },
            BackendError::CannotConnect,
        ),
        (
            "early-eof",
            RequestPlan {
                open: OpenAction::Continue,
                stream: Some(StreamFault {
                    at: StreamPoint::TokenResponse(1),
                    action: StreamAction::Close,
                    pause: false,
                }),
            },
            F::eof_error(),
        ),
    ] {
        let handle = control.request(id, plan);
        match bounded(
            "failed request ingress",
            router.generate(env.request(id, 3)),
        )
        .await
        {
            Ok(stream) => {
                failure(outputs(stream).await, &handle.tokens(), kind);
            }
            Err(error) => {
                assert!(handle.tokens().is_empty());
                assert!(
                    dynamo_runtime::error::match_error_chain(
                        error.as_ref(),
                        &[dynamo_backend_common::ErrorType::Backend(kind)],
                        &[],
                    ),
                    "{error}"
                );
            }
        }
        bounded("failed native request release", handle.wait(Event::Dropped)).await;
        healthy::<F>(&env, &router, &control, &format!("after-{id}")).await;
    }
    child.shutdown().await;
    env.withdrawn("backend", &router).await;
    peer.shutdown().await;
}

#[tokio::test]
async fn vllm_delayed_startup_publishes_only_after_native_readiness() {
    tokio::time::timeout(
        Duration::from_secs(60),
        delayed_startup_publishes_only_after_native_readiness::<vllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

async fn delayed_startup_publishes_only_after_native_readiness<F: ProcessFixture>() {
    let env = Environment::new().await;
    let control = Controller::default();
    let mut peer = F::start(
        control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            ..Default::default()
        },
    )
    .await;
    let mut gate = Gate::new(&peer.endpoint()).await;
    let mut child = env.spawn_env::<F>(&gate.endpoint, DisaggregationMode::Aggregated, 5);
    gate.accepted().await;
    assert!(env.cards().await.is_empty());
    assert!(env.registrations("backend").await.is_empty());
    gate.release();
    let router = env.ready("backend").await;
    healthy::<F>(&env, &router, &control, "after-delayed-start").await;
    child.shutdown().await;
    env.withdrawn("backend", &router).await;
    gate.shutdown().await;
    peer.shutdown().await;
}

#[tokio::test]
async fn vllm_failed_and_interrupted_startup_leave_no_registration() {
    tokio::time::timeout(
        Duration::from_secs(60),
        failed_and_interrupted_startup_leave_no_registration::<vllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

async fn failed_and_interrupted_startup_leave_no_registration<F: ProcessFixture>() {
    let env = Environment::new().await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let mut child = env.spawn::<F>(&endpoint, DisaggregationMode::Aggregated, 1);
    assert!(!child.exit().await.success(), "{}", child.logs());
    assert!(env.cards().await.is_empty());
    assert!(env.registrations("backend").await.is_empty());
    for is_interrupted in [false, true] {
        let env = Environment::new().await;
        let mut gate = Gate::new("http://127.0.0.1:1").await;
        let deadline = if is_interrupted { 30 } else { 1 };
        let mut child = env.spawn::<F>(&gate.endpoint, DisaggregationMode::Aggregated, deadline);
        gate.accepted().await;
        if is_interrupted {
            child.signal(libc::SIGTERM);
        }
        let status = child.exit().await;
        if is_interrupted {
            assert!(
                status.success(),
                "interrupted startup did not shut down cleanly\n{}",
                child.logs()
            );
        } else {
            assert!(
                !status.success(),
                "startup unexpectedly succeeded\n{}",
                child.logs()
            );
        }
        assert!(env.cards().await.is_empty());
        assert!(env.registrations("backend").await.is_empty());
        gate.shutdown().await;
    }
}

fn hold_after_token() -> RequestPlan {
    RequestPlan {
        open: OpenAction::Continue,
        stream: Some(StreamFault {
            at: StreamPoint::TokenResponse(1),
            action: StreamAction::Continue,
            pause: true,
        }),
    }
}

#[tokio::test]
async fn vllm_worker_cancel_and_consumer_drop_release_only_the_target() {
    tokio::time::timeout(
        Duration::from_secs(60),
        worker_cancel_and_consumer_drop_release_only_the_target::<vllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

async fn worker_cancel_and_consumer_drop_release_only_the_target<F: ProcessFixture>() {
    let env = Environment::new().await;
    let control = Controller::default();
    let mut peer = F::start(
        control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            speedup_ratio: 0.1,
            ..Default::default()
        },
    )
    .await;
    let mut child = env.spawn::<F>(&peer.endpoint(), DisaggregationMode::Aggregated, 5);
    let router = env.ready("backend").await;

    for (id, explicit, waiting_headers) in [
        ("cancel-open", true, true),
        ("cancel-stream", true, false),
        ("drop-stream", false, false),
    ] {
        let plan = if waiting_headers {
            RequestPlan {
                open: OpenAction::Hold,
                stream: None,
            }
        } else {
            hold_after_token()
        };
        let target = control.request(id, plan);
        let request = env.request(id, 10_000);
        let context = request.context();
        let generation = router.generate(request);
        tokio::pin!(generation);
        let mut stream = bounded("target request ingress", async {
            tokio::select! {
                result = &mut generation => Some(result.unwrap()),
                _ = target.wait(Event::Received) => None,
            }
        })
        .await;
        bounded("target native acceptance", target.wait(Event::Received)).await;
        if !waiting_headers {
            if stream.is_none() {
                stream = Some(
                    bounded("target response headers", &mut generation)
                        .await
                        .unwrap(),
                );
            }
            let first = bounded("target first token", stream.as_mut().unwrap().next())
                .await
                .unwrap()
                .into_data()
                .unwrap()
                .unwrap();
            assert_eq!(first.token_ids.len(), 1);
            bounded("target stream checkpoint", target.wait(Event::Checkpoint)).await;
            peer.scheduler_active().await;
        }
        let other_id = format!("other-{id}");
        let other = control.request(&other_id, hold_after_token());
        let mut other_stream = router.generate(env.request(&other_id, 3)).await.unwrap();
        let other_first = bounded("independent first token", other_stream.next())
            .await
            .unwrap()
            .into_data()
            .unwrap()
            .unwrap();
        bounded(
            "independent stream checkpoint",
            other.wait(Event::Checkpoint),
        )
        .await;
        if explicit {
            context.stop_generating();
            let cancelled = if let Some(stream) = stream {
                outputs(stream).await
            } else {
                let stream = bounded("cancel pending response headers", &mut generation)
                    .await
                    .unwrap();
                outputs(stream).await
            };
            assert!(
                cancelled
                    .iter()
                    .filter_map(|output| output.as_ref().ok())
                    .all(|output| !matches!(
                        output.finish_reason,
                        Some(FinishReason::Stop | FinishReason::Length)
                    ))
            );
        } else {
            assert!(
                !context.is_stopped(),
                "consumer drop must not call explicit cancellation"
            );
            drop(stream);
        }
        bounded("target remote release", target.wait(Event::Dropped)).await;
        assert!(!other.reached(Event::Dropped));
        assert_eq!(other.tokens().len(), 1);
        other.release();
        let mut other_outputs = vec![Ok(other_first)];
        other_outputs.extend(outputs(other_stream).await);
        terminal(other_outputs, &other.tokens(), 4, FinishReason::Length);
        peer.scheduler_idle().await;
        healthy::<F>(&env, &router, &control, &format!("recovery-{id}")).await;
    }
    child.shutdown().await;
    env.withdrawn("backend", &router).await;
    peer.shutdown().await;
}

#[tokio::test]
async fn vllm_sigterm_withdraws_worker_and_releases_active_native_request() {
    tokio::time::timeout(
        Duration::from_secs(60),
        sigterm_withdraws_worker_and_releases_active_native_request::<vllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

async fn sigterm_withdraws_worker_and_releases_active_native_request<F: ProcessFixture>() {
    let env = Environment::new().await;
    let control = Controller::default();
    let mut peer = F::start(
        control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            speedup_ratio: 0.1,
            ..Default::default()
        },
    )
    .await;
    let mut child = env.spawn_with_grace::<F>(&peer.endpoint(), DisaggregationMode::Aggregated);
    let router = env.ready("backend").await;
    let handle = control.request("shutdown-active", hold_after_token());
    let mut stream = router
        .generate(env.request("shutdown-active", 10_000))
        .await
        .unwrap();
    bounded("shutdown first token", stream.next())
        .await
        .unwrap()
        .into_data()
        .unwrap()
        .unwrap();
    bounded("shutdown native checkpoint", handle.wait(Event::Checkpoint)).await;
    peer.scheduler_active().await;
    child.signal(libc::SIGTERM);
    env.withdrawn("backend", &router).await;
    assert!(child.is_running(), "withdrawal must precede process exit");
    assert!(
        !handle.reached(Event::Dropped),
        "withdrawal must precede engine cleanup"
    );
    peer.scheduler_active().await;
    let tail = outputs(stream).await;
    assert!(
        tail.iter()
            .filter_map(|item| item.as_ref().ok())
            .all(|item| !matches!(
                item.finish_reason,
                Some(FinishReason::Stop | FinishReason::Length)
            ))
    );
    bounded("shutdown remote release", handle.wait(Event::Dropped)).await;
    peer.scheduler_idle().await;
    assert!(child.exit().await.success(), "{}", child.logs());
    let independent = peer.engine().await;
    independent.start(0).await.unwrap();
    independent.cleanup().await.unwrap();
    peer.shutdown().await;
}

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

async fn prefill_router_preserves_handoff_failure_and_cancellation<F: HandoffFixture>() {
    let env = Environment::new().await;
    let prefill_control = Controller::default();
    let decode_control = Controller::default();
    let mut prefill_peer = F::start(
        prefill_control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            disaggregation_mode: DisaggregationMode::Prefill,
            ..Default::default()
        },
    )
    .await;
    let mut decode_peer = F::start(
        decode_control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            disaggregation_mode: DisaggregationMode::Decode,
            ..Default::default()
        },
    )
    .await;
    let mut prefill = env.spawn::<F>(&prefill_peer.endpoint(), DisaggregationMode::Prefill, 5);
    let mut decode = env.spawn::<F>(&decode_peer.endpoint(), DisaggregationMode::Decode, 5);
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
        if F::HAS_BOOTSTRAP && role == WorkerType::Prefill {
            let endpoint = card.runtime_config.disaggregated_endpoint.as_ref().unwrap();
            assert!(
                endpoint
                    .bootstrap_host
                    .as_ref()
                    .is_some_and(|host| !host.is_empty())
            );
            assert_eq!(endpoint.bootstrap_port, Some(8998));
        }
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
        let observed_prefill = prefill_control.request(id, prefill_plan::<F>());
        let observed_decode = decode_control.request(id, RequestPlan::default());
        let stream = bounded(
            "prefill/decode handoff",
            router.generate(env.request(id, 3), decode_router.clone()),
        )
        .await
        .unwrap();
        if F::HAS_BOOTSTRAP {
            bounded(
                "prefill native acceptance",
                observed_prefill.wait(Event::Received),
            )
            .await;
            bounded(
                "decode native acceptance",
                observed_decode.wait(Event::Received),
            )
            .await;
            assert!(observed_prefill.tokens().is_empty());
            assert!(!observed_prefill.reached(Event::Dropped));
            observed_prefill.release();
        }
        terminal(
            outputs(stream).await,
            &observed_decode.tokens(),
            4,
            FinishReason::Length,
        );
        F::assert_handoff(&observed_prefill, &observed_decode, id);
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
    if !F::HAS_BOOTSTRAP {
        assert!(
            result.is_err(),
            "failed prefill must not produce a successful handoff"
        );
    }
    match result {
        Err(_) => {}
        Ok(stream) => {
            let response = outputs(stream).await;
            assert!(response.iter().any(Result::is_err));
            assert!(
                response
                    .iter()
                    .filter_map(|item| item.as_ref().ok())
                    .all(|output| {
                        !matches!(
                            output.finish_reason,
                            Some(FinishReason::Stop | FinishReason::Length)
                        )
                    })
            );
        }
    }
    if !F::HAS_BOOTSTRAP {
        assert!(!unsubmitted.reached(Event::Received));
    }
    bounded("failed prefill native release", failed.wait(Event::Dropped)).await;
    ready(&router).await;

    let prefill_observation = prefill_control.request("cancel-handoff", prefill_plan::<F>());
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
    bounded(
        "prefill accepted before handoff cancellation",
        prefill_observation.wait(Event::Received),
    )
    .await;
    context.stop_generating();
    assert!(prefill_observation.reached(Event::Received));
    if F::HAS_BOOTSTRAP {
        assert!(!prefill_observation.reached(Event::Dropped));
        F::assert_handoff(&prefill_observation, &decode_observation, "cancel-handoff");
        prefill_observation.release();
    }
    decode_observation.release();
    bounded("handoff cancellation completion", async {
        let stream = match stream {
            Some(stream) => stream,
            None => generation.await.unwrap(),
        };
        assert!(
            outputs(stream)
                .await
                .iter()
                .filter_map(|item| item.as_ref().ok())
                .all(|output| !matches!(
                    output.finish_reason,
                    Some(FinishReason::Stop | FinishReason::Length)
                ))
        );
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

fn prefill_plan<F: HandoffFixture>() -> RequestPlan {
    RequestPlan {
        open: if F::HAS_BOOTSTRAP {
            OpenAction::Hold
        } else {
            OpenAction::Continue
        },
        stream: None,
    }
}

#[tokio::test]
async fn vllm_prefill_router_preserves_handoff_failure_and_cancellation() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        prefill_router_preserves_handoff_failure_and_cancellation::<vllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn sglang_registration_and_errors_recover_through_worker_ingress() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        registration_and_errors_recover_through_worker_ingress::<support::sglang::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn sglang_delayed_startup_publishes_only_after_native_readiness() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        delayed_startup_publishes_only_after_native_readiness::<support::sglang::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn sglang_failed_and_interrupted_startup_leave_no_registration() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        failed_and_interrupted_startup_leave_no_registration::<support::sglang::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn sglang_worker_cancel_and_consumer_drop_release_only_the_target() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        worker_cancel_and_consumer_drop_release_only_the_target::<support::sglang::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn sglang_sigterm_withdraws_worker_and_releases_active_native_request() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        sigterm_withdraws_worker_and_releases_active_native_request::<support::sglang::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn sglang_prefill_router_preserves_handoff_failure_and_cancellation() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        prefill_router_preserves_handoff_failure_and_cancellation::<sglang::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn vllm_served_model_alias_is_published() {
    tokio::time::timeout(
        Duration::from_secs(60),
        served_model_alias_is_published::<vllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn sglang_served_model_alias_is_published() {
    tokio::time::timeout(
        Duration::from_secs(60),
        served_model_alias_is_published::<sglang::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn sglang_distinct_model_tokenizer_and_alias_are_published() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        distinct_model_tokenizer_and_alias_are_published(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn vllm_health_readiness_precedes_publication() {
    tokio::time::timeout(
        Duration::from_secs(60),
        health_readiness_precedes_publication::<vllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn sglang_health_readiness_precedes_publication() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        health_readiness_precedes_publication::<sglang::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn vllm_unhealthy_worker_never_registers() {
    tokio::time::timeout(
        Duration::from_secs(60),
        unhealthy_worker_never_registers::<vllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn sglang_unhealthy_worker_never_registers() {
    tokio::time::timeout(
        Duration::from_secs(60),
        unhealthy_worker_never_registers::<sglang::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn sglang_changed_role_never_registers() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        changed_role_never_registers(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

async fn served_model_alias_is_published<F: ProcessFixture>() {
    let env = Environment::new().await;
    let control = Controller::default();
    let mut peer = F::start(
        control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            ..Default::default()
        },
    )
    .await;
    peer.set_served_model_name("public-alias");
    let mut child = env.spawn::<F>(&peer.endpoint(), DisaggregationMode::Aggregated, 5);
    let router = env.ready("backend").await;
    let cards = env.cards().await;
    assert_eq!(cards.len(), 1);
    let card = &cards[0];
    assert_eq!(card.name(), "public-alias");
    assert_eq!(card.source_path(), env.model);
    assert_eq!(card.worker_type, Some(WorkerType::Aggregated));
    assert!(card.tokenizer.is_some());
    assert!(card.prompt_formatter.is_some());
    F::assert_registration(card);
    healthy_with_model::<F>(&env, &router, &control, "alias-request", "public-alias").await;
    child.shutdown().await;
    env.withdrawn("backend", &router).await;
    peer.shutdown().await;
}

async fn distinct_model_tokenizer_and_alias_are_published() {
    let env = Environment::new().await;
    let tokenizer = std::path::Path::new(&env.model).join("separate-tokenizer");
    std::fs::create_dir(&tokenizer).unwrap();
    for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
        std::fs::copy(
            std::path::Path::new(&env.model).join(file),
            tokenizer.join(file),
        )
        .unwrap();
    }
    let tokenizer_file = tokenizer.join("tokenizer.json");
    let mut tokenizer_json: Value =
        serde_json::from_str(&std::fs::read_to_string(&tokenizer_file).unwrap()).unwrap();
    tokenizer_json["model"]["vocab"]["separate"] = json!(3);
    std::fs::write(&tokenizer_file, tokenizer_json.to_string()).unwrap();
    let control = Controller::default();
    let mut peer = sglang::Fixture::start(
        control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            ..Default::default()
        },
    )
    .await;
    peer.override_discovery(
        json!({"tokenizer_path": tokenizer}),
        vec![json!({
            "served_model_name": "public-alias",
            "tool_call_parser": "hermes",
            "reasoning_parser": "deepseek_r1",
        })],
    );
    let mut child =
        env.spawn::<sglang::Fixture>(&peer.endpoint(), DisaggregationMode::Aggregated, 5);
    let router = env.ready("backend").await;
    let cards = env.cards().await;
    assert_eq!(cards.len(), 1);
    let card = &cards[0];
    assert_eq!(card.name(), "public-alias");
    assert_eq!(card.source_path(), tokenizer.to_str().unwrap());
    assert_eq!(card.worker_type, Some(WorkerType::Aggregated));
    assert_eq!(card.kv_cache_block_size, 4);
    assert_eq!(card.effective_context_length(), 32_768);
    assert_eq!(
        card.runtime_config.tool_call_parser.as_deref(),
        Some("hermes")
    );
    assert_eq!(
        card.runtime_config.reasoning_parser.as_deref(),
        Some("deepseek_r1")
    );
    let TokenizerKind::HfTokenizerJson(file) = card.tokenizer.as_ref().unwrap() else {
        panic!("expected the discovered tokenizer.json");
    };
    assert!(file.checksum_matches(&tokenizer_file));
    assert!(!file.checksum_matches(std::path::Path::new(&env.model).join("tokenizer.json")));
    assert!(card.prompt_formatter.is_some());
    let handle = healthy::<sglang::Fixture>(&env, &router, &control, "distinct-metadata").await;
    let native = handle.native_request().unwrap();
    let traceparent = &native.trace_headers["traceparent"];
    let fields = traceparent.split('-').collect::<Vec<_>>();
    assert_eq!(fields.len(), 4);
    assert_eq!(fields[0], "00");
    assert_eq!(fields[1].len(), 32);
    assert_eq!(fields[2].len(), 16);
    assert!(u128::from_str_radix(fields[1], 16).unwrap() != 0);
    assert!(u64::from_str_radix(fields[2], 16).unwrap() != 0);
    child.shutdown().await;
    env.withdrawn("backend", &router).await;
    peer.shutdown().await;
}

async fn health_readiness_precedes_publication<F: ProcessFixture>() {
    let env = Environment::new().await;
    let control = Controller::default();
    let mut peer = F::start(
        control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            ..Default::default()
        },
    )
    .await;
    peer.set_health(None);
    let mut child = env.spawn::<F>(&peer.endpoint(), DisaggregationMode::Aggregated, 5);
    peer.health_check_received().await;
    assert!(env.cards().await.is_empty());
    assert!(env.registrations("backend").await.is_empty());
    peer.set_health(Some(true));
    let router = env.ready("backend").await;
    healthy::<F>(&env, &router, &control, "after-health-readiness").await;
    child.shutdown().await;
    env.withdrawn("backend", &router).await;
    peer.shutdown().await;
}

async fn unhealthy_worker_never_registers<F: ProcessFixture>() {
    let env = Environment::new().await;
    let mut peer = F::start(
        Controller::default(),
        FixtureConfig {
            model: env.model.clone(),
            ..Default::default()
        },
    )
    .await;
    peer.set_health(Some(false));
    let mut child = env.spawn::<F>(&peer.endpoint(), DisaggregationMode::Aggregated, 1);
    peer.health_check_received().await;
    assert!(!child.exit().await.success(), "{}", child.logs());
    assert!(env.cards().await.is_empty());
    assert!(env.registrations("backend").await.is_empty());
    F::assert_unhealthy_startup(&child.logs());
    peer.shutdown().await;
}

async fn changed_role_never_registers() {
    let env = Environment::new().await;
    let mut peer = sglang::Fixture::start(
        Controller::default(),
        FixtureConfig {
            model: env.model.clone(),
            ..Default::default()
        },
    )
    .await;
    peer.override_discovery(
        Value::Null,
        vec![json!({}), json!({"disaggregation_mode": "prefill"})],
    );
    let mut child =
        env.spawn::<sglang::Fixture>(&peer.endpoint(), DisaggregationMode::Aggregated, 1);
    peer.health_check_received().await;
    assert!(!child.exit().await.success(), "{}", child.logs());
    assert!(env.cards().await.is_empty());
    assert!(env.registrations("backend").await.is_empty());
    assert!(
        child.logs().contains("role changed since bootstrap"),
        "{}",
        child.logs()
    );
    peer.shutdown().await;
}
