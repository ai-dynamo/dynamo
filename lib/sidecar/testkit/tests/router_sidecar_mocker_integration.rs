// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::{BackendError, DisaggregationMode, FinishReason, LLMEngine};
use dynamo_llm::discovery::{LoadThresholdHandle, ModelManager};
use dynamo_llm::kv_router::PrefillRouter;
use dynamo_llm::model_card::TokenizerKind;
use dynamo_llm::protocols::common::preprocessor::RoutingHints;
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

use process::{Environment, Frontend, Gate, outputs};
use support::{
    DiscoveryFixture, FixtureConfig, HandoffFixture, ProcessFixture, SidecarFixture, sglang, vllm,
};

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
    assert_eq!(
        error.class(),
        dynamo_backend_common::ErrorType::InvalidRequest,
        "{error}"
    );
    assert!(
        !dynamo_runtime::error::match_error_chain(
            &error,
            &[dynamo_backend_common::ErrorType::CannotConnect],
            &[],
        ),
        "{error}"
    );
    assert!(
        router
            .selectable_worker_ids()
            .is_ok_and(|ids| !ids.is_empty()),
        "a request-validation refusal must not quarantine the worker"
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

fn sse_events(body: &str) -> impl Iterator<Item = Value> + '_ {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str(data).ok())
}

async fn native_errors_reach_http_and_recover<F: ProcessFixture>() {
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
    let frontend = Frontend::start(&env).await;
    for (index, is_streaming) in [false, true].into_iter().enumerate() {
        let id = format!("00000000-0000-4000-8000-{:012}", index * 2 + 1);
        let handle = control.request(
            &id,
            if is_streaming {
                RequestPlan {
                    stream: Some(StreamFault {
                        at: StreamPoint::TokenResponse(1),
                        action: StreamAction::Fail,
                        pause: true,
                    }),
                    ..Default::default()
                }
            } else {
                RequestPlan {
                    open: OpenAction::Reject,
                    ..Default::default()
                }
            },
        );
        let mut response = bounded(
            "HTTP native failure response",
            frontend.generate(&env.model, &id, is_streaming).send(),
        )
        .await
        .unwrap();
        if is_streaming {
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            let mut body = String::new();
            bounded("HTTP client receives native token before failure", async {
                loop {
                    let chunk = response.chunk().await.unwrap().expect("token chunk");
                    body.push_str(&String::from_utf8_lossy(&chunk));
                    if sse_events(&body).any(|event| {
                        event["nvext"]["completion_token_ids"]
                            .as_array()
                            .is_some_and(|tokens| !tokens.is_empty())
                    }) {
                        break;
                    }
                }
            })
            .await;
            bounded("native failure checkpoint", handle.wait(Event::Checkpoint)).await;
            assert_eq!(handle.tokens().len(), 1);
            handle.release();
            body.push_str(
                &bounded("HTTP failed stream body", response.text())
                    .await
                    .unwrap(),
            );
            let events: Vec<_> = sse_events(&body).collect();
            let errors: Vec<_> = events
                .iter()
                .filter_map(|event| event.get("error"))
                .collect();
            assert_eq!(errors.len(), 1, "{body}");
            assert_eq!(errors[0]["code"], 503, "{body}");
            assert!(!body.contains("injected read failure"), "{body}");
            assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
            assert!(
                events
                    .iter()
                    .all(
                        |event| event["choices"].as_array().is_none_or(|choices| choices
                            .iter()
                            .all(|choice| choice["finish_reason"].is_null()))
                    ),
                "failed stream reported a successful finish: {body}"
            );
        } else {
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
            let body: Value = response.json().await.unwrap();
            assert_eq!(body["code"], 400, "{body}");
            assert_eq!(body["message"], "Invalid request", "{body}");
            assert!(
                handle.reached(Event::Received),
                "frontend rejected before native RPC"
            );
        }
        bounded(
            "failed native request released",
            handle.wait(Event::Dropped),
        )
        .await;
        peer.scheduler_idle().await;

        let recovery_id = format!("00000000-0000-4000-8000-{:012}", index * 2 + 2);
        let recovered = control.request(&recovery_id, RequestPlan::default());
        let response = bounded(
            "HTTP recovery response",
            frontend.generate(&env.model, &recovery_id, false).send(),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["choices"][0]["finish_reason"], "length", "{body}");
        assert_eq!(body["usage"]["completion_tokens"], 3, "{body}");
        assert_eq!(recovered.tokens().len(), 3);
        bounded(
            "HTTP recovery request released",
            recovered.wait(Event::Dropped),
        )
        .await;
        peer.scheduler_idle().await;
    }
    frontend.shutdown().await;
    child.shutdown().await;
    env.withdrawn("backend", &router).await;
    peer.shutdown().await;
}

#[tokio::test]
async fn vllm_native_errors_reach_http_and_recover() {
    tokio::time::timeout(
        Duration::from_secs(60),
        native_errors_reach_http_and_recover::<vllm::Fixture>(),
    )
    .await
    .expect("HTTP error scenario exceeded its overall deadline");
}

#[tokio::test]
async fn sglang_native_errors_reach_http_and_recover() {
    tokio::time::timeout(
        Duration::from_secs(60),
        native_errors_reach_http_and_recover::<sglang::Fixture>(),
    )
    .await
    .expect("HTTP error scenario exceeded its overall deadline");
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
    // Reserve the unavailable endpoint so another parallel mock cannot acquire it.
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let endpoint = format!("http://{}", socket.local_addr().unwrap());
    let mut child = env.spawn::<F>(&endpoint, DisaggregationMode::Aggregated, 1);
    assert!(!child.exit().await.success(), "{}", child.logs());
    assert!(env.cards().await.is_empty());
    assert!(env.registrations("backend").await.is_empty());
    drop(socket);
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
        assert_eq!(card.kv_cache_block_size, F::KV_BLOCK_SIZE);
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
        F::KV_BLOCK_SIZE,
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

async fn served_model_alias_is_published<F: DiscoveryFixture>() {
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
    // SGLang reports the parsers in GetModelInfo and the alias in GetServerInfo.
    let tokenizer_path = tokenizer.to_str().unwrap().to_string();
    peer.override_discovery(
        Some(Box::new(move |info| {
            info.tokenizer_path = tokenizer_path.clone();
            info.tool_call_parser = Some("hermes".into());
            info.reasoning_parser = Some("deepseek_r1".into());
        })),
        vec![Box::new(|info| {
            info.served_model_name = "public-alias".into()
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
    healthy::<sglang::Fixture>(&env, &router, &control, "distinct-metadata").await;
    child.shutdown().await;
    env.withdrawn("backend", &router).await;
    peer.shutdown().await;
}

async fn health_readiness_precedes_publication<F: DiscoveryFixture>() {
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

async fn unhealthy_worker_never_registers<F: DiscoveryFixture>() {
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
        None,
        vec![
            Box::new(|_| {}),
            Box::new(|info| info.disaggregation_mode = "prefill".into()),
        ],
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

/// Start the listeners of one SGLang node in `mode`.
async fn node(
    env: &Environment,
    control: &Controller<sglang::Adapter>,
    mode: DisaggregationMode,
    node: sglang::NodeConfig,
) -> Vec<sglang::Fixture> {
    sglang::Fixture::start_node(
        control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            disaggregation_mode: mode,
            ..Default::default()
        },
        node,
    )
    .await
}

async fn shutdown_node(node: &mut [sglang::Fixture]) {
    for listener in node {
        listener.shutdown().await;
    }
}

/// Wait until `env` has `count` model cards.
async fn worker_cards(env: &Environment, count: usize) {
    bounded("worker registrations", async {
        loop {
            if env.worker_cards().await.len() == count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
}

/// A PD deployment: `prefill` nodes and one decode listener, a sidecar child
/// per node, and a real round-robin PrefillRouter.
struct PdDeployment {
    env: Environment,
    prefill_control: Controller<sglang::Adapter>,
    decode_control: Controller<sglang::Adapter>,
    prefill: Vec<Vec<sglang::Fixture>>,
    decode: Vec<sglang::Fixture>,
    children: Vec<crate::process::Process>,
    prefill_router: Arc<crate::process::Router>,
    decode_router: Arc<crate::process::Router>,
    router: Arc<PrefillRouter>,
}

impl PdDeployment {
    async fn start(nodes: Vec<sglang::NodeConfig>) -> Self {
        let env = Environment::new().await;
        let prefill_control = Controller::default();
        let decode_control = Controller::default();
        let mut prefill = Vec::new();
        for config in nodes {
            prefill.push(node(&env, &prefill_control, DisaggregationMode::Prefill, config).await);
        }
        let decode = node(
            &env,
            &decode_control,
            DisaggregationMode::Decode,
            Default::default(),
        )
        .await;
        let mut children: Vec<_> = prefill
            .iter()
            .map(|listeners| {
                env.spawn::<sglang::Fixture>(
                    &listeners[0].endpoint(),
                    DisaggregationMode::Prefill,
                    5,
                )
            })
            .collect();
        children.push(env.spawn::<sglang::Fixture>(
            &decode[0].endpoint(),
            DisaggregationMode::Decode,
            5,
        ));
        let prefill_router = env.ready("prefill").await;
        let decode_router = env.ready("backend").await;
        worker_cards(&env, prefill.len() + 1).await;
        let (activation, activated) = tokio::sync::oneshot::channel();
        let router = PrefillRouter::new(
            activated,
            Arc::new(ModelManager::new()),
            RouterMode::RoundRobin,
            sglang::Fixture::KV_BLOCK_SIZE,
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
        Self {
            env,
            prefill_control,
            decode_control,
            prefill,
            decode,
            children,
            prefill_router,
            decode_router,
            router,
        }
    }

    /// The prefill workers by their first rank.
    async fn prefill_workers(&self) -> std::collections::BTreeMap<u32, u64> {
        self.env
            .worker_cards()
            .await
            .into_iter()
            .filter(|(_, card)| card.worker_type == Some(WorkerType::Prefill))
            .map(|(worker_id, card)| (card.runtime_config.data_parallel_start_rank, worker_id))
            .collect()
    }

    /// Hand one request through prefill to decode, and return what each leg
    /// put on the wire. A prefill worker with several DP ranks registers no
    /// bootstrap port, so the router dispatches decode only after the
    /// prefill's first response; the prefill is not held.
    async fn handoff(
        &self,
        id: &str,
        routing: Option<RoutingHints>,
    ) -> (
        dynamo_sglang_sidecar::proto::GenerateRequest,
        dynamo_sglang_sidecar::proto::GenerateRequest,
    ) {
        let prefill = self.prefill_control.request(id, RequestPlan::default());
        let decode = self.decode_control.request(id, RequestPlan::default());
        let mut request = self.env.request(id, 3);
        request.routing = routing;
        let stream = bounded(
            "prefill/decode handoff",
            self.router.generate(request, self.decode_router.clone()),
        )
        .await
        .unwrap();
        bounded("prefill native acceptance", prefill.wait(Event::Received)).await;
        bounded("decode native acceptance", decode.wait(Event::Received)).await;
        terminal(
            outputs(stream).await,
            &decode.tokens(),
            4,
            FinishReason::Length,
        );
        (
            prefill.native_request().unwrap(),
            decode.native_request().unwrap(),
        )
    }

    async fn shutdown(mut self) {
        for child in &mut self.children {
            child.shutdown().await;
        }
        self.env.withdrawn("backend", &self.decode_router).await;
        self.env.withdrawn("prefill", &self.prefill_router).await;
        for listeners in &mut self.prefill {
            shutdown_node(listeners).await;
        }
        shutdown_node(&mut self.decode).await;
    }
}

/// `(bootstrap_port, bootstrap_room)` on the wire.
fn port_and_room(wire: &dynamo_sglang_sidecar::proto::GenerateRequest) -> (i64, i64) {
    use dynamo_sglang_sidecar::proto as pb;
    let one = |value: Option<&pb::OptionalInt64OrList>| match value
        .and_then(|value| value.value.as_ref())
    {
        Some(pb::optional_int64_or_list::Value::One(value)) => value.value.unwrap(),
        other => panic!("expected one integer: {other:?}"),
    };
    (
        one(wire.bootstrap_port.as_ref()),
        one(wire.bootstrap_room.as_ref()),
    )
}

/// The prefill ran on one listener, and both legs carry that rank's bootstrap
/// port and a room congruent to it modulo `dp_size`; they name the rank only
/// when the router named it. Every node here has two listeners.
fn assert_named_rank(
    deployment: &PdDeployment,
    id: &str,
    dp_size: u32,
    router_named: bool,
    prefill: &dynamo_sglang_sidecar::proto::GenerateRequest,
    decode: &dynamo_sglang_sidecar::proto::GenerateRequest,
) -> u32 {
    let served: Vec<_> = (0..deployment.prefill.len())
        .flat_map(|node| (0..2).map(move |offset| (node, offset)))
        .filter(|&(node, offset)| {
            deployment.prefill[node][offset]
                .received()
                .contains(&id.to_string())
        })
        .collect();
    let [(node, offset)] = served[..] else {
        panic!("{id} reached prefill listeners {served:?}");
    };
    let rank = (2 * node + offset) as u32;
    let named = router_named.then_some(i64::from(rank));
    assert_eq!(prefill.disagg_prefill_dp_rank, named, "{id}");
    assert_eq!(decode.disagg_prefill_dp_rank, named, "{id}");
    let (port, room) = port_and_room(prefill);
    assert_eq!(port, 8998 + offset as i64, "{id}: the rank's own registry");
    assert_eq!(port_and_room(decode), (port, room));
    assert_eq!(room % i64::from(dp_size), i64::from(rank), "{id}");
    assert_eq!(prefill.routed_dp_rank, None);
    rank
}

// Built-in prefill routing picks a worker, not a rank. A worker with several
// DP ranks registers no bootstrap port, so the router takes each prefill's
// handoff, on either node; a pinned rank keeps its rank and port.
#[tokio::test]
async fn sglang_round_robin_prefill_hands_off_on_both_nodes() {
    tokio::time::timeout(Duration::from_secs(60), async {
        let deployment = PdDeployment::start(vec![
            sglang::NodeConfig {
                dp_size: Some(4),
                ..sglang::NodeConfig::listeners(2)
            },
            sglang::NodeConfig {
                dp_rank: 2,
                dp_size: Some(4),
                ..sglang::NodeConfig::listeners(2)
            },
        ])
        .await;
        let workers = deployment.prefill_workers().await;
        assert_eq!(workers.keys().copied().collect::<Vec<_>>(), [0, 2]);
        for card in deployment.env.cards().await {
            if card.worker_type == Some(WorkerType::Prefill) {
                assert_eq!(
                    card.runtime_config
                        .disaggregated_endpoint
                        .and_then(|endpoint| endpoint.bootstrap_port),
                    None
                );
            }
        }
        ready(&deployment.router).await;
        let mut ranks = std::collections::BTreeSet::new();
        for index in 0..16 {
            let id = format!("round-robin-{index}");
            let (prefill, decode) = deployment.handoff(&id, None).await;
            ranks.insert(assert_named_rank(
                &deployment,
                &id,
                4,
                false,
                &prefill,
                &decode,
            ));
            if ranks.len() == 2 {
                break;
            }
        }
        assert_eq!(ranks, std::collections::BTreeSet::from([0, 2]));
        // `x-dynamo-prefill-instance-id` + `x-dynamo-prefill-dp-rank`.
        let routing = RoutingHints {
            prefill_worker_id: Some(workers[&2]),
            prefill_dp_rank: Some(3),
            ..Default::default()
        };
        let (prefill, decode) = deployment.handoff("pinned-3", Some(routing)).await;
        assert_eq!(
            assert_named_rank(&deployment, "pinned-3", 4, true, &prefill, &decode),
            3
        );
        deployment.shutdown().await;
    })
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn trtllm_registration_and_errors_recover_through_worker_ingress() {
    tokio::time::timeout(
        Duration::from_secs(60),
        registration_and_errors_recover_through_worker_ingress::<support::trtllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn trtllm_delayed_startup_publishes_only_after_native_readiness() {
    tokio::time::timeout(
        Duration::from_secs(60),
        delayed_startup_publishes_only_after_native_readiness::<support::trtllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn trtllm_failed_and_interrupted_startup_leave_no_registration() {
    tokio::time::timeout(
        Duration::from_secs(60),
        failed_and_interrupted_startup_leave_no_registration::<support::trtllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn trtllm_worker_cancel_and_consumer_drop_release_only_the_target() {
    tokio::time::timeout(
        Duration::from_secs(60),
        worker_cancel_and_consumer_drop_release_only_the_target::<support::trtllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn trtllm_sigterm_withdraws_worker_and_releases_active_native_request() {
    tokio::time::timeout(
        Duration::from_secs(60),
        sigterm_withdraws_worker_and_releases_active_native_request::<support::trtllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn trtllm_prefill_router_preserves_handoff_failure_and_cancellation() {
    tokio::time::timeout(
        Duration::from_secs(60),
        prefill_router_preserves_handoff_failure_and_cancellation::<support::trtllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn trtllm_native_errors_reach_http_and_recover() {
    tokio::time::timeout(
        Duration::from_secs(60),
        native_errors_reach_http_and_recover::<support::trtllm::Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}
