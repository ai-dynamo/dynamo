// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{BackendError, DisaggregationMode, FinishReason};
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::pipeline::AsyncEngine;
use dynamo_sidecar_testkit::{
    assert::{failure, terminal},
    bounded,
    control::{Controller, Event, OpenAction, RequestPlan, StreamAction, StreamFault, StreamPoint},
};

use crate::process::{Environment, Gate, ProcessFixture, outputs};
use crate::support::FixtureConfig;

pub async fn healthy<F: ProcessFixture>(
    env: &Environment,
    router: &crate::process::Router,
    control: &Controller<F::Protocol>,
    id: &str,
) {
    bounded("router fault recovery", async {
        while router.selectable_worker_ids().is_err() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await;
    let handle = control.request(id, RequestPlan::default());
    let mut request = env.request(id, 3);
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
}

pub async fn registration_and_errors_recover_through_worker_ingress<F: ProcessFixture>() {
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

pub async fn delayed_startup_publishes_only_after_native_readiness<F: ProcessFixture>() {
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

pub async fn failed_and_interrupted_startup_leave_no_registration<F: ProcessFixture>() {
    let env = Environment::new().await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let mut child = env.spawn::<F>(&endpoint, DisaggregationMode::Aggregated, 1);
    assert!(!child.exit().await.success(), "{}", child.logs());
    assert!(env.cards().await.is_empty());
    assert!(env.registrations("backend").await.is_empty());
    for interrupt in [false, true] {
        let env = Environment::new().await;
        let mut gate = Gate::new("http://127.0.0.1:1").await;
        let mut child = env.spawn::<F>(&gate.endpoint, DisaggregationMode::Aggregated, 1);
        gate.accepted().await;
        if interrupt {
            child.signal(libc::SIGTERM);
        }
        assert!(
            !child.exit().await.success(),
            "startup unexpectedly succeeded\n{}",
            child.logs()
        );
        assert!(env.cards().await.is_empty());
        assert!(env.registrations("backend").await.is_empty());
        gate.shutdown().await;
    }
}
