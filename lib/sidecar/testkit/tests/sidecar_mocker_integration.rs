// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::testing::mock_context;
use dynamo_backend_common::{
    BackendError, DynamoError, FinishReason, GenerateContext, LLMEngine, LLMEngineOutput,
    PreprocessedRequest,
};
use dynamo_sidecar_testkit::assert::{failure, terminal};
use dynamo_sidecar_testkit::bounded;
use dynamo_sidecar_testkit::control::{
    Controller, Event, OpenAction, Protocol, RequestHandle, RequestPlan, StreamAction, StreamFault,
    StreamPoint,
};
use dynamo_sidecar_testkit::fixtures::{Outputs, collect, request};
use futures::{StreamExt, poll, stream::BoxStream};
use serde_json::{Value, json};

#[allow(dead_code)]
mod support;

use support::sglang::{self as sglang_fixture, http};
use support::vllm as vllm_fixture;
use support::{FixtureConfig, GenerateOpening, ProcessFixture, SidecarFixture, WireFixture};

fn after_token_responses(count: usize, action: StreamAction) -> RequestPlan {
    RequestPlan {
        stream: Some(StreamFault {
            at: StreamPoint::TokenResponse(count),
            action,
            pause: true,
        }),
        ..Default::default()
    }
}

async fn checkpoint_outputs<P: Protocol>(
    stream: &mut BoxStream<'static, Result<LLMEngineOutput, DynamoError>>,
    handle: &RequestHandle<P>,
) -> Outputs {
    bounded("consume native checkpoint prefix", async {
        let mut outputs = Vec::new();
        let mut tokens = Vec::new();
        let mut has_checkpoint = false;
        loop {
            if has_checkpoint {
                let expected = handle.tokens();
                assert!(!expected.is_empty());
                if tokens.len() >= expected.len() {
                    assert_eq!(tokens, expected);
                    return outputs;
                }
            }
            tokio::select! {
                _ = handle.wait(Event::Checkpoint), if !has_checkpoint => {
                    has_checkpoint = true;
                }
                output = stream.next() => {
                    let output = output.expect("stream ended before its checkpoint");
                    let chunk = output.as_ref().expect("checkpoint prefix must succeed");
                    assert!(chunk.finish_reason.is_none());
                    tokens.extend_from_slice(&chunk.token_ids);
                    outputs.push(output);
                }
            }
        }
    })
    .await
}

async fn healthy<F: SidecarFixture>(
    fixture: &F,
    engine: &F::Engine,
    control: &Controller<F::Protocol>,
) {
    let ctx = mock_context();
    let handle = control.request(ctx.id(), RequestPlan::default());
    let outputs = collect(
        engine,
        request("mocker-model", vec![11, 22, 33], 3),
        GenerateContext::new(ctx, None),
    )
    .await;
    terminal(outputs, &handle.tokens(), 3, FinishReason::Length);
    assert_eq!(handle.tokens().len(), 3);
    bounded("healthy remote drop", handle.wait(Event::Dropped)).await;
    fixture.scheduler_idle().await;
}

async fn finish<F: SidecarFixture>(fixture: &mut F, engine: &F::Engine) {
    bounded("sidecar cleanup", engine.cleanup()).await.unwrap();
    fixture.shutdown().await;
}

async fn streaming<F: SidecarFixture>() {
    let control = Controller::<F::Protocol>::default();
    let config = FixtureConfig {
        model: "alternate-mocker".into(),
        connections: 2,
        ..Default::default()
    };
    let mut fixture = F::start(control.clone(), config).await;
    let engine = fixture.engine().await;
    let metadata = engine.start(0).await.unwrap();
    assert_eq!(metadata.model, "alternate-mocker");
    let req = request("request-model", vec![10, 20, 30, 40, 50], 7);
    for plan in [
        RequestPlan::default(),
        RequestPlan {
            stream: Some(StreamFault {
                at: StreamPoint::Terminal,
                action: StreamAction::ReplayFirst,
                pause: false,
            }),
            ..Default::default()
        },
    ] {
        let ctx = mock_context();
        let handle = control.request(ctx.id(), plan);
        let outputs = collect(&engine, req.clone(), GenerateContext::new(ctx, None)).await;
        let native = handle.native_request().expect("native request");
        if let Some(model) = F::native_model(&native) {
            assert_eq!(model, metadata.model);
        }
        let tokens = handle.tokens();
        assert_eq!(
            tokens.len(),
            req.stop_conditions.max_tokens.unwrap() as usize
        );
        terminal(
            outputs,
            &tokens,
            req.token_ids.len() as u32,
            FinishReason::Length,
        );
        bounded("server stream drop", handle.wait(Event::Dropped)).await;
        assert_eq!(handle.reached(Event::Checkpoint), plan.stream.is_some());
        fixture.scheduler_idle().await;
    }
    finish(&mut fixture, &engine).await;
}

async fn failures<F: SidecarFixture>() {
    let control = Controller::<F::Protocol>::default();
    let mut fixture = F::start(control.clone(), FixtureConfig::default()).await;
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    for (plan, kind) in [
        (
            RequestPlan {
                open: OpenAction::Fail,
                ..Default::default()
            },
            BackendError::CannotConnect,
        ),
        (
            after_token_responses(1, StreamAction::Close),
            F::eof_error(),
        ),
        (
            after_token_responses(1, StreamAction::Fail),
            BackendError::CannotConnect,
        ),
    ] {
        let ctx = mock_context();
        let is_open_failure = matches!(plan.open, OpenAction::Fail);
        let handle = control.request(ctx.id(), plan);
        let req = request("mocker-model", vec![11, 22, 33], 6);
        let ctx = GenerateContext::new(ctx, None);
        let outputs = if is_open_failure {
            collect(&engine, req, ctx).await
        } else {
            let mut stream = engine.generate(req, ctx).await.unwrap();
            let mut outputs = checkpoint_outputs(&mut stream, &handle).await;
            handle.release();
            outputs.extend(bounded("failed stream completion", stream.collect::<Outputs>()).await);
            outputs
        };
        let tokens = handle.tokens();
        assert_eq!(tokens.is_empty(), is_open_failure);
        let error = failure(outputs, &tokens, kind);
        if kind == BackendError::CannotConnect {
            assert!(error.to_string().contains("injected"));
            assert!(error.to_string().contains("Generate"));
        }
        bounded("failed remote drop", handle.wait(Event::Dropped)).await;
        fixture.scheduler_idle().await;
        healthy(&fixture, &engine, &control).await;
    }
    finish(&mut fixture, &engine).await;
}

async fn cancellation<F: SidecarFixture>() {
    let control = Controller::<F::Protocol>::default();
    let mut fixture = F::start(control.clone(), FixtureConfig::default()).await;
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();

    let ctx = mock_context();
    let handle = control.request(ctx.id(), RequestPlan::default());
    ctx.stop_generating();
    terminal(
        collect(
            &engine,
            request("mocker-model", vec![11, 22, 33], 3),
            GenerateContext::new(ctx, None),
        )
        .await,
        &[],
        3,
        FinishReason::Cancelled,
    );
    assert!(!handle.reached(Event::Received));
    fixture.scheduler_idle().await;

    for cancellation in [Cancellation::Explicit, Cancellation::ConsumerDrop] {
        let ctx = mock_context();
        let handle = control.request(
            ctx.id(),
            RequestPlan {
                open: OpenAction::Hold,
                ..Default::default()
            },
        );
        let mut opening = Box::pin(engine.generate(
            request("mocker-model", vec![11, 22, 33], 3),
            GenerateContext::new(ctx.clone(), None),
        ));
        let outputs = match F::GENERATE_OPENING {
            GenerateOpening::WaitsForHeaders => {
                tokio::select! {
                    _ = bounded("pending native headers", handle.wait(Event::Received)) => {}
                    _ = &mut opening => panic!("generate returned before native headers"),
                }
                assert!(poll!(&mut opening).is_pending());
                match cancellation {
                    Cancellation::Explicit => {
                        ctx.stop_generating();
                        Some(
                            bounded("cancel pending opening", async {
                                opening.await.unwrap().collect::<Outputs>().await
                            })
                            .await,
                        )
                    }
                    Cancellation::ConsumerDrop => {
                        drop(opening);
                        None
                    }
                }
            }
            GenerateOpening::OnStreamPoll => {
                let stream = bounded("lazy generation opening", opening).await.unwrap();
                assert!(!handle.reached(Event::Received));
                let mut outputs = Box::pin(stream.collect::<Outputs>());
                tokio::select! {
                    _ = bounded("lazy native opening", handle.wait(Event::Received)) => {}
                    _ = &mut outputs => panic!("generation finished before cancellation"),
                }
                match cancellation {
                    Cancellation::Explicit => {
                        ctx.stop_generating();
                        Some(bounded("cancel lazy opening", outputs).await)
                    }
                    Cancellation::ConsumerDrop => {
                        drop(outputs);
                        None
                    }
                }
            }
        };
        if let Some(outputs) = outputs {
            terminal(outputs, &[], 3, FinishReason::Cancelled);
        }
        bounded("opening drop", handle.wait(Event::Dropped)).await;
        fixture.scheduler_idle().await;
    }
    read_isolation(&fixture, &engine, &control, Cancellation::Explicit).await;
    healthy(&fixture, &engine, &control).await;
    finish(&mut fixture, &engine).await;
}

#[derive(Clone, Copy)]
enum Cancellation {
    Explicit,
    ConsumerDrop,
}

async fn read_isolation<F: SidecarFixture>(
    fixture: &F,
    engine: &F::Engine,
    control: &Controller<F::Protocol>,
    cancellation: Cancellation,
) {
    let ctx_a = mock_context();
    let handle_a = control.request(ctx_a.id(), after_token_responses(2, StreamAction::Continue));
    let mut stream_a = engine
        .generate(
            request("mocker-model", vec![11, 22, 33], 8),
            GenerateContext::new(ctx_a.clone(), None),
        )
        .await
        .unwrap();
    let mut outputs_a = checkpoint_outputs(&mut stream_a, &handle_a).await;
    let tokens_a = handle_a.tokens();

    let ctx_b = mock_context();
    let handle_b = control.request(ctx_b.id(), after_token_responses(1, StreamAction::Continue));
    let mut stream_b = engine
        .generate(
            request("mocker-model", vec![90, 80, 70, 60, 50], 5),
            GenerateContext::new(ctx_b, None),
        )
        .await
        .unwrap();
    let mut outputs_b = checkpoint_outputs(&mut stream_b, &handle_b).await;
    let tokens_b = handle_b.tokens();
    assert!(poll!(stream_a.next()).is_pending());
    assert!(poll!(stream_b.next()).is_pending());

    match cancellation {
        Cancellation::ConsumerDrop => drop(stream_a),
        Cancellation::Explicit => {
            ctx_a.stop_generating();
            outputs_a
                .extend(bounded("request A cancellation", stream_a.collect::<Outputs>()).await);
            terminal(outputs_a, &tokens_a, 3, FinishReason::Cancelled);
        }
    }
    bounded("request A drop", handle_a.wait(Event::Dropped)).await;
    assert_eq!(handle_a.tokens(), tokens_a);
    assert!(!handle_b.reached(Event::Dropped));
    assert_eq!(handle_b.tokens(), tokens_b);
    assert!(poll!(stream_b.next()).is_pending());
    handle_b.release();
    outputs_b.extend(bounded("request B completion", stream_b.collect::<Outputs>()).await);
    assert_eq!(handle_b.tokens().len(), 5);
    terminal(outputs_b, &handle_b.tokens(), 5, FinishReason::Length);
    bounded("request B drop", handle_b.wait(Event::Dropped)).await;
    fixture.scheduler_idle().await;
}

async fn cleanup<F: SidecarFixture>() {
    let control = Controller::<F::Protocol>::default();
    let mut fixture = F::start(control.clone(), FixtureConfig::default()).await;
    {
        let engine = fixture.engine().await;
        let ctx = mock_context();
        let handle = control.request(ctx.id(), RequestPlan::default());
        failure(
            collect(
                &engine,
                request("mocker-model", vec![11, 22, 33], 3),
                GenerateContext::new(ctx, None),
            )
            .await,
            &[],
            BackendError::EngineShutdown,
        );
        engine.cleanup().await.unwrap();
        engine.cleanup().await.unwrap();
        assert!(!handle.reached(Event::Received));
        fixture.scheduler_idle().await;
    }
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    let ctx = mock_context();
    let handle = control.request(ctx.id(), after_token_responses(1, StreamAction::Continue));
    let mut stream = engine
        .generate(
            request("mocker-model", vec![11, 22, 33], 6),
            GenerateContext::new(ctx, None),
        )
        .await
        .unwrap();
    let mut outputs = checkpoint_outputs(&mut stream, &handle).await;
    assert!(poll!(stream.next()).is_pending());
    engine.cleanup().await.unwrap();
    engine.cleanup().await.unwrap();
    outputs.extend(bounded("cleanup cancellation", stream.collect::<Outputs>()).await);
    terminal(outputs, &handle.tokens(), 3, FinishReason::Cancelled);
    bounded("cleanup remote drop", handle.wait(Event::Dropped)).await;

    let ctx = mock_context();
    let unsubmitted = control.request(ctx.id(), RequestPlan::default());
    terminal(
        collect(
            &engine,
            request("mocker-model", vec![11, 22, 33], 3),
            GenerateContext::new(ctx, None),
        )
        .await,
        &[],
        3,
        FinishReason::Cancelled,
    );
    assert!(!unsubmitted.reached(Event::Received));
    fixture.scheduler_idle().await;
    let recovered = fixture.engine().await;
    recovered.start(0).await.unwrap();
    healthy(&fixture, &recovered, &control).await;
    finish(&mut fixture, &recovered).await;
}

async fn active_work<F: WireFixture>(cancellation: Cancellation) {
    let control = Controller::<F::Protocol>::default();
    let mut fixture = F::start(
        control.clone(),
        FixtureConfig {
            speedup_ratio: 0.1,
            connections: 2,
            ..Default::default()
        },
    )
    .await;
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    let ctx = mock_context();
    let handle = control.request(ctx.id(), after_token_responses(1, StreamAction::Continue));
    let mut stream = engine
        .generate(
            request("mocker-model", vec![11, 22, 33], 10_000),
            GenerateContext::new(ctx.clone(), None),
        )
        .await
        .unwrap();
    let mut outputs = checkpoint_outputs(&mut stream, &handle).await;
    let tokens = handle.tokens();
    fixture.scheduler_active().await;
    if matches!(cancellation, Cancellation::Explicit) {
        ctx.stop_generating();
        outputs.extend(
            bounded(
                "active cancellation terminal",
                stream.by_ref().collect::<Outputs>(),
            )
            .await,
        );
        terminal(outputs, &tokens, 3, FinishReason::Cancelled);
    }
    drop(stream);
    bounded("active request remote drop", handle.wait(Event::Dropped)).await;
    assert_eq!(handle.tokens(), tokens);
    fixture.scheduler_idle().await;
    healthy(&fixture, &engine, &control).await;
    if matches!(cancellation, Cancellation::ConsumerDrop) {
        read_isolation(&fixture, &engine, &control, Cancellation::ConsumerDrop).await;
        healthy(&fixture, &engine, &control).await;
    }
    finish(&mut fixture, &engine).await;
}

async fn cancellation_burst<F: WireFixture>() {
    let control = Controller::<F::Protocol>::default();
    let mut fixture = F::start(
        control.clone(),
        FixtureConfig {
            speedup_ratio: 0.1,
            connections: 2,
            ..Default::default()
        },
    )
    .await;
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    let requests = futures::future::join_all((0..8).map(|_| async {
        let ctx = mock_context();
        let handle = control.request(ctx.id(), after_token_responses(1, StreamAction::Continue));
        let mut stream = engine
            .generate(
                request("mocker-model", vec![11, 22, 33], 1_024),
                GenerateContext::new(ctx.clone(), None),
            )
            .await
            .unwrap();
        let outputs = checkpoint_outputs(&mut stream, &handle).await;
        (ctx, handle, stream, outputs)
    }))
    .await;
    assert_eq!(fixture.active_request_count(), requests.len());
    fixture.scheduler_active().await;
    futures::future::join_all(requests.into_iter().enumerate().map(
        |(index, (ctx, handle, stream, mut outputs))| async move {
            let tokens = handle.tokens();
            if index % 2 == 0 {
                ctx.stop_generating();
                outputs.extend(stream.collect::<Outputs>().await);
                terminal(outputs, &tokens, 3, FinishReason::Cancelled);
            } else {
                drop(stream);
            }
            handle.wait(Event::Dropped).await;
            assert_eq!(handle.tokens(), tokens);
        },
    ))
    .await;
    fixture.scheduler_idle().await;
    healthy(&fixture, &engine, &control).await;
    finish(&mut fixture, &engine).await;
}

async fn teardown_with_live_clients<F: WireFixture>() -> (Outputs, Vec<u32>) {
    let control = Controller::<F::Protocol>::default();
    let mut fixture = F::start(
        control.clone(),
        FixtureConfig {
            speedup_ratio: 0.1,
            ..Default::default()
        },
    )
    .await;
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    let ctx = mock_context();
    let handle = control.request(ctx.id(), after_token_responses(1, StreamAction::Continue));
    let mut stream = engine
        .generate(
            request("mocker-model", vec![11, 22, 33], 10_000),
            GenerateContext::new(ctx, None),
        )
        .await
        .unwrap();
    let mut outputs = checkpoint_outputs(&mut stream, &handle).await;
    fixture.scheduler_active().await;
    fixture.shutdown().await;
    bounded("owned RPC handler terminated", handle.wait(Event::Dropped)).await;
    fixture.scheduler_idle().await;
    outputs.extend(
        bounded(
            "client observes server termination",
            stream.collect::<Outputs>(),
        )
        .await,
    );
    engine.cleanup().await.unwrap();
    (outputs, handle.tokens())
}

macro_rules! enroll_baseline {
    ($backend:ident, $fixture:ty) => {
        mod $backend {
            use super::*;

            #[tokio::test]
            async fn stream_tokens_terminal_and_usage() {
                bounded("streaming conformance", streaming::<$fixture>()).await;
            }

            #[tokio::test]
            async fn open_failure_early_eof_and_read_failure() {
                bounded("failure conformance", failures::<$fixture>()).await;
            }

            #[tokio::test]
            async fn cancellation_before_open_during_open_and_during_read() {
                bounded("cancellation conformance", cancellation::<$fixture>()).await;
            }

            #[tokio::test]
            async fn cleanup_before_start_during_read_and_after_shutdown() {
                bounded("cleanup conformance", cleanup::<$fixture>()).await;
            }

            #[tokio::test]
            async fn explicit_cancel_releases_active_scheduler_work() {
                bounded(
                    "active cancellation and recovery",
                    active_work::<$fixture>(Cancellation::Explicit),
                )
                .await;
            }

            #[tokio::test]
            async fn consumer_drop_releases_active_scheduler_work() {
                bounded(
                    "consumer drop and recovery",
                    active_work::<$fixture>(Cancellation::ConsumerDrop),
                )
                .await;
            }

            #[tokio::test]
            async fn cancellation_burst_reuses_connection_pool() {
                bounded(
                    "cancellation burst and recovery",
                    cancellation_burst::<$fixture>(),
                )
                .await;
            }

            #[tokio::test]
            async fn request_fields_logprobs_and_usage() {
                bounded("request fields and responses", request_fields::<$fixture>()).await;
            }
        }
    };
}

enroll_baseline!(vllm, support::vllm::Fixture);
enroll_baseline!(sglang, support::sglang::Fixture);

#[tokio::test]
async fn vllm_teardown_terminates_handlers_with_clients_alive() {
    let (outputs, tokens) = bounded(
        "server teardown with live clients",
        teardown_with_live_clients::<vllm_fixture::Fixture>(),
    )
    .await;
    let error = failure(outputs, &tokens, BackendError::Unknown);
    assert!(error.to_string().contains("GenerateStream"));
}

async fn request_fields<F: ProcessFixture>() {
    let control = Controller::<F::Protocol>::default();
    let mut fixture = F::start(control.clone(), FixtureConfig::default()).await;
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    let mut req = request("mocker-model", vec![10, 20, 30, 40, 50], 7);
    F::configure_request(&mut req);
    let ctx = mock_context();
    let handle = control.request(ctx.id(), RequestPlan::default());
    let outputs = collect(&engine, req.clone(), GenerateContext::new(ctx, None)).await;
    F::assert_stream(&handle, &req, &outputs);
    terminal(
        outputs,
        &handle.tokens(),
        req.token_ids.len() as u32,
        FinishReason::Length,
    );
    bounded("wire fields remote drop", handle.wait(Event::Dropped)).await;
    fixture.scheduler_idle().await;
    finish(&mut fixture, &engine).await;
}

async fn native_rejection<F: WireFixture>() -> DynamoError {
    let control = Controller::<F::Protocol>::default();
    let mut fixture = F::start(control.clone(), FixtureConfig::default()).await;
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    let ctx = mock_context();
    let handle = control.request(ctx.id(), RequestPlan::default());
    let opening = engine
        .generate(
            request("mocker-model", vec![1, 2, 3], 32_769),
            GenerateContext::new(ctx, None),
        )
        .await;
    let outputs = match F::GENERATE_OPENING {
        GenerateOpening::WaitsForHeaders => vec![Err(opening
            .err()
            .expect("native rejection must fail before a response stream opens"))],
        GenerateOpening::OnStreamPoll => {
            bounded("native rejection", opening.unwrap().collect::<Outputs>()).await
        }
    };
    let error = failure(outputs, &[], BackendError::InvalidArgument);
    bounded("rejected request drop", handle.wait(Event::Dropped)).await;
    fixture.scheduler_idle().await;
    healthy(&fixture, &engine, &control).await;
    finish(&mut fixture, &engine).await;
    error
}

#[tokio::test]
async fn vllm_native_rejection_recovers_on_same_engine() {
    let error = bounded(
        "native rejection and recovery",
        native_rejection::<vllm_fixture::Fixture>(),
    )
    .await;
    assert!(error.to_string().contains("GenerateStream"));
    assert!(
        error
            .to_string()
            .contains("max_new_tokens must not exceed 32768")
    );
}

#[tokio::test]
async fn sglang_native_rejection_recovers_on_same_engine() {
    let error = bounded(
        "native rejection and recovery",
        native_rejection::<support::sglang::Fixture>(),
    )
    .await;
    assert!(error.to_string().contains("Generate"));
    assert!(
        error
            .to_string()
            .contains("prompt tokens (3) plus max_new_tokens (32769) exceed context_length 32768")
    );
}

#[tokio::test]
async fn sglang_teardown_terminates_handlers_with_clients_alive() {
    let (outputs, tokens) = bounded(
        "server teardown with live clients",
        teardown_with_live_clients::<support::sglang::Fixture>(),
    )
    .await;
    let error = failure(outputs, &tokens, BackendError::Unknown);
    assert!(error.to_string().contains("Generate"));
}

#[tokio::test]
async fn vllm_malformed_terminal_fails_then_recovers() {
    bounded("malformed terminal and recovery", async {
        use dynamo_vllm_sidecar::proto as pb;
        let control = Controller::<vllm_fixture::Adapter>::default();
        let mut fixture =
            vllm_fixture::Fixture::start(control.clone(), FixtureConfig::default()).await;
        let engine = fixture.engine().await;
        engine.start(0).await.unwrap();
        let ctx = mock_context();
        let handle = control.request(ctx.id(), RequestPlan::default());
        fixture.respond(
            ctx.id(),
            vec![
                pb::GenerateResponse {
                    outputs: Some(pb::SequenceOutput {
                        token_ids: vec![101],
                        num_tokens: 1,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                pb::GenerateResponse {
                    outputs: Some(pb::SequenceOutput {
                        finish_info: Some(pb::FinishInfo {
                            num_output_tokens: 1,
                            finish_reason: 999,
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
        );
        let error = failure(
            collect(
                &engine,
                request("mocker-model", vec![1, 2, 3], 3),
                GenerateContext::new(ctx, None),
            )
            .await,
            &[101],
            BackendError::Unknown,
        );
        assert!(error.to_string().contains("unknown finish reason 999"));
        bounded("malformed RPC released", handle.wait(Event::Dropped)).await;
        healthy(&fixture, &engine, &control).await;
        engine.cleanup().await.unwrap();
        fixture.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn sglang_malformed_terminal_fails_then_recovers() {
    bounded("malformed terminal and recovery", async {
        use dynamo_sglang_sidecar::proto as pb;
        let control = Controller::<support::sglang::Adapter>::default();
        let mut fixture =
            support::sglang::Fixture::start(control.clone(), FixtureConfig::default()).await;
        let engine = fixture.engine().await;
        engine.start(0).await.unwrap();
        let ctx = mock_context();
        let handle = control.request(ctx.id(), RequestPlan::default());
        fixture.respond(
            ctx.id(),
            vec![
                pb::GenerateResponse {
                    output_ids: vec![101],
                    ..Default::default()
                },
                pb::GenerateResponse {
                    finished: true,
                    ..Default::default()
                },
            ],
        );
        let error = failure(
            collect(
                &engine,
                request("mocker-model", vec![1, 2, 3], 3),
                GenerateContext::new(ctx, None),
            )
            .await,
            &[101],
            BackendError::Unknown,
        );
        assert!(error.to_string().contains("missing finish_reason"));
        bounded("malformed RPC released", handle.wait(Event::Dropped)).await;
        healthy(&fixture, &engine, &control).await;
        finish(&mut fixture, &engine).await;
    })
    .await;
}

fn http_request() -> PreprocessedRequest {
    let mut req = request("mocker-model", vec![11, 22, 33], 2);
    req.extra_args = Some(json!({"sglang_tito": {
        "sampling_params": {"max_new_tokens": 2, "custom_sampling": [0.5, null]},
        "future_field": {"opaque": [true, "unchanged"]}
    }}));
    req
}

fn assert_http_outputs(outputs: Outputs, request_id: &str) {
    let outputs: Vec<_> = outputs.into_iter().collect::<Result<_, _>>().unwrap();
    let expected = http::responses(request_id);
    assert_eq!(outputs.len(), expected.len());
    for (output, expected) in outputs.iter().zip(expected) {
        assert_eq!(
            output.engine_data,
            Some(json!({"sglang_response": expected}))
        );
        assert!(output.token_ids.is_empty());
    }
    assert!(outputs[0].finish_reason.is_none());
    assert_eq!(outputs[1].finish_reason, Some(FinishReason::Stop));
}

#[tokio::test]
async fn sglang_http_discovery_dispatch_and_opaque_payload() {
    let control = Controller::<sglang_fixture::Adapter>::default();
    let mut fixture =
        sglang_fixture::Fixture::start(control.clone(), FixtureConfig::default()).await;
    let http_control = Controller::<http::Adapter>::default();
    let mut http = http::Fixture::start(http_control.clone(), "200 OK").await;
    fixture.override_discovery(
        Value::Null,
        vec![json!({
            "port": http.port(), "incremental_streaming_output": true
        })],
    );
    let engine = fixture.engine().await;
    let metadata = bounded("HTTP discovery", engine.start(0)).await.unwrap();
    assert_eq!(
        metadata.runtime_data.get("sglang_generate"),
        Some(&json!(true))
    );

    let mut req = http_request();
    sglang_fixture::Fixture::configure_request(&mut req);
    req.routing.as_mut().unwrap().priority = Some(7);
    let mut expected = req.extra_args.as_ref().unwrap()["sglang_tito"].clone();
    req.extra_args.as_mut().unwrap()["sglang_tito"]["input_ids"] = json!([999]);
    req.extra_args.as_mut().unwrap()["sglang_tito"]["rid"] = json!("stale");
    req.extra_args.as_mut().unwrap()["sglang_tito"]["bootstrap_host"] = json!("stale");
    let ctx = mock_context();
    let id = ctx.id().to_string();
    expected["input_ids"] = json!(req.token_ids);
    expected["rid"] = json!(id);
    expected["stream"] = json!(true);
    expected["priority"] = json!(7);
    expected["routed_dp_rank"] = json!(3);
    expected["lora_path"] = json!("test-adapter");
    let native = http_control.request(&id, RequestPlan::default());
    let grpc = control.request(&id, RequestPlan::default());
    let (first_token, first_token_seen) = tokio::sync::watch::channel(false);
    let outputs = collect(&engine, req, GenerateContext::new(ctx, Some(first_token))).await;
    assert_http_outputs(outputs, &id);
    assert!(*first_token_seen.borrow());
    assert_eq!(native.native_request(), Some(expected));
    assert!(!grpc.reached(Event::Received));
    bounded("HTTP completion", native.wait(Event::Dropped)).await;
    healthy(&fixture, &engine, &control).await;
    assert!(http.aborted_requests().is_empty());
    finish(&mut fixture, &engine).await;
    http.shutdown().await;
}

#[tokio::test]
async fn sglang_http_unavailable_preserves_grpc() {
    let http_control = Controller::<http::Adapter>::default();
    let mut http = http::Fixture::start(http_control.clone(), "503 Service Unavailable").await;
    for server_info in [
        json!({}),
        json!({"port": http.port(), "incremental_streaming_output": true}),
    ] {
        let control = Controller::<sglang_fixture::Adapter>::default();
        let mut fixture =
            sglang_fixture::Fixture::start(control.clone(), FixtureConfig::default()).await;
        fixture.override_discovery(Value::Null, vec![server_info]);
        let engine = fixture.engine().await;
        let metadata = bounded("unavailable HTTP discovery", engine.start(0))
            .await
            .unwrap();
        assert!(!metadata.runtime_data.contains_key("sglang_generate"));
        let ctx = mock_context();
        let native = http_control.request(ctx.id(), RequestPlan::default());
        let grpc = control.request(ctx.id(), RequestPlan::default());
        let error = failure(
            collect(&engine, http_request(), GenerateContext::new(ctx, None)).await,
            &[],
            BackendError::InvalidArgument,
        );
        assert!(
            error
                .to_string()
                .contains("no ready incremental HTTP endpoint")
        );
        assert!(!native.reached(Event::Received));
        assert!(!grpc.reached(Event::Received));
        healthy(&fixture, &engine, &control).await;
        finish(&mut fixture, &engine).await;
    }
    http.shutdown().await;
}

async fn http_checkpoint(
    stream: &mut BoxStream<'static, Result<LLMEngineOutput, DynamoError>>,
    handle: &RequestHandle<http::Adapter>,
) -> Outputs {
    let output = bounded("HTTP first response", stream.next()).await.unwrap();
    assert!(output.as_ref().unwrap().finish_reason.is_none());
    bounded("HTTP checkpoint", handle.wait(Event::Checkpoint)).await;
    let native = handle.native_request().unwrap();
    assert_eq!(
        output.as_ref().unwrap().engine_data,
        Some(json!({"sglang_response": http::responses(native["rid"].as_str().unwrap())[0]}))
    );
    assert!(poll!(stream.next()).is_pending());
    vec![output]
}

#[tokio::test]
async fn sglang_http_cancellation_and_drop_isolate_requests() {
    http_cancellation_and_drop_isolate_requests(after_token_responses(1, StreamAction::Continue))
        .await;
}

#[tokio::test]
async fn sglang_http_cancellation_and_drop_before_headers_isolate_requests() {
    http_cancellation_and_drop_isolate_requests(RequestPlan {
        open: OpenAction::Hold,
        ..Default::default()
    })
    .await;
}

async fn http_cancellation_and_drop_isolate_requests(plan: RequestPlan) {
    let control = Controller::<sglang_fixture::Adapter>::default();
    let mut fixture =
        sglang_fixture::Fixture::start(control.clone(), FixtureConfig::default()).await;
    let http_control = Controller::<http::Adapter>::default();
    let mut http = http::Fixture::start(http_control.clone(), "200 OK").await;
    fixture.override_discovery(
        Value::Null,
        vec![json!({
            "port": http.port(), "incremental_streaming_output": true
        })],
    );
    let engine = fixture.engine().await;
    bounded("HTTP discovery", engine.start(0)).await.unwrap();
    let mut aborted = Vec::new();
    for cancellation in [Cancellation::Explicit, Cancellation::ConsumerDrop] {
        let ctx_a = mock_context();
        let handle_a = http_control.request(ctx_a.id(), plan);
        let mut stream_a = engine
            .generate(http_request(), GenerateContext::new(ctx_a.clone(), None))
            .await
            .unwrap();
        let mut outputs_a = if matches!(plan.open, OpenAction::Hold) {
            tokio::select! {
                _ = bounded("HTTP request accepted before headers", handle_a.wait(Event::Received)) => {}
                output = stream_a.next() => panic!("response arrived before headers: {output:?}"),
            }
            assert!(poll!(stream_a.next()).is_pending());
            Vec::new()
        } else {
            http_checkpoint(&mut stream_a, &handle_a).await
        };
        let tokens_a = handle_a.tokens();
        let ctx_b = mock_context();
        let id_b = ctx_b.id().to_string();
        let handle_b =
            http_control.request(&id_b, after_token_responses(1, StreamAction::Continue));
        let mut stream_b = engine
            .generate(http_request(), GenerateContext::new(ctx_b, None))
            .await
            .unwrap();
        let mut outputs_b = http_checkpoint(&mut stream_b, &handle_b).await;
        healthy(&fixture, &engine, &control).await;

        match cancellation {
            Cancellation::Explicit => {
                ctx_a.stop_generating();
                outputs_a.push(bounded("HTTP cancellation", stream_a.next()).await.unwrap());
                failure(outputs_a, &[], BackendError::Cancelled);
                http.wait_aborted(ctx_a.id()).await;
                drop(stream_a);
            }
            Cancellation::ConsumerDrop => drop(stream_a),
        }
        bounded("HTTP transport released", handle_a.wait(Event::Dropped)).await;
        http.wait_aborted(ctx_a.id()).await;
        aborted.push(ctx_a.id().to_string());
        assert_eq!(http.aborted_requests(), aborted);
        assert_eq!(handle_a.tokens(), tokens_a);
        assert!(!handle_b.reached(Event::Dropped));
        assert!(poll!(stream_b.next()).is_pending());
        handle_b.release();
        outputs_b.push(
            bounded("HTTP survivor terminal", stream_b.next())
                .await
                .unwrap(),
        );
        drop(stream_b);
        assert_http_outputs(outputs_b, &id_b);
        bounded("HTTP survivor released", handle_b.wait(Event::Dropped)).await;

        let ctx = mock_context();
        let id = ctx.id().to_string();
        assert_http_outputs(
            collect(&engine, http_request(), GenerateContext::new(ctx, None)).await,
            &id,
        );
        healthy(&fixture, &engine, &control).await;
        assert_eq!(http.aborted_requests(), aborted);
    }
    finish(&mut fixture, &engine).await;
    http.shutdown().await;
}
