// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::testing::mock_context;
use dynamo_backend_common::{
    BackendError, DisaggregationMode, DynamoError, FinishReason, GenerateContext, LLMEngine,
    LLMEngineOutput, PreprocessedRequest,
};
use dynamo_llm::protocols::common::preprocessor::{PrefillResult, RoutingHints};
use dynamo_sidecar_testkit::assert::{failure, terminal};
use dynamo_sidecar_testkit::bounded;
use dynamo_sidecar_testkit::control::{
    Controller, Event, OpenAction, Protocol, RequestHandle, RequestPlan, StreamAction, StreamFault,
    StreamPoint,
};
use dynamo_sidecar_testkit::fixtures::{Outputs, collect, request};
use futures::{StreamExt, poll, stream::BoxStream};
use serde_json::json;

#[allow(dead_code)]
mod support;

use support::sglang::{self as sglang_fixture, http};
use support::vllm as vllm_fixture;
use support::{
    DiscoveryFixture, FixtureConfig, GenerateOpening, ProcessFixture, SidecarFixture, WireFixture,
};

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

    {
        let ctx = mock_context();
        let handle = control.request(
            ctx.id(),
            RequestPlan {
                open: OpenAction::Hold,
                ..Default::default()
            },
        );
        let opening = engine.generate(
            request("mocker-model", vec![11, 22, 33], 3),
            GenerateContext::new(ctx.clone(), None),
        );
        tokio::pin!(opening);
        let outputs = match F::GENERATE_OPENING {
            GenerateOpening::WaitsForHeaders => {
                tokio::select! {
                    _ = bounded("pending native headers", handle.wait(Event::Received)) => {}
                    _ = &mut opening => panic!("generate returned before native headers"),
                }
                assert!(poll!(&mut opening).is_pending());
                ctx.stop_generating();
                bounded("cancel pending opening", async {
                    opening.await.unwrap().collect::<Outputs>().await
                })
                .await
            }
            GenerateOpening::OnStreamPoll => {
                let stream = bounded("lazy generation opening", opening).await.unwrap();
                assert!(!handle.reached(Event::Received));
                let outputs = stream.collect::<Outputs>();
                tokio::pin!(outputs);
                tokio::select! {
                    _ = bounded("lazy native opening", handle.wait(Event::Received)) => {}
                    _ = &mut outputs => panic!("generation finished before cancellation"),
                }
                ctx.stop_generating();
                bounded("cancel lazy opening", outputs).await
            }
        };
        terminal(outputs, &[], 3, FinishReason::Cancelled);
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
            async fn request_fields_logprobs_and_usage() {
                bounded("request fields and responses", request_fields::<$fixture>()).await;
            }
        }
    };
}

enroll_baseline!(vllm, support::vllm::Fixture);
enroll_baseline!(sglang, support::sglang::Fixture);
enroll_baseline!(trtllm, support::trtllm::Fixture);

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

async fn native_rejection<F: WireFixture>(max_tokens: u32) -> DynamoError {
    let control = Controller::<F::Protocol>::default();
    let mut fixture = F::start(control.clone(), FixtureConfig::default()).await;
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    let ctx = mock_context();
    let handle = control.request(ctx.id(), RequestPlan::default());
    let opening = engine
        .generate(
            request("mocker-model", vec![1, 2, 3], max_tokens),
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
        native_rejection::<vllm_fixture::Fixture>(1_000_001),
    )
    .await;
    assert!(error.to_string().contains("GenerateStream"));
    assert!(
        error
            .to_string()
            .contains("max_new_tokens must not exceed 1000000")
    );
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
async fn sglang_shutdown_releases_pending_health_check() {
    use dynamo_sglang_sidecar::proto::{
        HealthCheckRequest, sglang_service_client::SglangServiceClient,
    };

    let mut fixture =
        sglang_fixture::Fixture::start(Controller::default(), FixtureConfig::default()).await;
    fixture.set_health(None);
    let mut client = bounded(
        "connect health client",
        SglangServiceClient::connect(fixture.endpoint()),
    )
    .await
    .unwrap();
    let health = client.health_check(HealthCheckRequest {});
    tokio::pin!(health);
    tokio::select! {
        result = &mut health => panic!("health check completed before shutdown: {result:?}"),
        _ = fixture.health_check_received() => {},
    }
    fixture.shutdown().await;
    assert!(
        bounded("pending health check terminated", health)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn sglang_native_rejection_recovers_on_same_engine() {
    let error = bounded(
        "native rejection and recovery",
        native_rejection::<sglang_fixture::Fixture>(32_769),
    )
    .await;
    // The Rust server checks the context inside the open stream and reports
    // it as a coded error item.
    let message = error.to_string();
    assert!(message.contains("Generate"), "{message}");
    assert!(
        message.contains("maximum context length of 32768 tokens"),
        "{message}"
    );
    assert!(message.contains("(code 400)"), "{message}");
}

#[tokio::test]
async fn sglang_teardown_terminates_handlers_with_clients_alive() {
    let (outputs, tokens) = bounded(
        "server teardown with live clients",
        teardown_with_live_clients::<sglang_fixture::Fixture>(),
    )
    .await;
    let error = failure(outputs, &tokens, BackendError::Unknown);
    assert!(error.to_string().contains("Generate"));
}

#[tokio::test]
async fn sglang_malformed_terminal_fails_then_recovers() {
    bounded("malformed terminal and recovery", async {
        use sglang_fixture::token_frame;
        let control = Controller::<sglang_fixture::Adapter>::default();
        let mut fixture =
            sglang_fixture::Fixture::start(control.clone(), FixtureConfig::default()).await;
        let engine = fixture.engine().await;
        engine.start(0).await.unwrap();
        let ctx = mock_context();
        let handle = control.request(ctx.id(), RequestPlan::default());
        let mut terminal = token_frame(&[], None);
        if let Some(dynamo_sglang_sidecar::proto::generate_stream_item::Item::Frame(frame)) =
            terminal.item.as_mut()
        {
            frame.meta_info.as_mut().unwrap().finish_reason =
                Some(dynamo_sglang_sidecar::proto::FinishReason { kind: None });
        }
        fixture.respond(ctx.id(), vec![token_frame(&[101], None), terminal]);
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
        assert!(
            error
                .to_string()
                .contains("finish_reason is missing a type")
        );
        bounded("malformed RPC released", handle.wait(Event::Dropped)).await;
        healthy(&fixture, &engine, &control).await;
        finish(&mut fixture, &engine).await;
    })
    .await;
}

// The Rust server ends a request's stream with an error item carrying an HTTP
// status; each code keeps its error category whether or not a token preceded it.
// A 4xx is the request's fault and any other code the engine's, so neither is a
// transport failure that migration would retry.
#[tokio::test]
async fn sglang_stream_error_codes_map_before_and_after_a_token_then_recover() {
    bounded("in-stream error codes and recovery", async {
        use sglang_fixture::{error_item, token_frame};
        let control = Controller::<sglang_fixture::Adapter>::default();
        let mut fixture =
            sglang_fixture::Fixture::start(control.clone(), FixtureConfig::default()).await;
        let engine = fixture.engine().await;
        engine.start(0).await.unwrap();
        for (code, kind) in [
            (400, BackendError::InvalidArgument),
            (499, BackendError::Cancelled),
            (503, BackendError::Unknown),
        ] {
            for has_token in [false, true] {
                let ctx = mock_context();
                let handle = control.request(ctx.id(), RequestPlan::default());
                let mut items = Vec::new();
                if has_token {
                    items.push(token_frame(&[101], None));
                }
                items.push(error_item(code, "native failure"));
                fixture.respond(ctx.id(), items);
                let outputs = collect(
                    &engine,
                    request("mocker-model", vec![1, 2, 3], 3),
                    GenerateContext::new(ctx, None),
                )
                .await;
                let tokens: &[u32] = if has_token { &[101] } else { &[] };
                let error = failure(outputs, tokens, kind);
                assert!(
                    error.to_string().contains(&format!("(code {code})")),
                    "{error}"
                );
                assert_eq!(error.public_message(), None, "{code}");
                bounded("errored RPC released", handle.wait(Event::Dropped)).await;
            }
        }
        healthy(&fixture, &engine, &control).await;
        finish(&mut fixture, &engine).await;
    })
    .await;
}

// An abort the request did not ask for ends its stream with SGLang's 499
// item; Abort reports whether it matched an active generation.
#[tokio::test]
async fn sglang_abort_rpc_reports_499_and_match_flag() {
    bounded("Abort RPC", async {
        let control = Controller::<sglang_fixture::Adapter>::default();
        let mut fixture = sglang_fixture::Fixture::start(
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
        let id = ctx.id().to_string();
        let handle = control.request(&id, after_token_responses(1, StreamAction::Continue));
        let mut stream = engine
            .generate(
                request("mocker-model", vec![11, 22, 33], 10_000),
                GenerateContext::new(ctx.clone(), None),
            )
            .await
            .unwrap();
        let mut outputs = checkpoint_outputs(&mut stream, &handle).await;
        fixture.scheduler_active().await;
        engine.abort(ctx.clone()).await;
        fixture.wait_aborts(1).await;
        assert_eq!(fixture.aborts(), [(id.clone(), true)]);
        handle.release();
        outputs.extend(bounded("aborted stream", stream.collect::<Outputs>()).await);
        let error = failure(outputs, &handle.tokens(), BackendError::Cancelled);
        assert!(error.to_string().contains("(code 499)"), "{error}");
        bounded("aborted RPC released", handle.wait(Event::Dropped)).await;
        fixture.scheduler_idle().await;

        // A finished request matches nothing, and repeating Abort is harmless.
        engine.abort(ctx).await;
        fixture.wait_aborts(2).await;
        assert_eq!(fixture.aborts()[1], (id, false));
        healthy(&fixture, &engine, &control).await;
        finish(&mut fixture, &engine).await;
    })
    .await;
}

// A request runs on one listener. The gRPC Abort reaches every listener and
// matches only on the owner; native /generate aborts through its own
// listener only.
#[tokio::test]
async fn sglang_node_sidecar_aborts_on_the_owning_listener() {
    bounded("node abort", async {
        let control = Controller::<sglang_fixture::Adapter>::default();
        let http_control = Controller::<http::Adapter>::default();
        let mut node = sglang_fixture::Fixture::start_node(
            control.clone(),
            FixtureConfig {
                speedup_ratio: 0.1,
                ..Default::default()
            },
            sglang_fixture::NodeConfig::listeners(2),
        )
        .await;
        for listener in &mut node {
            listener.serve_http(http_control.clone()).await;
        }
        let engine = start_http_engine(&node[0]).await;
        let ctx = mock_context();
        let id = ctx.id().to_string();
        let handle = control.request(&id, after_token_responses(1, StreamAction::Continue));
        let mut req = request("mocker-model", vec![11, 22, 33], 10_000);
        req.routing = Some(RoutingHints {
            dp_rank: Some(1),
            ..Default::default()
        });
        let mut stream = engine
            .generate(req, GenerateContext::new(ctx.clone(), None))
            .await
            .unwrap();
        let mut outputs = checkpoint_outputs(&mut stream, &handle).await;
        engine.abort(ctx).await;
        node[0].wait_aborts(1).await;
        node[1].wait_aborts(1).await;
        assert_eq!(node[0].aborts(), [(id.clone(), false)]);
        assert_eq!(node[1].aborts(), [(id.clone(), true)]);
        handle.release();
        outputs.extend(bounded("aborted stream", stream.collect::<Outputs>()).await);
        let error = failure(outputs, &handle.tokens(), BackendError::Cancelled);
        assert!(error.to_string().contains("(code 499)"), "{error}");
        bounded("aborted RPC released", handle.wait(Event::Dropped)).await;

        // Native /generate: a dropped request is aborted through its
        // listener's Abort RPC only.
        let ctx = mock_context();
        let target =
            http_control.request(ctx.id(), after_token_responses(1, StreamAction::Continue));
        let mut req = http_request();
        req.routing = Some(RoutingHints {
            dp_rank: Some(1),
            ..Default::default()
        });
        let mut stream = engine
            .generate(req, GenerateContext::new(ctx.clone(), None))
            .await
            .unwrap();
        http_checkpoint(&mut stream, &target).await;
        drop(stream);
        node[1].http().wait_aborted(ctx.id()).await;
        bounded("HTTP target released", target.wait(Event::Dropped)).await;
        assert!(
            node[1]
                .http()
                .aborted_requests()
                .contains(&ctx.id().to_string())
        );
        assert_eq!(node[0].aborts(), [(id, false)]);
        for listener in &node {
            listener.scheduler_idle().await;
        }
        bounded("sidecar cleanup", engine.cleanup()).await.unwrap();
        for listener in &mut node {
            listener.shutdown().await;
        }
    })
    .await;
}

#[tokio::test]
async fn trtllm_teardown_terminates_handlers_with_clients_alive() {
    let (outputs, tokens) = bounded(
        "server teardown with live clients",
        teardown_with_live_clients::<support::trtllm::Fixture>(),
    )
    .await;
    let error = failure(outputs, &tokens, BackendError::Unknown);
    assert!(error.to_string().contains("Generate"));
}

#[tokio::test]
async fn trtllm_native_rejection_recovers_on_same_engine() {
    let error = bounded(
        "native rejection and recovery",
        native_rejection::<support::trtllm::Fixture>(1_000_001),
    )
    .await;
    assert!(
        error
            .to_string()
            .contains("exceeds the Mocker limit of 1000000"),
        "{error}"
    );
}

#[tokio::test]
async fn trtllm_malformed_terminal_fails_then_recovers() {
    bounded("malformed terminal and recovery", async {
        use dynamo_trtllm_sidecar::proto as pb;
        let control = Controller::<support::trtllm::Adapter>::default();
        let mut fixture =
            support::trtllm::Fixture::start(control.clone(), FixtureConfig::default()).await;
        let engine = fixture.engine().await;
        engine.start(0).await.unwrap();
        let ctx = mock_context();
        let handle = control.request(ctx.id(), RequestPlan::default());
        fixture.respond(
            ctx.id(),
            vec![
                pb::GenerateResponse {
                    event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
                        output_index: Some(0),
                        tokens: vec![pb::TokenInfo {
                            token_id: 101,
                            ..Default::default()
                        }],
                        ..Default::default()
                    })),
                    ..Default::default()
                },
                pb::GenerateResponse {
                    event: Some(pb::generate_response::Event::Finished(
                        pb::GenerationFinished {
                            output_index: Some(0),
                            reason: pb::FinishReason::Unspecified as i32,
                            ..Default::default()
                        },
                    )),
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
        assert!(error.to_string().contains("unspecified finish reason"));
        bounded("malformed RPC released", handle.wait(Event::Dropped)).await;
        healthy(&fixture, &engine, &control).await;
        finish(&mut fixture, &engine).await;
    })
    .await;
}

fn http_request() -> PreprocessedRequest {
    let mut req = request("mocker-model", vec![11, 22, 33], 2);
    req.extra_args = Some(json!({"sglang_tito": {"sampling_params": {"max_new_tokens": 2}}}));
    req
}

fn assert_http_completed(outputs: Outputs, request_id: &str) {
    let outputs = outputs.into_iter().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(outputs.len(), 2);
    assert!(outputs[0].finish_reason.is_none());
    assert_eq!(outputs[1].finish_reason, Some(FinishReason::Length));
    for (output, expected) in outputs.iter().zip(http::responses(request_id)) {
        assert_eq!(
            output.engine_data,
            Some(json!({"sglang_response": expected}))
        );
    }
}

async fn http_checkpoint(
    stream: &mut BoxStream<'static, Result<LLMEngineOutput, DynamoError>>,
    handle: &RequestHandle<http::Adapter>,
) -> Outputs {
    let first = bounded("HTTP first response", stream.next()).await.unwrap();
    assert!(first.as_ref().unwrap().finish_reason.is_none());
    bounded("HTTP checkpoint", handle.wait(Event::Checkpoint)).await;
    assert!(poll!(stream.next()).is_pending());
    vec![first]
}

// Native /generate readiness never fails startup: with an HTTP endpoint that
// never becomes ready, the worker warns, stays on gRPC and does not publish
// `sglang_generate`, and a native request fails. A multinode engine still
// publishes its worker group, whose lookup must not find the startup deadline
// already spent by the readiness wait.
#[tokio::test]
async fn sglang_native_http_unready_degrades_to_grpc() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    bounded("unready native HTTP", async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let unready = tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let mut request = [0_u8; 1024];
                    let _ = socket.read(&mut request).await;
                    let _ = socket
                        .write_all(
                            b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        )
                        .await;
                });
            }
        });
        let control = Controller::<sglang_fixture::Adapter>::default();
        let mut fixture =
            sglang_fixture::Fixture::start(control.clone(), FixtureConfig::default()).await;
        fixture.set_http_port(port);
        fixture.override_discovery(
            None,
            vec![Box::new(move |info| {
                let deployment = info.deployment_info.as_mut().unwrap();
                deployment.nnodes = 2;
                deployment.dist_init_addr = Some("127.0.0.1:29500".into());
            })],
        );
        let argv = vec![
            "dynamo-sglang-sidecar".to_string(),
            "--grpc-endpoint".to_string(),
            fixture.endpoint(),
            "--grpc-retry-interval-secs".to_string(),
            "1".to_string(),
            // The readiness wait spends what is left of it.
            "--grpc-startup-deadline-secs".to_string(),
            "3".to_string(),
        ];
        let engine = tokio::task::spawn_blocking(move || {
            dynamo_sglang_sidecar::SglangSidecarEngine::from_args(Some(argv))
                .unwrap()
                .0
        })
        .await
        .unwrap();
        let config = engine.start(0).await.unwrap();
        assert!(!config.runtime_data.contains_key("sglang_generate"));
        assert_eq!(
            config.runtime_data.get("sglang_worker_group_id"),
            Some(&json!("dist_init:tcp://127.0.0.1:29500"))
        );

        healthy(&fixture, &engine, &control).await;
        let error = engine
            .generate(http_request(), GenerateContext::new(mock_context(), None))
            .await
            .err()
            .expect("a native request must fail without a ready HTTP endpoint");
        assert_eq!(
            error.error_type(),
            dynamo_backend_common::ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(
            error
                .to_string()
                .contains("native SGLang Generate is unavailable"),
            "{error}"
        );
        unready.abort();
        finish(&mut fixture, &engine).await;
    })
    .await;
}

async fn start_http_engine(
    fixture: &sglang_fixture::Fixture,
) -> dynamo_sglang_sidecar::SglangSidecarEngine {
    let engine = fixture.engine().await;
    let config = engine.start(0).await.unwrap();
    assert_eq!(
        config.runtime_data.get("sglang_generate"),
        Some(&json!(true))
    );
    engine
}

/// SGLang's Rust server serves no `/abort_request`; the Abort RPC cancels
/// native requests.
#[tokio::test]
async fn sglang_http_cancellation_and_drop_abort_only_the_target() {
    bounded("HTTP cancellation phases and recovery", async {
        let control = Controller::<http::Adapter>::default();
        let mut fixture =
            sglang_fixture::Fixture::start_http(control.clone(), FixtureConfig::default()).await;
        let http = fixture.http();
        let engine = start_http_engine(&fixture).await;
        let mut aborted = Vec::new();
        for is_before_headers in [true, false] {
            for cancellation in [Cancellation::Explicit, Cancellation::ConsumerDrop] {
                let ctx = mock_context();
                let target = control.request(
                    ctx.id(),
                    if is_before_headers {
                        RequestPlan {
                            open: OpenAction::Hold,
                            ..Default::default()
                        }
                    } else {
                        after_token_responses(1, StreamAction::Continue)
                    },
                );
                let mut stream = engine
                    .generate(http_request(), GenerateContext::new(ctx.clone(), None))
                    .await
                    .unwrap();
                let mut outputs = if is_before_headers {
                    tokio::select! {
                        _ = target.wait(Event::Received) => {},
                        output = stream.next() => panic!("response arrived before headers: {output:?}"),
                    }
                    assert!(poll!(stream.next()).is_pending());
                    Vec::new()
                } else {
                    http_checkpoint(&mut stream, &target).await
                };
                let tokens = target.tokens();
                let survivor_ctx = mock_context();
                let survivor_id = survivor_ctx.id().to_string();
                let survivor = control.request(
                    &survivor_id,
                    after_token_responses(1, StreamAction::Continue),
                );
                let mut survivor_stream = engine
                    .generate(http_request(), GenerateContext::new(survivor_ctx, None))
                    .await
                    .unwrap();
                let mut survivor_outputs = http_checkpoint(&mut survivor_stream, &survivor).await;

                if matches!(cancellation, Cancellation::Explicit) {
                    ctx.stop_generating();
                    outputs.push(bounded("HTTP cancelled response", stream.next()).await.unwrap());
                    failure(outputs, &tokens, BackendError::Cancelled);
                }
                drop(stream);
                http.wait_aborted(ctx.id()).await;
                bounded("HTTP target transport released", target.wait(Event::Dropped)).await;
                aborted.push(ctx.id().to_string());
                assert_eq!(http.aborted_requests(), aborted);
                assert_eq!(target.tokens(), tokens);
                assert!(!survivor.reached(Event::Dropped));
                assert!(poll!(survivor_stream.next()).is_pending());
                survivor.release();
                survivor_outputs.extend(survivor_stream.collect::<Outputs>().await);
                assert_http_completed(survivor_outputs, &survivor_id);
                bounded("HTTP survivor released", survivor.wait(Event::Dropped)).await;

                let ctx = mock_context();
                let id = ctx.id().to_string();
                assert_http_completed(
                    collect(&engine, http_request(), GenerateContext::new(ctx, None)).await,
                    &id,
                );
                assert_eq!(http.aborted_requests(), aborted);
            }
        }
        finish(&mut fixture, &engine).await;
    })
    .await;
}

// Regression: the gate's false/zero values and exact forbidden sequences can
// disappear at the Dynamo-to-native protobuf boundary.
#[tokio::test]
async fn vllm_generation_controls_preserve_values_over_grpc() {
    bounded("vLLM engine generation controls", async {
        use dynamo_vllm_sidecar::{VllmSidecarEngine, proto as pb};
        use serde_json::json;
        let controller = Controller::<vllm_fixture::Adapter>::default();
        let mut fixture =
            vllm_fixture::Fixture::start(controller.clone(), FixtureConfig::default()).await;
        fixture.set_reasoning_parser("deepseek_r1");
        let endpoint = fixture.server.endpoint();
        let (engine, _) = tokio::task::spawn_blocking(move || {
            VllmSidecarEngine::try_from_args(vec![
                "sidecar".into(),
                "--grpc-endpoint".into(),
                endpoint,
                "--dyn-reasoning-parser".into(),
                "qwen3".into(),
            ])
            .unwrap()
        })
        .await
        .unwrap();
        let config = engine.start(0).await.unwrap();
        assert_eq!(
            config
                .runtime_data
                .get("tool_call_structural_tag_excludes_reasoning"),
            Some(&json!(true))
        );
        for control in [
            "default",
            "reasoning_parser_kwargs",
            "reasoning_ended",
            "thinking_token_budget",
            "bad_words_token_ids",
            "allowed_token_ids",
            "top_k",
            "min_p",
        ] {
            let mut req = request("mocker-model", vec![11, 22, 33], 1);
            match control {
                "reasoning_parser_kwargs" => {
                    req.extra_args = Some(json!({
                        "reasoning_parser_kwargs": {"chat_template_kwargs": {"enable_thinking": false}}
                    }))
                }
                "reasoning_ended" => req.extra_args = Some(json!({"reasoning_ended": false})),
                "thinking_token_budget" => req.stop_conditions.max_thinking_tokens = Some(0),
                "bad_words_token_ids" => {
                    req.extra_args =
                        Some(json!({"sampling_options": {"bad_words_token_ids": [[7, 11], [0]]}}))
                }
                "allowed_token_ids" => {
                    req.extra_args =
                        Some(json!({"sampling_options": {"allowed_token_ids": [7, 11]}}))
                }
                "top_k" => req.sampling_options.top_k = Some(-1),
                "min_p" => req.sampling_options.min_p = Some(0.0),
                _ => {}
            }
            let ctx = mock_context();
            let handle = controller.request(ctx.id(), RequestPlan::default());
            let outputs = collect(&engine, req, GenerateContext::new(ctx, None)).await;
            terminal(outputs, &handle.tokens(), 3, FinishReason::Length);
            let wire = handle.native_request().expect("native request");
            match control {
                "reasoning_parser_kwargs" => assert_eq!(
                    wire.engine_reasoning_gate
                        .as_ref().unwrap()
                        .chat_template_kwargs.as_ref().unwrap()
                        .fields["enable_thinking"].kind,
                    Some(prost_types_v14::value::Kind::BoolValue(false)),
                ),
                "reasoning_ended" => assert_eq!(
                    wire.engine_reasoning_gate.as_ref().unwrap().reasoning_ended,
                    Some(false)
                ),
                "thinking_token_budget" => assert_eq!(
                    wire.stopping.as_ref().unwrap().thinking_token_budget,
                    Some(0)
                ),
                "bad_words_token_ids" => assert_eq!(
                    wire.decoding.as_ref().unwrap().bad_words_token_ids,
                    vec![
                        pb::TokenIds { ids: vec![7, 11] },
                        pb::TokenIds { ids: vec![0] }
                    ]
                ),
                "allowed_token_ids" => assert_eq!(
                    wire.decoding.as_ref().unwrap().allowed_token_ids,
                    vec![7, 11]
                ),
                "top_k" => assert_eq!(wire.sampling.as_ref().unwrap().top_k, Some(0)),
                "min_p" => assert_eq!(wire.sampling.as_ref().unwrap().min_p, Some(0.0)),
                _ => {
                    assert!(wire.engine_reasoning_gate.is_none());
                    assert!(
                        wire.stopping
                            .as_ref()
                            .unwrap()
                            .thinking_token_budget
                            .is_none()
                    );
                    let sampling = wire.sampling.as_ref().unwrap();
                    assert_eq!(
                        (sampling.top_k, sampling.top_p, sampling.min_p),
                        (None, None, None)
                    );
                }
            }
        }
        engine.cleanup().await.unwrap();
        fixture.shutdown().await;
    })
    .await;
}

// Decode drains until KV transfer completes before its Abort RPC.
#[tokio::test]
async fn sglang_http_decode_cancellation_waits_for_transfer() {
    bounded("HTTP decode transfer-safe cancellation", async {
        let control = Controller::<http::Adapter>::default();
        let mut fixture = sglang_fixture::Fixture::start_http(
            control.clone(),
            FixtureConfig {
                disaggregation_mode: DisaggregationMode::Decode,
                ..Default::default()
            },
        )
        .await;
        let http = fixture.http();
        let engine = start_http_engine(&fixture).await;
        let mut aborted = Vec::new();
        for is_terminal_without_tokens in [false, true] {
            for cancellation in [Cancellation::Explicit, Cancellation::ConsumerDrop] {
                let ctx = mock_context();
                let target = control.request(
                    ctx.id(),
                    RequestPlan {
                        open: OpenAction::Hold,
                        ..after_token_responses(1, StreamAction::Continue)
                    },
                );
                let gate = http.hold_responses(
                    ctx.id(),
                    if is_terminal_without_tokens {
                        vec![json!({"output_ids": [], "text": "", "meta_info": {"id": ctx.id(), "finish_reason": {"type": "length"}}})]
                    } else {
                        http::responses(ctx.id()).into()
                    },
                );
                let mut req = http_request();
                req.prefill_result = Some(PrefillResult {
                    disaggregated_params: json!({
                        "bootstrap_host": "127.0.0.1",
                        "bootstrap_port": http.port(),
                        "bootstrap_room": 1,
                    }),
                    prompt_tokens_details: None,
                });
                let (first_token_tx, first_token) = tokio::sync::watch::channel(false);
                let mut stream = engine
                    .generate(req, GenerateContext::new(ctx.clone(), Some(first_token_tx)))
                    .await
                    .unwrap();
                tokio::select! {
                    _ = target.wait(Event::Received) => {},
                    output = stream.next() => panic!("response arrived before headers: {output:?}"),
                }
                assert!(poll!(stream.next()).is_pending());
                if matches!(cancellation, Cancellation::Explicit) {
                    ctx.stop_generating();
                    failure(
                        vec![bounded("HTTP decode cancellation", stream.next()).await.unwrap()],
                        &[],
                        BackendError::Cancelled,
                    );
                }
                drop(stream);

                for is_before_headers in [true, false] {
                    if !is_before_headers {
                        target.release();
                        gate.wait_empty_response().await;
                    }
                    assert!(
                        tokio::time::timeout(std::time::Duration::from_millis(100), async {
                            tokio::select! {
                                _ = http.wait_aborted(ctx.id()) => panic!("decode aborted before transfer completed"),
                                _ = target.wait(Event::Dropped) => panic!("decode transport dropped before transfer completed"),
                            }
                        })
                        .await
                        .is_err()
                    );
                    assert!(!*first_token.borrow());
                }
                gate.release();
                if !is_terminal_without_tokens {
                    http.wait_aborted(ctx.id()).await;
                    assert!(*first_token.borrow());
                    assert_eq!(target.tokens(), vec![101]);
                    aborted.push(ctx.id().to_string());
                }
                bounded("HTTP decode transport released", target.wait(Event::Dropped)).await;
                if is_terminal_without_tokens {
                    assert!(
                        tokio::time::timeout(
                            std::time::Duration::from_millis(100),
                            http.wait_aborted(ctx.id()),
                        )
                        .await
                        .is_err()
                    );
                }
                assert_eq!(http.aborted_requests(), aborted);
            }
        }
        finish(&mut fixture, &engine).await;
    })
    .await;
}

// Regression: native text strips tool-closing EOS tokens; exposing even empty
// text bypasses the frontend decoder and loses the tool delimiter.
#[tokio::test]
async fn vllm_visible_stop_tokens_preserve_tool_delimiters() {
    bounded("visible tool delimiter", async {
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
            vec![pb::GenerateResponse {
                outputs: Some(pb::SequenceOutput {
                    token_ids: vec![42],
                    num_tokens: 1,
                    finish_info: Some(pb::FinishInfo {
                        num_output_tokens: 1,
                        finish_reason: pb::finish_info::FinishReason::Stop as i32,
                        stop_reason: Some(pb::finish_info::StopReason::EosTokenId(42)),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        );
        let mut req = request("mocker-model", vec![11, 22, 33], 1);
        req.stop_conditions.stop_token_ids_visible = Some(vec![42]);
        req.stop_conditions.stop_token_ids_hidden = Some(vec![2]);
        req.output_options.skip_special_tokens = Some(false);
        let outputs = collect(&engine, req, GenerateContext::new(ctx, None)).await;
        assert!(
            outputs
                .iter()
                .all(|output| output.as_ref().unwrap().text.is_none())
        );
        terminal(outputs, &[42], 3, FinishReason::Stop);
        let wire = handle.native_request().unwrap();
        assert_eq!(wire.stopping.unwrap().stop_token_ids, [2, 42]);
        let response = wire.response.unwrap();
        assert_eq!(response.output_text, Some(false));
        assert!(response.output_token_ids);
        finish(&mut fixture, &engine).await;
    })
    .await;
}

// SGLang text trims the matched stop token, so the delimiter must arrive as an ID.
#[tokio::test]
async fn sglang_visible_stop_tokens_preserve_tool_delimiters() {
    bounded("visible tool delimiter", async {
        use dynamo_sglang_sidecar::proto as pb;
        use sglang_fixture::token_frame;
        let control = Controller::<sglang_fixture::Adapter>::default();
        let mut fixture =
            sglang_fixture::Fixture::start(control.clone(), FixtureConfig::default()).await;
        let engine = fixture.engine().await;
        engine.start(0).await.unwrap();
        let ctx = mock_context();
        let handle = control.request(ctx.id(), RequestPlan::default());
        fixture.respond(
            ctx.id(),
            vec![token_frame(
                &[42],
                Some(pb::finish_reason::Kind::Stop(pb::FinishStop {
                    matched: Some(pb::Matched {
                        value: Some(pb::matched::Value::Token(42)),
                    }),
                })),
            )],
        );
        let mut req = request("mocker-model", vec![11, 22, 33], 1);
        req.stop_conditions.stop_token_ids_visible = Some(vec![42]);
        req.stop_conditions.stop_token_ids_hidden = Some(vec![2]);
        let outputs = collect(&engine, req, GenerateContext::new(ctx, None)).await;
        assert!(
            outputs
                .iter()
                .all(|output| output.as_ref().unwrap().text.is_none())
        );
        terminal(outputs, &[42], 3, FinishReason::Stop);
        let wire = handle.native_request().unwrap();
        let Some(pb::sampling_params_or_list::Value::One(sampling)) =
            wire.sampling_params.and_then(|params| params.value)
        else {
            panic!("expected one set of sampling params");
        };
        assert_eq!(sampling.stop_token_ids.unwrap().items, [2, 42]);
        assert!(wire.text.is_none());
        finish(&mut fixture, &engine).await;
    })
    .await;
}
