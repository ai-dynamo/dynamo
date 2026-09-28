// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::testing::mock_context;
use dynamo_backend_common::{
    BackendError, DynamoError, FinishReason, GenerateContext, LLMEngine, LLMEngineOutput,
};
use dynamo_sidecar_testkit::assert::{failure, terminal};
use dynamo_sidecar_testkit::bounded;
use dynamo_sidecar_testkit::control::{
    Controller, Event, OpenAction, Protocol, RequestHandle, RequestPlan, StreamAction, StreamFault,
    StreamPoint,
};
use dynamo_sidecar_testkit::fixtures::{Outputs, collect, request};
use futures::{StreamExt, poll, stream::BoxStream};

mod support;

use support::{FixtureConfig, SidecarFixture};

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

async fn drained(fixture: &impl SidecarFixture) {
    bounded("Mocker request release", async {
        while fixture.active_request_count() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await;
}

async fn streaming<F: SidecarFixture>() {
    let control = Controller::<F::Protocol>::default();
    let config = FixtureConfig {
        model: "alternate-mocker".into(),
        connections: 2,
    };
    let mut fixture = bounded("Mocker startup", F::start(control.clone(), config)).await;
    let engine = bounded("sidecar construction", fixture.engine()).await;
    let metadata = bounded("sidecar startup", engine.start(0)).await.unwrap();
    assert_eq!(metadata.model, "alternate-mocker");
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
        let outputs = collect(
            &engine,
            request("request-model", vec![10, 20, 30, 40, 50], 7),
            GenerateContext::new(ctx, None),
        )
        .await;
        let tokens = handle.tokens();
        assert_eq!(tokens.len(), 7);
        terminal(outputs, &tokens, 5, FinishReason::Length);
        let native = handle.native_request().expect("native request");
        if let Some(model) = F::native_model(&native) {
            assert_eq!(model, metadata.model);
        }
        bounded("server stream drop", handle.wait(Event::Dropped)).await;
        assert_eq!(handle.reached(Event::Checkpoint), plan.stream.is_some());
        drained(&fixture).await;
    }
    bounded("sidecar cleanup", engine.cleanup()).await.unwrap();
    fixture.shutdown().await;
}

async fn failures<F: SidecarFixture>() {
    for (plan, early_eof) in [
        (
            RequestPlan {
                open: OpenAction::Fail,
                ..Default::default()
            },
            false,
        ),
        (after_token_responses(1, StreamAction::Close), true),
        (after_token_responses(1, StreamAction::Fail), false),
    ] {
        let control = Controller::<F::Protocol>::default();
        let mut fixture = bounded(
            "Mocker startup",
            F::start(control.clone(), FixtureConfig::default()),
        )
        .await;
        let engine = bounded("sidecar construction", fixture.engine()).await;
        bounded("sidecar startup", engine.start(0)).await.unwrap();
        let ctx = mock_context();
        let fails_open = matches!(plan.open, OpenAction::Fail);
        let handle = control.request(ctx.id(), plan);
        let req = request("mocker-model", vec![11, 22, 33], 6);
        let ctx = GenerateContext::new(ctx, None);
        let outputs = if fails_open {
            collect(&engine, req, ctx).await
        } else {
            let mut stream = bounded("generation opening", engine.generate(req, ctx))
                .await
                .unwrap();
            let mut outputs = checkpoint_outputs(&mut stream, &handle).await;
            handle.release();
            outputs.extend(bounded("failed stream completion", stream.collect::<Outputs>()).await);
            outputs
        };
        let kind = if early_eof {
            F::eof_error()
        } else {
            BackendError::CannotConnect
        };
        let tokens = handle.tokens();
        assert_eq!(tokens.is_empty(), fails_open);
        let error = failure(outputs, &tokens, kind);
        if !early_eof {
            assert!(error.to_string().contains("injected"));
            assert!(error.to_string().contains("Generate"));
        }
        bounded("server stream drop", handle.wait(Event::Dropped)).await;
        drained(&fixture).await;
        bounded("sidecar cleanup", engine.cleanup()).await.unwrap();
        fixture.shutdown().await;
    }
}

async fn cancellation<F: SidecarFixture>() {
    let control = Controller::<F::Protocol>::default();
    let mut fixture = bounded(
        "Mocker startup",
        F::start(control.clone(), FixtureConfig::default()),
    )
    .await;
    let engine = bounded("sidecar construction", fixture.engine()).await;
    bounded("sidecar startup", engine.start(0)).await.unwrap();

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

    let ctx = mock_context();
    let handle = control.request(
        ctx.id(),
        RequestPlan {
            open: OpenAction::Hold,
            ..Default::default()
        },
    );
    let outputs = collect(
        &engine,
        request("mocker-model", vec![11, 22, 33], 3),
        GenerateContext::new(ctx.clone(), None),
    );
    tokio::pin!(outputs);
    tokio::select! {
        _ = bounded("pending opening", handle.wait(Event::Received)) => {}
        _ = &mut outputs => panic!("generation finished before cancellation"),
    }
    ctx.stop_generating();
    terminal(outputs.await, &[], 3, FinishReason::Cancelled);
    bounded("opening drop", handle.wait(Event::Dropped)).await;

    let ctx_a = mock_context();
    let handle_a = control.request(ctx_a.id(), after_token_responses(2, StreamAction::Continue));
    let mut stream_a = bounded(
        "request A opening",
        engine.generate(
            request("mocker-model", vec![11, 22, 33], 8),
            GenerateContext::new(ctx_a.clone(), None),
        ),
    )
    .await
    .unwrap();
    let mut outputs_a = checkpoint_outputs(&mut stream_a, &handle_a).await;
    let tokens_a = handle_a.tokens();

    let ctx_b = mock_context();
    let handle_b = control.request(ctx_b.id(), after_token_responses(1, StreamAction::Continue));
    let mut stream_b = bounded(
        "request B opening",
        engine.generate(
            request("mocker-model", vec![90, 80, 70, 60, 50], 5),
            GenerateContext::new(ctx_b, None),
        ),
    )
    .await
    .unwrap();
    let mut outputs_b = checkpoint_outputs(&mut stream_b, &handle_b).await;
    let tokens_b = handle_b.tokens();
    assert!(poll!(stream_a.next()).is_pending());
    assert!(poll!(stream_b.next()).is_pending());

    ctx_a.stop_generating();
    outputs_a.extend(bounded("request A cancellation", stream_a.collect::<Outputs>()).await);
    assert_eq!(handle_a.tokens(), tokens_a);
    terminal(outputs_a, &tokens_a, 3, FinishReason::Cancelled);
    bounded("request A drop", handle_a.wait(Event::Dropped)).await;
    assert!(!handle_b.reached(Event::Dropped));
    assert_eq!(handle_b.tokens(), tokens_b);
    assert!(poll!(stream_b.next()).is_pending());

    handle_b.release();
    outputs_b.extend(bounded("request B completion", stream_b.collect::<Outputs>()).await);
    assert_eq!(handle_b.tokens().len(), 5);
    terminal(outputs_b, &handle_b.tokens(), 5, FinishReason::Length);
    bounded("request B drop", handle_b.wait(Event::Dropped)).await;
    drained(&fixture).await;
    bounded("sidecar cleanup", engine.cleanup()).await.unwrap();
    fixture.shutdown().await;
}

async fn cleanup<F: SidecarFixture>(needs_unstarted_check: bool) {
    let control = Controller::<F::Protocol>::default();
    let mut fixture = bounded(
        "Mocker startup",
        F::start(control.clone(), FixtureConfig::default()),
    )
    .await;
    if needs_unstarted_check {
        let engine = bounded("sidecar construction", fixture.engine()).await;
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
        bounded("cleanup before startup", engine.cleanup())
            .await
            .unwrap();
        bounded("repeated cleanup before startup", engine.cleanup())
            .await
            .unwrap();
        assert!(!handle.reached(Event::Received));
    }

    let engine = bounded("sidecar construction", fixture.engine()).await;
    bounded("sidecar startup", engine.start(0)).await.unwrap();
    let ctx = mock_context();
    let handle = control.request(ctx.id(), after_token_responses(1, StreamAction::Continue));
    let mut stream = bounded(
        "generation opening",
        engine.generate(
            request("mocker-model", vec![11, 22, 33], 6),
            GenerateContext::new(ctx, None),
        ),
    )
    .await
    .unwrap();
    let mut outputs = checkpoint_outputs(&mut stream, &handle).await;
    assert!(poll!(stream.next()).is_pending());
    bounded("sidecar cleanup", engine.cleanup()).await.unwrap();
    bounded("repeated sidecar cleanup", engine.cleanup())
        .await
        .unwrap();
    outputs.extend(bounded("stream cancellation", stream.collect::<Outputs>()).await);
    terminal(outputs, &handle.tokens(), 3, FinishReason::Cancelled);
    bounded("server stream drop", handle.wait(Event::Dropped)).await;
    drained(&fixture).await;
    fixture.shutdown().await;
}

macro_rules! enroll {
    ($backend:ident, $fixture:ty, $needs_unstarted_check:expr) => {
        mod $backend {
            use super::*;

            #[tokio::test]
            async fn stream_tokens_terminal_and_usage() {
                streaming::<$fixture>().await;
            }

            #[tokio::test]
            async fn open_failure_early_eof_and_read_failure() {
                failures::<$fixture>().await;
            }

            #[tokio::test]
            async fn cancellation_before_open_during_open_and_during_read() {
                cancellation::<$fixture>().await;
            }

            #[tokio::test]
            async fn cleanup_before_start_and_during_read() {
                cleanup::<$fixture>($needs_unstarted_check).await;
            }
        }
    };
}

enroll!(vllm, support::vllm::Fixture, false);
enroll!(sglang, support::sglang::Fixture, true);
