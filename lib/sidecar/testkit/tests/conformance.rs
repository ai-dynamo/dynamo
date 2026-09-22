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

mod support;

use support::{FixtureConfig, SidecarFixture, WireFixture, vllm};

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

async fn healthy(
    fixture: &impl WireFixture,
    engine: &impl LLMEngine,
    control: &Controller<impl dynamo_sidecar_testkit::control::Protocol>,
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

#[path = "conformance/cancellation.rs"]
mod cancellation;
#[path = "conformance/errors.rs"]
mod errors;
#[path = "conformance/lifecycle.rs"]
mod lifecycle;
#[path = "conformance/streaming.rs"]
mod streaming;

macro_rules! enroll_baseline {
    ($backend:ident, $fixture:ty) => {
        mod $backend {
            use super::*;
            type Fixture = $fixture;

            #[tokio::test]
            async fn stream_tokens_terminal_and_usage() {
                streaming::streaming::<Fixture>(
                    request("request-model", vec![10, 20, 30, 40, 50], 7),
                    None,
                )
                .await;
            }

            #[tokio::test]
            async fn open_failure_early_eof_and_read_failure() {
                for (plan, early_eof) in errors::failure_plans() {
                    let (mut fixture, engine, _) =
                        errors::failure_case::<Fixture>(plan, early_eof, None).await;
                    finish(&mut fixture, &engine).await;
                }
            }

            #[tokio::test]
            async fn cancellation_before_open_during_open_and_during_read() {
                let (mut fixture, engine, _) =
                    cancellation::cancellation::<Fixture>(FixtureConfig::default(), None, false)
                        .await;
                finish(&mut fixture, &engine).await;
            }

            #[tokio::test]
            async fn cleanup_before_start_and_during_read() {
                let (mut fixture, _, _) = lifecycle::cleanup::<Fixture>(true).await;
                fixture.shutdown().await;
            }
        }
    };
}

enroll_baseline!(sglang, support::sglang::Fixture);
