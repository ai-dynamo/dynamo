// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::testing::mock_context;
use dynamo_backend_common::{
    BackendError, DynamoError, ErrorType, FinishReason, GenerateContext, LLMEngine,
    LLMEngineOutput, PreprocessedRequest, StopConditions,
};
use futures::{StreamExt, poll};

use super::{Control, Fault, Gate, SidecarFixture, bounded};
use crate::Fixture;

type Outputs = Vec<Result<LLMEngineOutput, DynamoError>>;

fn request(tokens: Vec<u32>, max_tokens: u32) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("mocker-model".to_string())
        .token_ids(tokens)
        .sampling_options(Default::default())
        .output_options(Default::default())
        .stop_conditions(StopConditions {
            max_tokens: Some(max_tokens),
            ..Default::default()
        })
        .build()
        .unwrap()
}

async fn collect(
    engine: &impl LLMEngine,
    request: PreprocessedRequest,
    ctx: GenerateContext,
) -> Outputs {
    bounded(async {
        match engine.generate(request, ctx).await {
            Ok(stream) => stream.collect().await,
            Err(error) => vec![Err(error)],
        }
    })
    .await
}

fn assert_terminal(outputs: Outputs, tokens: &[u32], prompt_tokens: u32, reason: FinishReason) {
    let outputs: Vec<_> = outputs.into_iter().collect::<Result<_, _>>().unwrap();
    assert_eq!(
        outputs
            .iter()
            .flat_map(|o| &o.token_ids)
            .copied()
            .collect::<Vec<_>>(),
        tokens,
    );
    assert_eq!(
        outputs.iter().filter(|o| o.finish_reason.is_some()).count(),
        1
    );
    let terminal = outputs.last().expect("terminal output");
    assert_eq!(terminal.finish_reason, Some(reason));
    let usage = terminal.completion_usage.as_ref().expect("terminal usage");
    assert_eq!(
        (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens
        ),
        (
            prompt_tokens,
            tokens.len() as u32,
            prompt_tokens + tokens.len() as u32
        ),
    );
}

fn assert_failure(mut outputs: Outputs, tokens: &[u32], kind: BackendError) -> DynamoError {
    let error = outputs
        .pop()
        .expect("error output")
        .expect_err("truncation must fail");
    assert_eq!(error.error_type(), ErrorType::Backend(kind));
    let preceding: Vec<_> = outputs.into_iter().collect::<Result<_, _>>().unwrap();
    assert!(preceding.iter().all(|o| o.finish_reason.is_none()));
    assert_eq!(
        preceding
            .iter()
            .flat_map(|o| &o.token_ids)
            .copied()
            .collect::<Vec<_>>(),
        tokens,
    );
    error
}

async fn released(control: &Control, fixture: &Fixture) {
    control.wait(Gate::Dropped).await;
    bounded(async {
        while fixture.active_request_count() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await;
}

#[tokio::test]
async fn stream_tokens_terminal_and_usage() {
    let control = Control::new(Fault::ExtraAfterTerminal);
    let fixture = bounded(Fixture::start(control.clone())).await;
    let engine = bounded(fixture.engine()).await;
    bounded(engine.start(0)).await.unwrap();
    let ctx = mock_context();
    let outputs = collect(
        &engine,
        request(vec![10, 20, 30, 40, 50], 7),
        GenerateContext::new(ctx.clone(), None),
    )
    .await;
    let tokens = control.tokens();
    assert_eq!(tokens.len(), 7);
    assert_terminal(outputs, &tokens, 5, FinishReason::Length);
    assert_eq!(control.request_ids(), [ctx.id()]);
    released(&control, &fixture).await;
    bounded(engine.cleanup()).await.unwrap();
}

#[tokio::test]
async fn open_failure_early_eof_and_read_failure() {
    for fault in [Fault::OpenError, Fault::EarlyEof, Fault::ReadError] {
        let control = Control::new(fault);
        let fixture = bounded(Fixture::start(control.clone())).await;
        let engine = bounded(fixture.engine()).await;
        bounded(engine.start(0)).await.unwrap();
        let req = request(vec![11, 22, 33], 3);
        let ctx = GenerateContext::new(mock_context(), None);
        let outputs = if fault == Fault::OpenError {
            collect(&engine, req, ctx).await
        } else {
            let mut stream = bounded(engine.generate(req, ctx)).await.unwrap();
            let first = bounded(stream.next()).await.unwrap();
            assert_eq!(first.as_ref().unwrap().token_ids.len(), 1);
            control.wait(Gate::Read).await;
            control.release.notify_one();
            let mut outputs = vec![first];
            outputs.extend(bounded(stream.collect::<Outputs>()).await);
            outputs
        };
        let kind = if fault == Fault::EarlyEof {
            Fixture::eof_error()
        } else {
            BackendError::CannotConnect
        };
        let tokens = control.tokens();
        assert_eq!(tokens.len(), usize::from(fault != Fault::OpenError));
        let error = assert_failure(outputs, &tokens, kind);
        if fault != Fault::EarlyEof {
            assert!(error.to_string().contains("injected"));
            assert!(error.to_string().contains("Generate"));
        }
        released(&control, &fixture).await;
        bounded(engine.cleanup()).await.unwrap();
    }
}

#[tokio::test]
async fn cancellation_before_open_during_open_and_during_read() {
    for stage in [Gate::Idle, Gate::Open, Gate::Read] {
        let control = Control::new(if stage == Gate::Open {
            Fault::PendingOpen
        } else {
            Fault::PendingRead
        });
        let fixture = bounded(Fixture::start(control.clone())).await;
        let engine = bounded(fixture.engine()).await;
        bounded(engine.start(0)).await.unwrap();
        let ctx = mock_context();
        let req = request(vec![11, 22, 33], 3);
        if stage == Gate::Idle {
            ctx.stop_generating();
            let outputs = collect(&engine, req, GenerateContext::new(ctx, None)).await;
            assert_terminal(outputs, &[], 3, FinishReason::Cancelled);
            assert!(control.request_ids().is_empty());
        } else if stage == Gate::Open {
            let outputs = collect(&engine, req, GenerateContext::new(ctx.clone(), None));
            tokio::pin!(outputs);
            tokio::select! {
                _ = control.wait(Gate::Open) => {}
                _ = &mut outputs => panic!("generation finished before cancellation"),
            }
            ctx.stop_generating();
            assert_terminal(outputs.await, &[], 3, FinishReason::Cancelled);
            released(&control, &fixture).await;
        } else {
            let mut stream = bounded(engine.generate(req, GenerateContext::new(ctx.clone(), None)))
                .await
                .unwrap();
            let first = bounded(stream.next()).await.unwrap();
            assert_eq!(first.as_ref().unwrap().token_ids.len(), 1);
            control.wait(Gate::Read).await;
            assert!(poll!(stream.next()).is_pending());
            ctx.stop_generating();
            let mut outputs = vec![first];
            outputs.extend(bounded(stream.collect::<Outputs>()).await);
            assert_terminal(outputs, &control.tokens(), 3, FinishReason::Cancelled);
            released(&control, &fixture).await;
        }
        bounded(engine.cleanup()).await.unwrap();
    }
}

#[tokio::test]
async fn cleanup_before_start_and_during_read() {
    let control = Control::new(Fault::PendingRead);
    let fixture = bounded(Fixture::start(control.clone())).await;
    let engine = bounded(fixture.engine()).await;
    assert_failure(
        collect(
            &engine,
            request(vec![11, 22, 33], 3),
            GenerateContext::new(mock_context(), None),
        )
        .await,
        &[],
        BackendError::EngineShutdown,
    );
    bounded(engine.cleanup()).await.unwrap();
    bounded(engine.cleanup()).await.unwrap();
    assert!(control.request_ids().is_empty());

    let engine = bounded(fixture.engine()).await;
    bounded(engine.start(0)).await.unwrap();
    let mut stream = bounded(engine.generate(
        request(vec![11, 22, 33], 3),
        GenerateContext::new(mock_context(), None),
    ))
    .await
    .unwrap();
    let first = bounded(stream.next()).await.unwrap();
    assert_eq!(first.as_ref().unwrap().token_ids.len(), 1);
    control.wait(Gate::Read).await;
    assert!(poll!(stream.next()).is_pending());
    bounded(engine.cleanup()).await.unwrap();
    bounded(engine.cleanup()).await.unwrap();
    let mut outputs = vec![first];
    outputs.extend(bounded(stream.collect::<Outputs>()).await);
    assert_terminal(outputs, &control.tokens(), 3, FinishReason::Cancelled);
    released(&control, &fixture).await;
}
