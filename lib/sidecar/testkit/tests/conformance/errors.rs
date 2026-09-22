// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) fn failure_plans() -> [(RequestPlan, bool); 3] {
    [
        (
            RequestPlan {
                open: OpenAction::Fail,
                ..Default::default()
            },
            false,
        ),
        (after_token_responses(1, StreamAction::Close), true),
        (after_token_responses(1, StreamAction::Fail), false),
    ]
}

pub(super) async fn failure_case<F: SidecarFixture>(
    plan: RequestPlan,
    early_eof: bool,
    expected_tokens: Option<usize>,
) -> (F, F::Engine, Controller<F::Protocol>) {
    let control = Controller::<F::Protocol>::default();
    let fixture = bounded(
        "Mocker startup",
        F::start(control.clone(), FixtureConfig::default()),
    )
    .await;
    let engine = bounded("sidecar construction", fixture.engine()).await;
    bounded("sidecar startup", engine.start(0)).await.unwrap();
    let ctx = mock_context();
    let fails_open = matches!(plan.open, OpenAction::Fail);
    let handle = control.request(ctx.id(), plan);
    let req = request("mocker-model", vec![11, 22, 33], 3);
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
    if let Some(expected) = expected_tokens {
        assert_eq!(tokens.len(), expected);
    }
    let error = failure(outputs, &tokens, kind);
    if !early_eof {
        assert!(error.to_string().contains("injected"));
        assert!(error.to_string().contains("Generate"));
    }
    bounded("server stream drop", handle.wait(Event::Dropped)).await;
    drained(&fixture).await;
    (fixture, engine, control)
}

#[tokio::test]
async fn vllm_open_failure_early_eof_and_read_failure() {
    bounded("vllm_open_failure_early_eof_and_read_failure", async {
        for (plan, early_eof) in failure_plans() {
            let expected = usize::from(!matches!(plan.open, OpenAction::Fail));
            let (mut fixture, engine, control) =
                failure_case::<vllm::Fixture>(plan, early_eof, Some(expected)).await;
            healthy(&fixture, &engine, &control).await;
            finish(&mut fixture, &engine).await;
        }
    })
    .await;
}

async fn native_rejection<F: WireFixture>() {
    let control = Controller::<F::Protocol>::default();
    let mut fixture = F::start(control.clone(), FixtureConfig::default()).await;
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    let ctx = mock_context();
    let handle = control.request(ctx.id(), RequestPlan::default());
    let error = engine
        .generate(
            request("mocker-model", vec![1, 2, 3], 32_769),
            GenerateContext::new(ctx, None),
        )
        .await
        .err()
        .expect("native rejection must fail before a response stream opens");
    let error = failure(vec![Err(error)], &[], BackendError::InvalidArgument);
    assert!(
        error
            .to_string()
            .contains("max_new_tokens must not exceed 32768")
    );
    assert!(error.to_string().contains("GenerateStream"));
    bounded("rejected request drop", handle.wait(Event::Dropped)).await;
    fixture.scheduler_idle().await;
    healthy(&fixture, &engine, &control).await;
    engine.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn vllm_native_rejection_recovers_on_same_engine() {
    bounded(
        "native rejection and recovery",
        native_rejection::<vllm::Fixture>(),
    )
    .await;
}

#[tokio::test]
async fn vllm_malformed_terminal_fails_then_recovers() {
    bounded("malformed terminal and recovery", async {
        use dynamo_vllm_sidecar::proto as pb;
        let control = Controller::<vllm::Adapter>::default();
        let mut fixture = vllm::Fixture::start(control.clone(), FixtureConfig::default()).await;
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
