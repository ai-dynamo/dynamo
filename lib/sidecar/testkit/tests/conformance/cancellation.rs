// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) async fn cancellation<F: SidecarFixture>(
    config: FixtureConfig,
    expected_prefixes: Option<(usize, usize)>,
    needs_pending_open_check: bool,
) -> (F, F::Engine, Controller<F::Protocol>) {
    let control = Controller::<F::Protocol>::default();
    let fixture = bounded("Mocker startup", F::start(control.clone(), config)).await;
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
        if needs_pending_open_check {
            tokio::select! {
                _ = bounded("pending native headers", handle.wait(Event::Received)) => {}
                _ = &mut opening => panic!("generate returned before native headers"),
            }
            assert!(poll!(&mut opening).is_pending());
        }
        let outputs = async {
            opening
                .await
                .expect("held RPC must complete as a cancelled stream")
                .collect::<Outputs>()
                .await
        };
        tokio::pin!(outputs);
        tokio::select! {
            _ = bounded("pending opening", handle.wait(Event::Received)) => {}
            _ = &mut outputs => panic!("generation finished before cancellation"),
        }
        ctx.stop_generating();
        terminal(outputs.await, &[], 3, FinishReason::Cancelled);
        bounded("opening drop", handle.wait(Event::Dropped)).await;
    }

    read_isolation(&fixture, &engine, &control, false, expected_prefixes).await;
    (fixture, engine, control)
}

#[tokio::test]
async fn vllm_cancellation_before_open_during_open_and_during_read() {
    bounded(
        "vllm_cancellation_before_open_during_open_and_during_read",
        async {
            let (mut fixture, engine, control) = cancellation::<vllm::Fixture>(
                FixtureConfig {
                    connections: 2,
                    ..Default::default()
                },
                Some((2, 1)),
                true,
            )
            .await;
            healthy(&fixture, &engine, &control).await;
            finish(&mut fixture, &engine).await;
        },
    )
    .await;
}

async fn read_isolation<F: SidecarFixture>(
    fixture: &F,
    engine: &F::Engine,
    control: &Controller<F::Protocol>,
    drop_consumer: bool,
    expected_prefixes: Option<(usize, usize)>,
) {
    let ctx_a = mock_context();
    let handle_a = control.request(ctx_a.id(), after_token_responses(2, StreamAction::Continue));
    let mut stream_a = bounded(
        "request A opening",
        engine.generate(
            request("mocker-model", vec![11, 22, 33], 4),
            GenerateContext::new(ctx_a.clone(), None),
        ),
    )
    .await
    .unwrap();
    let mut outputs_a = checkpoint_outputs(&mut stream_a, &handle_a).await;
    let tokens_a = handle_a.tokens();
    if let Some((expected, _)) = expected_prefixes {
        assert_eq!(outputs_a.len(), expected);
        assert!(
            outputs_a
                .iter()
                .all(|output| output.as_ref().unwrap().token_ids.len() == 1)
        );
        assert_eq!(tokens_a.len(), expected);
    }

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
    if let Some((_, expected)) = expected_prefixes {
        assert_eq!(outputs_b.len(), expected);
        assert!(
            outputs_b
                .iter()
                .all(|output| output.as_ref().unwrap().token_ids.len() == 1)
        );
        assert_eq!(tokens_b.len(), expected);
    }
    assert!(poll!(stream_a.next()).is_pending());
    assert!(poll!(stream_b.next()).is_pending());

    if drop_consumer {
        drop(stream_a);
    } else {
        ctx_a.stop_generating();
        outputs_a.extend(bounded("request A cancellation", stream_a.collect::<Outputs>()).await);
        assert_eq!(handle_a.tokens(), tokens_a);
        terminal(outputs_a, &tokens_a, 3, FinishReason::Cancelled);
    }
    bounded("request A drop", handle_a.wait(Event::Dropped)).await;
    assert!(!handle_b.reached(Event::Dropped));
    assert_eq!(handle_b.tokens(), tokens_b);
    assert!(poll!(stream_b.next()).is_pending());

    handle_b.release();
    outputs_b.extend(bounded("request B completion", stream_b.collect::<Outputs>()).await);
    assert_eq!(handle_b.tokens().len(), 5);
    terminal(outputs_b, &handle_b.tokens(), 5, FinishReason::Length);
    bounded("request B drop", handle_b.wait(Event::Dropped)).await;
    drained(fixture).await;
}

async fn active_work<F: WireFixture>(explicit_cancel: bool) {
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
    let first = bounded("active request first token", stream.next())
        .await
        .unwrap()
        .unwrap();
    assert!(first.finish_reason.is_none());
    bounded("active request checkpoint", handle.wait(Event::Checkpoint)).await;
    fixture.scheduler_active().await;
    if explicit_cancel {
        ctx.stop_generating();
        let mut outputs = vec![Ok(first)];
        outputs.extend(
            bounded(
                "active cancellation terminal",
                stream.by_ref().collect::<Outputs>(),
            )
            .await,
        );
        terminal(outputs, &handle.tokens(), 3, FinishReason::Cancelled);
    }
    drop(stream);
    bounded("active request remote drop", handle.wait(Event::Dropped)).await;
    fixture.scheduler_idle().await;
    healthy(&fixture, &engine, &control).await;
    if !explicit_cancel {
        read_isolation(&fixture, &engine, &control, true, Some((2, 1))).await;
        healthy(&fixture, &engine, &control).await;
    }
    engine.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn vllm_explicit_cancel_releases_active_scheduler_work() {
    bounded(
        "active cancellation and recovery",
        active_work::<vllm::Fixture>(true),
    )
    .await;
}

#[tokio::test]
async fn vllm_consumer_drop_releases_active_scheduler_work() {
    bounded(
        "consumer drop and recovery",
        active_work::<vllm::Fixture>(false),
    )
    .await;
}
