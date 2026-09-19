// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

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

    read_isolation(&fixture, &engine, &control, false).await;
    bounded("sidecar cleanup", engine.cleanup()).await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn vllm_cancellation_before_open_during_open_and_during_read() {
    bounded(
        "vllm_cancellation_before_open_during_open_and_during_read",
        cancellation::<vllm::Fixture>(),
    )
    .await;
}

async fn read_isolation<F: SidecarFixture>(
    fixture: &F,
    engine: &F::Engine,
    control: &Controller<F::Protocol>,
    drop_consumer: bool,
) {
    let ctx_a = mock_context();
    let handle_a = control.request(ctx_a.id(), after_tokens(2, StreamAction::Continue));
    let mut stream_a = bounded(
        "request A opening",
        engine.generate(
            request("mocker-model", vec![11, 22, 33], 4),
            GenerateContext::new(ctx_a.clone(), None),
        ),
    )
    .await
    .unwrap();
    let mut outputs_a = Vec::new();
    for _ in 0..2 {
        let output = bounded("request A token", stream_a.next()).await.unwrap();
        assert_eq!(output.as_ref().unwrap().token_ids.len(), 1);
        outputs_a.push(output);
    }
    bounded("request A checkpoint", handle_a.wait(Event::Checkpoint)).await;

    let ctx_b = mock_context();
    let handle_b = control.request(ctx_b.id(), after_tokens(1, StreamAction::Continue));
    let mut stream_b = bounded(
        "request B opening",
        engine.generate(
            request("mocker-model", vec![90, 80, 70, 60, 50], 5),
            GenerateContext::new(ctx_b, None),
        ),
    )
    .await
    .unwrap();
    let first_b = bounded("request B token", stream_b.next()).await.unwrap();
    assert_eq!(first_b.as_ref().unwrap().token_ids.len(), 1);
    bounded("request B checkpoint", handle_b.wait(Event::Checkpoint)).await;
    assert!(poll!(stream_a.next()).is_pending());
    assert!(poll!(stream_b.next()).is_pending());

    if drop_consumer {
        drop(stream_a);
    } else {
        ctx_a.stop_generating();
        outputs_a.extend(bounded("request A cancellation", stream_a.collect::<Outputs>()).await);
        assert_eq!(handle_a.tokens().len(), 2);
        terminal(outputs_a, &handle_a.tokens(), 3, FinishReason::Cancelled);
    }
    bounded("request A drop", handle_a.wait(Event::Dropped)).await;
    assert!(!handle_b.reached(Event::Dropped));
    assert_eq!(handle_b.tokens().len(), 1);
    assert!(poll!(stream_b.next()).is_pending());

    handle_b.release();
    let mut outputs_b = vec![first_b];
    outputs_b.extend(bounded("request B completion", stream_b.collect::<Outputs>()).await);
    assert_eq!(handle_b.tokens().len(), 5);
    terminal(outputs_b, &handle_b.tokens(), 5, FinishReason::Length);
    bounded("request B drop", handle_b.wait(Event::Dropped)).await;
    drained(fixture).await;
    healthy(fixture, engine, control).await;
}

async fn active_work<F: SidecarFixture>(explicit_cancel: bool) {
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
    let handle = control.request(ctx.id(), after_tokens(1, StreamAction::Continue));
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
        read_isolation(&fixture, &engine, &control, true).await;
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
