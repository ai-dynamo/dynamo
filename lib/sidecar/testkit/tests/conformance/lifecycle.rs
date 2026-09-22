// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) async fn cleanup<F: SidecarFixture>(
    needs_unstarted_check: bool,
) -> (F, F::Engine, Controller<F::Protocol>) {
    let control = Controller::<F::Protocol>::default();
    let fixture = bounded(
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
            request("mocker-model", vec![11, 22, 33], 3),
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
    (fixture, engine, control)
}

#[tokio::test]
async fn vllm_cleanup_during_read_and_post_cleanup_admission() {
    bounded(
        "vllm_cleanup_during_read_and_post_cleanup_admission",
        async {
            let (mut fixture, engine, control) = cleanup::<vllm::Fixture>(false).await;
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
            fixture.shutdown().await;
        },
    )
    .await;
}

async fn teardown_with_live_clients<F: WireFixture>() {
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
    let first = bounded("teardown first token", stream.next())
        .await
        .unwrap()
        .unwrap();
    assert!(first.finish_reason.is_none());
    bounded("teardown checkpoint", handle.wait(Event::Checkpoint)).await;
    fixture.scheduler_active().await;
    fixture.shutdown().await;
    bounded("owned RPC handler terminated", handle.wait(Event::Dropped)).await;
    fixture.scheduler_idle().await;
    let remaining = bounded(
        "client observes server termination",
        stream.collect::<Outputs>(),
    )
    .await;
    let mut outputs = vec![Ok(first)];
    outputs.extend(remaining);
    let error = failure(outputs, &handle.tokens(), BackendError::Unknown);
    assert!(error.to_string().contains("GenerateStream"));
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn vllm_teardown_terminates_handlers_with_clients_alive() {
    bounded(
        "server teardown with live clients",
        teardown_with_live_clients::<vllm::Fixture>(),
    )
    .await;
}
