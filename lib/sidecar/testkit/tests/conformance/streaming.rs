// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

async fn streaming<F: SidecarFixture>() {
    let control = Controller::<F::Protocol>::default();
    let config = FixtureConfig {
        model: "alternate-mocker".into(),
        connections: 2,
        ..Default::default()
    };
    let mut fixture = bounded("Mocker startup", F::start(control.clone(), config)).await;
    let engine = bounded("sidecar construction", fixture.engine()).await;
    bounded("sidecar startup", engine.start(0)).await.unwrap();
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
        let mut req = request("alternate-mocker", vec![10, 20, 30, 40, 50], 7);
        req.sampling_options.temperature = Some(0.125);
        req.sampling_options.presence_penalty = Some(0.25);
        req.sampling_options.frequency_penalty = Some(0.75);
        req.output_options.logprobs = Some(2);
        req.output_options.prompt_logprobs = Some(1);
        let outputs = collect(&engine, req.clone(), GenerateContext::new(ctx, None)).await;
        F::assert_stream(&handle, &req, &outputs);
        let tokens = handle.tokens();
        assert_eq!(tokens.len(), 7);
        terminal(outputs, &tokens, 5, FinishReason::Length);
        bounded("server stream drop", handle.wait(Event::Dropped)).await;
        assert_eq!(handle.reached(Event::Checkpoint), plan.stream.is_some());
        drained(&fixture).await;
    }
    bounded("sidecar cleanup", engine.cleanup()).await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn vllm_tokens_terminal_logprobs_and_usage() {
    bounded(
        "vllm_tokens_terminal_logprobs_and_usage",
        streaming::<vllm::Fixture>(),
    )
    .await;
}
