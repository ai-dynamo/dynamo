// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared Dynamo-facing scenarios; native servers and protocol expectations stay with each mocker.

use std::{future::Future, sync::Arc, time::Duration};

use dynamo_backend_common::{
    AsyncEngineContext, BackendError, ErrorType, FinishReason, GenerateContext, LLMEngine,
    LLMEngineOutput, OutputOptions, PreprocessedRequest, SamplingOptions, StopConditions,
};
use dynamo_mocker::scheduler::MockerMetrics;
use futures::StreamExt;
use tokio::sync::watch;

pub use dynamo_backend_common::testing::mock_context as context;

const PHASE_TIMEOUT: Duration = Duration::from_secs(5);

async fn bounded<T>(phase: &str, future: impl Future<Output = T>) -> T {
    tokio::time::timeout(PHASE_TIMEOUT, future)
        .await
        .unwrap_or_else(|_| panic!("{phase} exceeded {PHASE_TIMEOUT:?}"))
}

pub fn request(max_tokens: u32) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("mocker-model".to_string())
        .token_ids(vec![11, 22, 33, 44])
        .stop_conditions(StopConditions {
            max_tokens: Some(max_tokens),
            ignore_eos: Some(true),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            temperature: Some(0.0),
            ..Default::default()
        })
        .output_options(OutputOptions {
            logprobs: Some(2),
            prompt_logprobs: Some(1),
            ..Default::default()
        })
        .build()
        .expect("valid token request")
}

async fn collect(
    engine: &impl LLMEngine,
    ctx: Arc<dyn AsyncEngineContext>,
) -> Vec<LLMEngineOutput> {
    bounded("generate and consume", async {
        engine
            .generate(request(3), GenerateContext::new(ctx, None))
            .await
            .expect("open generation")
            .map(|item| item.expect("successful stream item"))
            .collect()
            .await
    })
    .await
}

fn assert_completion(outputs: &[LLMEngineOutput]) {
    assert!(!outputs.is_empty(), "stream must not silently end");
    assert_eq!(
        outputs.iter().filter(|o| o.finish_reason.is_some()).count(),
        1
    );
    let terminal = outputs.last().unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
    assert_eq!(outputs.iter().map(|o| o.token_ids.len()).sum::<usize>(), 3);
    let usage = terminal.completion_usage.as_ref().expect("terminal usage");
    assert_eq!(
        (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens
        ),
        (4, 3, 7)
    );
}

/// Compare sidecar output with independently observed native tokens for the same request ID.
pub async fn streaming(
    engine: &impl LLMEngine,
    ctx: Arc<dyn AsyncEngineContext>,
    expected_tokens: &[u32],
    expected_logprobs: &[f64],
    top_logprobs: usize,
) {
    let outputs = collect(engine, ctx).await;
    assert_completion(&outputs);
    assert_eq!(
        outputs.len(),
        expected_tokens.len(),
        "native mocker streams one token per chunk"
    );
    assert!(outputs.iter().all(|output| output.token_ids.len() == 1));
    assert_eq!(
        outputs
            .iter()
            .flat_map(|o| o.token_ids.iter().copied())
            .collect::<Vec<_>>(),
        expected_tokens,
        "sidecar must preserve native token order without cumulative replay"
    );
    assert_eq!(
        outputs
            .iter()
            .flat_map(|o| o
                .log_probs
                .as_ref()
                .expect("selected logprobs")
                .iter()
                .copied())
            .collect::<Vec<_>>(),
        expected_logprobs,
        "sidecar must preserve selected logprob values and token alignment"
    );
    for output in &outputs {
        assert_eq!(
            output.log_probs.as_ref().expect("selected logprobs").len(),
            output.token_ids.len()
        );
        let alternatives = output.top_logprobs.as_ref().expect("top logprobs");
        assert_eq!(alternatives.len(), output.token_ids.len());
        assert!(
            alternatives
                .iter()
                .all(|tokens| tokens.len() == top_logprobs)
        );
    }
    assert!(
        outputs
            .last()
            .unwrap()
            .engine_data
            .as_ref()
            .expect("prompt data")["prompt_logprobs"]
            .is_array()
    );
}

pub async fn recovery(engine: &impl LLMEngine) {
    assert_completion(&collect(engine, context()).await);
}

/// Exercise native admission failure, including adapters that open the RPC lazily.
pub async fn rejection(engine: &impl LLMEngine) {
    bounded("native request rejection", async {
        let error = match engine
            .generate(request(1_000_001), GenerateContext::new(context(), None))
            .await
        {
            Err(error) => error,
            Ok(mut stream) => {
                let error = stream
                    .next()
                    .await
                    .expect("native failure must reach caller")
                    .expect_err("request exceeds peer output limit");
                assert!(stream.next().await.is_none(), "no output after rejection");
                error
            }
        };
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(
            error.to_string().contains("max_new_tokens"),
            "native reason was lost: {error}"
        );
    })
    .await;
    recovery(engine).await;
}

pub async fn cancellation(engine: &impl LLMEngine, active: impl Fn() -> usize) {
    bounded("cancel after first token", async {
        let ctx = context();
        let mut stream = engine
            .generate(
                request(10_000),
                GenerateContext::new(Arc::clone(&ctx), None),
            )
            .await
            .expect("open long generation");
        let first = stream
            .next()
            .await
            .expect("first chunk")
            .expect("first token");
        assert!(!first.token_ids.is_empty());
        assert!(
            first.finish_reason.is_none(),
            "request completed before cancellation"
        );
        assert!(active() > 0, "remote work completed before cancellation");
        ctx.stop_generating();
        let terminal = stream
            .next()
            .await
            .expect("cancelled terminal")
            .expect("cancellation is terminal output");
        assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
        let usage = terminal.completion_usage.as_ref().expect("cancelled usage");
        assert_eq!(usage.prompt_tokens, 4);
        assert_eq!(usage.completion_tokens as usize, first.token_ids.len());
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );
        assert!(
            terminal.token_ids.is_empty(),
            "no tokens after cancellation"
        );
        assert!(
            stream.next().await.is_none(),
            "no output after cancelled terminal"
        );
    })
    .await;
}

pub async fn consumer_drop(engine: &impl LLMEngine, active: impl Fn() -> usize) {
    bounded("abandon consumer after first token", async {
        let ctx = context();
        let mut stream = engine
            .generate(
                request(10_000),
                GenerateContext::new(Arc::clone(&ctx), None),
            )
            .await
            .expect("open long generation");
        let first = stream
            .next()
            .await
            .expect("first chunk")
            .expect("first token");
        assert!(!first.token_ids.is_empty());
        assert!(
            first.finish_reason.is_none(),
            "request completed before consumer drop"
        );
        assert!(active() > 0, "remote work completed before consumer drop");
        drop(stream);
        assert!(
            !ctx.is_stopped(),
            "consumer drop must be tested without explicit cancellation"
        );
    })
    .await;
}

pub async fn wait_idle(mut metrics: watch::Receiver<MockerMetrics>, active: impl Fn() -> usize) {
    let result = tokio::time::timeout(PHASE_TIMEOUT, async {
        let mut routes = tokio::time::interval(Duration::from_millis(10));
        loop {
            let idle = {
                let snapshot = metrics.borrow_and_update();
                active() == 0 && snapshot.running_requests == 0 && snapshot.waiting_requests == 0
            };
            if idle { break; }
            // Route removal can follow the last scheduler metrics notification.
            tokio::select! {
                changed = metrics.changed() => changed.expect("scheduler metrics closed before idle"),
                _ = routes.tick() => {},
            }
        }
    }).await;
    let snapshot = metrics.borrow();
    assert!(
        result.is_ok(),
        "remote work did not drain: active={}, running={}, waiting={}",
        active(),
        snapshot.running_requests,
        snapshot.waiting_requests
    );
}
