// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reference `LLMEngine` implementation.
//!
//! Generates rotating token IDs with configurable per-token latency. Useful
//! for testing the `Worker` lifecycle end-to-end and as a template for engine
//! leads implementing real backends.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use clap::Parser;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, CommonArgs, EngineConfig, FinishReason, LLMEngine,
    LLMEngineOutput, PreprocessedRequest, WorkerConfig, chunk, usage,
};
use futures::stream::BoxStream;

#[derive(Parser, Debug)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    about = "Reference Dynamo Rust backend — generates rotating token IDs."
)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    /// Friendly model name advertised to the frontend.
    #[arg(long, default_value = "sample-model")]
    model_name: String,

    /// HF repo or local path providing tokenizer + chat template. Leave
    /// empty for name-only registration (no templating).
    #[arg(long, default_value = "")]
    model_path: String,

    /// How many tokens to emit per request.
    #[arg(long, default_value_t = 16)]
    max_tokens: usize,

    /// Delay between emitted tokens, in milliseconds.
    #[arg(long, default_value_t = 10)]
    delay_ms: u64,
}

pub struct RotatingTokensEngine {
    model_name: String,
    max_tokens: usize,
    delay: Duration,
}

impl RotatingTokensEngine {
    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), BackendError> {
        let args = match argv {
            Some(a) => Args::try_parse_from(a),
            None => Args::try_parse(),
        }
        .map_err(|e| BackendError::invalid(e.to_string()))?;

        let engine = RotatingTokensEngine {
            model_name: args.model_name.clone(),
            max_tokens: args.max_tokens,
            delay: Duration::from_millis(args.delay_ms),
        };

        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            model_name: args.model_path,
            served_model_name: Some(args.model_name),
            ..Default::default()
        };
        Ok((engine, config))
    }
}

#[async_trait]
impl LLMEngine for RotatingTokensEngine {
    async fn start(&self) -> Result<EngineConfig, BackendError> {
        Ok(EngineConfig {
            model: self.model_name.clone(),
            served_model_name: Some(self.model_name.clone()),
            context_length: Some(2048),
            kv_cache_block_size: Some(16),
            total_kv_blocks: Some(1000),
            max_num_seqs: Some(64),
            max_num_batched_tokens: Some(2048),
        })
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, LLMEngineOutput>, BackendError> {
        let max_new = request
            .stop_conditions
            .max_tokens
            .map(|n| n as usize)
            .unwrap_or(self.max_tokens);
        let delay = self.delay;
        let prompt_len = request.token_ids.len() as u32;

        Ok(Box::pin(async_stream::stream! {
            for i in 0..max_new {
                if ctx.is_stopped() {
                    yield chunk::cancelled(usage(prompt_len, i as u32));
                    break;
                }
                tokio::time::sleep(delay).await;
                let token_id = ((i as u32) + 1) % 32000;

                if i == max_new - 1 {
                    yield chunk::terminal(
                        vec![token_id],
                        FinishReason::Length,
                        usage(prompt_len, max_new as u32),
                    );
                } else {
                    yield chunk::token(token_id);
                }
            }
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        tracing::debug!(
            request_id = ctx.id(),
            "rotating_tokens engine: abort requested"
        );
    }

    async fn cleanup(&self) -> Result<(), BackendError> {
        tracing::info!("rotating_tokens engine: cleanup invoked");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_backend_common::{SamplingOptions, StopConditions};
    use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
    use futures::StreamExt;

    fn build_engine(max_tokens: usize) -> RotatingTokensEngine {
        RotatingTokensEngine {
            model_name: "sample".to_string(),
            max_tokens,
            delay: Duration::from_millis(0),
        }
    }

    fn request(max_tokens: Option<u32>) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("sample".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions {
                max_tokens,
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(Default::default())
            .build()
            .unwrap()
    }

    #[test]
    fn args_parse_with_defaults() {
        let args = Args::try_parse_from(["bin"]).unwrap();
        assert_eq!(args.model_name, "sample-model");
        assert_eq!(args.max_tokens, 16);
        assert_eq!(args.delay_ms, 10);
        assert_eq!(args.common.namespace, "dynamo");
        assert_eq!(args.common.component, "backend");
    }

    #[test]
    fn args_parse_overrides() {
        let args = Args::try_parse_from([
            "bin",
            "--model-name",
            "foo",
            "--max-tokens",
            "3",
            "--namespace",
            "ns",
        ])
        .unwrap();
        assert_eq!(args.model_name, "foo");
        assert_eq!(args.max_tokens, 3);
        assert_eq!(args.common.namespace, "ns");
    }

    #[tokio::test]
    async fn generate_emits_rotating_token_ids() {
        let engine = build_engine(16);
        let ctx = Context::new(());
        let ctrl = ctx.context();
        let stream = engine
            .generate(request(Some(3)), ctrl)
            .await
            .expect("stream");
        let chunks: Vec<_> = stream.collect().await;

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].token_ids, vec![1]);
        assert_eq!(chunks[1].token_ids, vec![2]);
        assert_eq!(chunks[2].token_ids, vec![3]);
        assert!(matches!(
            chunks[2].finish_reason,
            Some(FinishReason::Length)
        ));
        let u = chunks[2].completion_usage.as_ref().unwrap();
        assert_eq!(u.prompt_tokens, 3);
        assert_eq!(u.completion_tokens, 3);
    }

    #[tokio::test]
    async fn generate_falls_back_to_engine_max_when_unset() {
        let engine = build_engine(2);
        let ctx = Context::new(());
        let ctrl = ctx.context();
        let stream = engine.generate(request(None), ctrl).await.expect("stream");
        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 2);
    }

    #[tokio::test]
    async fn generate_cancellation_yields_cancelled_chunk() {
        let engine = RotatingTokensEngine {
            model_name: "sample".to_string(),
            max_tokens: 100,
            delay: Duration::from_millis(20),
        };
        let ctx = Context::new(());
        let ctrl = ctx.context();
        let mut stream = engine
            .generate(request(Some(100)), ctrl.clone())
            .await
            .expect("stream");

        let _first = stream.next().await.expect("first chunk");
        ctrl.stop_generating();

        let rest: Vec<_> = stream.collect().await;
        assert_eq!(rest.len(), 1, "expected a single terminal cancelled chunk");
        assert!(rest[0].token_ids.is_empty());
        assert!(matches!(
            rest[0].finish_reason,
            Some(FinishReason::Cancelled)
        ));
    }

    #[tokio::test]
    async fn start_returns_advertised_metadata() {
        let engine = build_engine(16);
        let cfg = engine.start().await.unwrap();
        assert_eq!(cfg.model, "sample");
        assert_eq!(cfg.context_length, Some(2048));
        assert_eq!(cfg.kv_cache_block_size, Some(16));
    }

    #[tokio::test]
    async fn rotating_tokens_passes_conformance() {
        let engine = build_engine(4);
        dynamo_backend_common::testing::run_conformance(engine)
            .await
            .expect("rotating_tokens engine must satisfy conformance");
    }
}
