// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `LLMEngine` trait plus registration-metadata and output-construction helpers.
//!
//! The trait takes the same `PreprocessedRequest` / `LLMEngineOutput` types used
//! across preprocessing, routing, and the frontend — no separate data-shape
//! translation layer for Rust engines.
//!
//! Object-safety: every instance method takes `&self`. `Arc<dyn LLMEngine>` is
//! the handle `Worker` drives the lifecycle through.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::BackendError;

pub use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
pub use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
pub use dynamo_llm::protocols::common::{
    FinishReason, OutputOptions, SamplingOptions, StopConditions,
};
pub use dynamo_protocols::types::CompletionUsage;
pub use dynamo_runtime::engine::AsyncEngineContext;

/// Registration metadata returned by [`LLMEngine::start`].
///
/// `Worker` consumes this to build a `ModelDeploymentCard` and register the
/// model with discovery. `None` on an optional field means "don't advertise":
/// the router sees no value and falls back to round-robin (for scheduling
/// hints) or its configured defaults. Engines without a traditional KV cache
/// can leave `kv_cache_block_size` and `total_kv_blocks` unset.
#[derive(Clone, Debug, Default)]
pub struct EngineConfig {
    /// Canonical model identifier (e.g. HF repo name).
    pub model: String,
    /// Public-facing model name advertised to clients. Defaults to `model`.
    pub served_model_name: Option<String>,
    /// Maximum context length the engine supports, in tokens.
    pub context_length: Option<u32>,
    /// KV cache block size, in tokens. Used by KV-aware routing. `None`
    /// means the engine has no block-structured KV cache; KV-aware routing
    /// falls back to round-robin for this backend.
    pub kv_cache_block_size: Option<u32>,
    /// Total number of KV cache blocks available to the engine. `None`
    /// means "not advertised"; the planner treats the backend as having
    /// no KV-capacity hint.
    pub total_kv_blocks: Option<u64>,
    /// Maximum number of concurrent in-flight sequences.
    pub max_num_seqs: Option<u64>,
    /// Maximum tokens the engine will process in a single batched step.
    pub max_num_batched_tokens: Option<u64>,
}

/// Inference engine trait.
///
/// Lifecycle:
///   1. Construct the engine (typically via a backend-specific `from_args`).
///   2. `start()` — start the engine, return `EngineConfig` metadata.
///   3. `generate()` — called for each request (concurrent calls expected).
///   4. `abort()` — called when a request is cancelled (optional, default no-op).
///   5. `cleanup()` — called once on shutdown, release all resources.
#[async_trait]
pub trait LLMEngine: Send + Sync + 'static {
    /// Start the engine and return registration metadata.
    ///
    /// After this returns, the engine MUST be ready to accept `generate()`
    /// calls. `Worker` will register the model and begin serving immediately.
    /// Use interior mutability for any state allocated here.
    ///
    /// `start()` is async and may take minutes for real backends (e.g.
    /// compiling a model graph on an accelerator). Emit
    /// `tracing::info!` checkpoints so operators see progress — this
    /// call is otherwise a silent window between process launch and
    /// endpoint serving.
    async fn start(&self) -> Result<EngineConfig, BackendError>;

    /// Yield streaming response chunks for a single request.
    ///
    /// Called concurrently for multiple in-flight requests. The returned
    /// stream MUST poll `ctx.is_stopped()` between yields; on cancellation,
    /// emit a terminal chunk with `FinishReason::Cancelled`.
    ///
    /// Contract: exactly one terminal chunk (carrying both `finish_reason`
    /// and `completion_usage`) must be the last item yielded. In debug
    /// builds, the framework wraps the stream in a validator that panics
    /// on contract violations.
    ///
    /// The returned stream is `'static`: clone or move any state from
    /// `&self` or `request` into the stream body before constructing it
    /// (the `chunk::*` helpers in this module work well for this).
    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, LLMEngineOutput>, BackendError>;

    /// Abort an in-flight request (optional, default no-op).
    ///
    /// Called by the framework only when `ctx.stopped()` or `ctx.killed()`
    /// fires — i.e. when the client or operator explicitly cancels. It is
    /// NOT called when the response stream is simply dropped (e.g. TCP
    /// reset, consumer-side timeout without cancellation).
    ///
    /// For cleanup that must happen on ANY drop path (releasing an
    /// accelerator slot, freeing a request handle), put the release logic
    /// inside the `generate` stream body using RAII — a guard whose
    /// `Drop` runs when the stream is dropped, however that happens. Use
    /// `abort` only for out-of-band notifications (e.g. telling a remote
    /// scheduler to cancel compute early).
    async fn abort(&self, _ctx: Arc<dyn AsyncEngineContext>) {}

    /// Release all engine resources. Called once on shutdown.
    async fn cleanup(&self) -> Result<(), BackendError>;
}

/// Convenience constructors for [`LLMEngineOutput`]. Cuts per-chunk boilerplate.
pub mod chunk {
    use super::{CompletionUsage, FinishReason, LLMEngineOutput};

    /// Non-terminal chunk carrying a single token.
    pub fn token(id: u32) -> LLMEngineOutput {
        LLMEngineOutput {
            token_ids: vec![id],
            ..Default::default()
        }
    }

    /// Terminal chunk. `token_ids` may be empty or carry the final tokens.
    pub fn terminal(
        token_ids: Vec<u32>,
        finish_reason: FinishReason,
        usage: CompletionUsage,
    ) -> LLMEngineOutput {
        LLMEngineOutput {
            token_ids,
            finish_reason: Some(finish_reason),
            completion_usage: Some(usage),
            ..Default::default()
        }
    }

    /// Terminal chunk for a cancelled request. `token_ids` is empty.
    pub fn cancelled(usage: CompletionUsage) -> LLMEngineOutput {
        terminal(vec![], FinishReason::Cancelled, usage)
    }
}

/// Build a [`CompletionUsage`] from prompt and completion counts.
pub fn usage(prompt_tokens: u32, completion_tokens: u32) -> CompletionUsage {
    CompletionUsage {
        prompt_tokens,
        completion_tokens,
        // saturating_add: no real LLM context approaches u32::MAX, but
        // defensively avoid overflow panic in debug builds.
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
        prompt_tokens_details: None,
        completion_tokens_details: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_token_sets_only_token_ids() {
        let c = chunk::token(42);
        assert_eq!(c.token_ids, vec![42]);
        assert!(c.finish_reason.is_none());
        assert!(c.completion_usage.is_none());
    }

    #[test]
    fn chunk_terminal_sets_all_required_fields() {
        let c = chunk::terminal(vec![1, 2], FinishReason::Length, usage(10, 2));
        assert_eq!(c.token_ids, vec![1, 2]);
        assert!(matches!(c.finish_reason, Some(FinishReason::Length)));
        let u = c.completion_usage.unwrap();
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 2);
        assert_eq!(u.total_tokens, 12);
    }

    #[test]
    fn chunk_cancelled_has_empty_tokens_and_cancelled_reason() {
        let c = chunk::cancelled(usage(5, 3));
        assert!(c.token_ids.is_empty());
        assert!(matches!(c.finish_reason, Some(FinishReason::Cancelled)));
    }

    #[test]
    fn usage_sums_totals() {
        let u = usage(7, 11);
        assert_eq!(u.total_tokens, 18);
    }
}
