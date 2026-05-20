// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared runtime glue for Rust LLM backends.
//!
//! Two-type abstraction: [`LLMEngine`] (the engine trait an author implements)
//! and [`Worker`] (the runtime lifecycle owner), plus a [`run`] helper called
//! from each backend's `main.rs`.
//!
//! Engines work directly with [`PreprocessedRequest`] and [`LLMEngineOutput`]
//! — the same types the rest of the Rust pipeline uses.
//!
//! See `CLAUDE.md` in this crate for the design contract.

mod adapter;
pub mod args;
pub mod disagg;
pub mod engine;
pub mod error;
mod publisher;
pub mod run;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
#[cfg(debug_assertions)]
mod validate;
pub mod worker;

pub use args::CommonArgs;
pub use disagg::DisaggregationMode;
pub use engine::{
    AsyncEngineContext, BootstrapInfo, CompletionUsage, EngineConfig, FinishReason,
    GenerateContext, KvEventPublisher, KvEventSource, LLMEngine, LLMEngineOutput,
    LLMEngineOutputExt, Metrics, MetricsSource, OnPublisherReady, OutputOptions, PrefillResult,
    PreprocessedRequest, SamplingOptions, SnapshotFn, StopConditions, chunk, usage,
};
pub use error::{BackendError, DynamoError, ErrorType};
pub use run::run;
pub use worker::{RuntimeConfig, Worker, WorkerConfig};

/// Shape of `dynamo_llm::backend::ExecutionContext`; frontends assign one directly.
pub type InProcessEngine = std::sync::Arc<
    dyn dynamo_runtime::pipeline::AsyncEngine<
            dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
            dynamo_runtime::pipeline::ManyOut<
                dynamo_runtime::protocols::annotated::Annotated<LLMEngineOutput>,
            >,
            dynamo_runtime::pipeline::Error,
        >,
>;

/// Wrap an [`LLMEngine`] for embedding into a Dynamo frontend
/// (`InProcessTokens` engine type). Multi-worker deployments use [`run`].
pub fn wrap_in_process(
    engine: std::sync::Arc<dyn LLMEngine>,
    mode: DisaggregationMode,
) -> InProcessEngine {
    std::sync::Arc::new(adapter::EngineAdapter::new(engine, mode))
}
