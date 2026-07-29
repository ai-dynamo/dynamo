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
pub mod metrics;
mod publisher;
pub mod run;
pub mod snapshot_publisher;
pub mod telemetry;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
#[cfg(debug_assertions)]
mod validate;
pub mod worker;

pub use args::CommonArgs;
pub use disagg::DisaggregationMode;
pub use dynamo_llm::model_type::ModelInput;
pub use engine::{
    AsyncEngineContext, BootstrapInfo, CompletionUsage, ComponentSnapshot, EngineConfig,
    FinishReason, GenerateContext, GuidedDecodingOptions, HEALTH_CHECK_KEY, KvEventPublisher,
    KvEventSource, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, LlmRegistration, LogProbs,
    Metrics, MetricsBindings, MetricsCtx, OnPublisherReady, OnSnapshotPublisherReady,
    OutputOptions, PrefillResult, PreprocessedRequest, RawEngine, SamplingOptions, StopConditions,
    StopReason, TopLogprob, TopLogprobs, chunk, usage,
};
pub use error::{BackendError, DynamoError, ErrorType};
pub use metrics::{ComponentGauges, EngineMetrics, LifecycleGauges};
pub use run::{run, run_raw};
pub use snapshot_publisher::SnapshotPublisher;
pub use worker::{RuntimeConfig, Worker, WorkerConfig};
