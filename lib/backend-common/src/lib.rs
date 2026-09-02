// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared runtime glue for Rust LLM backends.
//!
//! Two-type abstraction: [`LLMEngine`] (the engine trait an author implements)
//! and [`Worker`] (the runtime lifecycle owner), plus a [`run()`] helper called
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
mod rl;
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
    AsyncEngineContext, BootstrapInfo, CompletionUsage, ComponentSnapshot, DraftCleanupOutcomeV1,
    DraftTransportDescriptorV1, EXTERNAL_SPECULATION_LIFECYCLE_ENGINE_DATA_KEY, Endpoint,
    EndpointId, EngineConfig, ExternalDraftBinding, ExternalSpeculationLifecycleV1, FinishReason,
    FirstTokenNotifier, GenerateContext, GuidedDecodingOptions, HEALTH_CHECK_KEY, KvEventPublisher,
    KvEventSource, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, LlmRegistration, LogProbs,
    Metrics, MetricsBindings, MetricsCtx, ModelRegistration, MultimodalData, OnPublisherReady,
    OnSnapshotPublisherReady, OutputOptions, PrefillResult, PreprocessedRequest, RawEngine,
    RouterHintEnvelope, SamplingOptions, SpeculativeDecodingRouterHintV1, StopConditions,
    StopReason, TopLogprob, TopLogprobs, WorkerRole, WorkerWithDpRank, chunk,
    new_external_speculation_incarnation, usage, validate_endpoint_id,
};
pub use error::{BackendError, DynamoError, ErrorType};
pub use metrics::{ComponentGauges, EngineMetrics, LifecycleGauges};
pub use rl::{RlAdminBaseUrl, RlWorkerMetadata};
pub use run::{run, run_raw};
pub use snapshot_publisher::SnapshotPublisher;
pub use worker::{RuntimeConfig, Worker, WorkerConfig};
