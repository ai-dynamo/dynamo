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
pub mod engine;
pub mod error;
pub mod run;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
#[cfg(debug_assertions)]
mod validate;
pub mod worker;

pub use args::CommonArgs;
pub use engine::{
    AsyncEngineContext, CompletionUsage, EngineConfig, FinishReason, LLMEngine, LLMEngineOutput,
    LLMEngineOutputExt, OutputOptions, PreprocessedRequest, SamplingOptions, StopConditions, chunk,
    usage,
};
pub use error::{BackendError, DynamoError, ErrorType};
pub use run::run;
pub use worker::{Worker, WorkerConfig};
