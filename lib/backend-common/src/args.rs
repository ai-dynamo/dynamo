// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Common CLI flags every Rust backend needs.
//!
//! Flag names match the Python runtime's `WorkerConfig` knobs so operators see
//! the same CLI surface across Rust and Python backends. Engines extend this
//! with their own `clap` `Args` using `#[command(flatten)]`.

use std::path::PathBuf;

use clap::Args;

use crate::disagg::DisaggregationMode;

/// CLI flags that every Rust backend needs. Flatten these into your engine's
/// `Args` struct:
///
/// ```ignore
/// #[derive(clap::Parser)]
/// struct EngineArgs {
///     #[clap(flatten)]
///     common: CommonArgs,
///
///     #[arg(long, default_value = "sample-model")]
///     model_name: String,
/// }
/// ```
#[derive(Args, Clone, Debug)]
pub struct CommonArgs {
    /// Dynamo namespace for discovery routing.
    #[arg(long, default_value = "dynamo", env = "DYN_NAMESPACE")]
    pub namespace: String,

    /// Component name within the namespace.
    #[arg(long, default_value = "backend", env = "DYN_COMPONENT")]
    pub component: String,

    /// Endpoint name exposed by this worker.
    #[arg(long, default_value = "generate", env = "DYN_ENDPOINT")]
    pub endpoint: String,

    /// Comma-separated list of supported endpoint types
    /// (`chat`, `completions`, `embeddings`, ...).
    #[arg(long, default_value = "chat,completions", env = "DYN_ENDPOINT_TYPES")]
    pub endpoint_types: String,

    /// Optional path to a custom Jinja chat template, used instead of the
    /// template shipped with the model.
    #[arg(long, env = "DYN_CUSTOM_JINJA_TEMPLATE")]
    pub custom_jinja_template: Option<PathBuf>,

    /// Dynamo frontend tool-call parser name for this model.
    #[arg(long = "dyn-tool-call-parser", env = "DYN_TOOL_CALL_PARSER")]
    pub dyn_tool_call_parser: Option<String>,

    /// Dynamo frontend reasoning parser name for this model.
    #[arg(long = "dyn-reasoning-parser", env = "DYN_REASONING_PARSER")]
    pub dyn_reasoning_parser: Option<String>,

    /// Exclude tools from the chat template when tool_choice is none.
    #[arg(
        long = "exclude-tools-when-tool-choice-none",
        env = "DYN_EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE",
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
    pub exclude_tools_when_tool_choice_none: bool,

    /// Disaggregation role: `agg` (default), `prefill`, `decode`, or `encode`.
    /// Prefill workers register with `ModelType::empty()` and
    /// `WorkerType::Prefill` regardless of `endpoint_types`; decode and encode
    /// workers do not advertise a local KV indexer. Encode workers register as
    /// `WorkerType::Encode` and are not exposed on the public chat/completions
    /// surface.
    #[arg(
        long,
        value_enum,
        default_value_t = DisaggregationMode::Aggregated,
        env = "DYN_DISAGGREGATION_MODE",
    )]
    pub disaggregation_mode: DisaggregationMode,

    /// Declare an upstream Encode peer in this worker's topology `needs`.
    /// Meaningful only on `--disaggregation-mode agg` and `prefill`.
    /// Setting it on `decode` or `encode` is rejected at startup.
    ///
    /// Scope: Rust backends consuming `CommonArgs` via clap. Python
    /// backends populate this from their own runtime config -- the Python
    /// shim does not read this env var.
    #[arg(long, default_value_t = false, env = "DYN_ROUTE_TO_ENCODER")]
    pub route_to_encoder: bool,
}
