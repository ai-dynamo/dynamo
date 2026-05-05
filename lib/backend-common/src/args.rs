// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Common CLI flags every Rust backend needs.
//!
//! Flag names match the Python runtime's `WorkerConfig` knobs so operators see
//! the same CLI surface across Rust and Python backends. Engines extend this
//! with their own `clap` `Args` using `#[command(flatten)]`.

use std::path::PathBuf;

use clap::Args;

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
}
