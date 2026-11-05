// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sidecar_common::SidecarArgs;

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-lpx-sidecar",
    about = "Run a Dynamo worker against an LPX engine's OpenEngine gRPC server"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub sidecar: SidecarArgs,

    /// Hugging Face model ID or local path used for tokenization and the chat
    /// template. The engine's own model paths are inside its container, so the
    /// sidecar cannot use them.
    #[arg(long)]
    pub model_path: String,

    /// The model name the engine must serve. The sidecar registers the name
    /// the engine reports and fails at startup when it differs from this one.
    /// Optional; it lets a runtime container written for the TensorRT-LLM
    /// sidecar, which takes this flag, keep its arguments.
    #[arg(long, env = "DYN_SERVED_MODEL_NAME")]
    pub served_model_name: Option<String>,

    /// Rank of this Pod in a multi-Pod engine; only rank 0 serves. On any
    /// other rank the sidecar stands by: it answers only the system health and
    /// live probes and never connects to the engine or to Dynamo.
    // Kubernetes resolves a rank bound to a missing Pod label to an empty
    // value. clap's u32 parser rejects it rather than applying the default,
    // which would give the engine a second serving rank 0.
    #[arg(long = "node-rank", env = "DYNAMO_RANK", default_value_t = 0)]
    pub node_rank: u32,
}
