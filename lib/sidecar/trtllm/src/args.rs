// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sidecar_common::SidecarArgs;

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-trtllm-sidecar",
    about = "Run a Dynamo worker against TensorRT-LLM's native gRPC TrtllmService"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub sidecar: SidecarArgs,

    /// Hugging Face model ID or local path used for tokenization and templates.
    #[arg(long)]
    pub model_path: String,

    /// Model maximum sequence length (input + output). Used to register the
    /// context length and to derive a default `max_tokens` when a request omits
    /// one. A value supplied here takes precedence over the context length
    /// TensorRT-LLM's `GetModelInfo` gRPC reports; that report is used only when
    /// this argument is omitted, and a disagreement is logged at WARN. Current
    /// releases report nothing usable (zero) or the maximum *input* length, so
    /// supply it here; otherwise requests that omit `max_tokens` are rejected.
    /// See the note in `convert.rs`.
    #[arg(long, env = "TRTLLM_CONTEXT_LENGTH")]
    pub context_length: Option<u32>,
}
