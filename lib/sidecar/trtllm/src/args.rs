// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sidecar_common::SidecarArgs;

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-trtllm-sidecar",
    about = "Run a Dynamo worker against TensorRT-LLM's OpenEngine gRPC server"
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
    /// `Control.GetModelInfo` reports; that report is used only when this
    /// argument is omitted, and a disagreement is logged at WARN. Supply this
    /// whenever the engine was started without `--max_seq_len`, because
    /// TensorRT-LLM then leaves `max_context_length` unset and the sidecar has
    /// no window to register. With neither source, requests that omit
    /// `max_tokens` are rejected.
    #[arg(long, env = "TRTLLM_CONTEXT_LENGTH", value_parser = clap::value_parser!(u32).range(1..))]
    pub context_length: Option<u32>,
}
