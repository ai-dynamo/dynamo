// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::CommonArgs;

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-vllm-sidecar",
    about = "Run a Dynamo worker against vLLM's native gRPC service"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub common: CommonArgs,

    /// vLLM gRPC endpoint as host:port or an http:// URL.
    #[arg(long, env = "VLLM_GRPC_ENDPOINT")]
    pub vllm_endpoint: String,

    /// Hugging Face model ID or local path used for tokenization and templates.
    #[arg(long)]
    pub model_path: String,
}
