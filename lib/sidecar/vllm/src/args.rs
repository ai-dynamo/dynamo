// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sidecar_common::SidecarArgs;

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-vllm-sidecar",
    about = "Run a Dynamo worker against vLLM's native gRPC service"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub sidecar: SidecarArgs,

    /// vLLM gRPC endpoint as host:port or an http:// URL.
    #[arg(long, env = "VLLM_GRPC_ENDPOINT")]
    pub vllm_endpoint: String,

    /// Direct HTTP control endpoint of the same vLLM engine.
    #[arg(long, env = "VLLM_HTTP_ENDPOINT")]
    pub admin_endpoint: Option<String>,

    /// Engine-advertised model identity published through RL discovery.
    #[arg(long, env = "DYN_RL_DISCOVERY_MODEL_NAME")]
    pub rl_discovery_model_name: Option<String>,

    /// Hugging Face model ID or local path used by Dynamo for model-card
    /// registration, tokenization, and chat templates. The released vLLM gRPC
    /// API does not expose this metadata, so it cannot be inferred from the
    /// endpoint. This flag is temporary and can be removed once vLLM exposes
    /// model and tokenizer metadata over gRPC.
    #[arg(long)]
    pub model_path: String,
}
