// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sidecar_common::{EngineMetricsArgs, HttpEndpoint, SidecarArgs};

use std::num::NonZeroU32;

fn parse_http_endpoint(raw: &str) -> Result<HttpEndpoint, String> {
    HttpEndpoint::parse(raw, "--vllm-http-endpoint").map_err(|error| error.to_string())
}

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-vllm-sidecar",
    about = "Run a Dynamo worker against vLLM's native gRPC service"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub sidecar: SidecarArgs,

    /// Optional vLLM HTTP base URL, used for RL compatibility operations and
    /// as the source of the engine's `/metrics`. Defaults to the gRPC host on
    /// vLLM's default HTTP port for metrics only.
    #[arg(
        long,
        env = "VLLM_HTTP_ENDPOINT",
        value_parser = parse_http_endpoint
    )]
    pub vllm_http_endpoint: Option<HttpEndpoint>,

    #[command(flatten)]
    pub engine_metrics: EngineMetricsArgs,

    /// Optional total RL process world size for vLLM versions that omit it from gRPC metadata.
    ///
    /// The value includes tensor, pipeline, prefill-context, and data parallelism.
    #[arg(long, env = "DYN_VLLM_RL_WORLD_SIZE")]
    pub vllm_rl_world_size: Option<NonZeroU32>,
}
