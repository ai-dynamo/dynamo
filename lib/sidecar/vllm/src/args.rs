// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sidecar_common::{HttpEndpoint, SidecarArgs};

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

    /// Optional controller-routable vLLM HTTP base URL for RL compatibility operations.
    #[arg(
        long,
        env = "VLLM_HTTP_ENDPOINT",
        value_parser = parse_http_endpoint
    )]
    pub vllm_http_endpoint: Option<HttpEndpoint>,

    /// Let the Dynamo frontend decode images and send them to vLLM as NIXL references.
    ///
    /// Takes effect on aggregated workers (also with `--route-to-encoder`) and Encode workers, and
    /// only when the vLLM gRPC server reports `supports_nixl_ref` (vllm-rs started with
    /// `--enable-grpc-nixl-media`, from a build with its `nixl-media` feature). Set it on every
    /// worker of a model, including its Encode workers.
    #[arg(
        long,
        default_value_t = false,
        env = "DYN_VLLM_FRONTEND_DECODING",
        value_parser = clap::builder::BoolishValueParser::new()
    )]
    pub frontend_decoding: bool,

    /// Optional total RL process world size for vLLM versions that omit it from gRPC metadata.
    ///
    /// The value includes tensor, pipeline, prefill-context, and data parallelism.
    #[arg(long, env = "DYN_VLLM_RL_WORLD_SIZE")]
    pub vllm_rl_world_size: Option<NonZeroU32>,
}
