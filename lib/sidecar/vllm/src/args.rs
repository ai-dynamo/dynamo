// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sidecar_common::{HttpEndpoint, SidecarArgs};

use std::num::NonZeroU32;

/// vLLM's NIXL connector default `kv_lease_duration`, in seconds.
pub(crate) const DEFAULT_DECODE_CANCEL_DRAIN_TIMEOUT_SECS: u64 = 30;

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

    /// Optional total RL process world size for vLLM versions that omit it from gRPC metadata.
    ///
    /// The value includes tensor, pipeline, prefill-context, and data parallelism.
    #[arg(long, env = "DYN_VLLM_RL_WORLD_SIZE")]
    pub vllm_rl_world_size: Option<NonZeroU32>,

    /// Seconds a cancelled decode request waits for vLLM's response headers
    /// and first token before the sidecar sends `Control.Abort` and releases
    /// it.
    ///
    /// The wait lets the request reach vLLM and finish its KV transfer. The
    /// default matches the NIXL connector's default `kv_lease_duration`
    /// (30 s), after which a prefill worker that hears no decode heartbeat
    /// frees the blocks itself. The abort is safe at any point: vLLM's NIXL
    /// scheduler frees the prefill blocks when it finishes an aborted request.
    /// If you raise `kv_lease_duration` in `kv_connector_extra_config`, raise
    /// this to match.
    #[arg(
        long,
        env = "DYN_VLLM_DECODE_CANCEL_DRAIN_TIMEOUT_SECS",
        default_value_t = DEFAULT_DECODE_CANCEL_DRAIN_TIMEOUT_SECS,
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    pub decode_cancel_drain_timeout_secs: u64,
}
