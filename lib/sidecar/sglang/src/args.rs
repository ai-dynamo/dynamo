// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Command-line arguments for the SGLang sidecar.

use dynamo_backend_common::{CommonArgs, DynamoError};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportArgs};

use crate::client;

/// Both serving and telemetry-only sidecars discover their local engine over gRPC.
#[derive(clap::Args, Debug, Clone)]
pub struct SglangSidecarArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Local engine gRPC endpoint. Falls back to SGLANG_GRPC_ENDPOINT.
    #[arg(long, env = "DYN_SIDECAR_GRPC_ENDPOINT")]
    pub grpc_endpoint: Option<String>,

    #[command(flatten)]
    pub grpc: GrpcTransportArgs,
}

impl SglangSidecarArgs {
    pub(crate) fn resolve_grpc_endpoint(&self) -> Result<GrpcEndpoint, DynamoError> {
        let endpoint = self
            .grpc_endpoint
            .clone()
            .or_else(|| std::env::var("SGLANG_GRPC_ENDPOINT").ok())
            .ok_or_else(|| {
                client::invalid_arg("sidecar requires --grpc-endpoint or SGLANG_GRPC_ENDPOINT")
            })?;
        GrpcEndpoint::parse(&endpoint, "--grpc-endpoint")
    }
}

/// Parsed sidecar arguments.
#[derive(clap::Parser, Debug, Clone)]
#[command(
    name = "dynamo-sglang-sidecar",
    about = "Dynamo sidecar for an out-of-process SGLang native gRPC server."
)]
pub struct Args {
    #[command(flatten)]
    pub sidecar: SglangSidecarArgs,

    /// Relay a follower node's local KV events without registering a request endpoint.
    #[arg(long)]
    pub telemetry_only: bool,

    /// Maximum wait for a matching leader registration in telemetry mode.
    #[arg(long, default_value_t = 1800, value_parser = clap::value_parser!(u64).range(1..))]
    pub leader_discovery_timeout_secs: u64,

    /// Reachable host that decode workers use to connect to a prefill worker's
    /// SGLang disaggregation bootstrap port. By default this is derived from
    /// SGLang's concrete `host`, then `dist_init_addr`, then a routable local
    /// address. This is required when discovery exposes only loopback or
    /// wildcard addresses.
    #[arg(long, env = "SGLANG_DISAGGREGATION_BOOTSTRAP_HOST")]
    pub bootstrap_host: Option<String>,
}
