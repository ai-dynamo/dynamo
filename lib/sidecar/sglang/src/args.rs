// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Command-line arguments for the SGLang sidecar.

use dynamo_backend_common::CommonArgs;
use dynamo_sidecar_common::GrpcTransportArgs;

use crate::context::SidecarContext;

/// SGLang alone supports a telemetry-only launch without a gRPC endpoint.
#[derive(clap::Args, Debug, Clone)]
pub struct SglangSidecarArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Engine gRPC endpoint, required in full mode. Falls back to SGLANG_GRPC_ENDPOINT.
    #[arg(long, env = "DYN_SIDECAR_GRPC_ENDPOINT")]
    pub grpc_endpoint: Option<String>,

    #[command(flatten)]
    pub grpc: GrpcTransportArgs,
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

    /// Versioned node-local context supplied by SGLang's managed sidecar launcher.
    #[arg(long, env = "SGLANG_SIDECAR_CONTEXT")]
    pub sidecar_context: Option<SidecarContext>,

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
