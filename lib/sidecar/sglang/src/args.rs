// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Command-line arguments for the SGLang sidecar.

use dynamo_sidecar_common::SidecarArgs;

/// Parsed sidecar arguments.
#[derive(clap::Parser, Debug, Clone)]
#[command(
    name = "dynamo-sglang-sidecar",
    about = "Dynamo sidecar for an out-of-process SGLang native gRPC server."
)]
pub struct Args {
    #[command(flatten)]
    pub sidecar: SidecarArgs,

    /// `host:port` (or URL) of SGLang's native `sglang.runtime.v1` service.
    #[arg(long, visible_alias = "grpc-endpoint", env = "SGLANG_GRPC_ENDPOINT")]
    pub sglang_endpoint: String,

    /// Reachable host that decode workers use to connect to a prefill worker's
    /// SGLang disaggregation bootstrap port. By default this is derived from
    /// SGLang's concrete `host`, then `dist_init_addr`, then a routable local
    /// address. This is required when discovery exposes only loopback or
    /// wildcard addresses.
    #[arg(long, env = "SGLANG_DISAGGREGATION_BOOTSTRAP_HOST")]
    pub bootstrap_host: Option<String>,
}
