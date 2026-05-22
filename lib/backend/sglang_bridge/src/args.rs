// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::CommonArgs;

#[derive(clap::Parser, Debug)]
#[command(
    name = "dynamo-sglang-bridge",
    about = "Dynamo sidecar bridge to SGLang's native gRPC server (sglang.runtime.v1)."
)]
pub struct Args {
    #[command(flatten)]
    pub common: CommonArgs,

    /// gRPC endpoint of the SGLang server.
    #[arg(long, env = "SGLANG_GRPC_ENDPOINT", default_value = "http://127.0.0.1:30000")]
    pub sglang_grpc_endpoint: String,

    /// Public-facing model name override. When unset, the bridge advertises
    /// whatever SGLang reports in GetServerInfo.
    #[arg(long)]
    pub served_model_name: Option<String>,

    /// Connect timeout for the initial gRPC dial, seconds.
    #[arg(long, default_value_t = 30)]
    pub connect_timeout_secs: u64,

    /// Bootstrap host advertised to decode workers. Falls back to
    /// dist_init_addr from SGLang's GetServerInfo, then to 127.0.0.1.
    #[arg(long, env = "DYN_BOOTSTRAP_HOST")]
    pub bootstrap_host: Option<String>,
}
