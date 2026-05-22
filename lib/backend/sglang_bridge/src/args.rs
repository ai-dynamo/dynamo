// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::CommonArgs;

#[derive(clap::Parser, Debug)]
#[command(name = "dynamo-sglang-bridge", about = "Bridge to SGLang's native gRPC server.")]
pub struct Args {
    #[command(flatten)]
    pub common: CommonArgs,

    #[arg(long, env = "SGLANG_GRPC_ENDPOINT", default_value = "http://127.0.0.1:30000")]
    pub sglang_grpc_endpoint: String,
}
