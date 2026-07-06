// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::CommonArgs;

#[derive(clap::Parser, Debug)]
#[command(
    name = "dynamo-sglang-rs-backend",
    about = "Dynamo SGLang backend for SMG's scheduler gRPC service."
)]
pub struct Args {
    #[command(flatten)]
    pub common: CommonArgs,

    /// HF repo name or local model path used to build Dynamo's MDC.
    #[arg(long, env = "DYN_MODEL_NAME")]
    pub model_name: Option<String>,

    /// Public-facing model name to register with Dynamo.
    #[arg(long, env = "DYN_SERVED_MODEL_NAME")]
    pub served_model_name: Option<String>,

    #[arg(
        long,
        env = "SGLANG_GRPC_ENDPOINT",
        default_value = "http://127.0.0.1:30000"
    )]
    pub sglang_grpc_endpoint: String,
}
