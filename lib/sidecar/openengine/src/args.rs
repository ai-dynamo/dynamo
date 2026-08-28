// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sidecar_common::SidecarArgs;

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-openengine-sidecar",
    about = "Run a Dynamo worker against an OpenEngine gRPC server"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub sidecar: SidecarArgs,

    /// Hugging Face model ID or local path used for tokenization and templates.
    /// Defaults to the single model advertised by OpenEngine GetServerInfo.
    #[arg(long)]
    pub model_path: Option<String>,
}
