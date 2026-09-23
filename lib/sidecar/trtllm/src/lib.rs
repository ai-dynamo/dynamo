// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for TensorRT-LLM's OpenEngine (`openengine.v1`) gRPC API.

mod args;
mod client;
mod convert;
mod disagg;
mod engine;
mod headless;
mod model;
mod node;
mod proto;

pub use engine::TrtllmSidecarEngine;

/// Discover the local engine role before constructing an inference worker.
pub fn run(argv: Vec<String>) -> anyhow::Result<()> {
    use clap::Parser;
    use dynamo_sidecar_common::SidecarStartupError;
    let args = args::Args::try_parse_from(argv).map_err(SidecarStartupError::from)?;
    let bootstrap = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let metadata = bootstrap
        .block_on(async {
            let client = client::TrtllmClient::connect(
                &args.sidecar.grpc_endpoint,
                args.sidecar.grpc.config(),
            )
            .await?;
            node::NodeMetadata::from_info(client.server_info().await?.as_ref())
        })
        .map_err(SidecarStartupError::from)?;
    drop(bootstrap);
    if let Some(node) = metadata.filter(|node| !node.leader) {
        headless::run(args, node)
    } else {
        let (engine, config) =
            TrtllmSidecarEngine::from_parsed(args).map_err(SidecarStartupError::from)?;
        dynamo_backend_common::run(std::sync::Arc::new(engine), config)
    }
}

#[cfg(test)]
mod tests;
