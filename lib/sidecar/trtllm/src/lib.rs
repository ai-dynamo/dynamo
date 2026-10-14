// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for TensorRT-LLM's OpenEngine (`openengine.v1`) gRPC API.

mod args;
mod client;
mod convert;
/// Disaggregation wire contract, shared with the Mocker server so the two do
/// not carry separate copies of the same keys.
#[doc(hidden)]
pub mod disagg;
mod engine;
mod headless;
mod model;
mod node;

/// Generated OpenEngine gRPC types, exposed for the Mocker server until
/// `openengine.v1` is published as a standalone protocol crate.
#[doc(hidden)]
pub mod proto;

pub use engine::TrtllmSidecarEngine;

/// Discover the local engine role before constructing an inference worker.
pub fn run(argv: Vec<String>) -> anyhow::Result<()> {
    use clap::Parser;
    use dynamo_backend_common::Worker;
    use dynamo_sidecar_common::SidecarStartupError;
    use std::sync::Arc;
    let args = args::Args::try_parse_from(argv).map_err(SidecarStartupError::from)?;
    let (engine, config) =
        TrtllmSidecarEngine::from_parsed(args.clone()).map_err(SidecarStartupError::from)?;
    let runtime_config = config.runtime.clone();
    let bootstrap = async move {
        let metadata = discover_node(&args)
            .await
            .map_err(SidecarStartupError::from)?;
        Ok::<_, anyhow::Error>((args, metadata))
    };
    dynamo_sidecar_common::run_with_drt(
        runtime_config,
        bootstrap,
        move |drt, (args, metadata), shutdown| async move {
            if let Some(node) = metadata.filter(|node| !node.leader) {
                headless::relay(drt, args, node, shutdown).await
            } else {
                Worker::new(Arc::new(engine), config)
                    .run_with_drt(drt, shutdown)
                    .await
                    .map_err(Into::into)
            }
        },
    )
}

async fn discover_node(
    args: &args::Args,
) -> Result<Option<node::NodeMetadata>, dynamo_backend_common::DynamoError> {
    let client =
        client::TrtllmClient::connect(&args.sidecar.grpc_endpoint, args.sidecar.grpc.config())
            .await?;
    node::NodeMetadata::from_info(client.server_info().await?.as_ref())
}

#[cfg(test)]
mod tests;
