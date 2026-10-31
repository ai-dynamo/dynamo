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
        let (engine, config) = if metadata.as_ref().is_some_and(|node| !node.leader) {
            (engine, config)
        } else {
            TrtllmSidecarEngine::bootstrap((engine, config))
                .await
                .map_err(SidecarStartupError::from)?
        };
        Ok::<_, anyhow::Error>((args, metadata, engine, config))
    };
    dynamo_sidecar_common::run_with_drt(
        runtime_config,
        bootstrap,
        move |drt, (args, metadata, engine, config), shutdown| async move {
            if let Some(node) = metadata.filter(|node| !node.leader) {
                headless::relay(drt, args, node, shutdown).await
            } else {
                let engine = Arc::new(engine);
                let worker =
                    Worker::new(engine.clone(), config).run_with_drt(drt, shutdown.clone());
                tokio::pin!(worker);
                tokio::select! {
                    result = &mut worker => result.map_err(Into::into),
                    result = engine.monitor_routing() => {
                        // Let Worker withdraw discovery and drain before exiting.
                        shutdown.cancel();
                        let stopped = worker.await;
                        result?;
                        stopped.map_err(Into::into)
                    }
                }
            }
        },
    )
}

async fn discover_node(
    args: &args::Args,
) -> Result<Option<node::NodeMetadata>, dynamo_backend_common::DynamoError> {
    let transport = args.sidecar.grpc.config();
    tokio::time::timeout(transport.startup_deadline, async {
        let client = client::TrtllmClient::connect(&args.sidecar.grpc_endpoint, transport).await?;
        node::NodeMetadata::from_info(client.server_info().await?.as_ref())
    })
    .await
    .map_err(|_| {
        dynamo_sidecar_common::connection_timeout(
            "OpenEngine node discovery exceeded the gRPC startup deadline",
        )
    })?
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod test_fixtures;
