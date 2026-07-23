// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;

use anyhow::Context;
use clap::Parser;
use dynamo_mocker::common::protocols::MockEngineArgs;
use dynamo_vllm_mocker::{MockerServerConfig, ServerMode, VllmMockerService};
use dynamo_vllm_sidecar::proto::generate_server::GenerateServer;

#[derive(Parser, Debug)]
#[command(
    name = "dynamo-vllm-mocker-server",
    about = "Run a CPU-only, Mocker-backed implementation of vLLM's native Generate gRPC API"
)]
struct Args {
    /// Address on which to expose the vLLM-compatible gRPC service.
    #[arg(long, default_value = "127.0.0.1:50051")]
    listen: SocketAddr,

    /// Model name accepted in Generate requests. The empty model used by the
    /// Dynamo vLLM sidecar is always accepted.
    #[arg(long, default_value = "mocker-model")]
    model: String,

    /// Wire-level serving role to emulate.
    #[arg(long, value_enum, default_value_t = ServerMode::Aggregated)]
    disaggregation_mode: ServerMode,

    /// Seed for deterministic synthetic token IDs and logprobs.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Maximum number of admitted RPCs, including requests queued by Mocker.
    #[arg(long, default_value_t = 256)]
    max_concurrent_requests: usize,

    /// Partial Mocker engine configuration as inline JSON or a JSON file path.
    #[arg(long)]
    extra_engine_args: Option<String>,
}

fn load_engine_args(value: Option<&str>) -> anyhow::Result<MockEngineArgs> {
    let args = match value {
        None => MockEngineArgs::default(),
        Some(value) if value.trim_start().starts_with('{') => {
            MockEngineArgs::from_json_str(value).map_err(anyhow::Error::msg)?
        }
        Some(path) => MockEngineArgs::from_json_file(std::path::Path::new(path))
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("failed to load --extra-engine-args from {path}"))?,
    };
    args.normalized().context("invalid Mocker engine arguments")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let engine_args = load_engine_args(args.extra_engine_args.as_deref())?;
    let service = VllmMockerService::new(
        MockerServerConfig {
            model: args.model,
            mode: args.disaggregation_mode,
            seed: args.seed,
            max_concurrent_requests: args.max_concurrent_requests,
        },
        engine_args,
    )?;

    tracing::info!(
        listen = %args.listen,
        model = %service.config().model,
        mode = %service.config().mode,
        "starting Mocker-backed vLLM gRPC server"
    );
    tonic::transport::Server::builder()
        .add_service(GenerateServer::new(service))
        .serve_with_shutdown(args.listen, async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
