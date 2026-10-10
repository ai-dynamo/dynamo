// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::Path;

use anyhow::{Context, bail};
use clap::Parser;
use dynamo_mocker::common::protocols::EngineType;
use dynamo_mocker::common::protocols::MockerConfig;
use dynamo_sglang_mocker::{MockerServerConfig, ServerMode, SglangMockerService};
use dynamo_sglang_sidecar::proto::sglang_service_server::SglangServiceServer;
use serde_json::{Map, Value};

#[derive(Parser, Debug)]
#[command(
    name = "dynamo-sglang-mocker-server",
    about = "Run a CPU-only, Mocker-backed implementation of SGLang's native gRPC API"
)]
struct Args {
    /// Address on which to expose the SGLang-compatible gRPC service.
    #[arg(long, default_value = "127.0.0.1:30001")]
    listen: SocketAddr,

    /// Model and tokenizer identity exposed by discovery RPCs.
    #[arg(long, default_value = "mocker-model")]
    model: String,

    /// Wire-level serving role to emulate.
    #[arg(long, value_enum, default_value_t = ServerMode::Aggregated)]
    disaggregation_mode: ServerMode,

    /// Seed for deterministic synthetic token IDs and logprobs.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Maximum model context length exposed by discovery RPCs.
    #[arg(long, default_value_t = 32_768)]
    context_length: u32,

    /// Maximum number of admitted RPCs, including requests queued by Mocker.
    #[arg(long, default_value_t = 256)]
    max_concurrent_requests: usize,

    /// Host published for the metadata-only disaggregation rendezvous.
    #[arg(long, default_value = "127.0.0.1")]
    bootstrap_host: String,

    /// Port published for the metadata-only disaggregation rendezvous.
    #[arg(long, default_value_t = 8_998)]
    bootstrap_port: u16,

    /// Partial Mocker engine configuration as inline JSON or a JSON file path.
    #[arg(long)]
    extra_engine_args: Option<String>,
}

fn load_engine_args(value: Option<&str>) -> anyhow::Result<MockerConfig> {
    let mut object = match value {
        None => Map::new(),
        Some(value) if value.trim_start().starts_with('{') => serde_json::from_str::<Value>(value)
            .context("failed to parse inline --extra-engine-args JSON")?
            .as_object()
            .cloned()
            .context("--extra-engine-args must be a JSON object")?,
        Some(path) => serde_json::from_str::<Value>(
            &std::fs::read_to_string(Path::new(path))
                .with_context(|| format!("failed to read --extra-engine-args from {path}"))?,
        )
        .with_context(|| format!("failed to parse --extra-engine-args from {path}"))?
        .as_object()
        .cloned()
        .context("--extra-engine-args must be a JSON object")?,
    };

    let rank = object
        .entry("engine")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .context("engine must be a JSON object")?;
    rank.entry("backend")
        .or_insert_with(|| Value::String("sglang".to_owned()));
    let args = MockerConfig::from_value(Value::Object(object))
        .context("invalid Mocker engine arguments")?;
    if args.backend != EngineType::Sglang {
        bail!("--extra-engine-args backend must be sglang");
    }
    Ok(args)
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
    let service = SglangMockerService::new(
        MockerServerConfig {
            model: args.model,
            mode: args.disaggregation_mode,
            seed: args.seed,
            context_length: args.context_length,
            max_concurrent_requests: args.max_concurrent_requests,
            bootstrap_host: args.bootstrap_host,
            bootstrap_port: args.bootstrap_port,
        },
        engine_args,
    )?;

    tracing::info!(
        listen = %args.listen,
        model = %service.config().model,
        mode = %service.config().mode,
        "starting Mocker-backed SGLang gRPC server"
    );
    tonic::transport::Server::builder()
        .add_service(SglangServiceServer::new(service))
        .serve_with_shutdown(args.listen, async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_loader_defaults_to_sglang() {
        let args = load_engine_args(Some(r#"{"engine":{"block_size":4}}"#)).unwrap();
        assert_eq!(args.backend, EngineType::Sglang);
        assert_eq!(args.block_size, 4);
    }
}
