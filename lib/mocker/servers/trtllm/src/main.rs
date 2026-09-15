// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::Path;

use anyhow::{Context, bail};
use clap::Parser;
#[cfg(test)]
use dynamo_mocker::common::protocols::EngineType;
use dynamo_mocker::common::protocols::MockEngineArgs;
use dynamo_trtllm_mocker::{MockerServerConfig, ServerMode, TrtllmMockerService};
use dynamo_trtllm_sidecar::proto::control_server::ControlServer;
use dynamo_trtllm_sidecar::proto::inference_server::InferenceServer;
use serde_json::{Map, Value};

#[derive(Parser, Debug)]
#[command(
    name = "dynamo-trtllm-mocker-server",
    about = "Run a CPU-only, Mocker-backed implementation of TensorRT-LLM's OpenEngine gRPC API"
)]
struct Args {
    /// Address on which to expose the OpenEngine gRPC services.
    #[arg(long, default_value = "127.0.0.1:50051")]
    listen: SocketAddr,

    /// Model identity exposed by Control.GetModelInfo. The sidecar's
    /// --model-path must match it or every request is rejected as NOT_FOUND.
    #[arg(long, default_value = "mocker-model")]
    model: String,

    /// Wire-level serving role to emulate.
    #[arg(long, value_enum, default_value_t = ServerMode::Aggregated)]
    disaggregation_mode: ServerMode,

    /// Seed for deterministic synthetic token IDs and logprobs.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Maximum context length exposed by Control.GetModelInfo. The sidecar
    /// refuses to start without a positive value, and turns an omitted
    /// max_tokens into `context_length - prompt_len`; combined with the
    /// TensorRT-LLM reserve-at-admission policy, a large value here makes
    /// capacity rejection likely on a small KV pool.
    #[arg(long, default_value_t = 32_768)]
    context_length: u32,

    /// Maximum number of admitted RPCs, including requests queued by Mocker.
    #[arg(long, default_value_t = 256)]
    max_concurrent_requests: usize,

    /// Host published in the disaggregated handoff's KV endpoint.
    #[arg(long, default_value = "127.0.0.1")]
    kv_host: String,

    /// Port published in the disaggregated handoff's KV endpoint.
    #[arg(long, default_value_t = 5_600)]
    kv_port: u16,

    /// Partial Mocker engine configuration as inline JSON or a JSON file path.
    #[arg(long)]
    extra_engine_args: Option<String>,
}

/// Rewrites the caller's object rather than deserializing it directly: the
/// serde default for `engine_type` is vllm, so a missing key would silently
/// select the wrong scheduler.
fn load_engine_args(value: Option<&str>) -> anyhow::Result<MockEngineArgs> {
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

    match object.get("engine_type") {
        None => {
            object.insert(
                "engine_type".to_string(),
                Value::String("trtllm".to_string()),
            );
        }
        Some(Value::String(engine_type)) if engine_type.eq_ignore_ascii_case("trtllm") => {}
        Some(engine_type) => {
            bail!("--extra-engine-args engine_type must be trtllm, got {engine_type}")
        }
    }

    MockEngineArgs::from_json_str(&Value::Object(object).to_string())
        .map_err(anyhow::Error::msg)?
        .normalized()
        .context("invalid Mocker engine arguments")
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
    let service = TrtllmMockerService::new(
        MockerServerConfig {
            model: args.model,
            mode: args.disaggregation_mode,
            seed: args.seed,
            context_length: args.context_length,
            max_concurrent_requests: args.max_concurrent_requests,
            kv_host: args.kv_host,
            kv_port: args.kv_port,
        },
        engine_args,
    )?;

    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<InferenceServer<TrtllmMockerService>>()
        .await;
    health_reporter
        .set_serving::<ControlServer<TrtllmMockerService>>()
        .await;

    tracing::info!(
        listen = %args.listen,
        model = %service.config().model,
        mode = %service.config().mode,
        "starting Mocker-backed TensorRT-LLM OpenEngine gRPC server"
    );
    tonic::transport::Server::builder()
        .add_service(health_service)
        .add_service(InferenceServer::new(service.clone()))
        .add_service(ControlServer::new(service.clone()))
        .serve_with_shutdown(args.listen, async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    // Stop the scheduler at a defined point and surface anything it collected,
    // rather than leaving it to whenever the last service clone is dropped.
    service.shutdown().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_loader_defaults_to_trtllm() {
        let args = load_engine_args(Some(r#"{"block_size":4}"#)).unwrap();
        assert_eq!(args.engine_type, EngineType::Trtllm);
        assert_eq!(args.block_size, 4);
    }

    #[test]
    fn engine_loader_materializes_the_trtllm_block_size() {
        let args = load_engine_args(None).unwrap();
        assert_eq!(args.engine_type, EngineType::Trtllm);
        assert_eq!(args.block_size, 32);
    }

    #[test]
    fn engine_loader_rejects_another_engine_type() {
        let error = load_engine_args(Some(r#"{"engine_type":"vllm"}"#)).unwrap_err();
        assert!(error.to_string().contains("must be trtllm"), "{error}");
    }
}
