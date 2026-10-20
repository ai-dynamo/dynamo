// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod bridge;
mod service;

use clap::{Parser, Subcommand, ValueEnum};
use dynamo_trtllm_sidecar::proto as pb;
use std::{net::SocketAddr, path::PathBuf};
use tokio::process::Command;

#[derive(Parser)]
#[command(about = "OpenEngine gRPC server backed by a DeepInfer engine owner")]
struct Args {
    #[command(subcommand)]
    action: Action,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Engine {
    Mock,
    Deepinfer,
}

#[derive(Subcommand)]
enum Action {
    Serve {
        #[arg(long, default_value = "127.0.0.1:30001")]
        listen: SocketAddr,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, value_enum, default_value = "mock")]
        engine: Engine,
        #[arg(long, default_value = "python3")]
        python: PathBuf,
        #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/engine_worker.py"))]
        worker_script: PathBuf,
        #[arg(long)]
        engine_args: Option<PathBuf>,
        #[arg(long, default_value_t = 8192, value_parser = clap::value_parser!(u32).range(1..))]
        context_length: u32,
        #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..))]
        max_requests: u32,
        #[arg(long, default_value_t = 2)]
        tick_ms: u64,
    },
    /// A real gRPC readiness/load check, also used by the launch script.
    Probe {
        #[arg(long)]
        endpoint: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    match args.action {
        Action::Probe { endpoint } => {
            let mut control = pb::control_client::ControlClient::connect(endpoint).await?;
            let health = control
                .health(pb::HealthRequest::default())
                .await?
                .into_inner();
            anyhow::ensure!(
                health.state == pb::HealthState::Ready as i32,
                "engine is not ready"
            );
            let info = control
                .get_server_info(pb::GetServerInfoRequest {})
                .await?
                .into_inner();
            let load = control
                .get_load(pb::GetLoadRequest::default())
                .await?
                .into_inner();
            println!(
                "{}",
                serde_json::json!({"engine":info.engine_name, "ready":true, "running_requests":load.running_requests})
            );
        }
        Action::Serve {
            listen,
            model_dir,
            engine,
            python,
            worker_script,
            engine_args,
            context_length,
            max_requests,
            tick_ms,
        } => {
            anyhow::ensure!(
                listen.ip().is_loopback(),
                "this local example requires a loopback listen address"
            );
            let model_dir = model_dir.canonicalize()?;
            let mut command = Command::new(python);
            command
                .arg("-u")
                .arg(worker_script)
                .arg("--model-dir")
                .arg(&model_dir)
                .arg("--engine")
                .arg(match engine {
                    Engine::Mock => "mock",
                    Engine::Deepinfer => "deepinfer",
                })
                .arg("--context-length")
                .arg(context_length.to_string())
                .arg("--tick-ms")
                .arg(tick_ms.to_string());
            if let Some(path) = engine_args {
                command.arg("--engine-args").arg(path);
            }
            let (bridge, worker) = bridge::Bridge::spawn(command, max_requests as usize).await?;
            let service = service::Service::new(
                bridge.clone(),
                model_dir.display().to_string(),
                context_length,
            );
            let (reporter, health) = tonic_health::server::health_reporter();
            reporter
                .set_serving::<pb::inference_server::InferenceServer<service::Service>>()
                .await;
            reporter
                .set_serving::<pb::control_server::ControlServer<service::Service>>()
                .await;
            let shutdown_bridge = bridge.clone();
            let shutdown = async move {
                let mut term =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("install SIGTERM handler");
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {},
                    _ = term.recv() => {},
                    _ = shutdown_bridge.failed.cancelled() => {},
                }
                shutdown_bridge.abort(None);
            };
            tracing::info!(%listen, ?engine, "DeepInfer OpenEngine server ready");
            let result = tonic::transport::Server::builder()
                .add_service(health)
                .add_service(
                    pb::inference_server::InferenceServer::new(service.clone())
                        .max_decoding_message_size(64 * 1024 * 1024),
                )
                .add_service(pb::control_server::ControlServer::new(service))
                .serve_with_shutdown(listen, shutdown)
                .await;
            bridge.shutdown();
            worker.await?;
            result?;
            anyhow::ensure!(
                !bridge.failed.is_cancelled(),
                "DeepInfer engine worker failed"
            );
        }
    }
    Ok(())
}
