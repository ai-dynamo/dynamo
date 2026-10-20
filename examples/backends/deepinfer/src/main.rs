// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-independent gRPC probe. Python hosts the serving extension.

use clap::{Parser, Subcommand};
use dynamo_trtllm_sidecar::proto as pb;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    Probe {
        #[arg(long)]
        endpoint: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Args::parse().action {
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
    }
    Ok(())
}
