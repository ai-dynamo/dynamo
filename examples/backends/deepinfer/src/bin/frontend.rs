// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin executable around the actual Dynamo HTTP frontend, without Python bindings.

use clap::Parser;
use dynamo_backend_common::RuntimeConfig;
use dynamo_llm::{
    discovery::ModelWatcher, entrypoint::RouterConfig, http::service::service_v2::HttpService,
    namespace::NamespaceFilter,
};
use dynamo_runtime::{DistributedRuntime, discovery::DiscoveryQuery, pipeline::RouterMode};
use std::sync::Arc;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    runtime: RuntimeConfig,
    #[arg(long, default_value_t = 8000)]
    http_port: u16,
    #[arg(long, default_value = "deepinfer")]
    namespace: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    args.runtime.apply_to_env();
    dynamo_sidecar_common::run_task(|runtime, shutdown| async move {
        let distributed =
            DistributedRuntime::new(runtime, args.runtime.to_distributed_config()?).await?;
        let service = HttpService::builder()
            .host("127.0.0.1")
            .port(args.http_port)
            .enable_chat_endpoints(true)
            .enable_cmpl_endpoints(true)
            .build()?;
        let watcher = Arc::new(ModelWatcher::new(
            distributed.clone(),
            service.state().manager_clone(),
            RouterConfig {
                router_mode: RouterMode::RoundRobin,
                ..Default::default()
            },
            0,
            None,
            None,
            None,
            service.state().metrics_clone(),
        ));
        let discovery = distributed
            .discovery()
            .list_and_watch(DiscoveryQuery::AllModels, Some(distributed.primary_token()))
            .await?;
        let watch = tokio::spawn(async move {
            watcher
                .watch(discovery, NamespaceFilter::Exact(args.namespace))
                .await;
        });
        tracing::info!(port = args.http_port, "Dynamo example frontend listening");
        let server = service.spawn(shutdown.clone()).await;
        let result = server.await?;
        watch.abort();
        let _ = watch.await;
        distributed.shutdown();
        result
    })
}
