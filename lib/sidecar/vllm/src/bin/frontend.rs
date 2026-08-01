// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo OpenAI-compatible frontend with the vLLM direct-gRPC dispatch provider
//! registered (the composition root).
//!
//! It discovers `dynamo-vllm-sidecar --direct` workers via etcd and — because
//! their model cards advertise `runtime_data["direct_backend"] = "vllm"` —
//! dispatches inference straight to each worker's `Generate` gRPC server,
//! bypassing the request plane, while `PushRouter` keeps instance selection,
//! fault detection, and migration.
//!
//! This is a thin composition root: it registers the provider, then delegates
//! to the same `run_input(Input::Http, EngineConfig::Dynamic)` path the stock
//! frontend uses. Keeping the registration here (rather than in the Python
//! bindings) keeps the generic `ai-dynamo` wheel free of the vLLM/tonic dep.

use std::sync::Arc;

use clap::Parser;
use dynamo_llm::discovery::register_direct_dispatch_provider;
use dynamo_llm::entrypoint::EngineConfig;
use dynamo_llm::entrypoint::input::{Input, run_input};
use dynamo_llm::local_model::LocalModelBuilder;
use dynamo_runtime::{DistributedRuntime, Runtime, logging};
use dynamo_vllm_sidecar::VllmDirectDispatchProvider;

#[derive(Parser, Debug)]
#[command(
    about = "Dynamo OpenAI frontend with the vLLM direct-gRPC dispatch provider registered"
)]
struct Args {
    /// Dynamo namespace to discover workers in.
    #[arg(long, default_value = "dynamo", env = "DYN_NAMESPACE")]
    namespace: String,

    /// HTTP host to bind (defaults to all interfaces).
    #[arg(long, env = "DYN_HTTP_HOST")]
    http_host: Option<String>,

    /// HTTP port for the OpenAI-compatible server.
    #[arg(long, default_value_t = 8000, env = "DYN_HTTP_PORT")]
    http_port: u16,

    /// Optional served model-name override. In dynamic mode the discovered
    /// worker model cards populate the served models, so this is usually unset.
    #[arg(long)]
    model_name: Option<String>,
}

fn main() -> anyhow::Result<()> {
    logging::init();
    let args = Args::parse();

    // Own the signal/runtime flow like the sidecar binaries (see
    // dynamo_backend_common::run): build the runtime, drive on the secondary.
    let runtime = Runtime::from_settings()?;
    let secondary = runtime.secondary();
    secondary.block_on(async move {
        // Composition root: register the vLLM provider BEFORE the discovery
        // watcher builds any routing, so direct-backend models resolve to a
        // GrpcDispatch instead of the request-plane router.
        register_direct_dispatch_provider(Arc::new(VllmDirectDispatchProvider::new()));

        let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

        // In dynamic mode the LocalModel is a frontend config holder (namespace,
        // http host/port); the served models come from etcd discovery.
        let local_model = LocalModelBuilder::default()
            .model_name(args.model_name)
            .http_host(args.http_host)
            .http_port(args.http_port)
            .namespace(Some(args.namespace))
            .build()
            .await?;

        let engine_config = EngineConfig::Dynamic {
            model: Box::new(local_model),
            chat_engine_factory: None,
            prefill_load_estimator: None,
        };

        let result = run_input(drt, Input::Http, engine_config).await;
        runtime.shutdown();
        result
    })
}
