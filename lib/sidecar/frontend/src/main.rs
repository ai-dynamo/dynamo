// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified Dynamo OpenAI frontend for the direct-gRPC path. It registers the
//! TensorRT-LLM, vLLM, and SGLang direct-dispatch providers, then runs the
//! standard `run_input(Input::Http, EngineConfig::Dynamic)` frontend. Each
//! discovered model card's `runtime_data["direct_backend"]` selects the matching
//! provider, so one binary serves all three engines' direct workers. Keeping the
//! registration here keeps the generic `ai-dynamo` wheel free of the engine/tonic
//! deps.

use std::sync::Arc;

use clap::Parser;
use dynamo_llm::discovery::register_direct_dispatch_provider;
use dynamo_llm::entrypoint::EngineConfig;
use dynamo_llm::entrypoint::input::{Input, run_input};
use dynamo_llm::local_model::LocalModelBuilder;
use dynamo_runtime::{DistributedRuntime, Runtime, logging};

#[derive(Parser, Debug)]
#[command(
    about = "Dynamo OpenAI frontend with the trtllm/vLLM/SGLang direct-gRPC dispatch providers registered"
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

    let runtime = Runtime::from_settings()?;
    let secondary = runtime.secondary();
    secondary.block_on(async move {
        // Composition root: register every direct-dispatch provider before the
        // discovery watcher builds routing, so each `direct_backend` model card
        // resolves to its engine's GrpcDispatch instead of the request plane.
        register_direct_dispatch_provider(Arc::new(
            dynamo_trtllm_sidecar::TrtllmDirectDispatchProvider::new(),
        ));
        register_direct_dispatch_provider(Arc::new(
            dynamo_vllm_sidecar::VllmDirectDispatchProvider::new(),
        ));
        register_direct_dispatch_provider(Arc::new(
            dynamo_sglang_sidecar::SglangDirectDispatchProvider::new(),
        ));

        let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

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
