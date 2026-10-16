// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use clap::{Parser, ValueEnum};
use dynamo_backend_common::RuntimeConfig;
use dynamo_llm::{
    local_model::register_model_card,
    model_card::ModelDeploymentCard,
    model_type::{ModelInput, ModelType},
    worker_type::WorkerType,
};
use dynamo_realtime_ws_sidecar::{Bridge, Profile};
use dynamo_runtime::{
    DistributedRuntime,
    pipeline::{
        BidirectionalStreamingEngine, ServerStreamingEngine,
        network::{Ingress, PushWorkHandler},
    },
    protocols::annotated::Annotated,
};
use serde_json::Value;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Asr,
    Omni,
    Tts,
}

#[derive(Parser)]
#[command(about = "Experimental Dynamo worker for unmodified upstream vLLM WebSockets")]
struct Args {
    #[arg(long, value_enum)]
    profile: Mode,
    #[arg(long)]
    model: String,
    #[arg(long)]
    upstream_url: String,
    #[arg(long, default_value = "dynamo")]
    namespace: String,
    #[arg(long, default_value = "backend")]
    component: String,
    #[arg(long, default_value = "generate")]
    endpoint: String,
    /// Override incoming PCM rate. Defaults: ASR 24000, Omni 16000.
    #[arg(long)]
    input_rate: Option<u32>,
    /// Upstream realtime max_output_tokens, if supported by the profile.
    #[arg(long)]
    max_output_tokens: Option<u32>,
    #[command(flatten)]
    runtime: RuntimeConfig,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    dynamo_sidecar_common::run_task(|runtime, shutdown| async move {
        let drt =
            DistributedRuntime::new(runtime.clone(), args.runtime.to_distributed_config()?).await?;
        let endpoint = drt
            .namespace(args.namespace)?
            .component(args.component)?
            .endpoint(args.endpoint);
        let profile = match args.profile {
            Mode::Asr => Profile::Asr,
            Mode::Omni => Profile::Omni,
            Mode::Tts => Profile::Tts,
        };
        let bridge = Arc::new(Bridge {
            profile,
            upstream_url: args.upstream_url,
            model: args.model.clone(),
            input_rate: args.input_rate.unwrap_or(if profile == Profile::Asr {
                24000
            } else {
                16000
            }),
            max_output_tokens: args.max_output_tokens,
        });
        bridge.validate()?;
        let ingress: Arc<dyn PushWorkHandler> = if profile == Profile::Tts {
            Ingress::for_engine(bridge as ServerStreamingEngine<Value, Annotated<Value>>)?
        } else {
            Ingress::for_engine(bridge as BidirectionalStreamingEngine<Value, Annotated<Value>>)?
        };
        let started = endpoint
            .endpoint_builder()
            .handler(ingress)
            .start_with_registration()
            .await?;
        let mut card = ModelDeploymentCard::with_name_only(&args.model);
        card.model_input = ModelInput::Text;
        card.model_type = if profile == Profile::Tts {
            ModelType::Audios
        } else {
            ModelType::Realtime
        };
        card.worker_type = Some(WorkerType::Aggregated);
        register_model_card(&endpoint, &card).await?;
        tracing::info!(
            model = args.model,
            ?profile,
            "Realtime WS sidecar registered"
        );
        tokio::select! {
            _ = shutdown.cancelled() => {},
            result = started.wait() => { result?; },
        }
        endpoint.unregister_endpoint_instance().await?;
        Ok(())
    })
}
