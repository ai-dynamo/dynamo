// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[cfg(any(feature = "vllm", feature = "sglang"))]
use std::{future::Future, pin::Pin};
use std::{io::Read, path::PathBuf, sync::Arc};

use dynamo_llm::{
    backend::ExecutionContext, engines::StreamingEngine, kv_router::publisher::KvMetricsPublisher,
    model_card::model::ModelDeploymentCard,
};
use dynamo_runtime::{protocols::Endpoint, DistributedRuntime};

mod flags;
pub use flags::Flags;
mod hub;
mod input;
#[cfg(any(feature = "vllm", feature = "sglang"))]
mod net;
mod opt;
pub use dynamo_llm::request_template::RequestTemplate;
pub use opt::{Input, Output};

/// How we identify a namespace/component/endpoint URL.
/// Technically the '://' is not part of the scheme but it eliminates several string
/// concatenations.
const ENDPOINT_SCHEME: &str = "dyn://";

/// When `in=text` the user doesn't need to know the model name, and doesn't need to provide it on
/// the command line. Hence it's optional, and defaults to this.
const INVISIBLE_MODEL_NAME: &str = "dynamo-run";

/// The component name for the KV publisher, if used
const KV_PUBLISHER_COMPONENT: &str = "kvpublisher";

/// How we identify a python string endpoint
#[cfg(feature = "python")]
const PYTHON_STR_SCHEME: &str = "pystr:";

/// How we identify a python token endpoint
#[cfg(feature = "python")]
const PYTHON_TOK_SCHEME: &str = "pytok:";

/// Prefix for Hugging Face model repository
const HF_SCHEME: &str = "hf://";

pub enum EngineConfig {
    /// An remote networked engine we don't know about yet
    Dynamic(Endpoint),

    /// A Full service engine does it's own tokenization and prompt formatting.
    StaticFull {
        service_name: String,
        engine: Arc<dyn StreamingEngine>,
        card: Box<ModelDeploymentCard>,
    },

    /// A core engine expects to be wrapped with pre/post processors that handle tokenization.
    StaticCore {
        service_name: String,
        engine: ExecutionContext,
        card: Box<ModelDeploymentCard>,
    },

    /// vllm multi-node doesn't run an engine on nodes other than 0. 'ray' does all the work.
    None,
}

/// Distributed system values
struct DynInput {
    endpoint_id: Endpoint,
    distributed_runtime: DistributedRuntime,
}

#[allow(unused_mut)]
pub async fn run(
    runtime: dynamo_runtime::Runtime,
    mut in_opt: Input, // mut because vllm and sglang multi-node can change it
    out_opt: Output,
    flags: Flags,
    #[allow(unused_variables)] zmq_socket_prefix: Option<String>,
) -> anyhow::Result<()> {
    let cancel_token = runtime.primary_token();

    // Turn relative paths into absolute paths and canonicalize them
    let mut model_path = flags
        .model_path_pos
        .clone()
        .or(flags.model_path_flag.clone())
        .and_then(|p| {
            // Check for hf:// prefix first
            if let Some(hf_path) = p.to_string_lossy().strip_prefix(HF_SCHEME) {
                return Some(PathBuf::from(hf_path));
            }
            if p.exists() {
                p.canonicalize().ok()
            } else {
                Some(p)
            }
        });

    // Serve the model under the name provided, or the name of the GGUF file or HF repo.
    let mut model_name = flags
        .model_name
        .clone()
        .or_else(|| {
            model_path
                .as_ref()
                .and_then(|p| p.iter().next_back())
                .map(|n| n.to_string_lossy().into_owned())
        })
        .or_else(|| {
            if in_opt == Input::Text {
                Some(INVISIBLE_MODEL_NAME.to_string())
            } else {
                None
            }
        });

    // If it's an HF repo download it
    if let Some(inner_model_path) = model_path.as_ref() {
        if !inner_model_path.exists() && !inner_model_path.is_absolute() {
            model_name = Some(inner_model_path.display().to_string());
            model_path = Some(hub::from_hf(inner_model_path).await?);
        }
    }

    // Load the model deployment card, if any
    // Only used by some engines, so without those feature flags it's unused.
    #[allow(unused_variables)]
    let maybe_card = match (&model_path, &flags.model_config) {
        // --model-config takes precedence
        (_, Some(model_config)) => {
            match ModelDeploymentCard::from_local_path(model_config, model_name.as_deref()).await {
                Ok(card) => Some(card),
                Err(e) => {
                    tracing::error!(
                        "Failed to load model card from --model-config path {}: {e}",
                        model_config.display(),
                    );
                    None
                }
            }
        }
        // If --model-path is an HF repo use that
        (Some(model_path), _) if model_path.is_dir() => {
            match ModelDeploymentCard::from_local_path(model_path, model_name.as_deref()).await {
                Ok(card) => Some(card),
                Err(e) => {
                    tracing::error!(
                        "Failed to load model card from --model-path {}: {e}",
                        model_path.display(),
                    );
                    None
                }
            }
        }
        (Some(model_path), _) if model_path.is_file() => {
            match ModelDeploymentCard::from_gguf(model_path, model_name.as_deref()).await {
                Ok(card) => Some(card),
                Err(e) => {
                    tracing::error!(
                        "Failed to load model card from GGUF {}: {e}",
                        model_path.display(),
                    );
                    None
                }
            }
        }
        // Otherwise we don't have one, but we only need it if we're tokenizing
        _ => {
            tracing::debug!(
                "No model card path provided (neither --model-config nor --model-path)"
            );
            None
        }
    };

    let dyn_input = match &in_opt {
        Input::Endpoint(endpoint_path) => {
            if model_path.as_ref().map(|mp| mp.is_file()).unwrap_or(false)
                && flags.model_config.is_none()
            {
                // TODO We need to convert tokenizer extract from GGUF file into something we can
                // publish to NATS. Ideally `tokenizer.json` directly, but otherwise an
                // intermediate format.
                tracing::error!("Serving GGUF files in a distributed system requires `--model-config <hf-repo-dir>` so that we can find the tokenzier config");
                return Ok(());
            }

            // If we are in a distributed system, we need to know our component upfront
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;
            let endpoint_id: Endpoint = endpoint_path.parse()?;
            Some(DynInput {
                endpoint_id,
                distributed_runtime,
            })
        }
        _ => None,
    };

    #[cfg(any(feature = "vllm", feature = "sglang"))]
    let mut extra: Option<Pin<Box<dyn Future<Output = ()> + Send>>> = None; // vllm and sglang sub-process

    let template = if let Some(path) = flags.request_template.as_ref() {
        let template = RequestTemplate::load(path)?;
        tracing::debug!("Using request template: {template:?}");
        Some(template)
    } else {
        None
    };

    // Create the engine matching `out`
    let engine_config = match out_opt {
        Output::EchoFull => {
            let Some(model_name) = model_name else {
                anyhow::bail!(
                    "Pass --model-name or --model-path so we know which model to imitate"
                );
            };
            EngineConfig::StaticFull {
                card: Box::new(ModelDeploymentCard::with_name_only(&model_name)),
                service_name: model_name,
                engine: dynamo_llm::engines::make_engine_full(),
            }
        }
        Output::EchoCore => {
            let Some(mut card) = maybe_card.clone() else {
                anyhow::bail!(
                    "out=echo_core need to find the tokenizer. Pass flag --model-path <path>"
                );
            };
            EngineConfig::StaticCore {
                service_name: card.service_name.clone(),
                engine: dynamo_llm::engines::make_engine_core(),
                card: Box::new(card),
            }
        }
        Output::Endpoint(path) => {
            let endpoint: Endpoint = path.parse()?;
            EngineConfig::Dynamic(endpoint)
        }
        #[cfg(feature = "mistralrs")]
        Output::MistralRs => {
            let Some(model_path) = model_path else {
                anyhow::bail!("out=mistralrs requires flag --model-path=<full-path-to-model-gguf>");
            };
            let Some(model_name) = model_name else {
                unreachable!("We checked model_path earlier, and set model_name from model_path");
            };
            EngineConfig::StaticFull {
                card: Box::new(ModelDeploymentCard::with_name_only(&model_name)),
                service_name: model_name,
                engine: dynamo_engine_mistralrs::make_engine(&model_path).await?,
            }
        }
        #[cfg(feature = "sglang")]
        Output::SgLang => {
            let Some(model_path) = model_path else {
                anyhow::bail!("out=sglang requires flag --model-path=<full-path-to-model-dir>");
            };
            if !model_path.is_dir() {
                anyhow::bail!("`--model-path should point at a HuggingFace repo checkout");
            }
            // Safety: Earlier we build maybe_card from model_path, which we checked right above
            let card = maybe_card.clone().unwrap();
            let Some(sock_prefix) = zmq_socket_prefix else {
                anyhow::bail!("sglang requires zmq_socket_prefix");
            };
            let node_conf = dynamo_llm::engines::MultiNodeConfig {
                num_nodes: flags.num_nodes,
                node_rank: flags.node_rank,
                leader_addr: flags.leader_addr.clone().unwrap_or_default(),
            };
            if node_conf.num_nodes > 1 {
                if let Ok(Some(if_name)) = net::get_primary_interface().await {
                    tracing::info!("If you see 'gloo' errors from sglang try setting these environment variables:");
                    tracing::info!("export GLOO_SOCKET_IFNAME={if_name}");
                    tracing::info!("export NCCL_SOCKET_IFNAME={if_name}");
                }
                if node_conf.node_rank != 0 {
                    // Follower nodes take input from leader node over pytorch distributed, not
                    // from user.
                    in_opt = Input::None;
                }
            }

            let (engine, sglang_process) = dynamo_engine_sglang::make_engine(
                cancel_token.clone(),
                &model_path,
                &sock_prefix,
                node_conf,
                flags.tensor_parallel_size,
                flags.base_gpu_id,
                flags.extra_engine_args.clone(),
            )
            .await?;
            extra = Some(Box::pin(async move {
                let _ = sglang_process.await;
            }));
            EngineConfig::StaticCore {
                service_name: card.service_name.clone(),
                engine,
                card: Box::new(card),
            }
        }
        #[cfg(feature = "vllm")]
        Output::Vllm0_7 => {
            if flags.base_gpu_id != 0 {
                anyhow::bail!("vllm does not support base_gpu_id. Set environment variable CUDA_VISIBLE_DEVICES instead.");
            }
            let Some(model_path) = model_path else {
                anyhow::bail!(
                    "out=vllm requires flag --model-path=<full-path-to-hf-repo-or-model-gguf>"
                );
            };
            let Some(card) = maybe_card.clone() else {
                anyhow::bail!(
                    "Unable to build tokenizer. out=vllm requires --model-path to be an HF repo with fast tokenizer (tokenizer.json) or a GGUF file"
                );
            };
            let Some(sock_prefix) = zmq_socket_prefix else {
                anyhow::bail!("vllm requires zmq_socket_prefix");
            };
            let node_conf = dynamo_llm::engines::MultiNodeConfig {
                num_nodes: flags.num_nodes,
                node_rank: flags.node_rank,
                leader_addr: flags.leader_addr.clone().unwrap_or_default(),
            };
            if node_conf.num_nodes > 1 {
                if let Ok(Some(if_name)) = net::get_primary_interface().await {
                    tracing::info!("If you see network errors from vllm try setting this environment variable:");
                    tracing::info!("export NCCL_SOCKET_IFNAME={if_name}");
                }
                if node_conf.node_rank != 0 {
                    // Only node 0 runs vllm, the others communicate over ray
                    in_opt = Input::None;
                }
            }
            if node_conf.node_rank == 0 {
                let kv_metrics_publisher = if let Some(dyn_input) = &dyn_input {
                    let kvp_component = dyn_input
                        .distributed_runtime
                        .namespace(dyn_input.endpoint_id.namespace.clone())?
                        .component(KV_PUBLISHER_COMPONENT)?;
                    let kvp = Arc::new(KvMetricsPublisher::new()?);
                    let kvp_inner = kvp.clone();
                    tokio::spawn(
                        async move { kvp_inner.create_endpoint(kvp_component, None).await },
                    );
                    Some(kvp)
                } else {
                    None
                };

                // vllm multi-node only the leader runs vllm
                let (engine, vllm_future) = dynamo_engine_vllm0_7::make_leader_engine(
                    cancel_token.clone(),
                    &model_path,
                    &sock_prefix,
                    node_conf,
                    flags.tensor_parallel_size,
                    flags.extra_engine_args.clone(),
                    kv_metrics_publisher,
                )
                .await?;
                extra = Some(Box::pin(async move {
                    let _ = vllm_future.await;
                }));
                EngineConfig::StaticCore {
                    service_name: card.service_name.clone(),
                    engine,
                    card: Box::new(card),
                }
            } else {
                // Nodes rank > 0 only run 'ray'
                let stop_future =
                    dynamo_engine_vllm0_7::start_follower(cancel_token.clone(), node_conf).await?;
                extra = Some(Box::pin(stop_future));
                EngineConfig::None
            }
        }

        #[cfg(feature = "vllm")]
        Output::Vllm | Output::Vllm0_8 => {
            if flags.base_gpu_id != 0 {
                anyhow::bail!("vllm does not support base_gpu_id. Set environment variable CUDA_VISIBLE_DEVICES instead.");
            }
            let Some(model_path) = model_path else {
                anyhow::bail!(
                    "out=vllm requires flag --model-path=<full-path-to-hf-repo-or-model-gguf>"
                );
            };
            let Some(card) = maybe_card.clone() else {
                anyhow::bail!(
                    "Unable to build tokenizer. out=vllm requires --model-path to be an HF repo with fast tokenizer (tokenizer.json) or a GGUF file"
                );
            };
            let node_conf = dynamo_llm::engines::MultiNodeConfig {
                num_nodes: flags.num_nodes,
                node_rank: flags.node_rank,
                leader_addr: flags.leader_addr.clone().unwrap_or_default(),
            };
            let engine = dynamo_engine_vllm0_8::make_engine(
                cancel_token.clone(),
                &model_path,
                node_conf,
                flags.tensor_parallel_size,
                flags.extra_engine_args.clone(),
            )
            .await?;
            EngineConfig::StaticCore {
                service_name: card.service_name.clone(),
                engine,
                card: Box::new(card),
            }
        }

        #[cfg(feature = "llamacpp")]
        Output::LlamaCpp => {
            let Some(model_path) = model_path else {
                anyhow::bail!("out=llamacpp requires flag --model-path=<full-path-to-model-gguf>");
            };
            if !model_path.is_file() {
                anyhow::bail!("--model-path should refer to a GGUF file. llama_cpp does not support safetensors.");
            }
            let Some(card) = maybe_card.clone() else {
                anyhow::bail!(
                    "Pass --model-config so we can find the tokenizer, should be an HF checkout."
                );
            };
            let engine =
                dynamo_engine_llamacpp::make_engine(cancel_token.clone(), &model_path).await?;
            EngineConfig::StaticCore {
                service_name: card.service_name.clone(),
                engine,
                card: Box::new(card),
            }
        }
        #[cfg(feature = "python")]
        Output::PythonStr(path_str) => {
            let Some(model_name) = &model_name else {
                anyhow::bail!("Provide model service name as `--model-name <this>`");
            };
            let py_args = flags.as_vec(&path_str, model_name);
            let p = std::path::PathBuf::from(path_str);
            let engine =
                dynamo_engine_python::make_string_engine(cancel_token.clone(), &p, py_args).await?;
            EngineConfig::StaticFull {
                service_name: model_name.to_string(),
                engine,
                card: Box::new(ModelDeploymentCard::with_name_only(model_name)),
            }
        }
        #[cfg(feature = "python")]
        Output::PythonTok(path_str) => {
            let Some(card) = maybe_card.clone() else {
                anyhow::bail!("Could not find tokenizer. Pass flag --model-path <path>");
            };
            let Some(model_name) = model_name else {
                unreachable!("If we have a card we must have a model name");
            };
            let py_args = flags.as_vec(&path_str, &model_name);
            let p = std::path::PathBuf::from(path_str);
            let engine =
                dynamo_engine_python::make_token_engine(cancel_token.clone(), &p, py_args).await?;
            EngineConfig::StaticCore {
                service_name: model_name.clone(),
                engine,
                card: Box::new(card),
            }
        }
    };

    match in_opt {
        Input::Http => {
            crate::input::http::run(runtime.clone(), flags, engine_config, template).await?;
        }
        Input::Text => {
            crate::input::text::run(runtime.clone(), flags, None, engine_config, template).await?;
        }
        Input::Stdin => {
            let mut prompt = String::new();
            std::io::stdin().read_to_string(&mut prompt).unwrap();
            crate::input::text::run(
                runtime.clone(),
                flags,
                Some(prompt),
                engine_config,
                template,
            )
            .await?;
        }
        Input::Batch(path) => {
            crate::input::batch::run(
                runtime.clone(),
                flags,
                maybe_card,
                path,
                engine_config,
                template,
            )
            .await?;
        }
        Input::Endpoint(path) => {
            let Some(dyn_input) = dyn_input else {
                unreachable!("We set dyn_input earlier");
            };
            crate::input::endpoint::run(dyn_input.distributed_runtime, path, engine_config).await?;
        }
        Input::None => {
            // Multi-node setup. The engine sub-process has been started and is talking
            // to it's node_rank 0 controller. We do nothing.
            // TODO: Acquire an etcd lease, we are running
            cancel_token.cancelled().await;
        }
    }

    #[cfg(any(feature = "vllm", feature = "sglang"))]
    // Allow engines to ask main thread to wait on an extra future.
    if let Some(extra) = extra {
        extra.await;
    }

    Ok(())
}
