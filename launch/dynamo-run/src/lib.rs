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

use std::io::Read;
#[cfg(any(feature = "vllm", feature = "sglang"))]
use std::{future::Future, pin::Pin};

use dynamo_llm::{
    backend::ExecutionContext, model_card::model::ModelDeploymentCard,
    types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine,
};
use dynamo_runtime::protocols::Endpoint;

mod flags;
pub use flags::Flags;
mod hub;
mod input;
#[cfg(any(feature = "vllm", feature = "sglang"))]
mod net;
mod opt;
mod output;
pub use opt::{Input, Output};

/// How we identify a namespace/component/endpoint URL.
/// Technically the '://' is not part of the scheme but it eliminates several string
/// concatenations.
const ENDPOINT_SCHEME: &str = "dyn://";

/// When `in=text` the user doesn't need to know the model name, and doesn't need to provide it on
/// the command line. Hence it's optional, and defaults to this.
const INVISIBLE_MODEL_NAME: &str = "dynamo-run";

/// How we identify a python string endpoint
#[cfg(feature = "python")]
const PYTHON_STR_SCHEME: &str = "pystr:";

/// How we identify a python token endpoint
#[cfg(feature = "python")]
const PYTHON_TOK_SCHEME: &str = "pytok:";

pub enum EngineConfig {
    /// An remote networked engine we don't know about yet
    Dynamic(Endpoint),

    /// A Full service engine does it's own tokenization and prompt formatting.
    StaticFull {
        service_name: String,
        engine: OpenAIChatCompletionsStreamingEngine,
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

#[allow(unused_mut)]
pub async fn run(
    runtime: dynamo_runtime::Runtime,
    mut in_opt: Input, // mut because vllm and sglang multi-node can change it
    out_opt: Output,
    flags: Flags,
    #[allow(unused_variables)] zmq_socket_prefix: Option<String>,
) -> anyhow::Result<()> {
    let cancel_token = runtime.primary_token();

    // Turn relative paths into absolute paths
    let mut model_path = flags
        .model_path_pos
        .clone()
        .or(flags.model_path_flag.clone())
        .and_then(|p| {
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
                .and_then(|p| p.iter().last())
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
        if !inner_model_path.exists() {
            model_name = inner_model_path
                .iter()
                .last()
                .map(|s| s.to_string_lossy().to_string());
            model_path = Some(hub::from_hf(inner_model_path).await?);
        }
    }

    // Load the model deployment card, if any
    // Only used by some engines, so without those feature flags it's unused.
    #[allow(unused_variables)]
    let (maybe_card_path, maybe_card) = match (&model_path, &flags.model_config) {
        // --model-config takes precedence
        (_, Some(model_config)) => {
            let card = ModelDeploymentCard::from_local_path(model_config, model_name.as_deref())
                .await
                .ok();
            (Some(model_config.clone()), card)
        }
        // If --model-path is an HF repo use that
        (Some(model_path), _) if model_path.is_dir() => {
            let card = ModelDeploymentCard::from_local_path(model_path, model_name.as_deref())
                .await
                .ok();
            (Some(model_path.clone()), card)
        }
        // Otherwise we don't have one, but we only need it if we're tokenizing
        _ => (None, None),
    };

    #[cfg(any(feature = "vllm", feature = "sglang"))]
    let mut extra: Option<Pin<Box<dyn Future<Output = ()> + Send>>> = None; // vllm and sglang sub-process

    // Create the engine matching `out`
    let engine_config = match out_opt {
        Output::EchoFull => {
            let Some(model_name) = model_name else {
                anyhow::bail!(
                    "Pass --model-name or --model-path so we know which model to imitate"
                );
            };
            EngineConfig::StaticFull {
                service_name: model_name,
                engine: output::echo_full::make_engine_full(),
            }
        }
        Output::EchoCore => {
            let Some(mut card) = maybe_card.clone() else {
                anyhow::bail!(
                    "out=echo_core need to find the tokenizer. Pass flag --model-path <path>"
                );
            };
            card.requires_preprocessing = true;
            EngineConfig::StaticCore {
                service_name: card.service_name.clone(),
                engine: output::echo_core::make_engine_core(),
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
                service_name: model_name,
                engine: dynamo_llm::engines::mistralrs::make_engine(&model_path).await?,
            }
        }
        #[cfg(feature = "sglang")]
        Output::SgLang => {
            use dynamo_llm::engines::sglang;
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
                leader_addr: flags.leader_addr.unwrap_or_default(),
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

            let (engine, sglang_process) = sglang::make_engine(
                cancel_token.clone(),
                &model_path,
                &sock_prefix,
                node_conf,
                flags.tensor_parallel_size,
                flags.base_gpu_id,
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
        Output::Vllm => {
            use dynamo_llm::engines::vllm;
            if flags.base_gpu_id != 0 {
                anyhow::bail!("vllm does not support base_gpu_id. Set environment variable CUDA_VISIBLE_DEVICES instead.");
            }
            let Some(model_path) = model_path else {
                anyhow::bail!(
                    "out=vllm requires flag --model-path=<full-path-to-hf-repo-or-model-gguf>"
                );
            };
            let Some(card_path) = maybe_card_path else {
                // If we have a gguf we also need a model card because we don't currently parse
                // tokenizer et al out of gguf.
                anyhow::bail!(
                    "Running GGUF files also requires a `--model-config` for the tokenizer et al."
                );
            };
            let Some(card) = maybe_card.clone() else {
                anyhow::bail!(
                    "out=vllm requires --model-path to be an HF repo, or for GGUF add flag --model-config <hf-repo>"
                );
            };
            let Some(sock_prefix) = zmq_socket_prefix else {
                anyhow::bail!("vllm requires zmq_socket_prefix");
            };
            let node_conf = dynamo_llm::engines::MultiNodeConfig {
                num_nodes: flags.num_nodes,
                node_rank: flags.node_rank,
                leader_addr: flags.leader_addr.unwrap_or_default(),
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
                // vllm multi-node only the leader runs vllm
                let (engine, vllm_future) = vllm::make_leader_engine(
                    cancel_token.clone(),
                    &card_path,
                    &model_path,
                    &sock_prefix,
                    node_conf,
                    flags.tensor_parallel_size,
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
                let stop_future = vllm::start_follower(cancel_token.clone(), node_conf).await?;
                extra = Some(Box::pin(stop_future));
                EngineConfig::None
            }
        }
        #[cfg(feature = "llamacpp")]
        Output::LlamaCpp => {
            use dynamo_llm::engines::llamacpp;
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
            let engine = llamacpp::make_engine(cancel_token.clone(), &model_path).await?;
            EngineConfig::StaticCore {
                service_name: card.service_name.clone(),
                engine,
                card: Box::new(card),
            }
        }
        #[cfg(feature = "trtllm")]
        Output::TrtLLM => {
            use dynamo_llm::engines::trtllm;
            let Some(model_path) = model_path else {
                anyhow::bail!("out=trtllm requires flag --model-path=<full-path-to-model-dir>");
            };
            if !model_path.is_dir() {
                anyhow::bail!(
                    "--model-path should point at a directory containing `.engine` files."
                );
            }
            // Safety: Earlier we build maybe_card from model_path, which we checked right above
            let card = maybe_card.clone().unwrap();
            let engine = trtllm::make_engine(model_path.display(), flags.tensor_parallel_size)?;
            EngineConfig::StaticCore {
                service_name: card.service_name.clone(),
                engine,
                card: Box::new(card),
            }
        }
        #[cfg(feature = "python")]
        Output::PythonStr(path_str) => {
            use dynamo_llm::engines::python;
            let Some(model_name) = model_name else {
                anyhow::bail!("Provide model service name as `--model-name <this>`");
            };
            let py_args = flags.as_vec(&path_str, &model_name);
            let p = std::path::PathBuf::from(path_str);
            let engine = python::make_string_engine(cancel_token.clone(), &p, py_args).await?;
            EngineConfig::StaticFull {
                service_name: model_name,
                engine,
            }
        }
        #[cfg(feature = "python")]
        Output::PythonTok(path_str) => {
            use dynamo_llm::engines::python;
            let Some(card) = maybe_card.clone() else {
                anyhow::bail!("Could not find tokenizer. Pass flag --model-path <path>");
            };
            let Some(model_name) = model_name else {
                unreachable!("If we have a card we must have a model name");
            };
            let py_args = flags.as_vec(&path_str, &model_name);
            let p = std::path::PathBuf::from(path_str);
            let engine = python::make_token_engine(cancel_token.clone(), &p, py_args).await?;
            EngineConfig::StaticCore {
                service_name: model_name.clone(),
                engine,
                card: Box::new(card),
            }
        }
    };

    match in_opt {
        Input::Http => {
            crate::input::http::run(runtime.clone(), flags.http_port, engine_config).await?;
        }
        Input::Text => {
            crate::input::text::run(runtime.clone(), cancel_token.clone(), None, engine_config)
                .await?;
        }
        Input::Stdin => {
            let mut prompt = String::new();
            std::io::stdin().read_to_string(&mut prompt).unwrap();
            crate::input::text::run(
                runtime.clone(),
                cancel_token.clone(),
                Some(prompt),
                engine_config,
            )
            .await?;
        }
        Input::Batch(path) => {
            crate::input::batch::run(
                runtime.clone(),
                cancel_token.clone(),
                maybe_card,
                path,
                engine_config,
            )
            .await?;
        }
        Input::Endpoint(path) => {
            crate::input::endpoint::run(runtime.clone(), path, engine_config).await?;
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
