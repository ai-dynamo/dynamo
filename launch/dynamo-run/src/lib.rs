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
use std::{io::Read, sync::Arc, time::Duration};

use anyhow::Context;
use dynamo_llm::{
    backend::ExecutionContext, engines::StreamingEngine, kv_router::publisher::KvMetricsPublisher,
    LocalModel,
};
use dynamo_runtime::{protocols::Endpoint, CancellationToken, DistributedRuntime};

mod flags;
pub use flags::Flags;
mod input;
#[cfg(any(feature = "vllm", feature = "sglang"))]
mod net;
mod opt;
pub use dynamo_llm::request_template::RequestTemplate;
pub use opt::{Input, Output};
mod subprocess;

/// When `in=text` the user doesn't need to know the model name, and doesn't need to provide it on
/// the command line. Hence it's optional, and defaults to this.
const INVISIBLE_MODEL_NAME: &str = "dynamo-run";

/// The component name for the KV publisher, if used
const KV_PUBLISHER_COMPONENT: &str = "kvpublisher";

const CHILD_STOP_TIMEOUT: Duration = Duration::from_secs(2);

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
        engine: Arc<dyn StreamingEngine>,
        model: Box<LocalModel>,
    },

    /// A core engine expects to be wrapped with pre/post processors that handle tokenization.
    StaticCore {
        engine: ExecutionContext,
        model: Box<LocalModel>,
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
    let maybe_path = flags
        .model_path_pos
        .clone()
        .or(flags.model_path_flag.clone());

    let local_model: LocalModel = match out_opt {
        // If output is an endpoint we are ingress and don't have a local model, but making an
        // empty one cleans up the code.
        Output::Endpoint(_) => Default::default(),

        // All other output types have a local model
        _ => {
            match &maybe_path {
                Some(model_path) => {
                    let maybe_model_name = if in_opt == Input::Text {
                        Some(INVISIBLE_MODEL_NAME.to_string())
                    } else {
                        flags.model_name.clone()
                    };
                    LocalModel::prepare(
                        model_path.to_str().context("Invalid UTF-8 in model path")?,
                        flags.model_config.as_deref(),
                        maybe_model_name.as_deref(),
                    )
                    .await?
                }
                None => {
                    // echo_full engine doesn't need a path
                    Default::default()
                }
            }
        }
    };

    let dyn_input = match &in_opt {
        Input::Endpoint(endpoint_path) => {
            if maybe_path.as_ref().map(|mp| mp.is_file()).unwrap_or(false)
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

    let mut extra: Option<Pin<Box<dyn Future<Output = ()> + Send>>> = None; // vllm and sglang sub-process

    let template = if let Some(path) = flags.request_template.as_ref() {
        let template = RequestTemplate::load(path)?;
        tracing::debug!("Using request template: {template:?}");
        Some(template)
    } else {
        None
    };

    // We may need it later
    let card = local_model.card().clone();

    // Create the engine matching `out`
    let engine_config = match out_opt {
        Output::Endpoint(path) => {
            let endpoint: Endpoint = path.parse()?;
            EngineConfig::Dynamic(endpoint)
        }
        Output::EchoFull => EngineConfig::StaticFull {
            model: Box::new(local_model),
            engine: dynamo_llm::engines::make_engine_full(),
        },
        Output::EchoCore => {
            let card = local_model.card();
            if !card.has_tokenizer() {
                anyhow::bail!(
                    "out=echo_core need to find the tokenizer. Pass flag --model-path <path>"
                );
            };
            EngineConfig::StaticCore {
                engine: dynamo_llm::engines::make_engine_core(),
                model: Box::new(local_model),
            }
        }
        #[cfg(feature = "mistralrs")]
        Output::MistralRs => EngineConfig::StaticFull {
            engine: dynamo_engine_mistralrs::make_engine(local_model.path()).await?,
            model: Box::new(local_model),
        },

        Output::SgLang => {
            if !local_model.path().is_dir() {
                // TODO Does sglang support GGUF? Can we make it work?
                anyhow::bail!("`--model-path should point at a HuggingFace repo checkout");
            }
            let (py_script, mut child) = match subprocess::start(
                subprocess::sglang::PY,
                local_model.path(),
                flags.tensor_parallel_size,
                if flags.base_gpu_id == 0 {
                    None
                } else {
                    Some(flags.base_gpu_id)
                },
                flags.extra_engine_args.as_deref(),
            )
            .await
            {
                Ok(x) => x,
                Err(err) => {
                    anyhow::bail!("Failed starting sglang sub-process: {err}");
                }
            };
            let cancel_token = cancel_token.clone();

            // Sub-process cleanup
            extra = Some(Box::pin(async move {
                stopper(cancel_token, child, py_script).await;
            }));
            let endpoint: Endpoint = subprocess::ENDPOINT.parse()?;
            EngineConfig::Dynamic(endpoint)
        }

        #[cfg(feature = "sglang")]
        Output::SgLangLegacy => {
            if !local_model.path().is_dir() {
                anyhow::bail!("`--model-path should point at a HuggingFace repo checkout");
            }
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
                local_model.path(),
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
                engine,
                model: Box::new(local_model),
            }
        }
        #[cfg(feature = "vllm")]
        Output::Vllm0_7 => {
            if flags.base_gpu_id != 0 {
                anyhow::bail!("vllm does not support base_gpu_id. Set environment variable CUDA_VISIBLE_DEVICES instead.");
            }
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
                    local_model.path(),
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
                    engine,
                    model: Box::new(local_model),
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
        Output::Vllm0_8 => {
            if flags.base_gpu_id != 0 {
                anyhow::bail!("vllm does not support base_gpu_id. Set environment variable CUDA_VISIBLE_DEVICES instead.");
            }
            let node_conf = dynamo_llm::engines::MultiNodeConfig {
                num_nodes: flags.num_nodes,
                node_rank: flags.node_rank,
                leader_addr: flags.leader_addr.clone().unwrap_or_default(),
            };
            let engine = dynamo_engine_vllm0_8::make_engine(
                cancel_token.clone(),
                local_model.path(),
                node_conf,
                flags.tensor_parallel_size,
                flags.extra_engine_args.clone(),
            )
            .await?;
            EngineConfig::StaticCore {
                engine,
                model: Box::new(local_model),
            }
        }

        // No feature flag because it uses a sub-process, it's very cheap to include
        Output::Vllm => {
            if flags.base_gpu_id != 0 {
                anyhow::bail!("vllm does not support base_gpu_id. Set environment variable CUDA_VISIBLE_DEVICES instead.");
            }
            let (py_script, mut child) = match subprocess::start(
                subprocess::vllm::PY,
                local_model.path(),
                flags.tensor_parallel_size,
                None, // base_gpu_id. vllm uses CUDA_VISIBLE_DEVICES instead
                flags.extra_engine_args.as_deref(),
            )
            .await
            {
                Ok(x) => x,
                Err(err) => {
                    anyhow::bail!("Failed starting vllm sub-process: {err}");
                }
            };
            let cancel_token = cancel_token.clone();

            // Sub-process cleanup
            extra = Some(Box::pin(async move {
                stopper(cancel_token, child, py_script).await;
            }));
            let endpoint: Endpoint = subprocess::ENDPOINT.parse()?;
            EngineConfig::Dynamic(endpoint)
        }

        #[cfg(feature = "llamacpp")]
        Output::LlamaCpp => {
            if !local_model.path().is_file() {
                anyhow::bail!("--model-path should refer to a GGUF file. llama_cpp does not support safetensors.");
            }
            let engine =
                dynamo_engine_llamacpp::make_engine(cancel_token.clone(), local_model.path())
                    .await?;
            EngineConfig::StaticCore {
                engine,
                model: Box::new(local_model),
            }
        }
        #[cfg(feature = "python")]
        Output::PythonStr(path_str) => {
            let card = local_model.card();
            let py_args = flags.as_vec(&path_str, &card.service_name);
            let p = std::path::PathBuf::from(path_str);
            let engine =
                dynamo_engine_python::make_string_engine(cancel_token.clone(), &p, py_args).await?;
            EngineConfig::StaticFull {
                engine,
                model: Box::new(local_model),
            }
        }
        #[cfg(feature = "python")]
        Output::PythonTok(path_str) => {
            let card = local_model.card();
            let py_args = flags.as_vec(&path_str, &card.service_name);
            let p = std::path::PathBuf::from(path_str);
            let engine =
                dynamo_engine_python::make_token_engine(cancel_token.clone(), &p, py_args).await?;
            EngineConfig::StaticCore {
                engine,
                model: Box::new(local_model),
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
            crate::input::batch::run(runtime.clone(), flags, card, path, engine_config, template)
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

    // Allow engines to ask main thread to wait on an extra future.
    // We use this to stop the vllm and sglang sub-process
    if let Some(extra) = extra {
        extra.await;
    }

    Ok(())
}

/// Wait for cancel_token to be cancelled, then stop the child as gracefully as possible.
/// Keeps the TempPath alive until the child is stopped.
async fn stopper(
    cancel_token: CancellationToken,
    mut child: tokio::process::Child,
    py_script: tempfile::TempPath,
) {
    cancel_token.cancelled().await;

    // Ask subprocess to stop gracefully
    if let Some(pid) = child.id() {
        unsafe { libc::kill(pid as i32, libc::SIGTERM) };
    }

    tokio::select! {
        exit = child.wait() => {
            tracing::trace!("vllm sub-process graceful exit");
            match exit {
                Ok(exit_status) if exit_status.success() => {}
                Ok(exit_status) => {
                    // This is nearly always 15 (SIGTERM)
                    tracing::trace!("vllm sub-process non-0 exit: {exit_status}");
                }
                Err(err) => {
                    tracing::warn!("vllm sub-process error getting exit status: {err}");
                }
            }
        }
        _ = tokio::time::sleep(CHILD_STOP_TIMEOUT) => {
            // It didn't stop in time, kill it
            child.kill().await.expect("Failed killing vllm subprocess");
            let _ = child.wait().await;
        }
    }
    // This temporary file contains the python script running the engine. It deletes on drop.
    // Keep it alive until the engine has stopped.
    drop(py_script);
}
