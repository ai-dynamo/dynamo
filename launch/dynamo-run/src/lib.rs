// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;
use std::{future::Future, pin::Pin};

use anyhow::Context as _;
use dynamo_llm::entrypoint::input::Input;
use dynamo_llm::entrypoint::EngineConfig;
use dynamo_llm::local_model::{LocalModel, LocalModelBuilder};
use dynamo_runtime::CancellationToken;
use dynamo_runtime::{DistributedRuntime, Runtime};

mod flags;
use either::Either;
pub use flags::Flags;
mod opt;
pub use dynamo_llm::request_template::RequestTemplate;
pub use opt::Output;
mod subprocess;

const CHILD_STOP_TIMEOUT: Duration = Duration::from_secs(2);

pub async fn run(
    runtime: Runtime,
    in_opt: Input,
    out_opt: Option<Output>,
    flags: Flags,
) -> anyhow::Result<()> {
    //
    // Configure
    //

    let mut builder = LocalModelBuilder::default();
    builder
        .model_path(
            flags
                .model_path_pos
                .clone()
                .or(flags.model_path_flag.clone()),
        )
        .model_name(flags.model_name.clone())
        .kv_cache_block_size(flags.kv_cache_block_size)
        // Only set if user provides. Usually loaded from tokenizer_config.json
        .context_length(flags.context_length)
        .http_port(Some(flags.http_port))
        .router_config(flags.router_config())
        .request_template(flags.request_template.clone());

    // If `in=dyn` we want the trtllm/sglang/vllm subprocess to listen on that endpoint.
    // If not, then the endpoint isn't exposed so we let LocalModel invent one.
    let mut rt = Either::Left(runtime.clone());
    if let Input::Endpoint(path) = &in_opt {
        builder.endpoint_id(Some(path.parse().with_context(|| path.clone())?));

        let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;
        rt = Either::Right(distributed_runtime);
    };

    let local_model = builder.build().await?;

    //
    // Create an engine
    //

    let out_opt = out_opt.unwrap_or_else(|| default_engine_for(&local_model));
    print_cuda(&out_opt);

    // Now that we know the output we're targeting, check if we expect it to work
    flags.validate(&local_model, &out_opt)?;

    // Make an engine from the local_model, flags and output.
    let (engine_config, extra) = engine_for(
        runtime.primary_token(),
        out_opt,
        flags.clone(),
        local_model,
        rt.clone(),
    )
    .await?;

    //
    // Run in from an input
    //
    dynamo_llm::entrypoint::input::run_input(rt, in_opt, engine_config).await?;

    // Allow engines to ask main thread to wait on an extra future.
    // We use this to stop the vllm and sglang sub-process
    if let Some(extra) = extra {
        extra.await;
    }

    Ok(())
}

type ExtraFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

/// Create the engine matching `out_opt`
/// Note validation happens in Flags::validate. In here assume everything is going to work.
async fn engine_for(
    cancel_token: CancellationToken,
    out_opt: Output,
    flags: Flags,
    local_model: LocalModel,
    rt: Either<Runtime, DistributedRuntime>,
) -> anyhow::Result<(EngineConfig, Option<ExtraFuture>)> {
    match out_opt {
        Output::Dynamic => Ok((EngineConfig::Dynamic(Box::new(local_model)), None)),
        Output::EchoFull => Ok((
            EngineConfig::StaticFull {
                model: Box::new(local_model),
                engine: dynamo_llm::engines::make_engine_full(),
            },
            None,
        )),
        Output::EchoCore => Ok((
            EngineConfig::StaticCore {
                engine: dynamo_llm::engines::make_engine_core(),
                model: Box::new(local_model),
            },
            None,
        )),
        #[cfg(feature = "mistralrs")]
        Output::MistralRs => Ok((
            EngineConfig::StaticFull {
                engine: dynamo_engine_mistralrs::make_engine(&local_model).await?,
                model: Box::new(local_model),
            },
            None,
        )),
        #[cfg(feature = "llamacpp")]
        Output::LlamaCpp => Ok((
            EngineConfig::StaticCore {
                engine: dynamo_engine_llamacpp::make_engine(cancel_token, &local_model).await?,
                model: Box::new(local_model),
            },
            None,
        )),
        // For multi-node config. vllm uses `ray`, see guide
        Output::Vllm => shell(subprocess::vllm::PY, cancel_token, local_model, flags, None).await,
        // For multi-node config. trtlllm uses `mpi`, see guide
        Output::Trtllm => {
            shell(
                subprocess::trtllm::PY,
                cancel_token,
                local_model,
                flags,
                None,
            )
            .await
        }
        Output::SgLang => {
            let multi_node_config = if flags.num_nodes > 1 {
                Some(dynamo_llm::engines::MultiNodeConfig {
                    num_nodes: flags.num_nodes,
                    node_rank: flags.node_rank,
                    leader_addr: flags.leader_addr.clone().unwrap_or_default(),
                })
            } else {
                None
            };
            shell(
                subprocess::sglang::PY,
                cancel_token,
                local_model,
                flags,
                multi_node_config,
            )
            .await
        }
        Output::Mocker => {
            let Either::Right(drt) = rt else {
                panic!("Mocker requires a distributed runtime to run.");
            };

            let args = flags.mocker_config();
            let endpoint = local_model.endpoint_id().clone();

            let engine =
                dynamo_llm::mocker::engine::make_mocker_engine(drt, endpoint, args).await?;

            Ok((
                EngineConfig::StaticCore {
                    engine,
                    model: Box::new(local_model),
                },
                None,
            ))
        }
    }
}

async fn shell(
    py_script: &'static str,
    cancel_token: CancellationToken,
    local_model: LocalModel,
    flags: Flags,
    multi_node_config: Option<dynamo_llm::engines::MultiNodeConfig>,
) -> anyhow::Result<(EngineConfig, Option<ExtraFuture>)> {
    let (py_script, child) =
        match subprocess::start(py_script, &local_model, flags.clone(), multi_node_config).await {
            Ok(x) => x,
            Err(err) => {
                anyhow::bail!("Failed starting engine sub-process: {err}");
            }
        };

    // Sub-process cleanup
    let extra: ExtraFuture = Box::pin(async move {
        stopper(cancel_token, child, py_script).await;
    });
    Ok((EngineConfig::Dynamic(Box::new(local_model)), Some(extra)))
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
            tracing::trace!("engine sub-process graceful exit");
            match exit {
                Ok(exit_status) if exit_status.success() => {}
                Ok(exit_status) => {
                    // This is nearly always 15 (SIGTERM)
                    tracing::trace!("engine sub-process non-0 exit: {exit_status}");
                }
                Err(err) => {
                    tracing::warn!("engine sub-process error getting exit status: {err}");
                }
            }
        }
        _ = tokio::time::sleep(CHILD_STOP_TIMEOUT) => {
            // It didn't stop in time, kill it
            child.kill().await.expect("Failed killing engine subprocess");
            let _ = child.wait().await;
        }
    }
    // This temporary file contains the python script running the engine. It deletes on drop.
    // Keep it alive until the engine has stopped.
    drop(py_script);
}

/// If the user will benefit from CUDA/Metal/Vulkan, remind them to build with it.
/// If they have it, celebrate!
// Only mistralrs and llamacpp need to be built with CUDA.
// The Python engines only need it at runtime.
#[cfg(any(feature = "mistralrs", feature = "llamacpp"))]
fn print_cuda(output: &Output) {
    // These engines maybe be compiled in, but are they the chosen one?
    match output {
        #[cfg(feature = "mistralrs")]
        Output::MistralRs => {}
        #[cfg(feature = "llamacpp")]
        Output::LlamaCpp => {}
        _ => {
            return;
        }
    }

    #[cfg(feature = "cuda")]
    {
        tracing::info!("CUDA on");
    }
    #[cfg(feature = "metal")]
    {
        tracing::info!("Metal on");
    }
    #[cfg(feature = "vulkan")]
    {
        tracing::info!("Vulkan on");
    }
    #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
    tracing::info!("CPU mode. Rebuild with `--features cuda|metal|vulkan` for better performance");
}

#[cfg(not(any(feature = "mistralrs", feature = "llamacpp")))]
fn print_cuda(_output: &Output) {}

fn default_engine_for(local_model: &LocalModel) -> Output {
    let default_engine = if local_model.card().is_gguf() {
        gguf_default()
    } else {
        safetensors_default()
    };
    tracing::info!(
        "Using default engine: {default_engine}. Use out=<engine> to specify one of {}",
        Output::available_engines().join(", ")
    );
    default_engine
}

fn gguf_default() -> Output {
    #[cfg(feature = "llamacpp")]
    {
        Output::LlamaCpp
    }

    #[cfg(all(feature = "mistralrs", not(feature = "llamacpp")))]
    {
        Output::MistralRs
    }

    #[cfg(not(any(feature = "mistralrs", feature = "llamacpp")))]
    {
        Output::EchoFull
    }
}

fn safetensors_default() -> Output {
    #[cfg(feature = "mistralrs")]
    {
        Output::MistralRs
    }

    #[cfg(not(feature = "mistralrs"))]
    {
        Output::EchoFull
    }
}
