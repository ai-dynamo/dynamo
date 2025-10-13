// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context as _;
use dynamo_llm::entrypoint::EngineConfig;
use dynamo_llm::entrypoint::input::Input;
use dynamo_llm::local_model::{LocalModel, LocalModelBuilder};
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::{DistributedRuntime, Runtime};

mod flags;
use either::Either;
pub use flags::Flags;
mod opt;
pub use dynamo_llm::request_template::RequestTemplate;
pub use opt::Output;

pub async fn run(
    runtime: Runtime,
    in_opt: Input,
    out_opt: Option<Output>,
    mut flags: Flags,
) -> anyhow::Result<()> {
    //
    // Download
    //

    let maybe_remote_repo = flags
        .model_path_pos
        .clone()
        .or_else(|| flags.model_path_flag.clone());
    let model_path = match maybe_remote_repo {
        None => None,
        Some(p) if p.exists() => {
            // Already a local path
            Some(p)
        }
        Some(p) => {
            // model_path might be an HF repo, not a local path. Resolve it by downloading.
            Some(LocalModel::fetch(&p.display().to_string(), false).await?)
        }
    };

    //
    // Configure
    //

    let mut builder = LocalModelBuilder::default();
    builder
        .model_name(flags.model_name.clone())
        .kv_cache_block_size(flags.kv_cache_block_size)
        // Only set if user provides. Usually loaded from tokenizer_config.json
        .context_length(flags.context_length)
        .http_port(flags.http_port)
        .tls_cert_path(flags.tls_cert_path.take())
        .tls_key_path(flags.tls_key_path.take())
        .router_config(Some(flags.router_config()))
        .request_template(flags.request_template.clone())
        .migration_limit(flags.migration_limit)
        .is_mocker(matches!(out_opt, Some(Output::Mocker)));

    // Only the worker has a model path
    if let Some(model_path) = model_path {
        builder.model_path(model_path);
    }

    // TODO: old, address this later:
    // If `in=dyn` we want the trtllm/sglang/vllm subprocess to listen on that endpoint.
    // If not, then the endpoint isn't exposed so we let LocalModel invent one.
    let mut rt = Either::Left(runtime.clone());
    if let Input::Endpoint(path) = &in_opt {
        builder.endpoint_id(Some(path.parse().with_context(|| path.clone())?));

        let dst_config = DistributedConfig::from_settings(flags.static_worker);
        let distributed_runtime = DistributedRuntime::new(runtime.clone(), dst_config).await?;
        rt = Either::Right(distributed_runtime);
    };
    if let Some(Output::Static(path)) = &out_opt {
        builder.endpoint_id(Some(path.parse().with_context(|| path.clone())?));
    }

    let local_model = builder.build().await?;

    //
    // Create an engine
    //

    let out_opt = out_opt.unwrap_or_else(|| default_engine_for(&local_model));
    print_cuda(&out_opt);

    // Now that we know the output we're targeting, check if we expect it to work
    flags.validate(&in_opt, &out_opt)?;

    // Make an engine from the local_model, flags and output.
    let engine_config = engine_for(out_opt, flags.clone(), local_model, rt.clone()).await?;

    // Run it from an input
    dynamo_llm::entrypoint::input::run_input(rt, in_opt, engine_config).await?;

    Ok(())
}

/// Create the engine matching `out_opt`
/// Note validation happens in Flags::validate. In here assume everything is going to work.
async fn engine_for(
    out_opt: Output,
    flags: Flags,
    local_model: LocalModel,
    rt: Either<Runtime, DistributedRuntime>,
) -> anyhow::Result<EngineConfig> {
    match out_opt {
        Output::Auto => {
            // Auto-discover backends
            Ok(EngineConfig::Dynamic(Box::new(local_model)))
        }
        Output::Static(_) => {
            // A single static backend, no etcd
            Ok(EngineConfig::StaticRemote(Box::new(local_model)))
        }
        Output::Echo => Ok(EngineConfig::StaticFull {
            model: Box::new(local_model),
            engine: dynamo_llm::engines::make_echo_engine(),
            is_static: flags.static_worker,
        }),
        #[cfg(feature = "mistralrs")]
        Output::MistralRs => Ok(EngineConfig::StaticFull {
            engine: dynamo_engine_mistralrs::make_engine(&local_model).await?,
            model: Box::new(local_model),
            is_static: flags.static_worker,
        }),
        Output::Mocker => {
            let Either::Right(drt) = rt else {
                panic!("Mocker requires a distributed runtime to run.");
            };

            let args = flags.mocker_config();
            let endpoint = local_model.endpoint_id().clone();

            let engine =
                dynamo_llm::mocker::engine::make_mocker_engine(drt, endpoint, args).await?;

            Ok(EngineConfig::StaticCore {
                engine,
                model: Box::new(local_model),
                is_static: flags.static_worker,
            })
        }
    }
}

/// If the user will benefit from CUDA or Metal, remind them to build with it.
/// If they have it, celebrate!
// Only mistralrs needs to be built with CUDA.
// The Python engines only need it at runtime.
#[cfg(feature = "mistralrs")]
fn print_cuda(output: &Output) {
    // These engines maybe be compiled in, but are they the chosen one?
    match output {
        #[cfg(feature = "mistralrs")]
        Output::MistralRs => {}
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
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    tracing::info!("CPU mode. Rebuild with `--features cuda|metal` for better performance");
}

#[cfg(not(feature = "mistralrs"))]
fn print_cuda(_output: &Output) {}

fn default_engine_for(_local_model: &LocalModel) -> Output {
    safetensors_default()
}

fn safetensors_default() -> Output {
    #[cfg(feature = "mistralrs")]
    {
        Output::MistralRs
    }

    #[cfg(not(feature = "mistralrs"))]
    {
        Output::Echo
    }
}
