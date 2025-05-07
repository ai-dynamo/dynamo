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

use std::env;

use clap::Parser;

use dynamo_run::{Input, Output};
use dynamo_runtime::logging;

const HELP: &str = r#"
dynamo-run is a single binary that wires together the various inputs (http, text, network) and workers (network, engine), that runs the services. It is the simplest way to use dynamo locally.

Example:
- cargo build --features cuda -p dynamo-run
- cd target/debug
- ./dynamo-run Qwen/Qwen2.5-3B-Instruct
- OR: ./dynamo-run /data/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
"#;

const USAGE: &str = "USAGE: dynamo-run in=[http|text|dyn://<path>|batch:<folder>] out=ENGINE_LIST|dyn://<path> [--http-port 8080] [--model-path <path>] [--model-name <served-model-name>] [--model-config <hf-repo>] [--tensor-parallel-size=1] [--num-nodes=1] [--node-rank=0] [--leader-addr=127.0.0.1:9876] [--base-gpu-id=0] [--extra-engine-args=args.json] [--router-mode random|round-robin]";

fn main() -> anyhow::Result<()> {
    // Set log level based on verbosity flag
    let log_level = match dynamo_run::Flags::try_parse() {
        Ok(flags) => match flags.verbosity {
            0 => "info",
            1 => "debug",
            2 => "trace",
            _ => {
                return Err(anyhow::anyhow!(
                    "Invalid verbosity level. Valid values are v (debug) or vv (trace)"
                ))
            }
        },
        Err(_) => "info",
    };

    if log_level != "info" {
        std::env::set_var("DYN_LOG", log_level);
    }

    logging::init();

    // max_worker_threads and max_blocking_threads from env vars or config file.
    let rt_config = dynamo_runtime::RuntimeConfig::from_settings()?;

    // One per process. Wraps a Runtime with holds two tokio runtimes.
    let worker = dynamo_runtime::Worker::from_config(rt_config)?;

    worker.execute(wrapper)
}

async fn wrapper(runtime: dynamo_runtime::Runtime) -> anyhow::Result<()> {
    let mut in_opt = None;
    let mut out_opt = None;
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty()
        || args[0] == "-h"
        || args[0] == "--help"
        || (args.iter().all(|arg| arg == "-v" || arg == "-vv"))
    {
        let engine_list = Output::available_engines().join("|");
        let usage = USAGE.replace("ENGINE_LIST", &engine_list);
        println!("{usage}");
        println!("{HELP}");
        return Ok(());
    }
    for arg in env::args().skip(1).take(2) {
        let Some((in_out, val)) = arg.split_once('=') else {
            // Probably we're defaulting in and/or out, and this is a flag
            continue;
        };
        match in_out {
            "in" => {
                in_opt = Some(val.try_into()?);
            }
            "out" => {
                out_opt = Some(val.try_into()?);
            }
            _ => {
                anyhow::bail!("Invalid argument, must start with 'in' or 'out. {USAGE}");
            }
        }
    }
    let mut non_flag_params = 1; // binary name
    let in_opt = match in_opt {
        Some(x) => {
            non_flag_params += 1;
            x
        }
        None => Input::default(),
    };
    let out_opt = match out_opt {
        Some(x) => {
            non_flag_params += 1;
            x
        }
        None => {
            let default_engine = Output::default(); // smart default based on feature flags
            tracing::info!(
                "Using default engine: {default_engine}. Use out=<engine> to specify one of {}",
                Output::available_engines().join(", ")
            );
            default_engine
        }
    };
    print_cuda(&out_opt);

    // Clap skips the first argument expecting it to be the binary name, so add it back
    // Note `--model-path` has index=1 (in lib.rs) so that doesn't need a flag.
    let flags = dynamo_run::Flags::try_parse_from(
        ["dynamo-run".to_string()]
            .into_iter()
            .chain(env::args().skip(non_flag_params)),
    )?;

    dynamo_run::run(runtime, in_opt, out_opt, flags).await
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
