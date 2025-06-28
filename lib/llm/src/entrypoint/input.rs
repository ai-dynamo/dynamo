// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module contains tools to gather a prompt from a user, forward it to an engine and return
//! the response.
//! See the Input enum for the inputs available. Input::Http (OpenAI compatible HTTP server)
//! and Input::Text (interactive chat) are good places to start.
//! The main entry point is `run_input`.

use std::{
    fmt,
    io::{IsTerminal as _, Read as _},
    path::PathBuf,
};

pub mod batch;
mod common;
pub mod endpoint;
pub mod http;
pub mod text;

use dynamo_runtime::{protocols::ENDPOINT_SCHEME, DistributedRuntime};

const BATCH_PREFIX: &str = "batch:";

/// The various ways of connecting prompts to an engine
#[derive(PartialEq)]
pub enum Input {
    /// Run an OpenAI compatible HTTP server
    Http,

    /// Single prompt on stdin
    Stdin,

    /// Interactive chat
    Text,

    /// Pull requests from a namespace/component/endpoint path.
    Endpoint(String),

    /// Batch mode. Run all the prompts, write the outputs, exit.
    Batch(PathBuf),
}

impl TryFrom<&str> for Input {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> anyhow::Result<Self> {
        match s {
            "http" => Ok(Input::Http),
            "text" => Ok(Input::Text),
            "stdin" => Ok(Input::Stdin),
            endpoint_path if endpoint_path.starts_with(ENDPOINT_SCHEME) => {
                Ok(Input::Endpoint(endpoint_path.to_string()))
            }
            batch_patch if batch_patch.starts_with(BATCH_PREFIX) => {
                let path = batch_patch.strip_prefix(BATCH_PREFIX).unwrap();
                Ok(Input::Batch(PathBuf::from(path)))
            }
            e => Err(anyhow::anyhow!("Invalid in= option '{e}'")),
        }
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Input::Http => "http",
            Input::Text => "text",
            Input::Stdin => "stdin",
            Input::Endpoint(path) => path,
            Input::Batch(path) => &path.display().to_string(),
        };
        write!(f, "{s}")
    }
}

impl Default for Input {
    fn default() -> Self {
        if std::io::stdin().is_terminal() {
            Input::Text
        } else {
            Input::Stdin
        }
    }
}

/// Run the given engine (EngineConfig) connected to an input.
/// Does not return until the input exits.
pub async fn run_input(
    in_opt: Input,
    runtime: dynamo_runtime::Runtime,
    engine_config: super::EngineConfig,
) -> anyhow::Result<()> {
    match in_opt {
        Input::Http => {
            http::run(runtime.clone(), engine_config).await?;
        }
        Input::Text => {
            text::run(runtime.clone(), None, engine_config).await?;
        }
        Input::Stdin => {
            let mut prompt = String::new();
            std::io::stdin().read_to_string(&mut prompt).unwrap();
            text::run(runtime.clone(), Some(prompt), engine_config).await?;
        }
        Input::Batch(path) => {
            batch::run(runtime.clone(), path, engine_config).await?;
        }
        Input::Endpoint(path) => {
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;
            endpoint::run(distributed_runtime, path, engine_config).await?;
        }
    }
    Ok(())
}
