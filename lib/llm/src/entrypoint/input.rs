// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module contains tools to gather a prompt from a user, forward it to an engine and return
//! the response.
//! See the Input enum for the inputs available. Input::Http (OpenAI compatible HTTP server)
//! and Input::Text (interactive chat) are good places to start.
//! The main entry point is `run_input`.

use std::{
    fmt,
    io::{IsTerminal as _, Read as _},
    str::FromStr,
};

mod common;
pub use common::{build_routed_pipeline, build_routed_pipeline_with_preprocessor};
pub mod endpoint;
pub mod grpc;
pub mod http;
pub mod text;

use dynamo_runtime::protocols::ENDPOINT_SCHEME;

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

    // Run an KServe compatible gRPC server
    Grpc,
}

impl FromStr for Input {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Input::try_from(s)
    }
}

impl TryFrom<&str> for Input {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> anyhow::Result<Self> {
        match s {
            "http" => Ok(Input::Http),
            "grpc" => Ok(Input::Grpc),
            "text" => Ok(Input::Text),
            "stdin" => Ok(Input::Stdin),
            endpoint_path if endpoint_path.starts_with(ENDPOINT_SCHEME) => {
                Ok(Input::Endpoint(endpoint_path.to_string()))
            }
            e => Err(anyhow::anyhow!("Invalid in= option '{e}'")),
        }
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Input::Http => "http",
            Input::Grpc => "grpc",
            Input::Text => "text",
            Input::Stdin => "stdin",
            Input::Endpoint(path) => path,
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
/// For Input::Endpoint pass a DistributedRuntime. For everything else pass either a Runtime or a
/// DistributedRuntime.
pub async fn run_input(
    drt: dynamo_runtime::DistributedRuntime,
    in_opt: Input,
    engine_config: super::EngineConfig,
) -> anyhow::Result<()> {
    if let Err(e) = crate::agents::trace::init_from_env_with_shutdown(drt.child_token()).await {
        tracing::warn!(error = %e, "Agent trace initialization failed; continuing without trace sink");
    }

    // Initialize audit bus + sink workers (off hot path; fan-out supported)
    if crate::audit::config::policy().enabled {
        let cap: usize = std::env::var("DYN_AUDIT_CAPACITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        crate::audit::bus::init(cap);
        crate::audit::sink::spawn_workers_from_env().await?;
        tracing::info!(cap, "Audit initialized");
    }

    match in_opt {
        Input::Http => {
            http::run(drt, engine_config).await?;
        }
        Input::Grpc => {
            grpc::run(drt, engine_config).await?;
        }
        Input::Text => {
            text::run(drt, None, engine_config).await?;
        }
        Input::Stdin => {
            let mut prompt = String::new();
            std::io::stdin().read_to_string(&mut prompt).unwrap();
            text::run(drt, Some(prompt), engine_config).await?;
        }
        Input::Endpoint(path) => {
            endpoint::run(drt, path, engine_config).await?;
        }
    }
    Ok(())
}
