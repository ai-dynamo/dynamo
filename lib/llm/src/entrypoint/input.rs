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
pub use common::{PreprocessedRouting, build_preprocessed_routing};
pub mod endpoint;
pub mod grpc;
pub mod http;
pub mod text;

use dynamo_runtime::protocols::ENDPOINT_SCHEME;

use crate::http::service::FrontendRouteExtension;

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
    run_input_with_frontend_route_extensions(drt, in_opt, engine_config, Vec::new()).await
}

/// Run the given engine (EngineConfig) connected to an input, with optional
/// system route extensions for the HTTP frontend.
pub async fn run_input_with_frontend_route_extensions(
    drt: dynamo_runtime::DistributedRuntime,
    in_opt: Input,
    engine_config: super::EngineConfig,
    frontend_route_extensions: Vec<FrontendRouteExtension>,
) -> anyhow::Result<()> {
    if let Err(e) = crate::request_trace::init_from_env_with_shutdown(drt.child_token()).await {
        tracing::warn!(error = %e, "Request trace initialization failed; continuing without trace sink");
    }
    if let Err(e) = crate::request_trace::start_tool_event_ingest_from_policy(
        drt.clone(),
        engine_config.local_model(),
    )
    .await
    {
        tracing::warn!(error = %e, "Request trace tool event ingest initialization failed; continuing without request trace tool events");
    }

    match in_opt {
        Input::Http => {
            http::run_with_frontend_route_extensions(drt, engine_config, frontend_route_extensions)
                .await?;
        }
        Input::Grpc => {
            if !frontend_route_extensions.is_empty() {
                anyhow::bail!("frontend route extensions are only supported by HTTP input");
            }
            grpc::run(drt, engine_config).await?;
        }
        Input::Text => {
            if !frontend_route_extensions.is_empty() {
                anyhow::bail!("frontend route extensions are only supported by HTTP input");
            }
            text::run(drt, None, engine_config).await?;
        }
        Input::Stdin => {
            if !frontend_route_extensions.is_empty() {
                anyhow::bail!("frontend route extensions are only supported by HTTP input");
            }
            let mut prompt = String::new();
            std::io::stdin().read_to_string(&mut prompt).unwrap();
            text::run(drt, Some(prompt), engine_config).await?;
        }
        Input::Endpoint(path) => {
            if !frontend_route_extensions.is_empty() {
                anyhow::bail!("frontend route extensions are only supported by HTTP input");
            }
            endpoint::run(drt, path, engine_config).await?;
        }
    }
    Ok(())
}
