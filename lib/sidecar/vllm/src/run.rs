// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{fmt, sync::Arc};

use clap::Parser;
use dynamo_backend_common::{Worker, run::shutdown_signal};
use dynamo_runtime::{
    DistributedRuntime, Runtime, SystemStatusProbePolicy, distributed::DistributedConfig,
};
use dynamo_sidecar_common::SidecarStartupError;

use crate::{args::Args, engine::PreparedStartup};

/// Preserve the launcher's distinction between CLI/engine bootstrap errors and
/// runtime/worker failures (SystemExit/ValueError versus RuntimeError in Python).
#[derive(Debug)]
pub enum RunError {
    Startup(SidecarStartupError),
    Runtime(anyhow::Error),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Startup(error) => error.fmt(f),
            Self::Runtime(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Startup(error) => Some(error),
            Self::Runtime(error) => Some(error.as_ref()),
        }
    }
}

/// Launch with runtime probes available before contacting the vLLM engine.
/// `argv` includes the program name, as in Clap's `try_parse_from`.
pub fn run(argv: Vec<String>) -> Result<(), RunError> {
    let args = Args::try_parse_from(argv).map_err(|error| RunError::Startup(error.into()))?;
    let prepared = PreparedStartup::new(args).map_err(|error| RunError::Startup(error.into()))?;
    prepared
        .config
        .validate()
        .map_err(|error| RunError::Runtime(error.into()))?;
    dynamo_runtime::logging::init();
    prepared.config.runtime.apply_to_env();
    let config = DistributedConfig::from_settings_with_overrides(
        prepared.config.runtime.discovery_backend.as_deref(),
        prepared.config.runtime.request_plane.as_deref(),
        prepared.config.runtime.event_plane.as_deref(),
    )
    .map_err(RunError::Runtime)?;
    let runtime = Runtime::from_settings().map_err(RunError::Runtime)?;
    runtime.secondary().block_on(async {
        let result = async {
            let (shutdown, _signals) = shutdown_signal(runtime.clone())
                .map_err(|error| RunError::Runtime(error.into()))?;
            let drt = tokio::select! {
                biased;
                _ = shutdown.cancelled() => return Ok(()),
                result = DistributedRuntime::new_with_probe_policy(
                    runtime.clone(), config, SystemStatusProbePolicy::RuntimeOnly,
                ) => result.map_err(RunError::Runtime)?,
            };
            let (engine, config) = tokio::select! {
                biased;
                _ = shutdown.cancelled() => return Ok(()),
                result = prepared.discover() => result.map_err(|error| RunError::Startup(error.into()))?,
            };
            Worker::new(Arc::new(engine), config).run_with_drt(drt, shutdown).await
                .map_err(|error| RunError::Runtime(error.into()))
        }.await;
        // This also cleans up partial construction and cancelled discovery.
        runtime.shutdown();
        runtime.primary_token().cancelled().await;
        result
    })
}
