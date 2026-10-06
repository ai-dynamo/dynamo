// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo SGLang sidecar.
//!
//! A [`SglangSidecarEngine`] implements [`dynamo_backend_common::LLMEngine`] by
//! proxying inference to an out-of-process SGLang engine over SGLang's native
//! `sglang.runtime.v1.SglangService` contract. Model identity, disaggregation
//! role, parallelism, KV block sizing, and context length are discovered from
//! the engine's gRPC metadata RPCs.
//!
//! The crate never depends on `sglang` or any engine crate. It uses
//! `dynamo-backend-common`, `dynamo-llm`, `dynamo-runtime`, `tonic`/`prost`,
//! `clap`, and tokio.

use std::sync::Arc;

use clap::Parser;
use dynamo_sidecar_common::SidecarStartupError;

use args::Args;
use headless::HeadlessSidecar;

pub mod args;
pub mod client;
pub mod engine;
mod headless;
mod native_http;

/// Generated SGLang gRPC types, temporarily exposed for the Mocker server
/// until SGLang publishes its upstream protocol package.
#[doc(hidden)]
pub mod proto;
mod protocol;

pub use engine::SglangSidecarEngine;

/// Parse and run the sidecar for both the Python launcher and Rust executable.
/// Startup errors retain their type so callers can preserve CLI exit codes and
/// distinguish invalid configuration from runtime failures.
pub fn run(argv: Vec<String>) -> anyhow::Result<()> {
    let args = Args::try_parse_from(argv).map_err(SidecarStartupError::from)?;
    SglangSidecarEngine::validate_args(&args).map_err(SidecarStartupError::from)?;
    dynamo_sidecar_common::run_task(|runtime, shutdown| async move {
        use dynamo_runtime::system_status_server::SystemProbePolicy;
        use dynamo_runtime::{DistributedRuntime, distributed::DistributedConfig};
        let startup = async {
            let drt = DistributedRuntime::new_with_probe_policy(
                runtime.clone(),
                DistributedConfig::try_from_settings()?,
                SystemProbePolicy::RuntimeOnly,
            )
            .await?;
            tracing::info!("Sidecar runtime connected; discovering engine metadata");
            let discovery = client::bootstrap_discover(
                &args.sidecar.grpc_endpoint,
                &args.sidecar.grpc.config(),
                false,
            )
            .await
            .map_err(SidecarStartupError::from)?;
            Ok::<_, anyhow::Error>((drt, discovery))
        };
        let runtime_shutdown = runtime.shutdown_started_token();
        let result = tokio::select! {
            biased;
            _ = shutdown.cancelled() => return Ok(()),
            _ = runtime_shutdown.cancelled() => anyhow::bail!("runtime shut down during sidecar initialization"),
            result = startup => result,
        };
        if runtime.is_shutting_down() {
            if shutdown.is_cancelled() {
                return Ok(());
            }
            anyhow::bail!("runtime shut down during sidecar initialization");
        }
        let (drt, discovery) = result?;
        match discovery {
            client::StartupDiscovery::Follower => {
                let follower =
                    HeadlessSidecar::from_args(args).map_err(SidecarStartupError::from)?;
                tokio::select! {
                    biased;
                    _ = shutdown.cancelled() => Ok(()),
                    _ = runtime_shutdown.cancelled() => anyhow::bail!("runtime shut down during follower relay"),
                    result = follower.run_inner(drt, shutdown.clone()) => result,
                }
            }
            client::StartupDiscovery::Leader(discovery) => {
                let (engine, config) = SglangSidecarEngine::from_discovered(args, *discovery)
                    .map_err(SidecarStartupError::from)?;
                dynamo_backend_common::Worker::new(Arc::new(engine), config)
                    .run_with_drt(drt, shutdown)
                    .await
                    .map_err(Into::into)
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_retains_structured_cli_exit() {
        let error = run(vec!["sidecar".into(), "--help".into()]).unwrap_err();
        let error = error.downcast::<SidecarStartupError>().unwrap();
        assert!(matches!(error, SidecarStartupError::Cli(error)
            if error.kind() == clap::error::ErrorKind::DisplayHelp));
    }
}
