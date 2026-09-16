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
//! The crate never depends on `sglang` or any engine crate — only
//! `dynamo-backend-common`, `tonic`/`prost`, `clap`, and tokio.

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
    if args.telemetry_only {
        HeadlessSidecar::from_args(args)
            .map_err(SidecarStartupError::from)?
            .run()
    } else {
        let (engine, config) =
            SglangSidecarEngine::from_parsed(args).map_err(SidecarStartupError::from)?;
        dynamo_backend_common::run(Arc::new(engine), config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_uses_headless_validation_before_grpc() {
        let error = run(vec![
            "sidecar".into(),
            "--telemetry-only".into(),
            "--grpc-endpoint".into(),
            "not-a-grpc-address".into(),
            // Stop at headless validation, before starting a runtime.
            "--route-to-encoder".into(),
        ])
        .unwrap_err();
        let error = error.downcast::<SidecarStartupError>().unwrap();
        assert!(matches!(error, SidecarStartupError::Dynamo(ref error)
            if error.to_string().contains("telemetry mode cannot register encoder or RL request routes")));
    }

    #[test]
    fn help_retains_structured_cli_exit() {
        let error = run(vec!["sidecar".into(), "--help".into()]).unwrap_err();
        let error = error.downcast::<SidecarStartupError>().unwrap();
        assert!(matches!(error, SidecarStartupError::Cli(error)
            if error.kind() == clap::error::ErrorKind::DisplayHelp));
    }
}
