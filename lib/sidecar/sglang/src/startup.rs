// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use clap::Parser;
use dynamo_backend_common::WorkerConfig;
use dynamo_sidecar_common::SidecarStartupError;

use crate::args::Args;
use crate::context::SidecarMode;
use crate::engine::SglangSidecarEngine;
use crate::headless::HeadlessSidecar;

/// Common startup dispatcher for the managed Python module and Rust executable.
pub struct SglangSidecar(Mode);

enum Mode {
    Full(Box<(SglangSidecarEngine, WorkerConfig)>),
    Telemetry(Box<HeadlessSidecar>),
}

impl SglangSidecar {
    pub fn try_from_args(argv: Vec<String>) -> Result<Self, SidecarStartupError> {
        let args = Args::try_parse_from(argv)?;
        let mode = if args
            .sidecar_context
            .as_ref()
            .is_some_and(|context| context.mode == SidecarMode::Telemetry)
        {
            Mode::Telemetry(Box::new(HeadlessSidecar::from_args(args)?))
        } else {
            Mode::Full(Box::new(SglangSidecarEngine::from_parsed(args)?))
        };
        Ok(Self(mode))
    }

    pub fn run(self) -> anyhow::Result<()> {
        match self.0 {
            Mode::Full(full) => {
                let (engine, config) = *full;
                dynamo_backend_common::run(Arc::new(engine), config)
            }
            Mode::Telemetry(headless) => headless.run(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::tests::context_json;

    #[test]
    fn telemetry_needs_no_grpc_and_does_not_parse_an_inherited_endpoint() {
        let sidecar = SglangSidecar::try_from_args(vec![
            "sidecar".into(),
            "--sidecar-context".into(),
            context_json("telemetry").to_string(),
            "--grpc-endpoint".into(),
            "not-a-grpc-address".into(),
        ])
        .unwrap();
        assert!(matches!(sidecar.0, Mode::Telemetry(_)));
    }

    #[test]
    fn help_retains_structured_cli_exit() {
        let result = SglangSidecar::try_from_args(vec!["sidecar".into(), "--help".into()]);
        assert!(
            matches!(result, Err(SidecarStartupError::Cli(error)) if error.kind() == clap::error::ErrorKind::DisplayHelp)
        );
    }
}
