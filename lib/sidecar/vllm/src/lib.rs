// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for vLLM's released native gRPC API.

mod args;
mod client;
mod convert;
mod engine;
mod lora;
mod media;
mod model;
mod speech;

#[doc(hidden)]
pub use vllm_proto as proto;

pub use engine::VllmSidecarEngine;

/// Shared mode dispatch for the executable and embedded Python launcher.
pub fn run(argv: Vec<String>) -> anyhow::Result<()> {
    use clap::FromArgMatches;
    use dynamo_sidecar_common::SidecarStartupError;

    let matches = command()
        .try_get_matches_from(argv)
        .map_err(SidecarStartupError::from)?;
    if matches.subcommand_name().is_some() {
        match media::Mode::from_arg_matches(&matches).map_err(SidecarStartupError::from)? {
            media::Mode::Realtime(args) => {
                let (runtime, bootstrap) =
                    media::RealtimeEngine::bootstrap(args).map_err(SidecarStartupError::from)?;
                dynamo_sidecar_common::run_bidirectional((runtime, bootstrap))
            }
            media::Mode::Speech(args) => {
                let (runtime, bootstrap) =
                    speech::SpeechEngine::bootstrap(args).map_err(SidecarStartupError::from)?;
                dynamo_sidecar_common::run_raw((runtime, bootstrap))
            }
        }
    } else {
        let args = args::Args::from_arg_matches(&matches).map_err(SidecarStartupError::from)?;
        dynamo_sidecar_common::run(
            VllmSidecarEngine::bootstrap(args).map_err(SidecarStartupError::from)?,
        )
    }
}

fn command() -> clap::Command {
    use clap::{CommandFactory, Subcommand};
    media::Mode::augment_subcommands(args::Args::command())
        .subcommand_negates_reqs(true)
        .args_conflicts_with_subcommands(true)
        .about("Dynamo sidecar for vLLM gRPC, realtime ASR/Omni, and speech services")
        .mut_subcommand("realtime", |command| {
            command.mut_arg("endpoint_types", |arg| arg.default_value("realtime"))
        })
        .mut_subcommand("speech", |command| {
            command.mut_arg("endpoint_types", |arg| arg.default_value("audios"))
        })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod test_fixtures;
