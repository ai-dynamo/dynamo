// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use dynamo_speech_nim_sidecar::{client, config::Args, worker};

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let runtime = args.mode.worker().runtime.clone();
    let bootstrap = async move {
        let channels = client::connect(&args.mode).await?;
        Ok((args.mode, channels))
    };
    dynamo_sidecar_common::run_with_drt(runtime, bootstrap, |drt, (mode, channels), shutdown| {
        worker::serve(mode, channels, drt, shutdown)
    })
}
