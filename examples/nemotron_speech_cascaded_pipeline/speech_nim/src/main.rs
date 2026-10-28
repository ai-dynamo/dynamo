// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use dynamo_speech_nim_sidecar::{config::Args, worker};

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    dynamo_sidecar_common::run_task(|runtime, shutdown| worker::run(args, runtime, shutdown))
}
