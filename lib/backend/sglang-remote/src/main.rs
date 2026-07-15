// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entry point for the `dynamo-sglang-remote` binary.
//!
//! Mirrors the mocker backend: parse the process CLI with clap, bootstrap-discover
//! the engine in `from_parsed_args` (building the
//! [`WorkerConfig`](dynamo_backend_common::WorkerConfig) `run` needs
//! synchronously), then hand the engine to the shared runtime harness.

use std::sync::Arc;

use clap::Parser;
use dynamo_sglang_remote::args::Args;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let (engine, config) = dynamo_sglang_remote::SglangRemoteEngine::from_parsed_args(args)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}
