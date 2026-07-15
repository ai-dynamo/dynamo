// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entry point for the `dynamo-sglang-remote` binary.
//!
//! Mirrors the mocker backend: bootstrap-discover the engine in `from_args`
//! (building the [`WorkerConfig`](dynamo_backend_common::WorkerConfig) `run`
//! needs synchronously), then hand the engine to the shared runtime harness.

use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let (engine, config) = dynamo_sglang_remote::SglangRemoteEngine::from_args(None)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}
