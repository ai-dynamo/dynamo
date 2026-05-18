// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entry point for the vLLM Rust backend.

use std::sync::Arc;

mod backend;
mod convert;
mod error;

fn main() -> anyhow::Result<()> {
    let (engine, config) = backend::VllmBackend::from_args(None)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}
