// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entry point for the vLLM Rust backend.

use std::sync::Arc;

mod vllm_engine;

fn main() -> anyhow::Result<()> {
    let (engine, config) = vllm_engine::VllmBackend::from_args(None)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}
