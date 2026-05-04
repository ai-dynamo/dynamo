// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entry point for the SGLang sidecar bridge.

use std::sync::Arc;

mod engine;
mod proto;

fn main() -> anyhow::Result<()> {
    let (engine, config) = engine::SglangBridge::from_args(None)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}
