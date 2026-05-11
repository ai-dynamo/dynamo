// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! B2 entry point: SGLang sidecar bridge as a standalone Dynamo worker.

use std::sync::Arc;

use dynamo_sglang_bridge::SglangBridge;

fn main() -> anyhow::Result<()> {
    let (engine, config) = SglangBridge::from_args(None)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}
