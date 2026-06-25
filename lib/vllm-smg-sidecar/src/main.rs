// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entry point for the `dynamo-vllm-smg-sidecar` binary.

use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let (engine, config) = dynamo_vllm_smg_sidecar::VllmSmgSidecarEngine::from_args(None)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}
