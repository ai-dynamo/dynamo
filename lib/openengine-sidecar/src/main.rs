// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let (engine, config) = dynamo_openengine_sidecar::OpenEngineSidecar::from_args(None)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}
