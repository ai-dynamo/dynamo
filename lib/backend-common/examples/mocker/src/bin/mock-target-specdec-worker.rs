// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_mocker_backend::specdec::target::TargetEngine;

fn main() -> anyhow::Result<()> {
    let (engine, config) = TargetEngine::from_args(None)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}
