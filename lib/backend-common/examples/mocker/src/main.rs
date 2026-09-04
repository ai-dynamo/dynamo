// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entry point for the mocker backend example.

use std::sync::Arc;

use dynamo_mocker_backend::engine::MockerBackend;

fn main() -> anyhow::Result<()> {
    let (engine, config) = MockerBackend::from_args(None)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}
