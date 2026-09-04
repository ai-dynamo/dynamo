// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_mocker_backend::specdec::draft::DraftEngine;

fn main() -> anyhow::Result<()> {
    let (engine, config) = DraftEngine::from_args(None)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}
