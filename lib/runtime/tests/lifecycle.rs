// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;

use dynamo_runtime::{Runtime, worker::Worker};

async fn hello_world(_runtime: Runtime) -> Result<()> {
    Ok(())
}

#[test]
fn test_lifecycle() {
    let worker = Worker::from_settings().unwrap();
    worker.execute(hello_world).unwrap();
}
