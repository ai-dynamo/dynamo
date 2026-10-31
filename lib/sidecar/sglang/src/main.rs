// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> anyhow::Result<()> {
    dynamo_sglang_sidecar::run(std::env::args().collect()).inspect_err(|error| {
        if let Some(dynamo_sidecar_common::SidecarStartupError::Cli(cli)) = error.downcast_ref() {
            cli.exit();
        }
    })
}
