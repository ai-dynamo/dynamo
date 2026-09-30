// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> anyhow::Result<()> {
    match dynamo_vllm_sidecar::run(std::env::args().collect()) {
        Err(dynamo_vllm_sidecar::RunError::Startup(
            dynamo_sidecar_common::SidecarStartupError::Cli(error),
        )) => error.exit(),
        result => result.map_err(Into::into),
    }
}
