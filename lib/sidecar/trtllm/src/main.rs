// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> anyhow::Result<()> {
    match dynamo_trtllm_sidecar::run(std::env::args().collect()) {
        Ok(()) => Ok(()),
        Err(error) => match error.downcast::<dynamo_sidecar_common::SidecarStartupError>() {
            Ok(dynamo_sidecar_common::SidecarStartupError::Cli(error)) => error.exit(),
            Ok(error) => Err(error.into()),
            Err(error) => Err(error),
        },
    }
}
