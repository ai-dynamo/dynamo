// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> anyhow::Result<()> {
    let sidecar =
        match dynamo_sglang_sidecar::SglangSidecar::try_from_args(std::env::args().collect()) {
            Ok(sidecar) => sidecar,
            Err(dynamo_sidecar_common::SidecarStartupError::Cli(error)) => error.exit(),
            Err(error) => return Err(error.into()),
        };
    sidecar.run()
}
