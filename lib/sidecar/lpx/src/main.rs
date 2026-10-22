// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_lpx_sidecar::LpxSidecarEngine;

fn main() -> anyhow::Result<()> {
    dynamo_lpx_sidecar::run(LpxSidecarEngine::from_cli()?)
}
