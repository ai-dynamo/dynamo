// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_lpx_sidecar::{Launch, LpxSidecarEngine};

fn main() -> anyhow::Result<()> {
    match LpxSidecarEngine::from_cli()? {
        Launch::Serve(bootstrap) => dynamo_lpx_sidecar::serve(bootstrap),
        Launch::Standby(rank) => dynamo_lpx_sidecar::run_standby(rank),
    }
}
