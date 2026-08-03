// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_vllm_sidecar::Launch;

fn main() -> anyhow::Result<()> {
    // `--direct`: register + health-gate the engine's gRPC and let the frontend
    // dispatch straight to it. Otherwise serve the Dynamo request plane.
    match dynamo_vllm_sidecar::launch_from_env()? {
        Launch::Direct(backend, config) => dynamo_direct_register::run_direct(backend, config),
        Launch::RequestPlane(engine, config) => dynamo_backend_common::run(engine, config),
    }
}
