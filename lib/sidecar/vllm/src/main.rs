// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "jemalloc")]
dynamo_sidecar_common::jemalloc_global_allocator!();

fn main() -> anyhow::Result<()> {
    dynamo_sidecar_common::run(dynamo_vllm_sidecar::VllmSidecarEngine::from_cli()?)
}
