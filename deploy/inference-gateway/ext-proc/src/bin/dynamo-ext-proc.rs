// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "custom-policy")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut registry =
        dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry::default();
    dynamo_worker_selection_policy_catalog::register(&mut registry)?;

    dynamo_ext_proc::run_with_worker_selection_policy_registry(registry).await
}

#[cfg(not(feature = "custom-policy"))]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dynamo_ext_proc::run().await
}
