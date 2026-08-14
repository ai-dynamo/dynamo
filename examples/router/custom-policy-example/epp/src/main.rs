// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_ext_proc::run_with_worker_selection_policy_registry;
use dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut registry = WorkerSelectionPolicyRegistry::default();
    dynamo_custom_policy_example_catalog::register(&mut registry)?;
    run_with_worker_selection_policy_registry(registry).await
}
