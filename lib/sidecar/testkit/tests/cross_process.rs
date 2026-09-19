// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "process/cancellation.rs"]
mod cancellation;
#[path = "process/handoff.rs"]
mod handoff;
#[path = "process/lifecycle.rs"]
mod lifecycle;
#[path = "support/process.rs"]
mod process;
#[allow(dead_code)]
mod support;
#[path = "process/vllm.rs"]
mod vllm;

use support::vllm::Fixture;

#[tokio::test]
async fn vllm_registration_and_errors_recover_through_worker_ingress() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        lifecycle::registration_and_errors_recover_through_worker_ingress::<Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn vllm_delayed_startup_publishes_only_after_native_readiness() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        lifecycle::delayed_startup_publishes_only_after_native_readiness::<Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn vllm_failed_and_interrupted_startup_leave_no_registration() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        lifecycle::failed_and_interrupted_startup_leave_no_registration::<Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn vllm_worker_cancel_and_consumer_drop_release_only_the_target() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        cancellation::worker_cancel_and_consumer_drop_release_only_the_target::<Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn vllm_sigterm_withdraws_worker_and_releases_active_native_request() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        cancellation::sigterm_withdraws_worker_and_releases_active_native_request::<Fixture>(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}

#[tokio::test]
async fn vllm_prefill_router_preserves_handoff_failure_and_cancellation() {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        handoff::vllm_prefill_router_preserves_handoff_failure_and_cancellation(),
    )
    .await
    .expect("process scenario exceeded its overall deadline");
}
