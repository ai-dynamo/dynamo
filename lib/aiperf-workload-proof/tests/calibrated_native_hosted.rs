// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone test target for the calibrated native-hosted control. The body
//! lives once in the shared support module so the full-scenario E2E can host
//! the same canonical role/spec/host path.

#![cfg(feature = "calibrated-proof")]

#[path = "../../../../../../aisimulate/.worktrees/aiperf-owned-runner-proof/crates/tests/aiperf-workload-runner/tests/support/run_fixture.rs"]
mod run_fixture;

#[path = "support/calibrated.rs"]
mod calibrated;

#[test]
fn calibrated_h100_fpm_drives_native_hosted_real_dynamo_kv_2p2d() {
    calibrated::calibrated_control().expect("calibrated native proof");
}
