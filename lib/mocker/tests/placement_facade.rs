// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercises `dynamo_mocker::placement` from outside the crate, the way an
//! external co-simulator (aiperf) actually reaches it. Every other test for
//! this facade compiles in-crate against `pub(crate)` visibility, so none of
//! them can catch a reachability regression: if `replay::mod` or
//! `offline::mod` is tightened back to a bare `mod`, or `pub(crate)` is
//! dropped from `extensions::mod`, this is the one test that fails.

use dynamo_mocker::placement::{KvReplayComposition, MockEngineArgs, provider_spec};

#[test]
fn placement_facade_builds_a_composition_and_names_its_provider() {
    let args = MockEngineArgs::default();
    let _composition = KvReplayComposition::aggregated(args, 1, None, None, None);

    let spec = provider_spec();
    assert_eq!(spec.provider, "dynamo_kv_router");
}
