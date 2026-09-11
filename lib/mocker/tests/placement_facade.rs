// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercises `dynamo_mocker::placement` from outside the crate, the way an
//! external co-simulator (aiperf) actually reaches it -- the one place that
//! catches a reachability regression in the pub(crate) chain underneath.

use dynamo_mocker::placement::{
    KvReplayComposition, KvReplayMetadata, KvRouterConfig, KvRouterPlacement, MockEngineArgs,
    MockEngineArgsBuilder, ReplayPrefillLoadEstimator, RouterEventBatch, RouterEventObservation,
    provider_spec,
};

#[test]
fn placement_facade_builds_a_composition_and_names_its_provider() {
    let args = MockEngineArgs::default();
    let _composition = KvReplayComposition::aggregated(args, 1, None, None, None);

    let spec = provider_spec();
    assert_eq!(spec.provider, "dynamo_kv_router");
}

/// `KvRouterPlacement` is the whole point of this facade -- aiperf injects
/// it as an `aisimulate_core::PlacementPolicy`.
#[test]
fn placement_facade_constructs_a_kv_router_placement() {
    let args = MockEngineArgs::default();
    KvRouterPlacement::new(&args, None, None, 1, None)
        .expect("a default MockEngineArgs with one worker must construct a placement");
}

/// Names every remaining facade re-export the tests above don't otherwise
/// touch, so removing any one of them fails this file to compile.
#[allow(dead_code)]
fn _remaining_facade_items_stay_nameable(
    _config: KvRouterConfig,
    _builder: MockEngineArgsBuilder,
    _estimator: Option<ReplayPrefillLoadEstimator>,
    _batch: RouterEventBatch,
    _observation: RouterEventObservation,
    _metadata: KvReplayMetadata,
) {
}
