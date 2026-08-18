// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_core::{
    EngineConfig, ReplayReport, ReplaySpec, Replayer, TimingModel, TimingModelConfig,
};

fn assert_public<T: ?Sized>() {}

#[test]
fn public_facade_exposes_stable_engine_and_replay_contracts() {
    assert_public::<EngineConfig>();
    assert_public::<ReplaySpec>();
    assert_public::<ReplayReport>();
    assert_public::<Replayer>();
    assert_public::<dyn TimingModel>();
    assert_public::<TimingModelConfig>();
}
