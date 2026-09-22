// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{BackendError, LLMEngine};
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs};
use dynamo_sidecar_testkit::control::{Controller, Protocol};

pub mod sglang;
pub mod vllm;

pub struct FixtureConfig {
    pub model: String,
    pub connections: usize,
}

impl Default for FixtureConfig {
    fn default() -> Self {
        Self {
            model: "mocker-model".into(),
            connections: 1,
        }
    }
}

pub trait SidecarFixture {
    type Engine: LLMEngine;
    type Protocol: Protocol;

    async fn start(control: Controller<Self::Protocol>, config: FixtureConfig) -> Self;
    async fn engine(&self) -> Self::Engine;
    fn eof_error() -> BackendError;
    fn native_model(request: &<Self::Protocol as Protocol>::Request) -> Option<&str>;
    fn active_request_count(&self) -> usize;
    async fn shutdown(&mut self);
}

fn fast_engine_args(engine_type: EngineType) -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(engine_type)
        .block_size(4)
        .num_gpu_blocks(4_096)
        .max_num_seqs(Some(64))
        .max_num_batched_tokens(Some(1_024))
        .speedup_ratio(0.0)
        .dp_size(1)
        .build()
        .unwrap()
}
