// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use dynamo_backend_common::{BackendError, DisaggregationMode, LLMEngine, PreprocessedRequest};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs};
use dynamo_mocker::scheduler::MockerMetrics;
use dynamo_sidecar_testkit::control::{Controller, Protocol, RequestHandle};
use dynamo_sidecar_testkit::fixtures::Outputs;

pub mod sglang;
pub mod vllm;

pub struct FixtureConfig {
    pub model: String,
    pub connections: usize,
    pub speedup_ratio: f64,
    pub disaggregation_mode: DisaggregationMode,
}

impl Default for FixtureConfig {
    fn default() -> Self {
        Self {
            model: "mocker-model".into(),
            connections: 1,
            speedup_ratio: 0.0,
            disaggregation_mode: DisaggregationMode::Aggregated,
        }
    }
}

pub enum GenerateOpening {
    WaitsForHeaders,
    OnStreamPoll,
}

pub trait SidecarFixture {
    type Engine: LLMEngine;
    type Protocol: Protocol;
    const GENERATE_OPENING: GenerateOpening;

    async fn start(control: Controller<Self::Protocol>, config: FixtureConfig) -> Self;
    async fn engine(&self) -> Self::Engine;
    fn eof_error() -> BackendError;
    fn native_model(request: &<Self::Protocol as Protocol>::Request) -> Option<&str>;
    fn active_request_count(&self) -> usize;
    async fn scheduler_idle(&self);
    async fn shutdown(&mut self);
}

pub trait WireFixture: SidecarFixture {
    fn assert_stream(
        handle: &RequestHandle<Self::Protocol>,
        request: &PreprocessedRequest,
        outputs: &Outputs,
    );
    async fn scheduler_active(&self);
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

pub trait ProcessFixture: WireFixture {
    fn endpoint(&self) -> String;
    fn command() -> Command;
    fn configure_request(request: &mut PreprocessedRequest);
    fn assert_registration(card: &ModelDeploymentCard);
}

pub fn sidecar_command(binary_name: &str, override_env: &str) -> Command {
    let binary = std::env::var_os(override_env)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join(binary_name)
        });
    assert!(
        binary.is_file(),
        "build {binary_name} first or set {override_env}: {}",
        binary.display()
    );
    Command::new(binary)
}

async fn wait_scheduler_idle(
    mut metrics: tokio::sync::watch::Receiver<MockerMetrics>,
    active_requests: impl Fn() -> usize,
) {
    let result = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let snapshot = metrics.borrow_and_update().clone();
            if active_requests() == 0
                && snapshot.running_requests == 0
                && snapshot.waiting_requests == 0
            {
                return;
            }
            tokio::select! {
                result = metrics.changed() => result.unwrap(),
                _ = tokio::time::sleep(Duration::from_millis(5)) => {},
            }
        }
    })
    .await;
    if result.is_err() {
        let snapshot = metrics.borrow();
        panic!(
            "Mocker did not drain: routes={}, running={}, waiting={}",
            active_requests(),
            snapshot.running_requests,
            snapshot.waiting_requests
        );
    }
}
