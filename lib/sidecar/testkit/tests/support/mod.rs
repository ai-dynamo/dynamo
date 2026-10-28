// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use dynamo_backend_common::{BackendError, DisaggregationMode, LLMEngine, PreprocessedRequest};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_mocker::common::protocols::{EngineType, MockerConfig};
use dynamo_mocker::scheduler::MockerMetrics;
use dynamo_sidecar_testkit::control::{Controller, Protocol, RequestHandle};
use dynamo_sidecar_testkit::fixtures::Outputs;

pub mod lpx;
pub mod sglang;
pub mod trtllm;
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

/// How a request refused before any response reaches `generate()`'s caller.
pub enum Rejection {
    /// `generate()` returns the error.
    Returned,
    /// `generate()` returns a stream whose only item is the error.
    FirstStreamItem,
}

pub trait SidecarFixture {
    type Engine: LLMEngine;
    type Protocol: Protocol;
    const GENERATE_OPENING: GenerateOpening;
    /// How the sidecar delivers a refusal it decides deterministically.
    const REJECTION: Rejection = Rejection::Returned;

    async fn start(control: Controller<Self::Protocol>, config: FixtureConfig) -> Self;
    async fn engine(&self) -> Self::Engine;
    fn eof_error() -> BackendError;
    /// A native stream that fails with UNAVAILABLE after its first tokens.
    fn read_error() -> BackendError {
        BackendError::CannotConnect
    }
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

fn fast_engine_args(engine_type: EngineType) -> MockerConfig {
    MockerConfig::from_value(serde_json::json!({
        "dp_size": 1,
        "engine": {
            "backend": engine_type,
            "block_size": 4,
            "num_gpu_blocks": 4_096,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 1_024,
            "speedup_ratio": 0.0
        }
    }))
    .unwrap()
}

pub trait ProcessFixture: WireFixture {
    /// An UNAVAILABLE stream error removes the worker from routing until the router reconciles.
    const QUARANTINES_ON_READ_ERROR: bool = false;

    fn endpoint(&self) -> String;
    fn command(model: &str) -> Command;
    fn configure_request(request: &mut PreprocessedRequest);
    fn assert_registration(card: &ModelDeploymentCard);
}

/// Backends that discover served aliases and gate publication on native health.
pub trait DiscoveryFixture: ProcessFixture {
    fn set_served_model_name(&self, name: &str);
    fn set_health(&self, is_healthy: Option<bool>);
    async fn health_check_received(&self);
    fn assert_unhealthy_startup(logs: &str);
}

/// A prefill fixture paired with the decode fixture `D`.
pub trait HandoffFixture<D: ProcessFixture = Self>: ProcessFixture {
    const HAS_BOOTSTRAP: bool;
    const KV_BLOCK_SIZE: u32;
    fn assert_handoff(
        prefill: &RequestHandle<Self::Protocol>,
        decode: &RequestHandle<D::Protocol>,
        id: &str,
    );
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

async fn wait_scheduler_active(
    fixture: &impl SidecarFixture,
    mut metrics: tokio::sync::watch::Receiver<MockerMetrics>,
) {
    dynamo_sidecar_testkit::bounded("Mocker active scheduler work", async {
        loop {
            let snapshot = metrics.borrow_and_update().clone();
            if snapshot.running_requests + snapshot.waiting_requests > 0 {
                assert!(fixture.active_request_count() > 0);
                return;
            }
            metrics.changed().await.unwrap();
        }
    })
    .await;
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
