// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A frontend that dials mock SGLang gRPC servers directly, with no sidecar:
//! the SGLang sidecar's engine runs inside the frontend, and each engine
//! reports its own role, model and KV publishers.

use std::time::Duration;

use dynamo_backend_common::DisaggregationMode;
use dynamo_backend_common::direct::{DirectEngineOptions, DirectEngineSpec, serve_direct_engines};
use dynamo_runtime::pipeline::RouterMode;
use dynamo_sglang_sidecar::direct::direct_engine_factory;
use dynamo_sidecar_testkit::{bounded, control::Controller};

#[path = "support/process.rs"]
#[allow(dead_code)]
mod process;
#[allow(dead_code)]
mod support;

use process::{Environment, Frontend};
use support::{FixtureConfig, ProcessFixture, SidecarFixture, sglang};

struct Engine {
    control: Controller<sglang::Adapter>,
    fixture: sglang::Fixture,
}

impl Engine {
    async fn start(env: &Environment, mode: DisaggregationMode) -> Self {
        let control = Controller::default();
        let fixture = sglang::Fixture::start(
            control.clone(),
            FixtureConfig {
                model: env.model.clone(),
                disaggregation_mode: mode,
                ..Default::default()
            },
        )
        .await;
        Self { control, fixture }
    }

    fn served(&self) -> usize {
        self.control.opened()
    }
}

/// A frontend that serves `engines` directly, with each engine's role taken
/// from the engine itself.
async fn direct_frontend(env: &Environment, engines: &[&Engine], mode: RouterMode) -> Frontend {
    let options = DirectEngineOptions {
        namespace: env.namespace.clone(),
        ..Default::default()
    };
    let specs = engines
        .iter()
        .map(|engine| DirectEngineSpec {
            url: engine.fixture.endpoint(),
        })
        .collect();
    bounded(
        "direct engine startup",
        serve_direct_engines(&env.runtime, specs, direct_engine_factory(options)),
    )
    .await
    .unwrap();
    Frontend::start_with_router_mode(env, mode).await
}

async fn complete(frontend: &Frontend, env: &Environment, prompt: &str) -> reqwest::StatusCode {
    bounded(
        "direct completion",
        frontend.complete(&env.model, prompt).send(),
    )
    .await
    .unwrap()
    .status()
}

#[tokio::test]
async fn round_robin_reaches_every_engine() {
    let env = Environment::process_local().await;
    let a = Engine::start(&env, DisaggregationMode::Aggregated).await;
    let b = Engine::start(&env, DisaggregationMode::Aggregated).await;
    let frontend = direct_frontend(&env, &[&a, &b], RouterMode::RoundRobin).await;

    for _ in 0..4 {
        assert_eq!(
            complete(&frontend, &env, "hello").await,
            reqwest::StatusCode::OK
        );
    }
    assert_eq!(a.served() + b.served(), 4);
    assert!(a.served() > 0 && b.served() > 0);
    frontend.shutdown().await;
}

#[tokio::test]
async fn kv_routing_follows_the_engines_own_kv_events() {
    let env = Environment::process_local().await;
    let a = Engine::start(&env, DisaggregationMode::Aggregated).await;
    let b = Engine::start(&env, DisaggregationMode::Aggregated).await;
    let frontend = direct_frontend(&env, &[&a, &b], RouterMode::KV).await;
    let prompt = ["hello world"; 16].join(" ");

    // The first request caches the prompt on one engine, which publishes the
    // blocks on its own ZMQ socket; the frontend subscribes to it directly.
    assert_eq!(
        complete(&frontend, &env, &prompt).await,
        reqwest::StatusCode::OK
    );
    let (cached, other) = if a.served() == 1 { (&a, &b) } else { (&b, &a) };
    tokio::time::sleep(Duration::from_secs(1)).await;

    for _ in 0..6 {
        assert_eq!(
            complete(&frontend, &env, &prompt).await,
            reqwest::StatusCode::OK
        );
    }
    assert_eq!((cached.served(), other.served()), (7, 0));
    frontend.shutdown().await;
}

#[tokio::test]
async fn disaggregated_requests_use_the_engines_that_report_prefill_and_decode() {
    let env = Environment::process_local().await;
    let prefill = Engine::start(&env, DisaggregationMode::Prefill).await;
    let decode = Engine::start(&env, DisaggregationMode::Decode).await;
    // Both are listed the same way: each engine says which role it serves.
    let frontend = direct_frontend(&env, &[&prefill, &decode], RouterMode::RoundRobin).await;

    for _ in 0..2 {
        assert_eq!(
            complete(&frontend, &env, "hello").await,
            reqwest::StatusCode::OK
        );
    }
    assert_eq!((prefill.served(), decode.served()), (2, 2));
    frontend.shutdown().await;
}
