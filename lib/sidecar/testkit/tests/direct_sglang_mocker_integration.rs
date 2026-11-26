// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A frontend that dials mock SGLang gRPC servers directly, with no sidecar:
//! the SGLang sidecar's engine runs inside the frontend, and each engine
//! reports its own role, model and KV publishers.

use std::time::Duration;

use dynamo_backend_common::DisaggregationMode;
use dynamo_backend_common::direct::{DirectEngineOptions, DirectEngineSpec, serve_direct_engines};
use dynamo_llm::worker_type::WorkerType;
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
/// from the engine itself. It is returned once every engine has served a
/// request through it: the host registers each engine on its own, and the
/// frontend, ready at its first model, admits each engine and the prefill role
/// on its own too.
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
    let frontend = Frontend::start_with_router_mode(env, mode).await;
    bounded("every direct engine serves", async {
        while engines.iter().any(|engine| engine.served() == 0) {
            // A request the frontend cannot route yet fails; the next one is sent.
            let _ = frontend.complete(&env.model, "hello").send().await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    frontend
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

    let (a_before, b_before) = (a.served(), b.served());
    for _ in 0..4 {
        assert_eq!(
            complete(&frontend, &env, "hello").await,
            reqwest::StatusCode::OK
        );
    }
    let (a_served, b_served) = (a.served() - a_before, b.served() - b_before);
    assert_eq!(a_served + b_served, 4);
    assert!(a_served > 0 && b_served > 0);
    frontend.shutdown().await;
}

#[tokio::test]
async fn kv_routing_follows_the_engines_own_kv_events() {
    let env = Environment::process_local().await;
    let a = Engine::start(&env, DisaggregationMode::Aggregated).await;
    let b = Engine::start(&env, DisaggregationMode::Aggregated).await;
    let frontend = direct_frontend(&env, &[&a, &b], RouterMode::KV).await;

    // A request caches its prompt on one engine, which publishes the blocks on
    // its own ZMQ socket; the frontend subscribes to it directly. Once the
    // router has indexed them, every repeat of the prompt lands on that
    // engine. Until then a repeat lands on either engine, so an attempt that
    // repeats too early starts over with a prompt neither engine has cached,
    // after a longer wait.
    const REPEATS: usize = 20;
    bounded("KV events route a prompt to its cache", async {
        for attempt in 0.. {
            let prompt = prompt(attempt);
            let (a_before, b_before) = (a.served(), b.served());
            assert_eq!(
                complete(&frontend, &env, &prompt).await,
                reqwest::StatusCode::OK
            );
            let (cached, other, other_before) = if a.served() > a_before {
                (&a, &b, b_before)
            } else {
                (&b, &a, a_before)
            };
            let cached_before = cached.served();
            tokio::time::sleep(Duration::from_millis(5) * 2_u32.pow(attempt)).await;
            while other.served() == other_before && cached.served() < cached_before + REPEATS {
                assert_eq!(
                    complete(&frontend, &env, &prompt).await,
                    reqwest::StatusCode::OK
                );
            }
            if other.served() == other_before {
                return;
            }
        }
    })
    .await;
    frontend.shutdown().await;
}

/// A prompt of eight KV blocks, none of which another attempt's prompt has:
/// the attempt number comes first, in binary, spelled in the two words the
/// test tokenizer knows.
fn prompt(attempt: u32) -> String {
    let words: Vec<_> = (0..32)
        .map(|bit| match attempt >> bit & 1 {
            0 => "hello",
            _ => "world",
        })
        .collect();
    words.join(" ")
}

#[tokio::test]
async fn disaggregated_requests_use_the_engines_that_report_prefill_and_decode() {
    let env = Environment::process_local().await;
    let prefill = Engine::start(&env, DisaggregationMode::Prefill).await;
    let decode = Engine::start(&env, DisaggregationMode::Decode).await;
    // Both are listed the same way: each engine says which role it serves.
    let frontend = direct_frontend(&env, &[&prefill, &decode], RouterMode::RoundRobin).await;

    // The mock reports only loopback for its host, so the prefill engine's
    // bootstrap host is the host it is dialed at. A sidecar would advertise its
    // own address.
    let cards = env.cards().await;
    let endpoint = cards
        .iter()
        .find(|card| card.worker_type == Some(WorkerType::Prefill))
        .and_then(|card| card.runtime_config.disaggregated_endpoint.as_ref())
        .expect("the prefill engine publishes its bootstrap endpoint");
    assert_eq!(endpoint.bootstrap_host.as_deref(), Some("127.0.0.1"));

    let (prefill_before, decode_before) = (prefill.served(), decode.served());
    for _ in 0..2 {
        assert_eq!(
            complete(&frontend, &env, "hello").await,
            reqwest::StatusCode::OK
        );
    }
    assert_eq!(
        (
            prefill.served() - prefill_before,
            decode.served() - decode_before
        ),
        (2, 2)
    );
    frontend.shutdown().await;
}
