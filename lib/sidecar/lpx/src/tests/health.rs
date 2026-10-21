// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The sidecar's readiness following the engine's `grpc.health.v1` status.
//!
//! Readiness here is what the sidecar's runtime-only readiness route reports;
//! the runtime's own probe tests cover the route itself.

use dynamo_runtime::component::Endpoint;
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::{DistributedRuntime, Runtime};
use tonic_health::ServingStatus;
use tonic_health::server::HealthReporter;

use super::*;

fn is_ready(endpoint: &Endpoint) -> bool {
    !endpoint
        .drt()
        .system_health()
        .lock()
        .is_runtime_readiness_withheld()
}

async fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting until {what}"));
}

/// Long enough for many of the sidecar's checks at the tests' retry interval:
/// a sidecar that would grant readiness has done so by then.
async fn several_retry_intervals() {
    tokio::time::sleep(Duration::from_millis(200)).await;
}

/// Started against `server` and following its health from the serving
/// endpoint a Worker hands the engine, on a process-local runtime.
async fn following(server: &FakeServer) -> (LpxSidecarEngine, Endpoint) {
    let engine = started(server, AGG).await;
    let runtime = Runtime::from_current().expect("current runtime");
    let endpoint = DistributedRuntime::new(runtime, DistributedConfig::process_local())
        .await
        .expect("process-local runtime")
        .namespace("lpx-health")
        .expect("namespace")
        .component("backend")
        .expect("component")
        .endpoint("generate");
    engine
        .on_endpoint_ready(endpoint.clone())
        .await
        .expect("follow engine health");
    (engine, endpoint)
}

/// LPX engines serve tonic-health's `grpc.health.v1` and report the whole
/// server `NOT_SERVING` until they admit requests and while they drain.
/// Readiness follows that. A watch the engine ends withdraws readiness until
/// the sidecar re-establishes it. A draining engine keeps its status, so
/// tonic-health never ends the watch itself: the sidecar releases it, or the
/// engine's graceful shutdown, which waits for every open stream, would never
/// finish. Cleanup stops following and gives readiness back to the runtime.
#[tokio::test]
async fn readiness_follows_the_engines_health_until_cleanup() {
    let mut health = HealthReporter::new();
    health
        .set_service_status("", ServingStatus::NotServing)
        .await;
    let server = FakeServer::start(FakeEngine::new(AGG), Some(health.clone())).await;
    let (engine, endpoint) = following(&server).await;
    several_retry_intervals().await;
    assert!(
        !is_ready(&endpoint),
        "readiness waits for the engine to report SERVING"
    );

    health.set_service_status("", ServingStatus::Serving).await;
    wait_until("a serving engine makes the sidecar ready", || {
        is_ready(&endpoint)
    })
    .await;

    // Clearing the status ends the open watch, as a lost connection would,
    // and every later watch fails until the status is back. Restored at once,
    // so the retry backoff stays far below the wait.
    health.clear_service_status("").await;
    wait_until("a lost watch withdraws readiness", || !is_ready(&endpoint)).await;
    health.set_service_status("", ServingStatus::Serving).await;
    wait_until("the re-established watch restores readiness", || {
        is_ready(&endpoint)
    })
    .await;

    health
        .set_service_status("", ServingStatus::NotServing)
        .await;
    wait_until("a draining engine withdraws readiness", || {
        !is_ready(&endpoint)
    })
    .await;
    server.shut_down_gracefully().await;

    engine.cleanup().await.expect("cleanup");
    wait_until("cleanup releases the engine's hold on readiness", || {
        is_ready(&endpoint)
    })
    .await;
}

/// LPX engines serve `grpc.health.v1`. One without it has health the sidecar
/// cannot read: tonic answers the watch `UNIMPLEMENTED`, and readiness stays
/// withdrawn.
#[tokio::test]
async fn an_engine_without_grpc_health_leaves_readiness_withdrawn() {
    let server = FakeServer::start(FakeEngine::new(AGG), None).await;
    let (engine, endpoint) = following(&server).await;
    several_retry_intervals().await;
    assert!(
        !is_ready(&endpoint),
        "UNIMPLEMENTED does not grant readiness"
    );
    engine.cleanup().await.expect("cleanup");
}
