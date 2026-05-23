// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reliability test for `ConsolidatorGuard::Drop` and the explicit
//! `InstanceLeader::shutdown_consolidator()` path.
//!
//! Codex stop-time review flag (2026-05-19): "consolidator shutdown is not
//! reliable". Before the fix, `Drop` spawned `Consolidator::shutdown()`
//! detached via `Handle::spawn(...)` and returned immediately. On runtime
//! teardown the spawned shutdown future was cancelled mid-flight, leaving
//! ZMQ sockets bound and background tasks dangling.
//!
//! After the fix `Drop` uses `block_in_place + block_on` on multi-thread
//! runtimes (synthesized current_thread fallback otherwise) so the join
//! handles are actually awaited before `Drop` returns.
//!
//! These tests validate the contract by binding the consolidator's egress
//! port, dropping the leader (or calling shutdown_consolidator), and
//! immediately re-binding the same port. If shutdown returned before the
//! socket was unbound, the rebind would fail with "Address already in use".

#![cfg(feature = "testing")]

use std::sync::Arc;
use std::time::Duration;

use kvbm_engine::G2;
use kvbm_engine::leader::{ConsolidatorParams, EventSource, InstanceLeader};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_logical::events::EventsManager;

const BLOCK_SIZE: usize = 16;
const BLOCK_COUNT: usize = 32;

fn new_velo_transport() -> Arc<velo::transports::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        velo::transports::tcp::TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

async fn new_velo() -> Arc<velo::Velo> {
    velo::Velo::builder()
        .add_transport(new_velo_transport())
        .build()
        .await
        .unwrap()
}

async fn make_leader(events_manager: Arc<EventsManager>) -> Arc<InstanceLeader> {
    let velo = new_velo().await;
    let registry = TestRegistryBuilder::new()
        .events_manager(events_manager)
        .build();
    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    Arc::new(
        InstanceLeader::builder()
            .messenger(velo.messenger().clone())
            .registry(registry)
            .g2_manager(g2)
            .workers(vec![])
            .build()
            .expect("build leader"),
    )
}

/// Try to bind a TCP listener on `port`. Returns Ok(()) if the bind succeeds
/// (port was free), Err otherwise. Linger close.
fn try_bind(port: u16) -> std::io::Result<()> {
    let listener = std::net::TcpListener::bind(format!("127.0.0.1:{port}"))?;
    drop(listener);
    Ok(())
}

/// Drop must run shutdown to completion: the egress port must be re-bindable
/// immediately after Drop returns.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn drop_releases_egress_port() {
    let egress_port = portpicker::pick_unused_port().expect("no free port");
    let egress_endpoint = format!("tcp://127.0.0.1:{egress_port}");

    let events_manager = Arc::new(EventsManager::builder().build());
    let leader = make_leader(events_manager.clone()).await;

    leader
        .with_consolidator(ConsolidatorParams {
            vllm_zmq_endpoint: None,
            egress_endpoint: egress_endpoint.clone(),
            engine_source: EventSource::Vllm,
            events_manager: events_manager.clone(),
        })
        .await
        .expect("with_consolidator");

    // Confirm the port is bound while the consolidator is alive.
    assert!(
        try_bind(egress_port).is_err(),
        "egress port {egress_port} should be bound while consolidator is alive"
    );

    // Drop the last InstanceLeader clone — ConsolidatorGuard::Drop must run
    // shutdown to completion. We give it a bounded budget to detect a hang.
    tokio::task::spawn_blocking(move || drop(leader))
        .await
        .expect("drop task must not panic");

    // Port should be re-bindable immediately. Allow a brief grace window for
    // TIME_WAIT to settle on platforms that need it, but no more.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        match try_bind(egress_port) {
            Ok(()) => break,
            Err(e) => {
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "egress port {egress_port} still bound 2s after Drop: {e} — \
                     shutdown was not awaited"
                );
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
}

/// Drop from a current_thread runtime must NOT panic and must NOT hang.
///
/// Regression test for the Codex stop-time review flag "current_thread Drop
/// shutdown can panic or hang". The pre-fix `drive_shutdown` on current_thread
/// tried to build a fresh tokio runtime via `Builder::new_current_thread()`
/// while already inside one — which panics with "Cannot start a runtime from
/// within a runtime".  The fix detaches the shutdown future via
/// `handle.spawn()` and emits a `warn!`.  Reliability on current_thread is
/// best-effort by construction (call `shutdown_consolidator` explicitly for
/// determinism), but Drop itself must complete cleanly.
#[test]
fn drop_on_current_thread_does_not_panic_or_hang() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build current_thread runtime");

    rt.block_on(async {
        let egress_port = portpicker::pick_unused_port().expect("no free port");
        let egress_endpoint = format!("tcp://127.0.0.1:{egress_port}");

        let events_manager = Arc::new(EventsManager::builder().build());
        let leader = make_leader(events_manager.clone()).await;

        leader
            .with_consolidator(ConsolidatorParams {
                vllm_zmq_endpoint: None,
                egress_endpoint: egress_endpoint.clone(),
                engine_source: EventSource::Vllm,
                events_manager: events_manager.clone(),
            })
            .await
            .expect("with_consolidator");

        // Drop must be near-instant: current_thread Drop spawns detached
        // shutdown. Far shorter than DROP_SHUTDOWN_TIMEOUT (5s) — that
        // budget only bounds the multi_thread / no-runtime arms.
        let drop_started = std::time::Instant::now();
        drop(leader);
        let drop_elapsed = drop_started.elapsed();
        assert!(
            drop_elapsed < Duration::from_secs(2),
            "Drop on current_thread should be near-instant (detached spawn); \
             took {drop_elapsed:?}"
        );

        // The detached shutdown is best-effort by design on current_thread;
        // we do NOT assert the port is freed.  See the reliability matrix
        // in lib/kvbm-engine/src/leader/consolidator.rs and the explicit
        // shutdown_consolidator() path for deterministic teardown.
    });
}

/// Drop from a non-runtime std::thread (while the original multi-thread
/// runtime is still alive) must signal cancel so the background tasks
/// self-terminate.  No leak.
///
/// Regression test for the Codex stop-time review flag "no-runtime Drop
/// can leak live consolidator tasks". The pre-fix `Err(_)` arm just did
/// `drop(c)` — but dropping a `CancellationToken` does NOT cancel cloned
/// child tokens that the spawned tasks already hold, and dropping
/// `JoinHandle`s detaches rather than aborts. Result: tasks keep running
/// (alive runtime keeps polling them), and the egress ZMQ port stays
/// bound.
///
/// The fix: `ConsolidatorGuard::Drop` calls `self.cancel.cancel()`
/// synchronously up-front, then enters the no-runtime arm. Tasks see the
/// cancel signal on their next poll by the (still-alive) runtime and
/// exit.
#[test]
fn drop_on_non_runtime_thread_signals_cancel() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build multi_thread runtime");

    let (leader, egress_port) = rt.block_on(async {
        let egress_port = portpicker::pick_unused_port().expect("no free port");
        let egress_endpoint = format!("tcp://127.0.0.1:{egress_port}");

        let events_manager = Arc::new(EventsManager::builder().build());
        let leader = make_leader(events_manager.clone()).await;

        leader
            .with_consolidator(ConsolidatorParams {
                vllm_zmq_endpoint: None,
                egress_endpoint: egress_endpoint.clone(),
                engine_source: EventSource::Vllm,
                events_manager: events_manager.clone(),
            })
            .await
            .expect("with_consolidator");

        (leader, egress_port)
    });

    // Confirm the port is bound while the consolidator is alive and the
    // runtime is still up.
    assert!(
        try_bind(egress_port).is_err(),
        "egress port {egress_port} should be bound while consolidator is alive"
    );

    // Drop the leader on a std::thread that has NO current tokio runtime.
    // The Drop impl's no-runtime arm fires; its synchronous cancel signal
    // is what saves us from a task leak.
    std::thread::spawn(move || {
        drop(leader);
    })
    .join()
    .expect("drop thread must not panic");

    // The cancel was signalled. The runtime is still alive and will poll
    // the tasks, which see cancel=true and exit. The egress port should
    // become free shortly — we give it up to 3 s.
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    loop {
        match try_bind(egress_port) {
            Ok(()) => break,
            Err(err) => {
                if std::time::Instant::now() >= deadline {
                    panic!(
                        "egress port {egress_port} still bound 3 s after \
                         non-runtime Drop: {err}. Cancel signal did not \
                         reach the background tasks — they leaked."
                    );
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    // Now we can shut the runtime down cleanly.
    drop(rt);
}

/// Explicit shutdown must release the egress port deterministically before
/// returning.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_consolidator_releases_egress_port() {
    let egress_port = portpicker::pick_unused_port().expect("no free port");
    let egress_endpoint = format!("tcp://127.0.0.1:{egress_port}");

    let events_manager = Arc::new(EventsManager::builder().build());
    let leader = make_leader(events_manager.clone()).await;

    leader
        .with_consolidator(ConsolidatorParams {
            vllm_zmq_endpoint: None,
            egress_endpoint: egress_endpoint.clone(),
            engine_source: EventSource::Vllm,
            events_manager: events_manager.clone(),
        })
        .await
        .expect("with_consolidator");

    let was_running = leader.shutdown_consolidator().await;
    assert!(
        was_running,
        "first shutdown_consolidator should return true"
    );

    // Port must be free *immediately* — no retry budget here. The whole
    // point of explicit shutdown is that the await is over only when the
    // sockets are actually closed.
    try_bind(egress_port)
        .expect("egress port must be free immediately after shutdown_consolidator");

    // Second call: idempotent — returns false (already shut down).
    let was_running_again = leader.shutdown_consolidator().await;
    assert!(
        !was_running_again,
        "second shutdown_consolidator should return false"
    );

    // Drop must be a no-op (consolidator already taken).
    drop(leader);
}
