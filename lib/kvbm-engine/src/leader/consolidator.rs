// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process kvbm-consolidator spawn surface for `InstanceLeader`.
//!
//! [`ConsolidatorParams`] bundles the three pieces the connector needs to
//! supply; [`InstanceLeader::with_consolidator`] builds the [`Consolidator`]
//! and stores it in a [`ConsolidatorGuard`] that shuts the background tasks
//! down when the last `InstanceLeader` clone is dropped.
//!
//! # Why async?
//!
//! `ConsolidatorBuilder::build()` internally calls `zmq_subscriber::spawn`
//! and `egress::zmq_publisher::spawn`, both of which bind/connect ZMQ sockets
//! on an executor thread and return `JoinHandle`s. There is no synchronous
//! path. The "prefer sync" memory-feedback applies when async is optional; it
//! is not optional here.
//!
//! # Drop strategy
//!
//! `InstanceLeader` is `#[derive(Clone)]`, so a plain `Drop` impl on the
//! struct would fire on every clone — useless for shutdown. Instead the
//! `Consolidator` is wrapped in [`ConsolidatorGuard`] whose `Drop` runs
//! `Consolidator::shutdown()` to completion so the background tasks exit
//! and the ZMQ sockets are unbound before `Drop` returns. The guard is held
//! behind `Arc<OnceLock<Arc<ConsolidatorGuard>>>` (same pattern as
//! `session_factory` / `modules` / etc. on `InstanceLeader`), so shutdown
//! fires exactly once when the last clone drops.
//!
//! **Reliability matrix** for Drop. In all arms, the cancel token is
//! signalled synchronously in `Drop::drop` BEFORE entering the arm —
//! ensuring background tasks see cancellation even when we can't await
//! them. The arm then decides how (or whether) to drive shutdown to
//! completion:
//!
//! | Caller context              | Strategy                                       | Reliability |
//! |-----------------------------|------------------------------------------------|-------------|
//! | multi-thread tokio runtime  | `block_in_place + block_on(shutdown)` w/ 5s timeout | Reliable (timeout-bounded). Production + smoke path. |
//! | current_thread tokio runtime| Detached `handle.spawn(shutdown())` + `warn!` | Best-effort wait. Cannot nest a fresh runtime (panics) and cannot `block_on` (deadlocks). Cancel was signalled — tasks self-terminate. Call [`InstanceLeader::shutdown_consolidator`] explicitly for determinism. |
//! | no tokio runtime active     | Cancel-then-drop                               | Reliable cancellation, asynchronous completion. Cancel signal propagates to tasks owned by the original (still-alive) runtime; tasks exit on next poll. No leak. |
//!
//! The 5-second timeout in the multi-thread arm prevents a wedged
//! background task (e.g. a blocked ZMQ recv that ignored its cancel token)
//! from hanging Drop forever. On timeout we abandon the tasks with a `warn!`
//! — but the cancel signal still propagates, so they will exit eventually.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::Result;
use kvbm_consolidator::{Consolidator, ConsolidatorBuilder, ConsolidatorHandle, EventSource};
use kvbm_logical::events::EventsManager;
use tokio_util::sync::CancellationToken;
use tracing::warn;

/// Maximum time `Drop` will block awaiting consolidator shutdown.
/// Bounds the worst-case Drop latency in case a background task is wedged.
const DROP_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

// ============================================================================
// Public types
// ============================================================================

/// Parameters required to spawn an in-process consolidator.
///
/// Passed to [`InstanceLeader::with_consolidator`].  All three fields are
/// mandatory; the method returns an error if called a second time.
pub struct ConsolidatorParams {
    /// ZMQ endpoint the consolidator subscribes to.
    ///
    /// vLLM publishes G1 events here (e.g. `"tcp://127.0.0.1:5557"`).
    /// Pass `None` to disable the ZMQ ingress (KVBM-only mode).
    pub vllm_zmq_endpoint: Option<String>,

    /// ZMQ endpoint the consolidator binds for output.
    ///
    /// The kv-router's `KvEventSubscriber` connects to this endpoint
    /// (e.g. `"tcp://0.0.0.0:57001"`).
    pub egress_endpoint: String,

    /// Source tag for ZMQ-ingress events.
    ///
    /// Used by the consolidator tracker to scope framework hash →
    /// `SequenceHash` translation.  For vLLM callers this is typically
    /// [`EventSource::Vllm`].
    pub engine_source: EventSource,

    /// Live `EventsManager` to subscribe to for G2/G3 KVBM events.
    ///
    /// Call [`EventsManager::subscribe`] on the manager wired into the
    /// `BlockRegistry` that this leader's `BlockManager` uses.  The
    /// consolidator consumes the resulting stream and forwards `Create`/
    /// `Remove` events to the tracker.
    pub events_manager: Arc<EventsManager>,
}

// ============================================================================
// ConsolidatorGuard — RAII wrapper that shuts down on last-clone drop
// ============================================================================

/// RAII guard that owns the running [`Consolidator`].
///
/// When the last clone of the owning `InstanceLeader` is dropped the guard
/// is dropped too, which detaches an async shutdown future so the background
/// tasks are cancelled without blocking the caller's thread.
pub(super) struct ConsolidatorGuard {
    inner: Mutex<Option<Consolidator>>,
    /// Cloned cancellation token. Lets `Drop` signal cancellation from
    /// synchronous contexts (no-runtime arm) without needing access to
    /// the moved-out `Consolidator`.
    cancel: CancellationToken,
}

impl ConsolidatorGuard {
    pub(super) fn new(consolidator: Consolidator) -> Self {
        let cancel = consolidator.cancel_token();
        Self {
            inner: Mutex::new(Some(consolidator)),
            cancel,
        }
    }

    /// Return a `ConsolidatorHandle` for direct event injection.
    ///
    /// Returns `None` if the consolidator has already been shut down.
    pub fn handle(&self) -> Option<ConsolidatorHandle> {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|c| c.handle())
    }

    /// Take the Consolidator out for explicit async shutdown.
    ///
    /// Returns `Some(Consolidator)` exactly once; subsequent calls (and the
    /// `Drop` impl) see `None` and are no-ops.
    pub(super) fn take(&self) -> Option<Consolidator> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner()).take()
    }
}

/// Run `Consolidator::shutdown` to completion using the right strategy for
/// the current execution context.  See the module-level reliability matrix.
///
/// Used only by `ConsolidatorGuard::Drop`.  The explicit
/// `InstanceLeader::shutdown_consolidator` path awaits the future directly
/// (no need for these contortions).
fn drive_shutdown_in_drop(c: Consolidator) {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => match handle.runtime_flavor() {
            tokio::runtime::RuntimeFlavor::MultiThread => {
                // Reliable: park as blocking thread, await join handles with
                // a bounded timeout so a wedged background task can't hang
                // Drop forever.
                let timed_shutdown = tokio::time::timeout(DROP_SHUTDOWN_TIMEOUT, c.shutdown());
                let result = tokio::task::block_in_place(|| handle.block_on(timed_shutdown));
                if result.is_err() {
                    warn!(
                        timeout_s = DROP_SHUTDOWN_TIMEOUT.as_secs(),
                        "ConsolidatorGuard: shutdown timed out in Drop; background tasks abandoned"
                    );
                }
            }
            _ => {
                // current_thread runtime — cannot block_in_place (deadlocks
                // the only worker) and cannot synthesize a fresh runtime
                // (would panic with "Cannot start a runtime from within a
                // runtime").  Spawn detached so the cancel-token at least
                // fires; the join is best-effort and may not complete before
                // runtime teardown.  Callers wanting determinism on
                // current_thread must use `shutdown_consolidator` explicitly.
                handle.spawn(async move {
                    c.shutdown().await;
                });
                warn!(
                    "ConsolidatorGuard dropped on current_thread runtime; shutdown is best-effort. \
                     Call InstanceLeader::shutdown_consolidator().await before drop for deterministic teardown."
                );
            }
        },
        Err(_) => {
            // No tokio runtime active on THIS thread.  Two sub-cases:
            //   a) the consolidator's original runtime is still alive (e.g.
            //      the leader was created on a multi-thread runtime and is
            //      now being dropped from a non-runtime std::thread). The
            //      background tasks are alive and we MUST signal cancel so
            //      they self-terminate. The cancel was already signalled
            //      synchronously in `Drop::drop` before we got here, so the
            //      tasks will exit on their next poll by the original
            //      runtime — no leak.
            //   b) the original runtime is gone too. The tasks were already
            //      aborted when the runtime was dropped; the cancel signal
            //      and our drop here are no-ops.
            //
            // Either way, dropping `c` here releases the `JoinHandle`s and
            // the parent `cancel` token. The child tokens cloned into the
            // tasks have already been told to cancel via `self.cancel`
            // (the same shared inner state), so no detach-leak.
            warn!(
                "ConsolidatorGuard dropped without an active tokio runtime; \
                 cancel signalled synchronously — background tasks will exit \
                 on the next poll by their owning runtime."
            );
            drop(c);
        }
    }
}

impl Drop for ConsolidatorGuard {
    fn drop(&mut self) {
        // Belt: always signal cancel up-front. This works in every arm of
        // `drive_shutdown_in_drop` and is the critical bit for the
        // no-runtime case where `drop(c)` alone would leak background tasks
        // (dropping `CancellationToken` does NOT cancel cloned child tokens,
        // and dropping `JoinHandle`s detaches rather than aborts — so without
        // this synchronous cancel, tasks owned by a still-alive original
        // runtime keep running after the leader is gone).
        //
        // For the multi-thread arm, `Consolidator::shutdown` will signal
        // cancel again — `cancel.cancel()` is idempotent so the double-signal
        // is a no-op. For the current_thread arm it primes the spawned
        // shutdown future. Suspenders: each arm also drives shutdown in the
        // way that's safe for its context.
        self.cancel.cancel();

        let consolidator = self.take();
        let Some(c) = consolidator else { return };
        drive_shutdown_in_drop(c);
    }
}

// ============================================================================
// Shared cell type (re-exported to instance.rs)
// ============================================================================

/// `Arc<OnceLock<…>>` cell for the consolidator guard.
///
/// Mirroring `session_factory` / `modules` / etc. — the cell starts empty and
/// is populated exactly once via [`InstanceLeader::with_consolidator`].
pub(super) type ConsolidatorCell = Arc<OnceLock<Arc<ConsolidatorGuard>>>;

/// Construct a fresh, empty consolidator cell.
pub(super) fn new_cell() -> ConsolidatorCell {
    Arc::new(OnceLock::new())
}

// ============================================================================
// Core spawn logic (called from InstanceLeader::with_consolidator)
// ============================================================================

/// Spawn the consolidator and install it in `cell`.
///
/// Returns `Err` if the cell was already populated (idempotency guard).
pub(super) async fn spawn_into_cell(
    cell: &ConsolidatorCell,
    params: ConsolidatorParams,
) -> Result<()> {
    // Idempotency — mirror the `set_session_factory` guard.
    if cell.get().is_some() {
        anyhow::bail!("consolidator already started");
    }

    // Subscribe to the KVBM event stream.
    let kvbm_stream = params.events_manager.subscribe();

    // Build the consolidator.
    let mut builder = ConsolidatorBuilder::new(params.egress_endpoint, params.engine_source)
        .kvbm_events(kvbm_stream);

    if let Some(zmq_endpoint) = params.vllm_zmq_endpoint {
        builder = builder.zmq_in(zmq_endpoint);
    }

    let consolidator = builder.build().await?;

    let guard = Arc::new(ConsolidatorGuard::new(consolidator));

    // `OnceLock::set` returns Err if already set — race with a concurrent
    // call.  We treat that as "already started" and drop our guard.
    if cell.set(guard).is_err() {
        anyhow::bail!("consolidator already started (concurrent call)");
    }

    Ok(())
}
