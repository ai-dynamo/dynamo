// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The [Runtime] module is the interface for [crate::component::Component]
//! to access shared resources. These include thread pool, memory allocators and other shared resources.
//!
//! The [Runtime] holds the primary [`CancellationToken`] which can be used to terminate all attached
//! [`crate::component::Component`].
//!
//! We expect in the future to offer topologically aware thread and memory resources, but for now the
//! set of resources is limited to the thread pool and cancellation token.
//!
//! Notes: We will need to do an evaluation on what is fully public, what is pub(crate) and what is
//! private; however, for now we are exposing most objects as fully public while the API is maturing.

use super::utils::GracefulShutdownTracker;
use crate::{
    compute,
    config::{self, RuntimeConfig},
};

use futures::Future;
use once_cell::sync::OnceCell;
use std::{
    mem::ManuallyDrop,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};
use tokio::{signal, sync::Mutex, task::JoinHandle};

pub use tokio_util::sync::CancellationToken;

const DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_SECS: u64 = 15 * 60;

/// Bound on joining post-cancellation teardown tasks. A lease revoke is one
/// round trip; anything longer means etcd is unreachable, in which case the
/// lease expires on its own TTL and waiting further only delays the exit.
const TEARDOWN_TASK_JOIN_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) fn graceful_shutdown_timeout() -> Duration {
    let timeout_secs = std::env::var(
        config::environment_names::runtime::DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
    )
    .ok()
    .and_then(|s| s.parse::<u64>().ok())
    .unwrap_or(DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_SECS);

    Duration::from_secs(timeout_secs)
}

/// Types of Tokio runtimes that can be used to construct a Dynamo [Runtime].
#[derive(Clone, Debug)]
enum RuntimeType {
    Shared(Arc<ManuallyDrop<tokio::runtime::Runtime>>),
    External(tokio::runtime::Handle),
}

/// Local [Runtime] which provides access to shared resources local to the physical node/machine.
#[derive(Debug, Clone)]
pub struct Runtime {
    id: Arc<String>,
    primary: RuntimeType,
    secondary: RuntimeType,
    cancellation_token: CancellationToken,
    endpoint_shutdown_token: CancellationToken,
    graceful_shutdown_tracker: Arc<GracefulShutdownTracker>,
    /// Bound the in-progress shutdown is draining endpoints with. Set once,
    /// before Phase 1, so the endpoint drain and the Phase 2 wait for it use
    /// the same number — otherwise Phase 3 can tear down the transports while
    /// a drain on a different clock is still running. Per-runtime rather than
    /// process-global so tests (and embedded runtimes) cannot see each other's.
    active_drain_timeout: Arc<std::sync::OnceLock<Duration>>,
    /// Background tasks that do their cleanup *after* the primary token is
    /// cancelled — today the etcd lease keep-alive, whose `lease.revoke()` runs
    /// on cancellation. Phase 3 only cancels; without joining these the process
    /// could exit with the lease still held, which is the stale-registration
    /// symptom this teardown exists to remove.
    teardown_tasks: Arc<std::sync::Mutex<Vec<JoinHandle<()>>>>,
    compute_pool: Option<Arc<compute::ComputePool>>,
    block_in_place_permits: Option<Arc<tokio::sync::Semaphore>>,
}

impl Runtime {
    fn new(runtime: RuntimeType, secondary: Option<RuntimeType>) -> anyhow::Result<Runtime> {
        // Initialise NVTX toggle once from environment (no-op when feature is off)
        crate::nvtx::init();

        // worker id
        let id = Arc::new(uuid::Uuid::new_v4().to_string());

        // create a cancellation token
        let cancellation_token = CancellationToken::new();

        // create endpoint shutdown token as a child of the main token
        let endpoint_shutdown_token = cancellation_token.child_token();

        // secondary runtime for background ectd/nats tasks
        let secondary = match secondary {
            Some(secondary) => secondary,
            None => {
                tracing::debug!("Created secondary runtime with single thread");
                RuntimeType::Shared(Arc::new(ManuallyDrop::new(
                    RuntimeConfig::single_threaded().create_runtime()?,
                )))
            }
        };

        // Initialize compute pool with default config
        // This will be properly configured when created from RuntimeConfig
        let compute_pool = None;
        let block_in_place_permits = None;

        Ok(Runtime {
            id,
            primary: runtime,
            secondary,
            cancellation_token,
            endpoint_shutdown_token,
            graceful_shutdown_tracker: Arc::new(GracefulShutdownTracker::new()),
            active_drain_timeout: Arc::new(std::sync::OnceLock::new()),
            teardown_tasks: Arc::new(std::sync::Mutex::new(Vec::new())),
            compute_pool,
            block_in_place_permits,
        })
    }

    fn new_with_config(
        runtime: RuntimeType,
        secondary: Option<RuntimeType>,
        config: &RuntimeConfig,
    ) -> anyhow::Result<Runtime> {
        let mut rt = Self::new(runtime, secondary)?;

        // Create compute pool from configuration
        let compute_config = crate::compute::ComputeConfig {
            num_threads: config.compute_threads,
            stack_size: config.compute_stack_size,
            thread_prefix: config.compute_thread_prefix.clone(),
            pin_threads: false,
        };

        // Check if compute pool is explicitly disabled
        if config.compute_threads == Some(0) {
            tracing::info!("Compute pool disabled (compute_threads = 0)");
        } else {
            match crate::compute::ComputePool::new(compute_config) {
                Ok(pool) => {
                    rt.compute_pool = Some(Arc::new(pool));
                    tracing::debug!(
                        "Initialized compute pool with {} threads",
                        rt.compute_pool.as_ref().unwrap().num_threads()
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to create compute pool: {}. CPU-intensive operations will use spawn_blocking",
                        e
                    );
                }
            }
        }

        // Initialize block_in_place semaphore based on actual worker threads
        let num_workers = config
            .num_worker_threads
            .unwrap_or_else(|| std::thread::available_parallelism().unwrap().get());
        // Reserve at least one thread for async work
        let permits = num_workers.saturating_sub(1).max(1);
        rt.block_in_place_permits = Some(Arc::new(tokio::sync::Semaphore::new(permits)));
        tracing::debug!(
            "Initialized block_in_place permits: {} (from {} worker threads)",
            permits,
            num_workers
        );

        Ok(rt)
    }

    /// Initialize thread-local compute context on the current thread
    /// This should be called on each Tokio worker thread
    pub fn initialize_thread_local(&self) {
        if let (Some(pool), Some(permits)) = (&self.compute_pool, &self.block_in_place_permits) {
            crate::compute::thread_local::initialize_context(Arc::clone(pool), Arc::clone(permits));
        }
        // Name this worker thread in the Nsight Systems timeline (no-op when nvtx feature is off)
        let thread_name = std::thread::current()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|| format!("tokio-worker-{:?}", std::thread::current().id()));
        crate::nvtx::name_current_thread_impl(&thread_name);
    }

    /// Initialize thread-local compute context on all worker threads using a barrier
    /// This ensures every worker thread has its thread-local context initialized
    pub async fn initialize_all_thread_locals(&self) -> anyhow::Result<()> {
        if let (Some(pool), Some(permits)) = (&self.compute_pool, &self.block_in_place_permits) {
            // First, detect how many worker threads we actually have
            let num_workers = self.detect_worker_thread_count().await;

            if num_workers == 0 {
                return Err(anyhow::anyhow!("No worker threads detected"));
            }

            // Create a barrier that all threads must reach
            let barrier = Arc::new(std::sync::Barrier::new(num_workers));
            let init_pool = Arc::clone(pool);
            let init_permits = Arc::clone(permits);

            // Spawn exactly one blocking task per worker thread
            let mut handles = Vec::new();
            for i in 0..num_workers {
                let barrier_clone = Arc::clone(&barrier);
                let pool_clone = Arc::clone(&init_pool);
                let permits_clone = Arc::clone(&init_permits);

                let handle = tokio::task::spawn_blocking(move || {
                    // Wait at barrier - ensures all threads are participating
                    barrier_clone.wait();

                    // Now initialize thread-local storage
                    crate::compute::thread_local::initialize_context(pool_clone, permits_clone);

                    // Get thread ID for logging
                    let thread_id = std::thread::current().id();
                    tracing::trace!(
                        "Initialized thread-local compute context on thread {:?} (worker {})",
                        thread_id,
                        i
                    );
                });
                handles.push(handle);
            }

            // Wait for all tasks to complete
            for handle in handles {
                handle.await?;
            }

            tracing::info!(
                "Successfully initialized thread-local compute context on {} worker threads",
                num_workers
            );
        } else {
            tracing::debug!("No compute pool configured, skipping thread-local initialization");
        }
        Ok(())
    }

    /// Detect the number of worker threads in the runtime
    async fn detect_worker_thread_count(&self) -> usize {
        use parking_lot::Mutex;
        use std::collections::HashSet;

        let thread_ids = Arc::new(Mutex::new(HashSet::new()));
        let mut handles = Vec::new();

        // Spawn many blocking tasks to ensure we hit all threads
        // We use spawn_blocking because it runs on worker threads
        let num_probes = 100;
        for _ in 0..num_probes {
            let ids = Arc::clone(&thread_ids);
            let handle = tokio::task::spawn_blocking(move || {
                let thread_id = std::thread::current().id();
                ids.lock().insert(thread_id);
            });
            handles.push(handle);
        }

        // Wait for all probes to complete
        for handle in handles {
            let _ = handle.await;
        }

        let count = thread_ids.lock().len();
        tracing::debug!("Detected {count} worker threads in runtime");
        count
    }

    pub fn from_current() -> anyhow::Result<Runtime> {
        Runtime::from_handle(tokio::runtime::Handle::current())
    }

    pub fn from_handle(handle: tokio::runtime::Handle) -> anyhow::Result<Runtime> {
        let primary = RuntimeType::External(handle.clone());
        let secondary = RuntimeType::External(handle);
        Runtime::new(primary, Some(secondary))
    }

    /// Like [`Runtime::from_handle`], but also attaches the compute pool and `block_in_place`
    /// permits that `config` implies, the way [`Runtime::from_settings`] does.
    ///
    /// For when the Tokio runtime is owned elsewhere — a process-wide `OnceCell`, say — so only a
    /// handle can be borrowed, but the [`RuntimeConfig`] behind it is known.
    pub fn from_handle_with_config(
        handle: tokio::runtime::Handle,
        config: &RuntimeConfig,
    ) -> anyhow::Result<Runtime> {
        let primary = RuntimeType::External(handle.clone());
        let secondary = RuntimeType::External(handle);
        Runtime::new_with_config(primary, Some(secondary), config)
    }

    /// Create a [`Runtime`] instance from the settings
    /// See [`config::RuntimeConfig::from_settings`]
    pub fn from_settings() -> anyhow::Result<Runtime> {
        let config = config::RuntimeConfig::from_settings()?;
        let runtime = Arc::new(ManuallyDrop::new(config.create_runtime()?));
        let primary = RuntimeType::Shared(runtime.clone());
        let secondary = RuntimeType::External(runtime.handle().clone());
        Runtime::new_with_config(primary, Some(secondary), &config)
    }

    /// Create a [`Runtime`] with two single-threaded async tokio runtime
    pub fn single_threaded() -> anyhow::Result<Runtime> {
        let config = config::RuntimeConfig::single_threaded();
        let owned = RuntimeType::Shared(Arc::new(ManuallyDrop::new(config.create_runtime()?)));
        Runtime::new(owned, None)
    }

    /// Returns the unique identifier for the [`Runtime`]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns a [`tokio::runtime::Handle`] for the primary/application thread pool
    pub fn primary(&self) -> tokio::runtime::Handle {
        self.primary.handle()
    }

    /// Returns a [`tokio::runtime::Handle`] for the secondary/background thread pool
    pub fn secondary(&self) -> tokio::runtime::Handle {
        self.secondary.handle()
    }

    /// Access the primary [`CancellationToken`] for the [`Runtime`]
    pub fn primary_token(&self) -> CancellationToken {
        self.cancellation_token.clone()
    }

    /// Creates a child [`CancellationToken`] tied to the life-cycle of the [`Runtime`]'s endpoint shutdown token.
    pub fn child_token(&self) -> CancellationToken {
        self.endpoint_shutdown_token.child_token()
    }

    /// Get access to the graceful shutdown tracker
    /// Bound for an endpoint's in-flight drain.
    ///
    /// During shutdown this is the value Phase 2 is waiting with, so the drain
    /// and the wait for it cannot disagree. Outside shutdown — an endpoint
    /// unregistered on its own, e.g. a sleeping worker — it is the runtime
    /// default, which is the behaviour that existed before.
    /// Register a task whose cleanup runs on primary-token cancellation, so
    /// [`shutdown_and_wait`](Self::shutdown_and_wait) joins it after Phase 3.
    pub(crate) fn register_teardown_task(&self, handle: JoinHandle<()>) {
        if let Ok(mut tasks) = self.teardown_tasks.lock() {
            tasks.push(handle);
        }
    }

    pub(crate) fn endpoint_drain_timeout(&self) -> Duration {
        self.active_drain_timeout
            .get()
            .copied()
            .unwrap_or_else(graceful_shutdown_timeout)
    }

    pub(crate) fn graceful_shutdown_tracker(&self) -> Arc<GracefulShutdownTracker> {
        self.graceful_shutdown_tracker.clone()
    }

    /// Get access to the compute pool for CPU-intensive operations
    ///
    /// Returns None if the compute pool was not initialized (e.g., due to configuration error)
    pub fn compute_pool(&self) -> Option<&Arc<crate::compute::ComputePool>> {
        self.compute_pool.as_ref()
    }

    /// Shuts down the [`Runtime`] instance.
    ///
    /// Fire-and-forget: the three-phase sequence is spawned and this returns
    /// immediately. A caller that exits the process straight after (as a
    /// `main` typically does) will terminate before the phases complete, so
    /// the endpoint inflight drain and the etcd lease revoke in Phase 2/3 are
    /// not guaranteed to run. Use [`shutdown_and_wait`](Self::shutdown_and_wait)
    /// when the teardown must actually finish.
    pub fn shutdown(&self) {
        let sequence = self.shutdown_sequence(None);
        self.primary().spawn(sequence);
    }

    /// [`shutdown`](Self::shutdown) that resolves once the three-phase
    /// sequence has finished, so Phase 3 has run and the endpoint inflight
    /// drain is complete before the caller proceeds.
    ///
    /// `drain_timeout` bounds Phase 2 (the wait for outstanding graceful
    /// tasks); `None` uses `DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS`.
    /// Bounding is expressed here rather than by wrapping this call in a
    /// `timeout`, because `tokio::time::timeout` cancels by *dropping* the
    /// future — which would skip Phase 3 and leave the transports connected,
    /// the exact failure this method exists to prevent. Phase 3 always runs.
    ///
    /// Phase 3 cancels the primary token, which *signals* transport teardown.
    /// The tasks whose cleanup is that signal — notably the etcd keep-alive
    /// task and its `lease.revoke()` — are joined afterwards, separately
    /// bounded by [`TEARDOWN_TASK_JOIN_TIMEOUT`], so an unreachable etcd delays
    /// the exit by seconds rather than indefinitely.
    pub async fn shutdown_and_wait(&self, drain_timeout: Option<Duration>) {
        self.shutdown_sequence(drain_timeout).await
    }

    /// The three-phase teardown shared by [`shutdown`](Self::shutdown) and
    /// [`shutdown_and_wait`](Self::shutdown_and_wait). Returns an owned future
    /// so the fire-and-forget path can spawn it.
    fn shutdown_sequence(
        &self,
        drain_timeout: Option<Duration>,
    ) -> impl std::future::Future<Output = ()> + Send + 'static {
        tracing::info!("Runtime shutdown initiated");

        let tracker = self.graceful_shutdown_tracker.clone();
        let main_token = self.cancellation_token.clone();
        let endpoint_token = self.endpoint_shutdown_token.clone();
        let active_drain_timeout = self.active_drain_timeout.clone();
        let teardown_tasks = self.teardown_tasks.clone();

        async move {
            // Resolve the one bound this shutdown will use, *before* Phase 1.
            // The endpoint drain that Phase 2 waits on runs inside the endpoint
            // cleanup task, which reads this — without it the drain used
            // `DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` (900s) while Phase 2
            // waited with the caller's bound (30s for a worker), so Phase 3
            // cancelled the primary token and tore down NATS/etcd with the
            // drain still running. Two clocks that could never agree.
            let timeout = drain_timeout.unwrap_or_else(graceful_shutdown_timeout);
            let _ = active_drain_timeout.set(timeout);

            // Phase 1: Cancel endpoint shutdown token to stop accepting new requests
            tracing::info!("Phase 1: Cancelling endpoint shutdown token");
            endpoint_token.cancel();

            // Phase 2: Wait for all graceful endpoints to complete
            tracing::info!("Phase 2: Waiting for graceful endpoints to complete");

            let count = tracker.get_count();
            tracing::info!("Active graceful endpoints: {count}");

            if count != 0
                && tokio::time::timeout(timeout, tracker.wait_for_completion())
                    .await
                    .is_err()
            {
                let remaining = tracker.get_count();
                tracing::error!(
                    timeout_secs = timeout.as_secs(),
                    remaining_endpoints = remaining,
                    "Graceful endpoint shutdown timed out; proceeding with runtime teardown"
                );
            }

            // Phase 3: Now connections will be disconnected to backend services (e.g. NATS/ETCD) by cancelling the main token
            tracing::info!("Phase 3: Connections to backend services will now be disconnected");
            main_token.cancel();

            // Phase 3 only *signals* teardown. Join the tasks whose cleanup is
            // that signal — the etcd lease keep-alive issues `lease.revoke()`
            // here — or the process can exit with the lease still held and the
            // instance advertised until the TTL expires.
            //
            // Bounded, and the bound is what makes a self-await safe: the
            // lease-loss path calls `Runtime::shutdown` from inside the
            // keep-alive task itself. That call only spawns this sequence, so
            // the task does finish — but a future caller who awaited from such
            // a task would otherwise hang here forever.
            let pending: Vec<JoinHandle<()>> = teardown_tasks
                .lock()
                .map(|mut tasks| std::mem::take(&mut *tasks))
                .unwrap_or_default();
            if !pending.is_empty() {
                tracing::debug!(count = pending.len(), "Joining teardown tasks");
                let joined = futures::future::join_all(pending);
                if tokio::time::timeout(TEARDOWN_TASK_JOIN_TIMEOUT, joined)
                    .await
                    .is_err()
                {
                    tracing::warn!(
                        timeout_secs = TEARDOWN_TASK_JOIN_TIMEOUT.as_secs(),
                        "Timed out joining teardown tasks; a lease may not have been revoked"
                    );
                }
            }
        }
    }
}

impl RuntimeType {
    /// Get [`tokio::runtime::Handle`] to runtime
    pub fn handle(&self) -> tokio::runtime::Handle {
        match self {
            RuntimeType::External(rt) => rt.clone(),
            RuntimeType::Shared(rt) => rt.handle().clone(),
        }
    }
}

/// Handle dropping a tokio runtime from an async context.
///
/// When used from the Python bindings the runtime will be dropped from (I think) Python's asyncio.
/// Tokio does not allow this and will panic. That panic prevents logging from printing it's last
/// messages, which makes knowing what went wrong very difficult.
///
/// This is the panic:
/// > pyo3_runtime.PanicException: Cannot drop a runtime in a context where blocking is not allowed.
/// > This happens when a runtime is dropped from within an asynchronous context.
///
/// Hence we wrap the runtime in a ManuallyDrop and use tokio's alternative shutdown if we detect
/// that we are inside an async runtime.
impl Drop for RuntimeType {
    fn drop(&mut self) {
        match self {
            RuntimeType::External(_) => {}
            RuntimeType::Shared(arc) => {
                let Some(md_runtime) = Arc::get_mut(arc) else {
                    // Only drop if we are the only owner of the shared pointer, meaning
                    // one strong count and no weak count.
                    return;
                };
                if tokio::runtime::Handle::try_current().is_ok() {
                    // We are inside an async runtime.
                    let tokio_runtime = unsafe { ManuallyDrop::take(md_runtime) };
                    tokio_runtime.shutdown_background();
                } else {
                    // We are not inside an async context, dropping the runtime is safe.
                    //
                    // We never reach this case. I'm not sure why, something about the interaction
                    // with pyo3 and Python lifetimes.
                    //
                    // Process is gone so doesn't really matter, but TODO now that we realize it.
                    unsafe { ManuallyDrop::drop(md_runtime) };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::environment_names::runtime as env_runtime;

    #[tokio::test(start_paused = true)]
    async fn shutdown_cancels_main_token_after_graceful_timeout() {
        temp_env::async_with_vars(
            [(
                env_runtime::DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
                Some("5"),
            )],
            async {
                let runtime = Runtime::from_current().unwrap();
                let tracker = runtime.graceful_shutdown_tracker();
                let _guard = tracker.register_task();
                let main_token = runtime.primary_token();
                let endpoint_token = runtime.child_token();

                runtime.shutdown();
                tokio::task::yield_now().await;

                assert!(endpoint_token.is_cancelled());
                assert!(!main_token.is_cancelled());
                assert_eq!(tracker.get_count(), 1);

                tokio::time::advance(Duration::from_secs(4)).await;
                tokio::task::yield_now().await;

                assert!(!main_token.is_cancelled());

                tokio::time::advance(Duration::from_secs(1)).await;
                tokio::task::yield_now().await;

                assert!(main_token.is_cancelled());
                assert_eq!(tracker.get_count(), 1);
            },
        )
        .await;
    }

    /// `shutdown()` is fire-and-forget: it only spawns the sequence, so a
    /// caller that exits straight after can terminate before Phase 3 runs.
    /// Asserted *after* yielding — before a yield the spawned task provably
    /// has not run, so an immediate assert would hold no matter what
    /// `shutdown()` did.
    #[tokio::test(start_paused = true)]
    async fn shutdown_leaves_the_sequence_outstanding_after_yielding() {
        let runtime = Runtime::from_current().unwrap();
        let main_token = runtime.primary_token();
        let endpoint_token = runtime.child_token();

        // No graceful task registered: Phase 2 has nothing to wait for, so
        // the only thing keeping Phase 3 from completing is that the caller
        // never awaited the sequence.
        runtime.shutdown();
        assert!(!main_token.is_cancelled());

        tokio::task::yield_now().await;
        // The spawned sequence did get scheduled...
        assert!(endpoint_token.is_cancelled(), "Phase 1 must have run");
        // ...but nothing tied its completion to the caller. Contrast
        // `shutdown_and_wait`, which resolves only once Phase 3 is done.
        assert!(
            main_token.is_cancelled(),
            "sanity: with no graceful tasks the spawned sequence runs to \
             completion once scheduled — the hazard is that the caller may \
             exit before this point, which shutdown_and_wait fixes"
        );
    }

    /// Phase 3 only *cancels* the primary token; the etcd keep-alive task does
    /// its `lease.revoke()` in response. Returning without joining that task let
    /// the process exit with the lease still held, leaving the instance
    /// advertised until its TTL expired.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_and_wait_joins_post_cancellation_teardown_tasks() {
        let runtime = Runtime::from_current().unwrap();
        let revoked = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Stands in for the lease keep-alive: cleanup runs only once the
        // primary token is cancelled, and takes a moment to finish.
        let token = runtime.primary_token();
        let flag = revoked.clone();
        runtime.register_teardown_task(tokio::spawn(async move {
            token.cancelled().await;
            tokio::time::sleep(Duration::from_millis(300)).await;
            flag.store(true, Ordering::SeqCst);
        }));

        runtime
            .shutdown_and_wait(Some(Duration::from_millis(50)))
            .await;

        assert!(
            revoked.load(Ordering::SeqCst),
            "shutdown_and_wait returned before the post-cancellation cleanup finished"
        );
    }

    /// The bound must be applied *inside* Phase 2, not wrapped around the
    /// call. `tokio::time::timeout` cancels by dropping the future, so a
    /// wrapping timeout would skip Phase 3 and leave the transports up —
    /// reintroducing the bug this method exists to fix.
    #[tokio::test(start_paused = true)]
    async fn shutdown_and_wait_runs_phase_three_even_when_the_drain_times_out() {
        let runtime = Runtime::from_current().unwrap();
        let tracker = runtime.graceful_shutdown_tracker();
        // Never released: Phase 2 will hit its bound.
        let _guard = tracker.register_task();
        let main_token = runtime.primary_token();

        runtime
            .shutdown_and_wait(Some(Duration::from_secs(5)))
            .await;

        assert!(
            main_token.is_cancelled(),
            "Phase 3 must run even though the Phase 2 drain timed out"
        );
        assert_eq!(tracker.get_count(), 1, "the stuck task is still counted");
    }

    /// `shutdown_and_wait()` is the contract `run.rs` depends on: it must not
    /// resolve until Phase 3 has cancelled the main token, so transport
    /// teardown is complete before the process exits.
    #[tokio::test(start_paused = true)]
    async fn shutdown_and_wait_resolves_only_after_phase_three() {
        let runtime = Runtime::from_current().unwrap();
        let tracker = runtime.graceful_shutdown_tracker();
        let guard = tracker.register_task();
        let main_token = runtime.primary_token();
        let endpoint_token = runtime.child_token();

        let waiter = {
            let runtime = runtime.clone();
            tokio::spawn(async move { runtime.shutdown_and_wait(None).await })
        };

        tokio::task::yield_now().await;
        assert!(endpoint_token.is_cancelled(), "Phase 1 must have run");
        assert!(
            !waiter.is_finished(),
            "must still be waiting on the outstanding graceful task"
        );
        assert!(!main_token.is_cancelled());

        // Releasing the last registration lets Phase 2 complete.
        drop(guard);
        waiter.await.unwrap();

        assert!(main_token.is_cancelled(), "Phase 3 must have run");
    }

    /// A stuck graceful task must not hang teardown forever — Phase 2 is
    /// bounded, and `shutdown_and_wait` inherits that bound.
    #[tokio::test(start_paused = true)]
    async fn shutdown_and_wait_is_bounded_by_the_phase_two_timeout() {
        temp_env::async_with_vars(
            [(
                env_runtime::DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
                Some("5"),
            )],
            async {
                let runtime = Runtime::from_current().unwrap();
                let tracker = runtime.graceful_shutdown_tracker();
                // Never released: stands in for an endpoint that never drains.
                let _guard = tracker.register_task();
                let main_token = runtime.primary_token();

                tokio::time::timeout(Duration::from_secs(3600), runtime.shutdown_and_wait(None))
                    .await
                    .expect("shutdown_and_wait must be bounded; it hung past the outer guard");

                assert!(main_token.is_cancelled());
            },
        )
        .await;
    }
}
