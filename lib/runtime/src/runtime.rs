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
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::{signal, sync::Mutex, task::JoinHandle};

pub use tokio_util::sync::CancellationToken;

const DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_SECS: u64 = 15 * 60;

/// Slack added on top of [`graceful_shutdown_timeout`] to bound how long the teardown
/// thread keeps an owned Tokio runtime alive waiting for the shutdown coordinator.
/// A coordinator that never reaches phase 3 then leaks one parked thread instead of
/// pinning the executor and all of its worker threads for the life of the process.
const TEARDOWN_WAIT_MARGIN: Duration = Duration::from_secs(30);

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

/// Shutdown state shared by a [`Runtime`] and every clone of it.
#[derive(Debug)]
struct ShutdownState {
    /// Latched by the first [`Runtime::shutdown`] call so that later calls cannot start a
    /// second coordinator task or a second teardown thread.
    initiated: AtomicBool,

    /// Cancelled by the coordinator once phase 3 has cancelled the primary token. It is
    /// awaitable from async code and also observable from a plain thread.
    complete: CancellationToken,
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
    shutdown_state: Arc<ShutdownState>,
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
            shutdown_state: Arc::new(ShutdownState {
                initiated: AtomicBool::new(false),
                complete: CancellationToken::new(),
            }),
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
    pub(crate) fn graceful_shutdown_tracker(&self) -> Arc<GracefulShutdownTracker> {
        self.graceful_shutdown_tracker.clone()
    }

    /// Get access to the compute pool for CPU-intensive operations
    ///
    /// Returns None if the compute pool was not initialized (e.g., due to configuration error)
    pub fn compute_pool(&self) -> Option<&Arc<crate::compute::ComputePool>> {
        self.compute_pool.as_ref()
    }

    /// A [`CancellationToken`] that is cancelled once the shutdown phases started by
    /// [`Runtime::shutdown`] have completed, that is once phase 3 has cancelled the
    /// primary token. It stays un-cancelled if `shutdown` was never called.
    pub fn shutdown_complete_token(&self) -> CancellationToken {
        self.shutdown_state.complete.clone()
    }

    /// Resolves once the shutdown phases started by [`Runtime::shutdown`] have completed.
    /// Never resolves if `shutdown` was never called.
    pub async fn shutdown_complete(&self) {
        self.shutdown_state.complete.cancelled().await
    }

    /// Shuts down the [`Runtime`] instance.
    ///
    /// Phase 1 runs on the calling thread, so the endpoint shutdown token is always
    /// cancelled by the time this returns. Phase 2 (waiting for registered graceful
    /// endpoints, bounded by `DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS`) and phase 3
    /// (cancelling the primary token) run on the primary runtime;
    /// [`Runtime::shutdown_complete`] resolves once phase 3 has run.
    ///
    /// When this [`Runtime`] owns its Tokio runtime, dropping the last [`Runtime`] handle
    /// no longer destroys that runtime before the phases reach that point — without this,
    /// dropping the last handle from an async context calls Tokio's `shutdown_background`
    /// and discards the queued coordinator.
    ///
    /// That keep-alive guarantee holds against a *drop*, which is what this fixes; it is
    /// not a guarantee against process exit. The thread that holds the owned runtime open
    /// is detached, so a `main` that returns immediately after calling this still
    /// terminates with the phases in flight. A caller that has to wait for them should
    /// await [`Runtime::shutdown_complete`] or observe
    /// [`Runtime::shutdown_complete_token`]. Externally owned runtimes are never torn
    /// down here.
    ///
    /// Calling this more than once is a no-op after the first call.
    pub fn shutdown(&self) {
        // Phase 1 runs here rather than in the coordinator task so that it cannot be lost
        // along with that task, and ahead of the `initiated` latch so that the
        // postcondition also holds for a second caller that races the first one between
        // the latch and this line. `CancellationToken::cancel` is idempotent, so the
        // repeat costs a redundant call and nothing else.
        self.endpoint_shutdown_token.cancel();

        if self.shutdown_state.initiated.swap(true, Ordering::SeqCst) {
            tracing::debug!("Runtime shutdown already initiated; ignoring repeat request");
            return;
        }

        tracing::info!("Runtime shutdown initiated");
        tracing::info!("Phase 1: Cancelled endpoint shutdown token");

        let tracker = self.graceful_shutdown_tracker.clone();
        let main_token = self.cancellation_token.clone();
        let complete = self.shutdown_state.complete.clone();

        let owns_executor = matches!(self.primary, RuntimeType::Shared(_))
            || matches!(self.secondary, RuntimeType::Shared(_));

        // Nothing is ever sent on this channel. The teardown thread blocks on the receiver
        // so that dropping the coordinator task — whether it finished or was discarded —
        // wakes it immediately instead of leaving it on the timeout.
        let (coordinator_alive, coordinator_done) = if owns_executor {
            let (tx, rx) = std::sync::mpsc::channel::<()>();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        // Use the runtime handle to spawn the task
        let handle = self.primary();
        handle.spawn(async move {
            let _coordinator_alive = coordinator_alive;

            // Phase 2: Wait for all graceful endpoints to complete
            tracing::info!("Phase 2: Waiting for graceful endpoints to complete");

            let count = tracker.get_count();
            tracing::info!("Active graceful endpoints: {count}");

            if count != 0 {
                let timeout = graceful_shutdown_timeout();
                if tokio::time::timeout(timeout, tracker.wait_for_completion())
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
            }

            // Phase 3: Now connections will be disconnected to backend services (e.g. NATS/ETCD) by cancelling the main token
            tracing::info!("Phase 3: Connections to backend services will now be disconnected");
            main_token.cancel();
            complete.cancel();
        });

        if let Some(coordinator_done) = coordinator_done {
            self.spawn_owned_teardown(coordinator_done);
        }
    }

    /// Hand clones of the owned Tokio runtimes to a thread that outlives the shutdown
    /// phases and then drops them where blocking is legal.
    ///
    /// Only [`Runtime::shutdown`] may call this. An owned runtime that is dropped without
    /// a `shutdown` call — `transports::etcd` and `storage::kv::etcd` both build one,
    /// `block_on` it and drop it — must keep the plain drop behaviour.
    fn spawn_owned_teardown(&self, coordinator_done: std::sync::mpsc::Receiver<()>) {
        // Holding these clones is what keeps `Arc::get_mut` in `Drop for RuntimeType` from
        // succeeding on the caller's thread, so a drop from an async context short-circuits
        // instead of calling `shutdown_background` on the executor the coordinator is
        // queued on.
        let primary = self.primary.clone();
        let secondary = self.secondary.clone();
        let wait_bound = graceful_shutdown_timeout() + TEARDOWN_WAIT_MARGIN;

        let spawned = std::thread::Builder::new()
            .name("dyn-rt-teardown".to_string())
            .spawn(move || {
                if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
                    coordinator_done.recv_timeout(wait_bound)
                {
                    tracing::error!(
                        wait_secs = wait_bound.as_secs(),
                        "Shutdown coordinator did not finish in time; tearing down the owned runtime anyway"
                    );
                }

                // This thread is not inside a Tokio context, so these drops take the
                // blocking branch of `Drop for RuntimeType`.
                drop(primary);
                drop(secondary);
            });

        if let Err(err) = spawned {
            tracing::error!(
                %err,
                "Failed to spawn the runtime teardown thread; the owned runtime keeps the \
                 pre-existing drop behaviour and its shutdown phases may not complete"
            );
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
                    // This is the branch the teardown thread spawned by `Runtime::shutdown`
                    // takes, so it is the normal end of life for an owned runtime that was
                    // shut down: a real blocking drop after the shutdown phases have run.
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

    /// Occupy the sole worker thread of the runtime behind `handle`, so that anything
    /// queued on it afterwards provably cannot be polled. Returns the sender that
    /// releases the worker again.
    ///
    /// Without this the reproduction is a coin flip: `RuntimeConfig::single_threaded`
    /// builds a multi-thread runtime with one worker, so that worker may or may not get
    /// to the coordinator before the runtime is torn down.
    fn occupy_sole_worker(handle: &tokio::runtime::Handle) -> std::sync::mpsc::Sender<()> {
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();

        handle.spawn(async move {
            let _ = started_tx.send(());
            let _ = release_rx.recv();
        });
        started_rx.recv().expect("worker occupying task never ran");

        release_tx
    }

    /// Call `shutdown()` on a runtime that owns its executor and immediately drop the last
    /// handle from an async context, then check that phase 1 already happened and that the
    /// coordinator still reaches phase 3.
    async fn assert_owned_shutdown_survives_immediate_drop(runtime: Runtime) {
        let tracker = runtime.graceful_shutdown_tracker();
        let guard = tracker.register_task();
        let main_token = runtime.primary_token();
        let endpoint_token = runtime.child_token();
        let shutdown_complete = runtime.shutdown_complete_token();

        let release_worker = occupy_sole_worker(&runtime.primary());

        runtime.shutdown();

        // No yield before these: the coordinator task cannot have run, so phase 1 must have
        // happened on this thread.
        assert!(
            endpoint_token.is_cancelled(),
            "phase 1 must complete before shutdown() returns"
        );
        assert!(!main_token.is_cancelled());
        assert_eq!(tracker.get_count(), 1);

        // Dropping the last owner from an async context is what used to call
        // `shutdown_background` on the executor the coordinator is queued on.
        drop(runtime);

        // Release the graceful task before the worker, so phase 2 observes a zero count and
        // returns without waiting. That keeps the test off both the wall clock and a
        // `Notify` wakeup, either of which would make it timing-dependent.
        drop(guard);
        let _ = release_worker.send(());

        tokio::time::timeout(Duration::from_secs(30), shutdown_complete.cancelled())
            .await
            .expect("shutdown coordinator never reached phase 3");
        assert!(main_token.is_cancelled());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_threaded_runtime_completes_shutdown_after_immediate_drop() {
        assert_owned_shutdown_survives_immediate_drop(Runtime::single_threaded().unwrap()).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn from_settings_runtime_completes_shutdown_after_immediate_drop() {
        temp_env::async_with_vars(
            [(env_runtime::DYN_RUNTIME_NUM_WORKER_THREADS, Some("1"))],
            async {
                assert_owned_shutdown_survives_immediate_drop(Runtime::from_settings().unwrap())
                    .await;
            },
        )
        .await;
    }

    /// Regression coverage for the executor-lifetime half of the fix, i.e. for
    /// `owns_executor`, the `coordinator_alive`/`coordinator_done` channel, and
    /// [`Runtime::spawn_owned_teardown`].
    ///
    /// The two tests above only observe phase 1, which is why this one exists: it looks at
    /// the owned executor *while the shutdown phases are still in flight* rather than
    /// after they have finished. The graceful guard is what creates that window — phase 2
    /// parks on it, so the coordinator cannot reach phase 3 and the teardown thread cannot
    /// let go of the executor while the probe runs.
    ///
    /// Without `spawn_owned_teardown`, `Arc::get_mut` in `Drop for RuntimeType` succeeds on
    /// the last handle, the drop happens in an async context, and `shutdown_background()`
    /// destroys that executor right there — so the probe task is never polled and this test
    /// is red. Observing the executor only *after* completion cannot tell the two apart:
    /// the executor is dead by then either way, and under the bug it died sooner.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn owned_executor_outlives_the_drop_while_phases_are_in_flight() {
        const PROBE: &str = "polled by the owned executor";

        let runtime = Runtime::single_threaded().unwrap();
        let tracker = runtime.graceful_shutdown_tracker();

        // Held across the drop and the probe below, so phase 2 has something to wait for.
        let guard = tracker.register_task();

        let main_token = runtime.primary_token();
        let shutdown_complete = runtime.shutdown_complete_token();
        // A `Handle` does not keep its runtime alive, so spawning on this after the drop
        // observes the executor's real lifetime rather than extending it.
        let owned_executor = runtime.primary();

        runtime.shutdown();

        // The last owner, dropped from an async context with no intervening yield: the
        // exact motion from the issue.
        drop(runtime);

        assert!(
            !main_token.is_cancelled(),
            "phase 3 must not have run yet, or the window this test needs does not exist"
        );

        // The load-bearing observation. Queue work on the owned executor *now*, with the
        // phases still in flight, and require that the executor is alive enough to run it.
        let probe = owned_executor.spawn(async { PROBE });
        match tokio::time::timeout(Duration::from_secs(10), probe).await {
            Ok(Ok(value)) => assert_eq!(value, PROBE),
            Ok(Err(join_error)) => panic!(
                "the owned executor was torn down while the shutdown phases were still in \
                 flight: {join_error}"
            ),
            Err(_) => panic!(
                "the owned executor never polled work queued after the last handle was \
                 dropped, so it is no longer running the shutdown phases"
            ),
        }

        // Release phase 2 and confirm the coordinator did survive to phase 3.
        drop(guard);
        tokio::time::timeout(Duration::from_secs(30), shutdown_complete.cancelled())
            .await
            .expect("shutdown coordinator never reached phase 3");
        assert!(main_token.is_cancelled());
    }
}
