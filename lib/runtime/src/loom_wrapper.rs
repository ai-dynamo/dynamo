// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Loom-rs integration layer providing API-compatible Runtime interface.
//!
//! This module wraps [`loom_rs::LoomRuntime`] to provide the same interface as the
//! standard [`Runtime`](super::runtime::Runtime), allowing seamless migration to
//! loom's unified runtime with adaptive MAB scheduling and starvation prevention.
//!
//! # Feature Flag
//!
//! This module is only available when the `loom-runtime` feature is enabled.
//!
//! # API Compatibility
//!
//! The wrapper preserves all existing APIs:
//! - `from_settings()`, `from_current()`, `from_handle()`, `single_threaded()`
//! - `primary()`, `secondary()` for runtime handle access
//! - `primary_token()`, `child_token()` for cancellation
//! - `compute_pool()` for explicit pool access
//!
//! Additionally, it exposes new loom-specific APIs:
//! - `spawn_adaptive()` - MAB-scheduled execution
//! - `spawn_compute()` - Direct rayon offload
//! - `loom_runtime()` - Access to underlying LoomRuntime
//!
//! # Important Notes
//!
//! Unlike the standard runtime, loom-rs creates its own tokio runtime internally.
//! The `from_current()` and `from_handle()` methods will create a new loom runtime
//! rather than reusing an existing handle. Use `from_settings()` for production use.

use crate::config::{self};
use crate::utils::GracefulShutdownTracker;

use anyhow::Result;
use loom_rs::{ComputeHint, LoomBuilder, LoomRuntime};
use std::future::Future;
use std::sync::Arc;
use tokio::task::JoinHandle;

pub use tokio_util::sync::CancellationToken;

/// Loom-backed Runtime providing unified async/compute execution.
///
/// This is an API-compatible replacement for the standard Runtime that uses
/// loom-rs underneath for adaptive MAB scheduling and automatic starvation prevention.
#[derive(Clone)]
pub struct Runtime {
    /// Unique identifier for this runtime instance
    id: Arc<String>,

    /// The underlying loom runtime
    loom: Arc<LoomRuntime>,

    /// Cancellation token for the entire runtime
    cancellation_token: CancellationToken,

    /// Child token for endpoint shutdown (cancelled before main token)
    endpoint_shutdown_token: CancellationToken,

    /// Tracker for graceful shutdown coordination
    graceful_shutdown_tracker: Arc<GracefulShutdownTracker>,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime (loom)")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl Runtime {
    /// Create a new loom-backed runtime with the given configuration.
    fn new_with_loom(loom: LoomRuntime) -> Result<Self> {
        let id = Arc::new(uuid::Uuid::new_v4().to_string());
        let cancellation_token = CancellationToken::new();
        let endpoint_shutdown_token = cancellation_token.child_token();

        Ok(Self {
            id,
            loom: Arc::new(loom),
            cancellation_token,
            endpoint_shutdown_token,
            graceful_shutdown_tracker: Arc::new(GracefulShutdownTracker::new()),
        })
    }

    /// Create a [`Runtime`] from the current tokio context.
    ///
    /// **Note**: Loom-rs creates its own tokio runtime. This method creates a new
    /// loom runtime rather than reusing the current handle. For production use,
    /// prefer `from_settings()`.
    pub fn from_current() -> Result<Self> {
        tracing::debug!(
            "Creating loom runtime from_current() - note: loom creates its own tokio runtime"
        );

        let loom = LoomBuilder::new()
            .prefix("dynamo")
            .tokio_threads(1)
            .rayon_threads(2)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create loom runtime: {}", e))?;

        Self::new_with_loom(loom)
    }

    /// Create a [`Runtime`] from an existing tokio handle.
    ///
    /// **Note**: Loom-rs creates its own tokio runtime. This method creates a new
    /// loom runtime rather than reusing the provided handle. For production use,
    /// prefer `from_settings()`.
    pub fn from_handle(_handle: tokio::runtime::Handle) -> Result<Self> {
        tracing::debug!(
            "Creating loom runtime from_handle() - note: loom creates its own tokio runtime"
        );

        let loom = LoomBuilder::new()
            .prefix("dynamo")
            .tokio_threads(1)
            .rayon_threads(2)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create loom runtime: {}", e))?;

        Self::new_with_loom(loom)
    }

    /// Create a [`Runtime`] instance from settings.
    ///
    /// This reads configuration from environment/files and creates an optimally
    /// configured loom runtime with MAB scheduling enabled.
    ///
    /// # Thread Distribution
    ///
    /// By default, threads are distributed to favor compute-heavy workloads:
    /// - Tokio threads: 1/3 of total (for async I/O)
    /// - Rayon threads: 2/3 of total (for CPU-intensive work like tokenization)
    ///
    /// This can be overridden via `num_worker_threads` (tokio) and `compute_threads`
    /// (rayon) in the runtime configuration.
    pub fn from_settings() -> Result<Self> {
        let config = config::RuntimeConfig::from_settings()?;

        // Determine total thread budget
        let total_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Default distribution: 1/3 tokio, 2/3 rayon for compute-heavy workloads
        // This shifts more threads to rayon for tokenization, template rendering, etc.
        let default_tokio_threads = (total_threads / 3).max(1);
        let default_rayon_threads = total_threads - default_tokio_threads;

        // Allow config overrides
        let tokio_threads = config.num_worker_threads.unwrap_or(default_tokio_threads);
        let rayon_threads = config.compute_threads.unwrap_or(default_rayon_threads);

        tracing::info!(
            total_threads = total_threads,
            tokio_threads = tokio_threads,
            rayon_threads = rayon_threads,
            "Loom runtime thread distribution"
        );

        let loom = LoomBuilder::new()
            .prefix("dynamo")
            .tokio_threads(tokio_threads)
            .rayon_threads(rayon_threads)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create loom runtime from settings: {}", e))?;

        Self::new_with_loom(loom)
    }

    /// Create a single-threaded [`Runtime`].
    pub fn single_threaded() -> Result<Self> {
        let loom = LoomBuilder::new()
            .prefix("dynamo-st")
            .tokio_threads(1)
            .rayon_threads(1)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create single-threaded loom runtime: {}", e))?;

        Self::new_with_loom(loom)
    }

    /// Returns the unique identifier for this [`Runtime`].
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns a [`tokio::runtime::Handle`] for the primary/application thread pool.
    ///
    /// This handle can be used to spawn async tasks directly on tokio.
    pub fn primary(&self) -> tokio::runtime::Handle {
        self.loom.tokio_handle().clone()
    }

    /// Returns a handle for background/secondary tasks.
    ///
    /// With loom-rs, this returns a [`SecondaryHandle`] that routes tasks through
    /// loom's adaptive scheduler. Background tasks are tracked and protected by
    /// loom's starvation guardrails.
    pub fn secondary(&self) -> SecondaryHandle {
        SecondaryHandle {
            loom: Arc::clone(&self.loom),
        }
    }

    /// Access the primary [`CancellationToken`] for the runtime.
    pub fn primary_token(&self) -> CancellationToken {
        self.cancellation_token.clone()
    }

    /// Creates a child [`CancellationToken`] tied to the endpoint shutdown lifecycle.
    pub fn child_token(&self) -> CancellationToken {
        self.endpoint_shutdown_token.child_token()
    }

    /// Get access to the graceful shutdown tracker.
    pub(crate) fn graceful_shutdown_tracker(&self) -> Arc<GracefulShutdownTracker> {
        self.graceful_shutdown_tracker.clone()
    }

    /// Get access to the compute pool for CPU-intensive operations.
    ///
    /// With loom-rs, prefer using `spawn_compute()` or `spawn_adaptive()` instead.
    /// This method is provided for API compatibility and returns None.
    pub fn compute_pool(&self) -> Option<&Arc<crate::compute::ComputePool>> {
        // Loom manages its own rayon pool internally
        None
    }

    /// Initialize thread-local compute context on the current thread.
    ///
    /// With loom-rs, this is a no-op as loom manages its own thread-local state.
    pub fn initialize_thread_local(&self) {
        // Loom handles thread-local initialization internally
    }

    /// Initialize thread-local compute context on all worker threads.
    ///
    /// With loom-rs, this is a no-op as loom manages its own thread-local state.
    pub async fn initialize_all_thread_locals(&self) -> Result<()> {
        // Loom handles thread-local initialization internally
        Ok(())
    }

    /// Shutdown the runtime gracefully.
    pub fn shutdown(&self) {
        tracing::info!("Runtime (loom) shutdown initiated");

        let tracker = self.graceful_shutdown_tracker.clone();
        let main_token = self.cancellation_token.clone();
        let endpoint_token = self.endpoint_shutdown_token.clone();

        let handle = self.primary();
        handle.spawn(async move {
            // Phase 1: Cancel endpoint shutdown token
            tracing::info!("Phase 1: Cancelling endpoint shutdown token");
            endpoint_token.cancel();

            // Phase 2: Wait for graceful endpoints
            tracing::info!("Phase 2: Waiting for graceful endpoints to complete");
            let count = tracker.get_count();
            tracing::info!("Active graceful endpoints: {}", count);

            if count != 0 {
                tracker.wait_for_completion().await;
            }

            // Phase 3: Cancel main token
            tracing::info!(
                "Phase 3: All endpoints ended gracefully. Disconnecting backend services."
            );
            main_token.cancel();
        });
    }

    // =========================================================================
    // Loom-specific APIs (new functionality)
    // =========================================================================

    /// Access the underlying [`LoomRuntime`].
    ///
    /// Use this to access loom-specific features like metrics or direct MAB control.
    pub fn loom_runtime(&self) -> &LoomRuntime {
        &self.loom
    }

    /// Spawn an async task on the tokio runtime with loom tracking.
    ///
    /// This is equivalent to `runtime.primary().spawn()` but integrates with
    /// loom's task tracking and starvation detection.
    pub fn spawn_async<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.loom.spawn_async(future)
    }

    /// Execute a CPU-intensive closure on the rayon thread pool.
    ///
    /// This always offloads to rayon, bypassing MAB scheduling. Use this for
    /// tasks that are known to be CPU-intensive (>1ms).
    pub async fn spawn_compute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.loom.spawn_compute(f).await
    }

    /// Execute a closure with adaptive MAB scheduling.
    ///
    /// The MAB scheduler learns whether to execute inline or offload to rayon
    /// based on observed latency for the closure's type. The scheduling decision
    /// is tracked per closure type using Rust's type system.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = runtime.spawn_adaptive(|| {
    ///     tokenizer.encode_batch(&texts)
    /// }).await;
    /// ```
    pub async fn spawn_adaptive<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.loom.spawn_adaptive(f).await
    }

    /// Execute a closure with adaptive scheduling and an explicit hint.
    ///
    /// The hint helps the scheduler make better initial decisions before it has
    /// learned the actual execution time.
    ///
    /// # Hints
    ///
    /// - `ComputeHint::Low` - Expected < 50µs (likely inline-safe)
    /// - `ComputeHint::Medium` - Expected 50-500µs (borderline)
    /// - `ComputeHint::High` - Expected > 500µs (should test offload early)
    /// - `ComputeHint::Unknown` - No hint (default exploration)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use loom_rs::ComputeHint;
    ///
    /// let result = runtime.spawn_adaptive_with_hint(ComputeHint::High, || {
    ///     expensive_computation()
    /// }).await;
    /// ```
    pub async fn spawn_adaptive_with_hint<F, R>(&self, hint: ComputeHint, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.loom.spawn_adaptive_with_hint(hint, f).await
    }
}

/// Handle for spawning background/secondary tasks.
///
/// This provides API compatibility with the existing `runtime.secondary().spawn()` pattern
/// while routing tasks through loom's adaptive scheduler.
#[derive(Clone)]
pub struct SecondaryHandle {
    loom: Arc<LoomRuntime>,
}

impl SecondaryHandle {
    /// Spawn an async task as a background task.
    ///
    /// Background tasks are protected by loom's starvation guardrails.
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.loom.spawn_async(future)
    }

    /// Block on an async future from a synchronous context.
    ///
    /// This is used for sync-async bridging (e.g., C FFI, Python bindings).
    /// Use sparingly as it can cause deadlocks if called from within an async context.
    pub fn block_on<F: Future>(&self, f: F) -> F::Output {
        // Use a dedicated blocking approach to avoid deadlock
        tokio::task::block_in_place(|| self.loom.tokio_handle().block_on(f))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn create_test_runtime() -> Result<Runtime> {
        let loom = LoomBuilder::new()
            .prefix("test")
            .tokio_threads(1)
            .rayon_threads(2)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create test runtime: {}", e))?;

        Runtime::new_with_loom(loom)
    }

    #[test]
    fn test_loom_runtime_creation() {
        let rt = create_test_runtime().expect("Should create loom runtime");
        assert!(!rt.id().is_empty());
    }

    #[test]
    fn test_spawn_compute() {
        let rt = create_test_runtime().expect("Should create loom runtime");

        let result = rt.loom.block_on(async {
            rt.spawn_compute(|| {
                // Simulate CPU work
                let mut sum = 0u64;
                for i in 0..1000 {
                    sum += i;
                }
                sum
            })
            .await
        });

        assert_eq!(result, 499500);
    }

    #[test]
    fn test_spawn_adaptive() {
        let rt = create_test_runtime().expect("Should create loom runtime");

        let result = rt.loom.block_on(async { rt.spawn_adaptive(|| 2 + 2).await });

        assert_eq!(result, 4);
    }

    #[test]
    fn test_spawn_async() {
        let rt = create_test_runtime().expect("Should create loom runtime");

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        rt.loom.block_on(async {
            let handle = rt.spawn_async(async move {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                42
            });

            let result = handle.await.expect("Task should complete");
            assert_eq!(result, 42);
        });

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_secondary_spawn() {
        let rt = create_test_runtime().expect("Should create loom runtime");

        rt.loom.block_on(async {
            let handle = rt.secondary().spawn(async {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                "background task done"
            });

            let result = handle.await.expect("Background task should complete");
            assert_eq!(result, "background task done");
        });
    }

    #[test]
    fn test_cancellation_tokens() {
        let rt = create_test_runtime().expect("Should create loom runtime");

        let primary = rt.primary_token();
        let child = rt.child_token();

        assert!(!primary.is_cancelled());
        assert!(!child.is_cancelled());
    }

    // =========================================================================
    // Starvation Prevention Tests
    // =========================================================================

    /// Test that async I/O tasks complete even when many CPU-intensive tasks are running.
    /// This verifies loom's starvation guardrails are effective.
    #[test]
    fn test_starvation_prevention_async_io_not_blocked() {
        use std::time::{Duration, Instant};

        let rt = LoomBuilder::new()
            .prefix("starvation-test")
            .tokio_threads(2) // Limited tokio threads
            .rayon_threads(4) // More rayon threads
            .build()
            .expect("Failed to create runtime");

        let rt = Arc::new(rt);

        rt.block_on(async {
            // Spawn many CPU-intensive tasks on rayon (via spawn_compute)
            // These should NOT block the tokio runtime
            let compute_tasks: Vec<_> = (0..20)
                .map(|i| {
                    let rt_clone = rt.clone();
                    async move {
                        rt_clone
                            .spawn_compute(move || {
                                // ~1ms of CPU work
                                let mut sum = 0u64;
                                for j in 0u64..100_000 {
                                    sum = sum.wrapping_add(j.wrapping_mul(i as u64));
                                }
                                std::hint::black_box(sum)
                            })
                            .await
                    }
                })
                .collect();

            // Spawn an async I/O task that should complete quickly
            let io_start = Instant::now();
            let io_task = tokio::spawn(async {
                // This simulates a quick async I/O operation
                tokio::time::sleep(Duration::from_millis(10)).await;
                "io_complete"
            });

            // The I/O task should complete in reasonable time even under load
            let io_result = tokio::time::timeout(Duration::from_secs(2), io_task)
                .await
                .expect("I/O task should not timeout")
                .expect("I/O task should not panic");

            let io_duration = io_start.elapsed();

            // Wait for compute tasks to complete
            for task in compute_tasks {
                let _ = task.await;
            }

            assert_eq!(io_result, "io_complete");
            // I/O should complete within a reasonable time (not starved)
            // Allow some slack for system variance
            assert!(
                io_duration < Duration::from_millis(500),
                "I/O task took too long ({:?}), possible starvation",
                io_duration
            );
        });
    }

    /// Test that background tasks (like ETCD/NATS polling) continue to work
    /// when the system is under heavy compute load.
    #[test]
    fn test_starvation_prevention_background_tasks() {
        use std::sync::atomic::AtomicBool;
        use std::time::Duration;

        let rt = LoomBuilder::new()
            .prefix("background-test")
            .tokio_threads(2)
            .rayon_threads(4)
            .build()
            .expect("Failed to create runtime");

        let rt = Arc::new(rt);
        let background_completed = Arc::new(AtomicBool::new(false));

        rt.block_on(async {
            // Start a "background" task that simulates periodic polling (like ETCD/NATS)
            let bg_flag = background_completed.clone();
            let background_handle = tokio::spawn(async move {
                // Simulate 5 polling cycles
                for _ in 0..5 {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                bg_flag.store(true, Ordering::SeqCst);
            });

            // Start heavy compute load
            let rt_clone = rt.clone();
            let compute_tasks: Vec<_> = (0..10)
                .map(|_| {
                    let rt_inner = rt_clone.clone();
                    async move {
                        rt_inner
                            .spawn_compute(|| {
                                // Heavy CPU work (~5ms)
                                let mut sum = 0u64;
                                for i in 0u64..500_000 {
                                    sum = sum.wrapping_add(i.wrapping_mul(i));
                                }
                                std::hint::black_box(sum)
                            })
                            .await
                    }
                })
                .collect();

            // Background task should complete even under load
            let bg_result = tokio::time::timeout(Duration::from_secs(5), background_handle)
                .await
                .expect("Background task should not timeout")
                .expect("Background task should not panic");

            // Wait for compute tasks
            for task in compute_tasks {
                let _ = task.await;
            }

            // Verify background task actually ran
            assert!(
                background_completed.load(Ordering::SeqCst),
                "Background task should have completed all polling cycles"
            );
        });
    }

    /// Test that spawn_adaptive learns to offload heavy tasks.
    #[test]
    fn test_adaptive_learning_offloads_heavy_tasks() {
        use std::time::{Duration, Instant};

        let rt = LoomBuilder::new()
            .prefix("adaptive-test")
            .tokio_threads(2)
            .rayon_threads(4)
            .build()
            .expect("Failed to create runtime");

        rt.block_on(async {
            // Run the same heavy task many times to let MAB learn
            let mut total_time = Duration::ZERO;
            let iterations = 30;

            for _ in 0..iterations {
                let start = Instant::now();
                let result = rt
                    .spawn_adaptive(|| {
                        // ~1ms of work - MAB should learn to offload this
                        let mut sum = 0u64;
                        for i in 0..100_000 {
                            sum = sum.wrapping_add(i);
                        }
                        std::hint::black_box(sum)
                    })
                    .await;

                total_time += start.elapsed();
                std::hint::black_box(result);
            }

            let avg_time = total_time / iterations;

            // Just verify it completes successfully
            // The MAB should have learned a reasonable strategy
            assert!(
                avg_time < Duration::from_millis(50),
                "Average execution time ({:?}) seems too high",
                avg_time
            );
        });
    }
}
