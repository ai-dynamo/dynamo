// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Zero-overhead macros for compute task execution with optional validation
//!
//! These macros provide size-aware execution strategies:
//! - `compute_small!`: Direct inline execution for tasks <100μs
//! - `compute_medium!`: Semaphore-guarded block_in_place for tasks 100μs-1ms
//! - `compute_large!`: Rayon offload for tasks >1ms
//!
//! When the `compute-validation` feature is enabled, these macros will
//! time execution and emit warnings if tasks are misclassified.

/// Execute a small compute task (<100μs) directly inline.
///
/// This macro has zero overhead and simply executes the expression directly.
/// When validation is enabled, it will warn if the task takes >100μs.
///
/// # Example
/// ```
/// # use dynamo_runtime::compute_small;
/// let result = compute_small!(2 + 2);
/// assert_eq!(result, 4);
/// ```
#[macro_export]
macro_rules! compute_small {
    ($expr:expr) => {{
        #[cfg(feature = "compute-validation")]
        let _start = std::time::Instant::now();

        let result = $expr; // Direct execution, zero overhead

        #[cfg(feature = "compute-validation")]
        $crate::compute::validation::validate_small(_start.elapsed());

        result
    }};
}

/// Execute a medium compute task (100μs-1ms) with intelligent scheduling.
///
/// This macro first tries to use thread-local context if available (on Tokio worker threads).
/// If no thread-local context, it requires a pool parameter.
///
/// # Example
/// ```ignore
/// # use dynamo_runtime::{compute_medium, compute::ComputePool};
/// # async fn example(pool: &ComputePool) {
/// // With thread-local context (on worker thread)
/// let result = compute_medium!({
///     (0..1000).map(|i| i * 2).sum::<i32>()
/// }).await;
///
/// // Or with explicit pool (fallback)
/// let result = compute_medium!(pool, {
///     (0..1000).map(|i| i * 2).sum::<i32>()
/// }).await;
/// # }
/// ```
#[macro_export]
macro_rules! compute_medium {
    // Thread-local version (no pool parameter)
    ($expr:expr) => {{
        #[cfg(feature = "compute-validation")]
        let _start = std::time::Instant::now();

        let result = async {
            // Try thread-local context first
            if let Ok(_permit) = $crate::compute::thread_local::try_acquire_block_permit() {
                // Got permit - use block_in_place
                Ok(tokio::task::block_in_place(|| {
                    let r = $expr;
                    drop(_permit); // Release ASAP
                    r
                }))
            } else if let Some(pool) = $crate::compute::thread_local::get_pool() {
                // No permit but have pool - offload
                pool.execute(|| $expr).await
            } else {
                // No context available - fall back to inline execution
                // This may block the async runtime but ensures the macro always works
                tracing::warn!("compute_medium: No thread-local context, executing inline (may block async runtime)");
                Ok($expr)
            }
        }
        .await?;

        #[cfg(feature = "compute-validation")]
        $crate::compute::validation::validate_medium(_start.elapsed());

        result
    }};

    // Explicit pool version (fallback)
    ($pool:expr, $expr:expr) => {{
        #[cfg(feature = "compute-validation")]
        let _start = std::time::Instant::now();

        let result = async {
            // Try thread-local permits first, fall back to pool
            if let Ok(_permit) = $crate::compute::thread_local::try_acquire_block_permit() {
                // Got permit - use block_in_place
                Ok(tokio::task::block_in_place(|| {
                    let r = $expr;
                    drop(_permit); // Release ASAP
                    r
                }))
            } else {
                // No permit available - offload to provided pool
                $pool.execute(|| $expr).await
            }
        }
        .await?;

        #[cfg(feature = "compute-validation")]
        $crate::compute::validation::validate_medium(_start.elapsed());

        result
    }};
}

/// Execute a large compute task (>1ms) on the Rayon thread pool.
///
/// This macro always offloads to Rayon as the overhead is negligible
/// compared to the computation time.
///
/// # Example
/// ```ignore
/// # use dynamo_runtime::{compute_large, compute::ComputePool};
/// # async fn example(pool: &ComputePool) {
/// // With thread-local context
/// let result = compute_large!({
///     expensive_matrix_multiplication()
/// }).await;
///
/// // Or with explicit pool
/// let result = compute_large!(pool, {
///     expensive_matrix_multiplication()
/// }).await;
/// # }
/// ```
#[macro_export]
macro_rules! compute_large {
    // Thread-local version
    ($expr:expr) => {{
        #[cfg(feature = "compute-validation")]
        let _start = std::time::Instant::now();

        let result = async {
            if let Some(pool) = $crate::compute::thread_local::get_pool() {
                pool.execute(|| $expr).await
            } else {
                // No pool available - fall back to inline execution
                // Warning: Large tasks inline will severely block the async runtime
                tracing::warn!("compute_large: No thread-local context, executing inline (will block async runtime!)");
                Ok($expr)
            }
        }
        .await?;

        #[cfg(feature = "compute-validation")]
        $crate::compute::validation::validate_large(_start.elapsed());

        result
    }};

    // Explicit pool version
    ($pool:expr, $expr:expr) => {{
        #[cfg(feature = "compute-validation")]
        let _start = std::time::Instant::now();

        let result = $pool.execute(|| $expr).await?;

        #[cfg(feature = "compute-validation")]
        $crate::compute::validation::validate_large(_start.elapsed());

        result
    }};
}
