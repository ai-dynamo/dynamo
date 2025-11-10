// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compute pool implementation with tokio-rayon integration
//!
//! The `ComputePool` allows multiple async tasks to concurrently submit different
//! types of parallel work to a shared Rayon thread pool. This enables efficient
//! CPU utilization without manual thread management.
//!
//! # Concurrent Usage Example
//!
//! ```ignore
//! use std::sync::Arc;
//! use dynamo_runtime::compute::ComputePool;
//! use rayon::prelude::*;
//!
//! async fn concurrent_processing(pool: Arc<ComputePool>) {
//!     // Task 1: Using scope for dynamic task generation
//!     let task1 = tokio::spawn({
//!         let pool = pool.clone();
//!         async move {
//!             pool.execute_scoped(|scope| {
//!                 // Dynamically spawn tasks based on runtime conditions
//!                 for i in 0..100 {
//!                     scope.spawn(move |_| {
//!                         // CPU-intensive work
//!                         let mut sum = 0u64;
//!                         for j in 0..1000 {
//!                             sum += (i * j) as u64;
//!                         }
//!                         sum
//!                     });
//!                 }
//!             }).await
//!         }
//!     });
//!
//!     // Task 2: Using parallel iterators for batch processing
//!     let task2 = tokio::spawn({
//!         let pool = pool.clone();
//!         async move {
//!             let data: Vec<u32> = (0..10000).collect();
//!             pool.install(|| {
//!                 data.par_chunks(100)
//!                     .map(|chunk| chunk.iter().sum::<u32>())
//!                     .collect::<Vec<_>>()
//!             }).await
//!         }
//!     });
//!
//!     // Both tasks run concurrently, sharing the same thread pool
//!     let (result1, result2) = tokio::join!(task1, task2);
//! }
//! ```
//!
//! # Thread Pool Sharing
//!
//! The Rayon thread pool uses work-stealing to efficiently distribute work from
//! multiple concurrent sources:
//!
//! - Tasks from `scope.spawn()` are pushed to thread-local deques
//! - Parallel iterators distribute work across all threads
//! - Idle threads steal work from busy threads
//! - No coordination needed between different parallelization patterns

use super::{ComputeConfig, ComputeMetrics};
use anyhow::Result;
use async_trait::async_trait;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// A compute pool that manages CPU-intensive operations
#[derive(Clone)]
pub struct ComputePool {
    /// The underlying Rayon thread pool
    pool: Arc<rayon::ThreadPool>,

    /// Metrics for monitoring compute operations
    metrics: Arc<ComputeMetrics>,

    /// Configuration used to create this pool
    config: ComputeConfig,
}

impl std::fmt::Debug for ComputePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputePool")
            .field("num_threads", &self.pool.current_num_threads())
            .field("metrics", &self.metrics)
            .field("config", &self.config)
            .finish()
    }
}

impl ComputePool {
    /// Create a new compute pool with the given configuration
    pub fn new(config: ComputeConfig) -> Result<Self> {
        let pool = config.build_pool()?;
        let metrics = Arc::new(ComputeMetrics::new());

        Ok(Self {
            pool: Arc::new(pool),
            metrics,
            config,
        })
    }

    /// Create a compute pool with default configuration
    pub fn with_defaults() -> Result<Self> {
        Self::new(ComputeConfig::default())
    }

    /// Execute a synchronous computation on the thread pool
    ///
    /// This method is designed to be called from within `spawn_blocking` or other
    /// synchronous contexts. It has minimal overhead as it directly uses Rayon
    /// without the async bridge.
    ///
    /// # Example
    /// ```ignore
    /// # use dynamo_runtime::compute::ComputePool;
    /// # let pool = ComputePool::new(Default::default()).unwrap();
    /// tokio::task::spawn_blocking(move || {
    ///     pool.execute_sync(|| {
    ///         // CPU-intensive work
    ///         expensive_computation()
    ///     })
    /// });
    /// ```
    pub fn execute_sync<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(f)
    }

    /// Execute a compute task in the Rayon pool
    ///
    /// This bridges from async context to the Rayon thread pool,
    /// allowing CPU-intensive work to run without blocking Tokio workers.
    ///
    /// Note: This method has ~25μs overhead for small tasks due to the async
    /// channel communication. For very small computations (<100μs), consider
    /// running directly on Tokio or using `spawn_blocking` with `execute_sync`.
    pub async fn execute<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.metrics.record_task_start();
        let start = std::time::Instant::now();

        // Use tokio-rayon to bridge to the compute pool
        let pool = self.pool.clone();
        let result = tokio_rayon::spawn(move || pool.install(f)).await;

        self.metrics.record_task_completion(start.elapsed());
        Ok(result)
    }

    /// Execute a function with a Rayon scope
    ///
    /// This allows spawning multiple parallel tasks within the scope,
    /// with the guarantee that all tasks complete before returning.
    pub async fn execute_scoped<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&rayon::Scope) -> R + Send + 'static,
        R: Send + 'static,
    {
        self.metrics.record_task_start();
        let start = std::time::Instant::now();

        let pool = self.pool.clone();
        let result = tokio_rayon::spawn(move || {
            pool.install(|| {
                let mut result = None;
                rayon::scope(|s| {
                    result = Some(f(s));
                });
                result.unwrap()
            })
        })
        .await;

        self.metrics.record_task_completion(start.elapsed());
        Ok(result)
    }

    /// Execute a function with a FIFO scope
    ///
    /// Similar to execute_scoped, but tasks are prioritized in FIFO order
    /// rather than the default LIFO order.
    pub async fn execute_scoped_fifo<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&rayon::ScopeFifo) -> R + Send + 'static,
        R: Send + 'static,
    {
        self.metrics.record_task_start();
        let start = std::time::Instant::now();

        let pool = self.pool.clone();
        let result = tokio_rayon::spawn(move || {
            pool.install(|| {
                let mut result = None;
                rayon::scope_fifo(|s| {
                    result = Some(f(s));
                });
                result.unwrap()
            })
        })
        .await;

        self.metrics.record_task_completion(start.elapsed());
        Ok(result)
    }

    /// Join two computations in parallel
    pub async fn join<F1, F2, R1, R2>(&self, f1: F1, f2: F2) -> Result<(R1, R2)>
    where
        F1: FnOnce() -> R1 + Send + 'static,
        F2: FnOnce() -> R2 + Send + 'static,
        R1: Send + 'static,
        R2: Send + 'static,
    {
        self.execute(move || rayon::join(f1, f2)).await
    }

    /// Get metrics for this compute pool
    pub fn metrics(&self) -> &ComputeMetrics {
        &self.metrics
    }

    /// Get the number of threads in the pool
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Install this pool as the Rayon pool for the given closure
    ///
    /// This method is essential for using Rayon's parallel iterators (like `par_iter`,
    /// `par_chunks`, etc.) with this specific thread pool. Any parallel iterator
    /// operations within the closure will execute on this pool's threads.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rayon::prelude::*;
    ///
    /// // Process data using parallel iterators
    /// let result = pool.install(|| {
    ///     data.par_chunks(100)
    ///         .map(|chunk| process_chunk(chunk))
    ///         .collect::<Vec<_>>()
    /// }).await?;
    /// ```
    ///
    /// # Concurrent Usage
    ///
    /// Multiple async tasks can call `install()` concurrently on the same pool.
    /// The Rayon work-stealing scheduler will efficiently distribute work from
    /// all concurrent operations:
    ///
    /// ```ignore
    /// // These can run concurrently without interference
    /// let task1 = pool.install(|| data1.par_iter().map(f1).collect());
    /// let task2 = pool.install(|| data2.par_chunks(50).map(f2).collect());
    /// ```
    pub async fn install<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let pool = self.pool.clone();
        self.metrics.record_task_start();
        let start = std::time::Instant::now();

        let result = tokio_rayon::spawn(move || pool.install(f)).await;

        self.metrics.record_task_completion(start.elapsed());
        Ok(result)
    }
}

/// A handle to a compute task that's currently running
pub struct ComputeHandle<T> {
    inner: Pin<Box<dyn Future<Output = T> + Send>>,
}

impl<T> ComputeHandle<T> {
    /// Create a new compute handle from a future
    pub(crate) fn new<F>(future: F) -> Self
    where
        F: Future<Output = T> + Send + 'static,
    {
        Self {
            inner: Box::pin(future),
        }
    }
}

impl<T> Future for ComputeHandle<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
    }
}

/// Extension trait for ComputePool with additional patterns
#[async_trait]
pub trait ComputePoolExt {
    /// Process items in parallel batches
    async fn parallel_batch<T, F, R>(
        &self,
        items: Vec<T>,
        batch_size: usize,
        f: F,
    ) -> Result<Vec<R>>
    where
        T: Send + Sync + 'static,
        F: Fn(&[T]) -> Vec<R> + Send + Sync + 'static,
        R: Send + 'static;

    /// Map over items in parallel using Rayon's par_iter
    async fn parallel_map<T, F, R>(&self, items: Vec<T>, f: F) -> Result<Vec<R>>
    where
        T: Send + Sync + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static;
}

#[async_trait]
impl ComputePoolExt for ComputePool {
    async fn parallel_batch<T, F, R>(
        &self,
        items: Vec<T>,
        batch_size: usize,
        f: F,
    ) -> Result<Vec<R>>
    where
        T: Send + Sync + 'static,
        F: Fn(&[T]) -> Vec<R> + Send + Sync + 'static,
        R: Send + 'static,
    {
        use rayon::prelude::*;

        self.install(move || items.par_chunks(batch_size).flat_map(f).collect())
            .await
    }

    async fn parallel_map<T, F, R>(&self, items: Vec<T>, f: F) -> Result<Vec<R>>
    where
        T: Send + Sync + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        use rayon::prelude::*;

        self.install(move || items.into_par_iter().map(f).collect())
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    #[tokio::test]
    async fn test_compute_pool_execute() {
        let pool = ComputePool::with_defaults().unwrap();

        let result = pool
            .execute(|| {
                // Simulate CPU-intensive work
                let mut sum = 0u64;
                for i in 0..1000 {
                    sum += i;
                }
                sum
            })
            .await
            .unwrap();

        assert_eq!(result, 499500);
    }

    #[tokio::test]
    async fn test_compute_pool_join() {
        let pool = ComputePool::with_defaults().unwrap();

        let (a, b) = pool.join(|| 2 + 2, || 3 * 3).await.unwrap();

        assert_eq!(a, 4);
        assert_eq!(b, 9);
    }

    #[tokio::test]
    async fn test_compute_pool_execute_sync() {
        let pool = Arc::new(ComputePool::with_defaults().unwrap());

        // Test using execute_sync from spawn_blocking
        let pool_clone = pool.clone();
        let result = tokio::task::spawn_blocking(move || {
            pool_clone.execute_sync(|| {
                let mut sum = 0u64;
                for i in 0..1000 {
                    sum += i;
                }
                sum
            })
        })
        .await
        .unwrap();

        assert_eq!(result, 499500);
    }

    #[tokio::test]
    async fn test_compute_pool_scoped() {
        use std::sync::mpsc;

        let pool = ComputePool::with_defaults().unwrap();

        let mut result = pool
            .execute_scoped(|scope| {
                let (tx, rx) = mpsc::channel();

                for i in 0..4 {
                    let tx = tx.clone();
                    scope.spawn(move |_| {
                        tx.send((i, i * 2)).unwrap();
                    });
                }

                drop(tx); // Close sender so receiver can finish

                let mut results = vec![0; 4];
                for (i, val) in rx {
                    results[i] = val;
                }
                results
            })
            .await
            .unwrap();

        // Results may be in any order due to parallel execution
        result.sort();
        assert_eq!(result, vec![0, 2, 4, 6]);
    }
}
