// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compute module for CPU-intensive operations using Rayon
//!
//! This module provides a dedicated compute thread pool for CPU-bound work,
//! integrating Rayon's fork-join parallelism with Tokio's async runtime.
//!
//! Key features:
//! - Dedicated Rayon thread pool for compute operations
//! - Seamless async-to-sync bridging via tokio-rayon
//! - Scope-based parallelism for complex computational graphs
//! - Metrics and monitoring for compute operations
//!
#![doc = include_str!("../../docs/rayon-tokio-strategy.md")]

use anyhow::Result;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub mod macros;
pub mod metrics;
pub mod pool;
pub mod thread_local;
#[cfg(feature = "compute-validation")]
pub mod validation;

pub use metrics::ComputeMetrics;
pub use pool::{ComputeHandle, ComputePool, ComputePoolExt};

/// Configuration for the compute thread pool
#[derive(Debug, Clone)]
pub struct ComputeConfig {
    /// Number of threads in the Rayon pool (defaults to num_cpus / 2)
    pub num_threads: Option<usize>,

    /// Stack size for compute threads (defaults to 2MB)
    pub stack_size: Option<usize>,

    /// Thread name prefix (defaults to "compute")
    pub thread_prefix: String,

    /// Whether to pin threads to CPU cores
    pub pin_threads: bool,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            num_threads: None,                 // Will use num_cpus / 2
            stack_size: Some(2 * 1024 * 1024), // 2MB
            thread_prefix: "compute".to_string(),
            pin_threads: false,
        }
    }
}

impl ComputeConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if let Some(num_threads) = self.num_threads
            && num_threads == 0
        {
            return Err(anyhow::anyhow!(
                "Number of compute threads cannot be 0. Use None to disable compute pool entirely."
            ));
        }

        if let Some(stack_size) = self.stack_size
            && stack_size < 128 * 1024
        {
            return Err(anyhow::anyhow!(
                "Stack size too small: {}KB. Minimum recommended: 128KB",
                stack_size / 1024
            ));
        }

        Ok(())
    }

    /// Create a ThreadPoolBuilder from this configuration
    pub(crate) fn build_pool(&self) -> Result<rayon::ThreadPool> {
        // Validate configuration first
        self.validate()?;

        let mut builder = ThreadPoolBuilder::new();

        // Set number of threads with better logic for minimum parallelism
        let num_threads = self.num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| {
                    let total_cores = n.get();
                    // Use half the cores, but ensure we have at least 2 threads
                    // for meaningful parallelism, and cap at 16 for efficiency
                    (total_cores / 2).clamp(2, 16)
                })
                .unwrap_or(2) // Fallback to 2 threads if detection fails
        });
        builder = builder.num_threads(num_threads);

        // Set stack size if specified
        if let Some(stack_size) = self.stack_size {
            builder = builder.stack_size(stack_size);
        }

        // Set thread name prefix
        let prefix = self.thread_prefix.clone();
        let thread_counter = Arc::new(AtomicU64::new(0));
        builder = builder.thread_name(move |_| {
            let id = thread_counter.fetch_add(1, Ordering::SeqCst);
            format!("{}-{}", prefix, id)
        });

        // TODO: Add CPU pinning if requested
        // if self.pin_threads {
        //     builder = builder.start_handler(|idx| {
        //         // Pin thread to CPU core
        //     });
        // }

        builder
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create Rayon thread pool: {}", e))
    }
}

/// Helper trait for scope-based operations
pub trait ScopeExecutor {
    /// Execute a function within a Rayon scope
    fn execute_in_scope<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&rayon::Scope) -> R + Send,
        R: Send;
}

/// Helper functions for common parallel patterns
pub mod patterns {
    use super::*;

    /// Execute two functions in parallel and return both results
    pub async fn parallel_join<F1, F2, R1, R2>(
        pool: &ComputePool,
        f1: F1,
        f2: F2,
    ) -> Result<(R1, R2)>
    where
        F1: FnOnce() -> R1 + Send + 'static,
        F2: FnOnce() -> R2 + Send + 'static,
        R1: Send + 'static,
        R2: Send + 'static,
    {
        pool.execute(move || rayon::join(f1, f2)).await
    }

    /// Execute multiple functions in parallel using scope
    pub async fn parallel_map<F, T, R>(pool: &ComputePool, items: Vec<T>, f: F) -> Result<Vec<R>>
    where
        F: Fn(T) -> R + Sync + Send + 'static,
        T: Send + 'static,
        R: Send + 'static,
    {
        use rayon::prelude::*;
        pool.execute(move || items.into_par_iter().map(f).collect())
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_config_default() {
        let config = ComputeConfig::default();
        assert_eq!(config.thread_prefix, "compute");
        assert_eq!(config.stack_size, Some(2 * 1024 * 1024));
        assert!(!config.pin_threads);
    }

    #[test]
    fn test_build_pool() {
        let config = ComputeConfig {
            num_threads: Some(2),
            ..Default::default()
        };

        let pool = config.build_pool().unwrap();
        assert_eq!(pool.current_num_threads(), 2);
    }
}
