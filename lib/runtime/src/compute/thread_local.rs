// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thread-local storage for compute resources
//!
//! This module provides thread-local access to compute resources (Rayon pool and semaphore)
//! for Tokio worker threads. This eliminates the need to pass Runtime or ComputePool
//! references through async function calls.

use super::ComputePool;
use std::cell::RefCell;
use std::sync::Arc;
use tokio::sync::Semaphore;

thread_local! {
    /// Thread-local compute context available on Tokio worker threads
    static COMPUTE_CONTEXT: RefCell<Option<ComputeContext>> = const { RefCell::new(None) };
}

/// Compute resources available to a Tokio worker thread
#[derive(Clone)]
pub struct ComputeContext {
    /// The Rayon compute pool
    pub pool: Arc<ComputePool>,
    /// Semaphore for block_in_place permits
    pub block_in_place_permits: Arc<Semaphore>,
}

/// Initialize the thread-local compute context
///
/// This should be called from the Tokio runtime's `on_thread_start` callback
pub fn initialize_context(pool: Arc<ComputePool>, permits: Arc<Semaphore>) {
    COMPUTE_CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = Some(ComputeContext {
            pool,
            block_in_place_permits: permits,
        });
    });
}

/// Access the thread-local compute context
///
/// Returns None if called from a non-worker thread or if context wasn't initialized
pub fn with_context<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&ComputeContext) -> R,
{
    COMPUTE_CONTEXT.with(|ctx| ctx.borrow().as_ref().map(f))
}

/// Try to acquire a block_in_place permit from thread-local context
///
/// Returns Ok(permit) if successful, Err if no context or no permits available
pub fn try_acquire_block_permit() -> Result<tokio::sync::OwnedSemaphorePermit, &'static str> {
    with_context(|ctx| {
        ctx.block_in_place_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| "No permits available")
    })
    .ok_or("No compute context on this thread")?
}

/// Get the compute pool from thread-local context
///
/// Returns None if called from a non-worker thread
pub fn get_pool() -> Option<Arc<ComputePool>> {
    with_context(|ctx| ctx.pool.clone())
}

/// Check if the current thread has compute context initialized
///
/// Returns true if the thread-local context is initialized with a compute pool
/// and semaphore permits, meaning the compute macros will offload work.
/// Returns false if macros would fall back to inline execution.
pub fn has_compute_context() -> bool {
    with_context(|_| ()).is_some()
}

/// Assert that the current thread has compute context initialized
///
/// Panics if the thread-local context is not initialized.
/// Use this to ensure compute macros will offload work rather than run inline.
pub fn assert_compute_context() {
    if !has_compute_context() {
        panic!(
            "Thread-local compute context not initialized! \
             Compute macros will fall back to inline execution. \
             Call Runtime::initialize_thread_local() on worker threads."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uninitialized_context() {
        // Should return None when context not initialized
        assert!(get_pool().is_none());
        assert!(try_acquire_block_permit().is_err());
        assert!(!has_compute_context());
    }

    #[test]
    #[should_panic(expected = "Thread-local compute context not initialized")]
    fn test_assert_compute_context_panics() {
        // Should panic when context not initialized
        assert_compute_context();
    }
}
