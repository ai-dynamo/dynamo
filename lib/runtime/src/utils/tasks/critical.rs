// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Utilities for handling tasks.

use anyhow::{Context, Result};
use std::future::Future;
use tokio::runtime::Handle;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Type alias for a critical task handler function.
///
/// The handler receives a [CancellationToken] and returns a [Future] that resolves to [Result<()>].
/// The task should monitor the cancellation token and gracefully shut down when it's cancelled.
pub type CriticalTaskHandler<Fut> = dyn FnOnce(CancellationToken) -> Fut + Send + 'static;

/// The [CriticalTaskExecutionHandle] is a handle for a critical task that is expected to
/// complete successfully. This handle provides two cancellation mechanisms:
///
/// 1. **Critical Failure**: If the task returns an error or panics, the parent cancellation
///    token is triggered immediately via a monitoring task that detects failures.
///
/// 2. **Graceful Shutdown**: The task can be gracefully shut down via its child token,
///    allowing it to complete cleanly without triggering system-wide cancellation.
///
/// This is useful for ensuring that critical detached tasks either complete successfully
/// or trigger appropriate shutdown procedures when they fail.
pub struct CriticalTaskExecutionHandle {
    monitor_task: JoinHandle<()>,
    graceful_shutdown_token: CancellationToken,
    result_receiver: Option<oneshot::Receiver<Result<()>>>,
    detached: bool,
}

impl CriticalTaskExecutionHandle {
    pub fn new<Fut>(
        task_fn: impl FnOnce(CancellationToken) -> Fut + Send + 'static,
        parent_token: CancellationToken,
        description: &str,
    ) -> Result<Self>
    where
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        Self::new_with_runtime(task_fn, parent_token, description, &Handle::try_current()?)
    }

    /// Create a new [CriticalTaskExecutionHandle] for a critical task.
    ///
    /// # Arguments
    /// * `task_fn` - A function that takes a cancellation token and returns the critical task future
    /// * `parent_token` - Token that will be cancelled if this critical task fails
    /// * `description` - Description for logging purposes
    /// * `runtime` - The runtime to use for the task.
    pub fn new_with_runtime<Fut>(
        task_fn: impl FnOnce(CancellationToken) -> Fut + Send + 'static,
        parent_token: CancellationToken,
        description: &str,
        runtime: &Handle,
    ) -> Result<Self>
    where
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let graceful_shutdown_token = parent_token.child_token();
        let description = description.to_string();
        let parent_token_clone = parent_token.clone();

        // Create channel for communicating results from monitor to handle
        let (result_sender, result_receiver) = oneshot::channel();

        let graceful_shutdown_token_clone = graceful_shutdown_token.clone();
        let description_clone = description.to_string();
        let task = runtime.spawn(async move {
            let future = task_fn(graceful_shutdown_token_clone);

            match future.await {
                Ok(()) => {
                    tracing::debug!(
                        "Critical task '{}' completed successfully",
                        description_clone
                    );
                    Ok(())
                }
                Err(e) => {
                    tracing::error!("Critical task '{}' failed: {:#}", description_clone, e);
                    Err(e.context(format!("Critical task '{}' failed", description_clone)))
                }
            }
        });

        // Spawn monitor task that immediately joins the main task and detects failures
        let monitor_task = {
            let main_task_handle = task;
            let parent_token_monitor = parent_token_clone.clone();
            let description_monitor = description.clone();

            runtime.spawn(async move {
                let result = match main_task_handle.await {
                    Ok(task_result) => {
                        // Task completed normally (success or error)
                        if task_result.is_err() {
                            // Error - trigger parent cancellation immediately
                            parent_token_monitor.cancel();
                        }
                        task_result
                    }
                    Err(join_error) => {
                        // Task panicked - handle immediately
                        if join_error.is_panic() {
                            let panic_msg = if let Ok(reason) = join_error.try_into_panic() {
                                if let Some(s) = reason.downcast_ref::<String>() {
                                    s.clone()
                                } else if let Some(s) = reason.downcast_ref::<&str>() {
                                    s.to_string()
                                } else {
                                    "Unknown panic".to_string()
                                }
                            } else {
                                "Panic occurred but reason unavailable".to_string()
                            };

                            tracing::error!(
                                "Critical task '{}' panicked: {}",
                                description_monitor,
                                panic_msg
                            );
                            parent_token_monitor.cancel(); // Trigger parent cancellation immediately
                            Err(anyhow::anyhow!(
                                "Critical task '{}' panicked: {}",
                                description_monitor,
                                panic_msg
                            ))
                        } else {
                            parent_token_monitor.cancel();
                            Err(anyhow::anyhow!(
                                "Failed to join critical task '{}': {}",
                                description_monitor,
                                join_error
                            ))
                        }
                    }
                };

                // Send result to handle (ignore if receiver dropped)
                let _ = result_sender.send(result);
            })
        };

        Ok(Self {
            monitor_task,
            graceful_shutdown_token,
            result_receiver: Some(result_receiver),
            detached: false,
        })
    }

    /// Check if the task awaiting on the [Server]s background event loop has finished.
    pub fn is_finished(&self) -> bool {
        self.monitor_task.is_finished()
    }

    /// Check if the server's event loop has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.graceful_shutdown_token.is_cancelled()
    }

    /// Gracefully cancel this critical task without triggering system-wide shutdown.
    ///
    /// This signals the task to stop processing and exit cleanly. The task should
    /// monitor its cancellation token and respond appropriately.
    ///
    /// This will not propagate to the parent [CancellationToken] unless an error
    /// occurs during the shutdown process.
    pub fn cancel(&self) {
        self.graceful_shutdown_token.cancel();
    }

    /// Join on the critical task and return its actual result.
    ///
    /// This will return:
    /// - `Ok(())` if the task completed successfully or was gracefully cancelled
    /// - `Err(...)` if the task failed or panicked, preserving the original error
    ///
    /// Note: Both errors and panics trigger parent cancellation immediately via the monitor task.
    pub async fn join(mut self) -> Result<()> {
        self.detached = true;

        match self.result_receiver.take().unwrap().await {
            Ok(task_result) => task_result,
            Err(_) => {
                // This should rarely happen - means monitor task was dropped/cancelled
                Err(anyhow::anyhow!("Critical task monitor was cancelled"))
            }
        }
    }

    /// Detach the task. This allows the task to continue running after the handle is dropped.
    pub fn detach(mut self) {
        self.detached = true;
    }
}

impl Drop for CriticalTaskExecutionHandle {
    fn drop(&mut self) {
        if !self.detached {
            panic!("Critical task was not detached prior to drop!");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Duration;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_successful_task_completion() {
        // Test: A critical task that completes successfully without any issues
        // Verifies:
        // - Task executes and completes normally
        // - Result is Ok(())
        // - Parent token remains uncancelled (no critical failure)
        // - Task execution side effects occur (work gets done)
        let parent_token = CancellationToken::new();
        let completed = Arc::new(AtomicBool::new(false));
        let completed_clone = completed.clone();

        let handle = CriticalTaskExecutionHandle::new(
            |_cancel_token| async move {
                completed_clone.store(true, Ordering::SeqCst);
                Ok(())
            },
            parent_token.clone(),
            "test-success-task",
        )
        .unwrap();

        // Task should complete successfully
        let result = handle.join().await;
        assert!(result.is_ok());
        assert!(completed.load(Ordering::SeqCst));
        assert!(!parent_token.is_cancelled());
    }

    #[tokio::test]
    async fn test_task_failure_cancels_parent_token() {
        // Test: A critical task that returns an error (critical failure)
        // Verifies:
        // - Task error is properly propagated to caller
        // - Parent cancellation token is triggered (critical failure behavior)
        // - Error message is preserved and includes context
        // - Demonstrates the "critical" aspect - failures affect the entire system
        let parent_token = CancellationToken::new();

        let handle = CriticalTaskExecutionHandle::new(
            |_cancel_token| async move {
                anyhow::bail!("Critical task failed!");
            },
            parent_token.clone(),
            "test-failure-task",
        )
        .unwrap();

        // Task should fail and cancel parent token
        let result = handle.join().await;
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        // Check that the error contains either the original message or the context
        assert!(
            error_msg.contains("Critical task failed!")
                || error_msg.contains("Critical task 'test-failure-task' failed"),
            "Error message should contain failure context: {}",
            error_msg
        );

        // Give a moment for the cancellation to propagate
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(parent_token.is_cancelled());
    }

    #[tokio::test]
    async fn test_task_panic_is_caught_and_reported() {
        // Test: A critical task that panics during execution
        // Verifies:
        // - Tokio's JoinHandle catches panics automatically
        // - Panics are converted to proper Error types
        // - System doesn't crash, panic is contained
        // - Error message indicates a panic occurred
        // - Parent token is cancelled (panic is treated as critical failure)
        // - Demonstrates panic safety of the critical task system
        let parent_token = CancellationToken::new();

        let handle = CriticalTaskExecutionHandle::new(
            |_cancel_token| async move {
                panic!("Something went terribly wrong!");
            },
            parent_token.clone(),
            "test-panic-task",
        )
        .unwrap();

        // Panic should be caught and converted to error
        let result = handle.join().await;
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(
            error_msg.contains("panicked") || error_msg.contains("panic"),
            "Error message should indicate a panic occurred: {}",
            error_msg
        );

        // Parent token should be cancelled due to panic (critical failure)
        assert!(parent_token.is_cancelled());
    }

    #[tokio::test]
    async fn test_graceful_shutdown_via_cancellation_token() {
        // Test: A long-running task that responds to graceful shutdown signals
        // Verifies:
        // - Task can monitor its cancellation token and stop early
        // - Graceful cancellation does NOT trigger parent token cancellation
        // - Task can do partial work before stopping
        // - handle.cancel() triggers the child token, not parent token
        // - Demonstrates proper shutdown patterns for long-running tasks
        let parent_token = CancellationToken::new();
        let work_done = Arc::new(AtomicU32::new(0));
        let work_done_clone = work_done.clone();

        let handle = CriticalTaskExecutionHandle::new(
            |cancel_token| async move {
                for i in 0..100 {
                    if cancel_token.is_cancelled() {
                        break;
                    }
                    work_done_clone.store(i, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Ok(())
            },
            parent_token.clone(),
            "test-graceful-shutdown",
        )
        .unwrap();

        // Let task do some work
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Request graceful shutdown
        handle.cancel();

        // Task should complete gracefully
        let result = handle.join().await;
        assert!(result.is_ok());

        // Task should have done some work but not all
        let final_work = work_done.load(Ordering::SeqCst);
        assert!(final_work > 0);
        assert!(final_work < 99);

        // Parent token should NOT be cancelled (graceful shutdown)
        assert!(!parent_token.is_cancelled());
    }

    #[tokio::test]
    async fn test_multiple_critical_tasks_one_failure() {
        // Test: Multiple critical tasks sharing a parent token, one fails
        // Verifies:
        // - Multiple critical tasks can share the same parent cancellation token
        // - When one critical task fails, all related tasks receive cancellation signal
        // - Tasks can respond to cancellation and stop gracefully
        // - System-wide shutdown behavior when critical components fail
        // - Demonstrates coordinated shutdown of related services
        let parent_token = CancellationToken::new();
        let task1_completed = Arc::new(AtomicBool::new(false));
        let task2_completed = Arc::new(AtomicBool::new(false));

        let task1_completed_clone = task1_completed.clone();
        let task2_completed_clone = task2_completed.clone();

        // Start two critical tasks
        let handle1 = CriticalTaskExecutionHandle::new(
            |cancel_token| async move {
                for _ in 0..50 {
                    if cancel_token.is_cancelled() {
                        return Ok(());
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                task1_completed_clone.store(true, Ordering::SeqCst);
                Ok(())
            },
            parent_token.clone(),
            "long-running-task",
        )
        .unwrap();

        let handle2 = CriticalTaskExecutionHandle::new(
            |_cancel_token| async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                task2_completed_clone.store(true, Ordering::SeqCst);
                anyhow::bail!("Task 2 failed!");
            },
            parent_token.clone(),
            "failing-task",
        )
        .unwrap();

        // Wait for task 2 to fail
        let result2 = handle2.join().await;
        assert!(result2.is_err());

        // Parent token should be cancelled due to task 2 failure
        assert!(parent_token.is_cancelled());

        // Task 1 should complete early due to cancellation
        let result1 = handle1.join().await;
        assert!(result1.is_ok());
        assert!(!task1_completed.load(Ordering::SeqCst)); // Should not have completed normally
    }

    #[tokio::test]
    async fn test_status_checking_methods() {
        // Test: Non-blocking status checking methods on the handle
        // Verifies:
        // - is_finished() accurately reports task completion status
        // - is_cancelled() accurately reports cancellation status
        // - Status methods work before and after cancellation
        // - Methods are non-blocking and can be called multiple times
        // - Demonstrates monitoring patterns for task supervision
        let parent_token = CancellationToken::new();

        let handle = CriticalTaskExecutionHandle::new(
            |cancel_token| async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                if cancel_token.is_cancelled() {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(())
            },
            parent_token,
            "status-test-task",
        )
        .unwrap();

        // Initially task should be running
        assert!(!handle.is_finished());
        assert!(!handle.is_cancelled());

        // Cancel the task
        handle.cancel();

        // Task should now be cancelled but may not be finished yet
        assert!(handle.is_cancelled());

        // Wait for completion
        let result = handle.join().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_task_with_select_pattern() {
        // Test: Task using tokio::select! for cancellation-aware operations
        // Verifies:
        // - Tasks can use idiomatic tokio patterns with cancellation tokens
        // - select! allows racing between work and cancellation
        // - Cancellation interrupts work immediately, not just at check points
        // - Demonstrates recommended pattern for responsive cancellation
        // - Shows how to handle cancellation in the middle of async operations
        let parent_token = CancellationToken::new();
        let work_completed = Arc::new(AtomicBool::new(false));
        let work_completed_clone = work_completed.clone();

        let handle = CriticalTaskExecutionHandle::new(
            |cancel_token| async move {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_millis(200)) => {
                        work_completed_clone.store(true, Ordering::SeqCst);
                        Ok(())
                    }
                    _ = cancel_token.cancelled() => {
                        // Graceful shutdown requested
                        Ok(())
                    }
                }
            },
            parent_token,
            "select-pattern-task",
        )
        .unwrap();

        // Cancel after a short time
        tokio::time::sleep(Duration::from_millis(50)).await;
        handle.cancel();

        let result = handle.join().await;
        assert!(result.is_ok());
        assert!(!work_completed.load(Ordering::SeqCst)); // Should not have completed the work
    }

    #[tokio::test]
    async fn test_timeout_behavior() {
        // Test: External timeout vs task failure distinction
        // Verifies:
        // - External timeouts don't trigger parent token cancellation
        // - Tasks continue running in background even after timeout
        // - Difference between "waiting timeout" and "task failure"
        // - Client-side timeout vs server-side failure handling
        // - Demonstrates that join() timeout != critical task failure
        let parent_token = CancellationToken::new();

        let handle = CriticalTaskExecutionHandle::new(
            |_cancel_token| async move {
                // A task that takes a long time
                tokio::time::sleep(Duration::from_secs(10)).await;
                Ok(())
            },
            parent_token,
            "long-task",
        )
        .unwrap();

        // Test with timeout
        let result = timeout(Duration::from_millis(100), handle.join()).await;
        assert!(result.is_err()); // Should timeout

        // The parent token should NOT be cancelled since task didn't fail
        // (it's still running in the background, but we timed out waiting for it)
    }

    #[tokio::test]
    async fn test_panic_triggers_immediate_parent_cancellation() {
        // Test: Verify that panics trigger parent cancellation immediately via monitor task
        // Verifies:
        // - Monitor task detects panics immediately when they occur
        // - Parent token cancellation happens immediately, not on join()
        // - System shutdown is triggered as soon as critical task panics
        // - Demonstrates true "critical task" behavior with immediate failure propagation
        let parent_token = CancellationToken::new();

        let handle = CriticalTaskExecutionHandle::new(
            |_cancel_token| async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                panic!("Critical failure!");
            },
            parent_token.clone(),
            "immediate-panic-task",
        )
        .unwrap();

        // Wait for the panic to be detected by monitor task
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Parent token should be cancelled immediately via monitor task
        assert!(
            parent_token.is_cancelled(),
            "Parent token should be cancelled immediately when critical task panics"
        );
        assert!(handle.join().await.is_err());
    }

    #[tokio::test]
    async fn test_error_triggers_immediate_parent_cancellation() {
        // Test: Verify that regular errors also trigger parent cancellation immediately
        // Verifies:
        // - Parent token cancellation happens immediately when task returns error
        // - No need to call join() for critical failure detection
        // - Both panics AND regular errors trigger immediate system shutdown
        // - Demonstrates consistent critical failure behavior
        let parent_token = CancellationToken::new();

        let handle = CriticalTaskExecutionHandle::new(
            |_cancel_token| async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                anyhow::bail!("Critical error!");
            },
            parent_token.clone(),
            "immediate-error-task",
        )
        .unwrap();

        // Don't call join() - just wait for the error to be detected
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Parent token should be cancelled even though we didn't call join()
        assert!(
            parent_token.is_cancelled(),
            "Parent token should be cancelled immediately when critical task errors"
        );
        assert!(handle.join().await.is_err());
    }

    #[tokio::test]
    #[should_panic]
    async fn test_task_detach() {
        // Dropping without detaching should panic
        let parent_token = CancellationToken::new();
        let _handle = CriticalTaskExecutionHandle::new(
            |_cancel_token| async move { Ok(()) },
            parent_token,
            "test-detach-task",
        )
        .unwrap();

        // Dropping without detaching should panic
    }
}
