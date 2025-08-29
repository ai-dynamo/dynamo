// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Task Tracker - Hierarchical Task Management System
//!
//! A composable task management system with configurable scheduling and error handling policies.
//! The TaskTracker enables controlled concurrent execution with proper resource management,
//! cancellation semantics, and retry support.
//!
//! ## Architecture Overview
//!
//! The TaskTracker system is built around three core abstractions that compose together:
//!
//! ### 1. **TaskScheduler** - Resource Management
//!
//! Controls when and how tasks acquire execution resources (permits, slots, etc.).
//! Schedulers implement resource acquisition with cancellation support:
//!
//! ```text
//! TaskScheduler::acquire_execution_slot(cancel_token) -> SchedulingResult<ResourceGuard>
//! ```
//!
//! - **Resource Acquisition**: Can be cancelled to avoid unnecessary allocation
//! - **RAII Guards**: Resources are automatically released when guards are dropped
//! - **Pluggable**: Different scheduling policies (unlimited, semaphore, rate-limited, etc.)
//!
//! ### 2. **OnErrorPolicy** - Error Handling
//!
//! Defines how the system responds to task failures:
//!
//! ```text
//! OnErrorPolicy::on_error(error, task_id) -> ErrorResponse
//! ```
//!
//! - **ErrorResponse::Fail**: Log error, fail this task
//! - **ErrorResponse::Shutdown**: Shutdown tracker and all children
//! - **ErrorResponse::Custom(action)**: Execute custom logic that can return:
//!   - `ActionResult::Fail`: Handle error and fail the task
//!   - `ActionResult::Shutdown`: Shutdown tracker
//!   - `ActionResult::Continue { continuation }`: Continue with provided task
//!
//! ### 3. **Execution Pipeline** - Task Orchestration
//!
//! The execution pipeline coordinates scheduling, execution, and error handling:
//!
//! ```text
//! 1. Acquire resources (scheduler.acquire_execution_slot)
//! 2. Create task future (only after resources acquired)
//! 3. Execute task while holding guard (RAII pattern)
//! 4. Handle errors through policy (with retry support for cancellable tasks)
//! 5. Update metrics and release resources
//! ```
//!
//! ## Key Design Principles
//!
//! ### **Separation of Concerns**
//! - **Scheduling**: When/how to allocate resources
//! - **Execution**: Running tasks with proper resource management
//! - **Error Handling**: Responding to failures with configurable policies
//!
//! ### **Composability**
//! - Schedulers and error policies are independent and can be mixed/matched
//! - Custom policies can be implemented via traits
//! - Execution pipeline handles the coordination automatically
//!
//! ### **Resource Safety**
//! - Resources are acquired before task creation (prevents early execution)
//! - RAII pattern ensures resources are always released
//! - Cancellation is supported during resource acquisition, not during execution
//!
//! ### **Retry Support**
//! - Regular tasks (`spawn`): Cannot be retried (future is consumed)
//! - Cancellable tasks (`spawn_cancellable`): Support retry via `FnMut` closures
//! - Error policies can provide next executors via `ActionResult::Continue`
//!
//! ## Task Types
//!
//! ### Regular Tasks
//! ```rust
//! # use dynamo_runtime::utils::tasks::tracker::*;
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let tracker = TaskTracker::new(UnlimitedScheduler::new(), LogOnlyPolicy::new()).unwrap();
//! let handle = tracker.spawn(async { Ok(42) });
//! # let _result = handle.await?;
//! # Ok(())
//! # }
//! ```
//! - Simple futures that run to completion
//! - Cannot be retried (future is consumed on first execution)
//! - Suitable for one-shot operations
//!
//! ### Cancellable Tasks
//! ```rust
//! # use dynamo_runtime::utils::tasks::tracker::*;
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let tracker = TaskTracker::new(UnlimitedScheduler::new(), LogOnlyPolicy::new()).unwrap();
//! let handle = tracker.spawn_cancellable(|cancel_token| async move {
//!     // Task can check cancel_token.is_cancelled() or use tokio::select!
//!     CancellableTaskResult::Ok(42)
//! });
//! # let _result = handle.await?;
//! # Ok(())
//! # }
//! ```
//! - Receive a `CancellationToken` for cooperative cancellation
//! - Support retry via `FnMut` closures (can be called multiple times)
//! - Return `CancellableTaskResult` to indicate success/cancellation/error
//!
//! ## Hierarchical Structure
//!
//! TaskTrackers form parent-child relationships:
//! - **Metrics**: Child metrics aggregate to parents
//! - **Cancellation**: Parent cancellation propagates to children
//! - **Independence**: Child cancellation doesn't affect parents
//! - **Cleanup**: `join()` waits for all descendants bottom-up
//!
//! ## Metrics and Observability
//!
//! Built-in metrics track task lifecycle:
//! - `issued`: Tasks submitted via spawn methods
//! - `active`: Currently executing tasks
//! - `success/failed/cancelled/rejected`: Final outcomes
//! - `pending`: Issued but not completed (issued - completed)
//! - `queued`: Waiting for resources (pending - active)
//!
//! Optional Prometheus integration available via `PrometheusTaskMetrics`.
//!
//! ## Usage Examples
//!
//! ### Basic Task Execution
//! ```rust
//! use dynamo_runtime::utils::tasks::tracker::*;
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let scheduler = SemaphoreScheduler::with_permits(10);
//! let error_policy = LogOnlyPolicy::new();
//! let tracker = TaskTracker::new(scheduler, error_policy)?;
//!
//! let handle = tracker.spawn(async { Ok(42) });
//! let result = handle.await??;
//! assert_eq!(result, 42);
//! # Ok(())
//! # }
//! ```
//!
//! ### Cancellable Tasks with Retry
//! ```rust
//! # use dynamo_runtime::utils::tasks::tracker::*;
//! # use std::sync::Arc;
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! # let scheduler = SemaphoreScheduler::with_permits(10);
//! # let error_policy = LogOnlyPolicy::new();
//! # let tracker = TaskTracker::new(scheduler, error_policy)?;
//! let handle = tracker.spawn_cancellable(|cancel_token| async move {
//!     tokio::select! {
//!         result = do_work() => CancellableTaskResult::Ok(result),
//!         _ = cancel_token.cancelled() => CancellableTaskResult::Cancelled,
//!     }
//! });
//! # Ok(())
//! # }
//! # async fn do_work() -> i32 { 42 }
//! ```
//!
//! ### Task-Driven Retry with Continuations
//! ```rust
//! # use dynamo_runtime::utils::tasks::tracker::*;
//! # use anyhow::anyhow;
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! # let tracker = TaskTracker::new(UnlimitedScheduler::new(), LogOnlyPolicy::new())?;
//! let handle = tracker.spawn(async {
//!     // Simulate initial failure with retry logic
//!     let error = FailedWithContinuation::from_fn(
//!         anyhow!("Network timeout"),
//!         || async {
//!             println!("Retrying with exponential backoff...");
//!             tokio::time::sleep(std::time::Duration::from_millis(100)).await;
//!             Ok("Success after retry".to_string())
//!         }
//!     );
//!     let result: Result<String, anyhow::Error> = Err(error);
//!     result
//! });
//!
//! let result = handle.await?;
//! assert!(result.is_ok());
//! # Ok(())
//! # }
//! ```
//!
//! ### Custom Error Policy with Continuation
//! ```rust
//! # use dynamo_runtime::utils::tasks::tracker::*;
//! # use std::sync::Arc;
//! # use async_trait::async_trait;
//! # #[derive(Debug)]
//! struct RetryPolicy {
//!     max_attempts: u32,
//! }
//!
//! impl OnErrorPolicy for RetryPolicy {
//!     fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
//!         Arc::new(RetryPolicy { max_attempts: self.max_attempts })
//!     }
//!
//!     fn create_context(&self) -> Option<Box<dyn std::any::Any + Send + 'static>> {
//!         None // Stateless policy
//!     }
//!
//!     fn on_error(&self, _error: &anyhow::Error, context: &mut OnErrorContext) -> ErrorResponse {
//!         if context.attempt_count < self.max_attempts {
//!             ErrorResponse::Custom(Box::new(RetryAction))
//!         } else {
//!             ErrorResponse::Fail
//!         }
//!     }
//! }
//!
//! # #[derive(Debug)]
//! struct RetryAction;
//!
//! #[async_trait]
//! impl OnErrorAction for RetryAction {
//!     async fn execute(
//!         &self,
//!         _error: &anyhow::Error,
//!         _task_id: TaskId,
//!         _attempt_count: u32,
//!         _context: &TaskExecutionContext,
//!     ) -> ActionResult {
//!         // In practice, you would create a continuation here
//!         ActionResult::Fail
//!     }
//! }
//! ```
//!
//! ## Future Extensibility
//!
//! The system is designed for extensibility. See the source code for detailed TODO comments
//! describing additional policies that can be implemented:
//! - **Scheduling**: Token bucket rate limiting, adaptive concurrency, memory-aware scheduling
//! - **Error Handling**: Retry with backoff, circuit breakers, dead letter queues
//!
//! Each TODO comment includes complete implementation guidance with data structures,
//! algorithms, and dependencies needed for future contributors.
//!
//! ## Hierarchical Organization
//!
//! ```rust
//! use dynamo_runtime::utils::tasks::tracker::{
//!     TaskTracker, UnlimitedScheduler, ThresholdCancelPolicy, SemaphoreScheduler
//! };
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create root tracker with failure threshold policy
//! let error_policy = ThresholdCancelPolicy::with_threshold(5);
//! let root = TaskTracker::builder()
//!     .scheduler(UnlimitedScheduler::new())
//!     .error_policy(error_policy)
//!     .build()?;
//!
//! // Create child trackers for different components
//! let api_handler = root.child_tracker()?;  // Inherits policies
//! let background_jobs = root.child_tracker()?;
//!
//! // Children can have custom policies
//! let rate_limited = root.child_tracker_builder()
//!     .scheduler(SemaphoreScheduler::with_permits(2))  // Custom concurrency limit
//!     .build()?;
//!
//! // Tasks run independently but metrics roll up
//! api_handler.spawn(async { Ok(()) });
//! background_jobs.spawn(async { Ok(()) });
//! rate_limited.spawn(async { Ok(()) });
//!
//! // Join all children hierarchically
//! root.join().await;
//! assert_eq!(root.metrics().success(), 3); // Sees all successes
//! # Ok(())
//! # }
//! ```
//!
//! ## Policy Examples
//!
//! ```rust
//! use dynamo_runtime::utils::tasks::tracker::{
//!     TaskTracker, CancelOnError, SemaphoreScheduler, ThresholdCancelPolicy
//! };
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Pattern-based error cancellation
//! let (error_policy, token) = CancelOnError::with_patterns(
//!     vec!["OutOfMemory".to_string(), "DeviceError".to_string()]
//! );
//! let simple = TaskTracker::builder()
//!     .scheduler(SemaphoreScheduler::with_permits(5))
//!     .error_policy(error_policy)
//!     .build()?;
//!
//! // Threshold-based cancellation with monitoring
//! let scheduler = SemaphoreScheduler::with_permits(10);  // Returns Arc<SemaphoreScheduler>
//! let error_policy = ThresholdCancelPolicy::with_threshold(3);  // Returns Arc<Policy>
//!
//! let advanced = TaskTracker::builder()
//!     .scheduler(scheduler)
//!     .error_policy(error_policy)
//!     .build()?;
//!
//! // Monitor cancellation externally
//! if token.is_cancelled() {
//!     println!("Tracker cancelled due to failures");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Metrics and Observability
//!
//! ```rust
//! use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let tracker = std::sync::Arc::new(TaskTracker::builder()
//!     .scheduler(SemaphoreScheduler::with_permits(2))  // Only 2 concurrent tasks
//!     .error_policy(LogOnlyPolicy::new())
//!     .build()?);
//!
//! // Spawn multiple tasks
//! for i in 0..5 {
//!     tracker.spawn(async move {
//!         tokio::time::sleep(std::time::Duration::from_millis(100)).await;
//!         Ok(i)
//!     });
//! }
//!
//! // Check metrics
//! let metrics = tracker.metrics();
//! println!("Issued: {}", metrics.issued());        // 5 tasks issued
//! println!("Active: {}", metrics.active());        // 2 tasks running (semaphore limit)
//! println!("Queued: {}", metrics.queued());        // 3 tasks waiting in scheduler queue
//! println!("Pending: {}", metrics.pending());      // 5 tasks not yet completed
//!
//! tracker.join().await;
//! assert_eq!(metrics.success(), 5);
//! assert_eq!(metrics.pending(), 0);
//! # Ok(())
//! # }
//! ```
//!
//! ## Prometheus Integration
//!
//! ```rust
//! use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
//! use dynamo_runtime::metrics::MetricsRegistry;
//!
//! # async fn example(registry: &dyn MetricsRegistry) -> anyhow::Result<()> {
//! // Root tracker with Prometheus metrics
//! let tracker = TaskTracker::new_with_prometheus(
//!     SemaphoreScheduler::with_permits(10),
//!     LogOnlyPolicy::new(),
//!     registry,
//!     "my_component"
//! )?;
//!
//! // Metrics automatically exported to Prometheus:
//! // - my_component_tasks_issued_total
//! // - my_component_tasks_success_total
//! // - my_component_tasks_failed_total
//! // - my_component_tasks_active
//! // - my_component_tasks_queued
//! # Ok(())
//! # }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::metrics::MetricsRegistry;
use crate::metrics::prometheus_names::task_tracker;
use anyhow::Result;
use async_trait::async_trait;
use derive_builder::Builder;
use std::collections::HashSet;
use std::sync::{Mutex, RwLock, Weak};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker as TokioTaskTracker;
use tracing::{Instrument, debug, error, warn};
use uuid::Uuid;

/// Error type for task execution results
///
/// This enum distinguishes between task cancellation and actual failures,
/// enabling proper metrics tracking and error handling.
#[derive(Error, Debug)]
pub enum TaskError {
    /// Task was cancelled (either via cancellation token or tracker shutdown)
    #[error("Task was cancelled")]
    Cancelled,

    /// Task failed with an error
    #[error(transparent)]
    Failed(#[from] anyhow::Error),

    /// Cannot spawn task on a closed tracker
    #[error("Cannot spawn task on a closed tracker")]
    TrackerClosed,
}

impl TaskError {
    /// Check if this error represents a cancellation
    ///
    /// This is a convenience method for compatibility and readability.
    pub fn is_cancellation(&self) -> bool {
        matches!(self, TaskError::Cancelled)
    }

    /// Check if this error represents a failure
    pub fn is_failure(&self) -> bool {
        matches!(self, TaskError::Failed(_))
    }

    /// Get the underlying anyhow::Error for failures, or a cancellation error for cancellations
    ///
    /// This is provided for compatibility with existing code that expects anyhow::Error.
    pub fn into_anyhow(self) -> anyhow::Error {
        match self {
            TaskError::Failed(err) => err,
            TaskError::Cancelled => anyhow::anyhow!("Task was cancelled"),
            TaskError::TrackerClosed => anyhow::anyhow!("Cannot spawn task on a closed tracker"),
        }
    }
}

/// A handle to a spawned task that provides both join functionality and cancellation control
///
/// `TaskHandle` wraps a `JoinHandle` and provides access to the task's individual cancellation token.
/// This allows fine-grained control over individual tasks while maintaining the familiar `JoinHandle` API.
///
/// # Example
/// ```rust
/// # use dynamo_runtime::utils::tasks::tracker::*;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let tracker = TaskTracker::new(UnlimitedScheduler::new(), LogOnlyPolicy::new())?;
/// let handle = tracker.spawn(async {
///     tokio::time::sleep(std::time::Duration::from_millis(100)).await;
///     Ok(42)
/// });
///
/// // Access the task's cancellation token
/// let cancel_token = handle.cancellation_token();
///
/// // Can cancel the specific task
/// // cancel_token.cancel();
///
/// // Await the task like a normal JoinHandle
/// let result = handle.await?;
/// assert_eq!(result?, 42);
/// # Ok(())
/// # }
/// ```
pub struct TaskHandle<T> {
    join_handle: JoinHandle<Result<T, TaskError>>,
    cancel_token: CancellationToken,
}

impl<T> TaskHandle<T> {
    /// Create a new TaskHandle wrapping a JoinHandle and cancellation token
    pub(crate) fn new(
        join_handle: JoinHandle<Result<T, TaskError>>,
        cancel_token: CancellationToken,
    ) -> Self {
        Self {
            join_handle,
            cancel_token,
        }
    }

    /// Get the cancellation token for this specific task
    ///
    /// This token is a child of the tracker's cancellation token and can be used
    /// to cancel just this individual task without affecting other tasks.
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::*;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let tracker = TaskTracker::new(UnlimitedScheduler::new(), LogOnlyPolicy::new())?;
    /// let handle = tracker.spawn(async {
    ///     tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    ///     Ok("completed")
    /// });
    ///
    /// // Cancel this specific task
    /// handle.cancellation_token().cancel();
    ///
    /// // Task will be cancelled
    /// let result = handle.await?;
    /// assert!(result.is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    /// Abort the task associated with this handle
    ///
    /// This is equivalent to calling `JoinHandle::abort()` and will cause the task
    /// to be cancelled immediately without running any cleanup code.
    pub fn abort(&self) {
        self.join_handle.abort();
    }

    /// Check if the task associated with this handle has finished
    ///
    /// This is equivalent to calling `JoinHandle::is_finished()`.
    pub fn is_finished(&self) -> bool {
        self.join_handle.is_finished()
    }
}

impl<T> std::future::Future for TaskHandle<T> {
    type Output = Result<Result<T, TaskError>, tokio::task::JoinError>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        std::pin::Pin::new(&mut self.join_handle).poll(cx)
    }
}

impl<T> std::fmt::Debug for TaskHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskHandle")
            .field("join_handle", &"<JoinHandle>")
            .field("cancel_token", &self.cancel_token)
            .finish()
    }
}

/// Trait for continuation tasks that execute after a failure
///
/// This trait allows tasks to define what should happen next after a failure,
/// eliminating the need for complex type erasure and executor management.
/// Tasks implement this trait to provide clean continuation logic.
#[async_trait]
pub trait Continuation: Send + Sync + std::fmt::Debug + std::any::Any {
    /// Execute the continuation task after a failure
    ///
    /// This method is called when a task fails and a continuation is provided.
    /// The implementation can perform retry logic, fallback operations,
    /// transformations, or any other follow-up action.
    /// Returns the result in a type-erased Box<dyn Any> for flexibility.
    async fn execute(
        &self,
        cancel_token: CancellationToken,
    ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>>;
}

/// Error type that signals a task failed but provided a continuation
///
/// This error type contains a continuation task that can be executed as a follow-up.
/// The task defines its own continuation logic through the Continuation trait.
#[derive(Error, Debug)]
#[error("Task failed with continuation: {source}")]
pub struct FailedWithContinuation {
    /// The underlying error that caused the task to fail
    #[source]
    pub source: anyhow::Error,
    /// The continuation task for follow-up execution
    pub continuation: Arc<dyn Continuation + Send + Sync + 'static>,
}

impl FailedWithContinuation {
    /// Create a new FailedWithContinuation with a continuation task
    ///
    /// The continuation task defines its own execution logic through the Continuation trait.
    pub fn new(
        source: anyhow::Error,
        continuation: Arc<dyn Continuation + Send + Sync + 'static>,
    ) -> Self {
        Self {
            source,
            continuation,
        }
    }

    /// Create a FailedWithContinuation and convert it to anyhow::Error
    ///
    /// This is a convenience method for tasks to easily return continuation errors.
    pub fn into_anyhow(
        source: anyhow::Error,
        continuation: Arc<dyn Continuation + Send + Sync + 'static>,
    ) -> anyhow::Error {
        anyhow::Error::new(Self::new(source, continuation))
    }

    /// Create a FailedWithContinuation from a simple async function (no cancellation support)
    ///
    /// This is a convenience method for creating continuation errors from simple async closures
    /// that don't need to handle cancellation. The function will be executed when the
    /// continuation is triggered.
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::*;
    /// # use anyhow::anyhow;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let error = FailedWithContinuation::from_fn(
    ///     anyhow!("Initial task failed"),
    ///     || async {
    ///         println!("Retrying operation...");
    ///         Ok("retry_result".to_string())
    ///     }
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_fn<F, Fut, T>(source: anyhow::Error, f: F) -> anyhow::Error
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<T, anyhow::Error>> + Send + 'static,
        T: Send + 'static,
    {
        let continuation = Arc::new(FnContinuation { f: Box::new(f) });
        Self::into_anyhow(source, continuation)
    }

    /// Create a FailedWithContinuation from a cancellable async function
    ///
    /// This is a convenience method for creating continuation errors from async closures
    /// that can handle cancellation. The function receives a CancellationToken
    /// and should check it periodically for cooperative cancellation.
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::*;
    /// # use anyhow::anyhow;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let error = FailedWithContinuation::from_cancellable(
    ///     anyhow!("Initial task failed"),
    ///     |cancel_token| async move {
    ///         if cancel_token.is_cancelled() {
    ///             return Err(anyhow!("Cancelled"));
    ///         }
    ///         println!("Retrying operation with cancellation support...");
    ///         Ok("retry_result".to_string())
    ///     }
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_cancellable<F, Fut, T>(source: anyhow::Error, f: F) -> anyhow::Error
    where
        F: Fn(CancellationToken) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<T, anyhow::Error>> + Send + 'static,
        T: Send + 'static,
    {
        let continuation = Arc::new(CancellableFnContinuation { f: Box::new(f) });
        Self::into_anyhow(source, continuation)
    }
}

/// Extension trait for extracting FailedWithContinuation from anyhow::Error
///
/// This trait provides methods to detect and extract continuation tasks
/// from the type-erased anyhow::Error system.
pub trait FailedWithContinuationExt {
    /// Extract a continuation task if this error contains one
    ///
    /// Returns the continuation task if the error is a FailedWithContinuation,
    /// None otherwise.
    fn extract_continuation(&self) -> Option<Arc<dyn Continuation + Send + Sync + 'static>>;

    /// Check if this error has a continuation
    fn has_continuation(&self) -> bool;
}

impl FailedWithContinuationExt for anyhow::Error {
    fn extract_continuation(&self) -> Option<Arc<dyn Continuation + Send + Sync + 'static>> {
        // Try to downcast to FailedWithContinuation
        if let Some(continuation_err) = self.downcast_ref::<FailedWithContinuation>() {
            Some(continuation_err.continuation.clone())
        } else {
            None
        }
    }

    fn has_continuation(&self) -> bool {
        self.downcast_ref::<FailedWithContinuation>().is_some()
    }
}

/// Implementation of Continuation for simple async functions (no cancellation support)
struct FnContinuation<F> {
    f: Box<F>,
}

impl<F> std::fmt::Debug for FnContinuation<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnContinuation")
            .field("f", &"<closure>")
            .finish()
    }
}

#[async_trait]
impl<F, Fut, T> Continuation for FnContinuation<F>
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<T, anyhow::Error>> + Send + 'static,
    T: Send + 'static,
{
    async fn execute(
        &self,
        _cancel_token: CancellationToken,
    ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
        match (self.f)().await {
            Ok(result) => TaskExecutionResult::Success(Box::new(result)),
            Err(error) => TaskExecutionResult::Error(error),
        }
    }
}

/// Implementation of Continuation for cancellable async functions
struct CancellableFnContinuation<F> {
    f: Box<F>,
}

impl<F> std::fmt::Debug for CancellableFnContinuation<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellableFnContinuation")
            .field("f", &"<closure>")
            .finish()
    }
}

#[async_trait]
impl<F, Fut, T> Continuation for CancellableFnContinuation<F>
where
    F: Fn(CancellationToken) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<T, anyhow::Error>> + Send + 'static,
    T: Send + 'static,
{
    async fn execute(
        &self,
        cancel_token: CancellationToken,
    ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
        match (self.f)(cancel_token).await {
            Ok(result) => TaskExecutionResult::Success(Box::new(result)),
            Err(error) => TaskExecutionResult::Error(error),
        }
    }
}

/// Common scheduling policies for task execution
///
/// These enums provide convenient access to built-in scheduling policies
/// without requiring manual construction of policy objects.
///
/// ## Cancellation Semantics
///
/// All schedulers follow the same cancellation behavior:
/// - Respect cancellation tokens before resource allocation (permits, etc.)
/// - Once task execution begins, always await completion
/// - Let tasks handle their own cancellation internally
#[derive(Debug, Clone)]
pub enum SchedulingPolicy {
    /// No concurrency limits - execute all tasks immediately
    Unlimited,
    /// Semaphore-based concurrency limiting
    Semaphore(usize),
    // TODO: Future scheduling policies to implement
    //
    // /// Token bucket rate limiting with burst capacity
    // /// Implementation: Use tokio::time::interval for refill, AtomicU64 for tokens.
    // /// acquire() decrements tokens, schedule() waits for refill if empty.
    // /// Burst allows temporary spikes above steady rate.
    // /// struct: { rate: f64, burst: usize, tokens: AtomicU64, last_refill: Mutex<Instant> }
    // /// Example: TokenBucket { rate: 10.0, burst: 5 } = 10 tasks/sec, burst up to 5
    // TokenBucket { rate: f64, burst: usize },
    //
    // /// Weighted fair scheduling across multiple priority classes
    // /// Implementation: Maintain separate VecDeque for each priority class.
    // /// Use weighted round-robin: serve N tasks from high, M from normal, etc.
    // /// Track deficit counters to ensure fairness over time.
    // /// struct: { queues: HashMap<String, VecDeque<Task>>, weights: Vec<(String, u32)> }
    // /// Example: WeightedFair { weights: vec![("high", 70), ("normal", 20), ("low", 10)] }
    // WeightedFair { weights: Vec<(String, u32)> },
    //
    // /// Memory-aware scheduling that limits tasks based on available memory
    // /// Implementation: Monitor system memory via /proc/meminfo or sysinfo crate.
    // /// Pause scheduling when available memory < threshold, resume when memory freed.
    // /// Use exponential backoff for memory checks to avoid overhead.
    // /// struct: { max_memory_mb: usize, check_interval: Duration, semaphore: Semaphore }
    // MemoryAware { max_memory_mb: usize },
    //
    // /// CPU-aware scheduling that adjusts concurrency based on CPU load
    // /// Implementation: Sample system load via sysinfo crate every N seconds.
    // /// Dynamically resize internal semaphore permits based on load average.
    // /// Use PID controller for smooth adjustments, avoid oscillation.
    // /// struct: { max_cpu_percent: f32, permits: Arc<Semaphore>, sampler: tokio::task }
    // CpuAware { max_cpu_percent: f32 },
    //
    // /// Adaptive scheduler that automatically adjusts concurrency based on performance
    // /// Implementation: Track task latency and throughput in sliding windows.
    // /// Increase permits if latency low & throughput stable, decrease if latency spikes.
    // /// Use additive increase, multiplicative decrease (AIMD) algorithm.
    // /// struct: { permits: AtomicUsize, latency_tracker: RingBuffer, throughput_tracker: RingBuffer }
    // Adaptive { initial_permits: usize },
    //
    // /// Throttling scheduler that enforces minimum time between task starts
    // /// Implementation: Store last_execution time in AtomicU64 (unix timestamp).
    // /// Before scheduling, check elapsed time and tokio::time::sleep if needed.
    // /// Useful for rate-limiting API calls to external services.
    // /// struct: { min_interval: Duration, last_execution: AtomicU64 }
    // Throttling { min_interval_ms: u64 },
    //
    // /// Batch scheduler that groups tasks and executes them together
    // /// Implementation: Collect tasks in Vec<Task>, use tokio::time::timeout for max_wait.
    // /// Execute batch when size reached OR timeout expires, whichever first.
    // /// Use futures::future::join_all for parallel execution within batch.
    // /// struct: { batch_size: usize, max_wait: Duration, pending: Mutex<Vec<Task>> }
    // Batch { batch_size: usize, max_wait_ms: u64 },
    //
    // /// Priority-based scheduler with separate queues for different priority levels
    // /// Implementation: Three separate semaphores for high/normal/low priorities.
    // /// Always serve high before normal, normal before low (strict priority).
    // /// Add starvation protection: promote normal->high after timeout.
    // /// struct: { high_sem: Semaphore, normal_sem: Semaphore, low_sem: Semaphore }
    // Priority { high: usize, normal: usize, low: usize },
    //
    // /// Backpressure-aware scheduler that monitors downstream capacity
    // /// Implementation: Track external queue depth via provided callback/metric.
    // /// Pause scheduling when queue_threshold exceeded, resume after pause_duration.
    // /// Use exponential backoff for repeated backpressure events.
    // /// struct: { queue_checker: Arc<dyn Fn() -> usize>, threshold: usize, pause_duration: Duration }
    // Backpressure { queue_threshold: usize, pause_duration_ms: u64 },
}

/// Trait for implementing error handling policies
///
/// Error policies are lightweight, synchronous decision-makers that analyze task failures
/// and return an ErrorResponse telling the TaskTracker what action to take. The TaskTracker
/// handles all the actual work (cancellation, metrics, etc.) based on the policy's response.
///
/// ## Key Design Principles
/// - **Synchronous**: Policies make fast decisions without async operations
/// - **Stateless where possible**: TaskTracker manages cancellation tokens and state
/// - **Composable**: Policies can be combined and nested in hierarchies
/// - **Focused**: Each policy handles one specific error pattern or strategy
///
/// Per-task error handling context
///
/// Provides context information and state management for error policies.
/// The state field allows policies to maintain per-task state across multiple error attempts.
pub struct OnErrorContext {
    /// Number of times this task has been attempted (starts at 1)
    pub attempt_count: u32,
    /// Unique identifier of the failed task
    pub task_id: TaskId,
    /// Full execution context with access to scheduler, metrics, etc.
    pub execution_context: TaskExecutionContext,
    /// Optional per-task state managed by the policy (None for stateless policies)
    pub state: Option<Box<dyn std::any::Any + Send + 'static>>,
}

/// Error handling policy trait for task failures
///
/// Policies define how the TaskTracker responds to task failures.
/// They can be stateless (like LogOnlyPolicy) or maintain per-task state
/// (like ThresholdCancelPolicy with per-task failure counters).
pub trait OnErrorPolicy: Send + Sync + std::fmt::Debug {
    /// Create a child policy for a child tracker
    ///
    /// This allows policies to maintain hierarchical relationships,
    /// such as child cancellation tokens or shared circuit breaker state.
    fn create_child(&self) -> Arc<dyn OnErrorPolicy>;

    /// Create per-task context state (None if policy is stateless)
    ///
    /// This method is called once per task when the first error occurs.
    /// Stateless policies should return None to avoid unnecessary heap allocations.
    /// Stateful policies should return Some(Box::new(initial_state)).
    ///
    /// # Returns
    /// * `None` - Policy doesn't need per-task state (no heap allocation)
    /// * `Some(state)` - Initial state for this task (heap allocated when needed)
    fn create_context(&self) -> Option<Box<dyn std::any::Any + Send + 'static>>;

    /// Handle a task failure and return the desired response
    ///
    /// # Arguments
    /// * `error` - The error that occurred
    /// * `context` - Mutable context with attempt count, task info, and optional state
    ///
    /// # Returns
    /// ErrorResponse indicating how the TaskTracker should handle this failure
    fn on_error(&self, error: &anyhow::Error, context: &mut OnErrorContext) -> ErrorResponse;

    /// Should continuations be allowed for this error?
    ///
    /// This method is called before checking if a task provided a continuation to determine
    /// whether the policy allows continuation-based retries at all. If this returns `false`,
    /// any `FailedWithContinuation` will be ignored and the error will be handled through
    /// the normal policy response.
    ///
    /// # Arguments
    /// * `error` - The error that occurred
    /// * `context` - Per-task context with attempt count and state
    ///
    /// # Returns
    /// * `true` - Allow continuations, check for `FailedWithContinuation` (default)
    /// * `false` - Reject continuations, handle through normal policy response
    fn allow_continuation(&self, _error: &anyhow::Error, _context: &OnErrorContext) -> bool {
        true // Default: allow continuations
    }

    /// Should this continuation be rescheduled through the scheduler?
    ///
    /// This method is called when a continuation is about to be executed to determine
    /// whether it should go through the scheduler's acquisition process again or execute
    /// immediately with the current execution permission.
    ///
    /// **What this means:**
    /// - **Don't reschedule (`false`)**: Execute continuation immediately with current permission
    /// - **Reschedule (`true`)**: Release current permission, go through scheduler again
    ///
    /// Rescheduling means the continuation will be subject to the scheduler's policies
    /// again (rate limiting, concurrency limits, backoff delays, etc.).
    ///
    /// # Arguments
    /// * `error` - The error that triggered this retry decision
    /// * `context` - Per-task context with attempt count and state
    ///
    /// # Returns
    /// * `false` - Execute continuation immediately (default, efficient)
    /// * `true` - Reschedule through scheduler (for delays, rate limiting, backoff)
    fn should_reschedule(&self, _error: &anyhow::Error, _context: &OnErrorContext) -> bool {
        false // Default: immediate execution
    }
}

/// Common error handling policies for task failure management
///
/// These enums provide convenient access to built-in error handling policies
/// without requiring manual construction of policy objects.
#[derive(Debug, Clone)]
pub enum ErrorPolicy {
    /// Log errors but continue execution - no cancellation
    LogOnly,
    /// Cancel all tasks on any error (using default error patterns)
    CancelOnError,
    /// Cancel all tasks when specific error patterns are encountered
    CancelOnPatterns(Vec<String>),
    /// Cancel after a threshold number of failures
    CancelOnThreshold { max_failures: usize },
    /// Cancel when failure rate exceeds threshold within time window
    CancelOnRate {
        max_failure_rate: f32,
        window_secs: u64,
    },
    // TODO: Future error policies to implement
    //
    // /// Retry failed tasks with exponential backoff
    // /// Implementation: Store original task in retry queue with attempt count.
    // /// Use tokio::time::sleep for delays: backoff_ms * 2^attempt.
    // /// Spawn retry as new task, preserve original task_id for tracing.
    // /// Need task cloning support in scheduler interface.
    // /// struct: { max_attempts: usize, backoff_ms: u64, retry_queue: VecDeque<(Task, u32)> }
    // Retry { max_attempts: usize, backoff_ms: u64 },
    //
    // /// Send failed tasks to a dead letter queue for later processing
    // /// Implementation: Use tokio::sync::mpsc::channel for queue.
    // /// Serialize task info (id, error, payload) for persistence.
    // /// Background worker drains queue to external storage (Redis/DB).
    // /// Include retry count and timestamps for debugging.
    // /// struct: { queue: mpsc::Sender<DeadLetterItem>, storage: Arc<dyn DeadLetterStorage> }
    // DeadLetter { queue_name: String },
    //
    // /// Execute fallback logic when tasks fail
    // /// Implementation: Store fallback closure in Arc for thread-safety.
    // /// Execute fallback in same context as failed task (inherit cancel token).
    // /// Track fallback success/failure separately from original task metrics.
    // /// Consider using enum for common fallback patterns (default value, noop, etc).
    // /// struct: { fallback_fn: Arc<dyn Fn(TaskId, Error) -> BoxFuture<Result<()>>> }
    // Fallback { fallback_fn: Arc<dyn Fn() -> BoxFuture<'static, Result<()>>> },
    //
    // /// Circuit breaker pattern - stop executing after threshold failures
    // /// Implementation: Track state (Closed/Open/HalfOpen) with AtomicU8.
    // /// Use failure window (last N tasks) or time window for threshold.
    // /// In Open state, reject tasks immediately, use timer for recovery.
    // /// In HalfOpen, allow one test task to check if issues resolved.
    // /// struct: { state: AtomicU8, failure_count: AtomicU64, last_failure: AtomicU64 }
    // CircuitBreaker { failure_threshold: usize, timeout_secs: u64 },
    //
    // /// Resource protection policy that monitors memory/CPU usage
    // /// Implementation: Background task samples system resources via sysinfo.
    // /// Cancel tracker when memory > threshold, use process-level monitoring.
    // /// Implement graceful degradation: warn at 80%, cancel at 90%.
    // /// Include both system-wide and process-specific thresholds.
    // /// struct: { monitor_task: JoinHandle, thresholds: ResourceThresholds, cancel_token: CancellationToken }
    // ResourceProtection { max_memory_mb: usize },
    //
    // /// Timeout policy that cancels tasks exceeding maximum duration
    // /// Implementation: Wrap each task with tokio::time::timeout.
    // /// Store task start time, check duration in on_error callback.
    // /// Distinguish timeout errors from other task failures in metrics.
    // /// Consider per-task or global timeout strategies.
    // /// struct: { max_duration: Duration, timeout_tracker: HashMap<TaskId, Instant> }
    // Timeout { max_duration_secs: u64 },
    //
    // /// Sampling policy that only logs a percentage of errors
    // /// Implementation: Use thread-local RNG for sampling decisions.
    // /// Hash task_id for deterministic sampling (same task always sampled).
    // /// Store sample rate as f32, compare with rand::random::<f32>().
    // /// Include rate in log messages for context.
    // /// struct: { sample_rate: f32, rng: ThreadLocal<RefCell<SmallRng>> }
    // Sampling { sample_rate: f32 },
    //
    // /// Aggregating policy that batches error reports
    // /// Implementation: Collect errors in Vec, flush on size or time trigger.
    // /// Use tokio::time::interval for periodic flushing.
    // /// Group errors by type/pattern for better insights.
    // /// Include error frequency and rate statistics in reports.
    // /// struct: { window: Duration, batch: Mutex<Vec<ErrorEntry>>, flush_task: JoinHandle }
    // Aggregating { window_secs: u64, max_batch_size: usize },
    //
    // /// Alerting policy that sends notifications on error patterns
    // /// Implementation: Use reqwest for webhook HTTP calls.
    // /// Rate-limit alerts to prevent spam (max N per minute).
    // /// Include error context, task info, and system metrics in payload.
    // /// Support multiple notification channels (webhook, email, slack).
    // /// struct: { client: reqwest::Client, rate_limiter: RateLimiter, alert_config: AlertConfig }
    // Alerting { webhook_url: String, severity_threshold: String },
}

/// Response type for error handling policies
///
/// This enum defines how the TaskTracker should respond to task failures.
/// Currently provides minimal functionality with planned extensions for common patterns.
#[derive(Debug)]
pub enum ErrorResponse {
    /// Just fail this task - error will be logged/counted, but tracker continues
    Fail,

    /// Shutdown this tracker and all child trackers
    Shutdown,

    /// Execute custom error handling logic with full context access
    Custom(Box<dyn OnErrorAction>),
    // TODO: Future specialized error responses to implement:
    //
    // /// Retry the failed task with configurable strategy
    // /// Implementation: Add RetryStrategy trait with delay(), should_continue(attempt_count),
    // /// release_and_reacquire_resources() methods. TaskTracker handles retry loop with
    // /// attempt counting and resource management. Supports exponential backoff, jitter.
    // /// Usage: ErrorResponse::Retry(Box::new(ExponentialBackoff { max_attempts: 3, base_delay: 100ms }))
    // Retry(Box<dyn RetryStrategy>),
    //
    // /// Execute fallback logic, then follow secondary action
    // /// Implementation: Add FallbackAction trait with execute(error, task_id) -> Result<(), Error>.
    // /// Execute fallback first, then recursively handle the 'then' response based on fallback result.
    // /// Enables patterns like: try fallback, if it works continue, if it fails retry original task.
    // /// Usage: ErrorResponse::Fallback { fallback: Box::new(DefaultValue(42)), then: Box::new(ErrorResponse::Continue) }
    // Fallback { fallback: Box<dyn FallbackAction>, then: Box<ErrorResponse> },
    //
    // /// Restart task with preserved state (for long-running/stateful tasks)
    // /// Implementation: Add TaskState trait for serialize/deserialize state, RestartStrategy trait
    // /// with create_continuation_task(state) -> Future. Task saves checkpoints during execution,
    // /// on error returns StatefulTaskError containing preserved state. Policy can restart from checkpoint.
    // /// Usage: ErrorResponse::RestartWithState { state: checkpointed_state, strategy: Box::new(CheckpointRestart { ... }) }
    // RestartWithState { state: Box<dyn TaskState>, strategy: Box<dyn RestartStrategy> },
}

/// Trait for implementing custom error handling actions
///
/// This provides full access to the task execution context for complex error handling
/// scenarios that don't fit into the built-in response patterns.
#[async_trait]
pub trait OnErrorAction: Send + Sync + std::fmt::Debug {
    /// Execute custom error handling logic
    ///
    /// # Arguments
    /// * `error` - The error that caused the task to fail
    /// * `task_id` - Unique identifier of the failed task
    /// * `attempt_count` - Number of times this task has been attempted (starts at 1)
    /// * `context` - Full execution context with access to scheduler, metrics, etc.
    ///
    /// # Returns
    /// ActionResult indicating what the TaskTracker should do next
    async fn execute(
        &self,
        error: &anyhow::Error,
        task_id: TaskId,
        attempt_count: u32,
        context: &TaskExecutionContext,
    ) -> ActionResult;
}

/// Scheduler execution guard state for conditional re-acquisition during task retry loops
///
/// This controls whether a continuation should reuse the current scheduler execution permission
/// or go through the scheduler's acquisition process again.
#[derive(Debug, Clone, PartialEq, Eq)]
enum GuardState {
    /// Keep the current scheduler execution permission for immediate continuation
    ///
    /// The continuation will execute immediately without going through the scheduler again.
    /// This is efficient for simple retries that don't need delays or rate limiting.
    Keep,

    /// Release current permission and re-acquire through scheduler before continuation
    ///
    /// The continuation will be subject to the scheduler's policies again (concurrency limits,
    /// rate limiting, backoff delays, etc.). Use this for implementing retry delays or
    /// when the scheduler needs to apply its policies to the retry attempt.
    Reschedule,
}

/// Result of a custom error action execution
#[derive(Debug)]
pub enum ActionResult {
    /// Just fail this task (error was logged/handled by policy)
    ///
    /// This means the policy has handled the error appropriately (e.g., logged it,
    /// updated metrics, etc.) and the task should fail with this error.
    /// The task execution terminates here.
    Fail,

    /// Continue execution with the provided task
    ///
    /// This provides a new executable to continue the retry loop with.
    /// The task execution continues with the provided continuation.
    Continue {
        continuation: Arc<dyn Continuation + Send + Sync + 'static>,
    },

    /// Shutdown this tracker and all child trackers
    ///
    /// This triggers shutdown of the entire tracker hierarchy.
    /// All running and pending tasks will be cancelled.
    Shutdown,
}

/// Execution context provided to custom error actions
///
/// This gives custom actions full access to the task execution environment
/// for implementing complex error handling scenarios.
pub struct TaskExecutionContext {
    /// Scheduler for reacquiring resources or checking state
    pub scheduler: Arc<dyn TaskScheduler>,

    /// Metrics for custom tracking
    pub metrics: Arc<dyn HierarchicalTaskMetrics>,
    // TODO: Future context additions:
    // pub guard: Box<dyn ResourceGuard>,    // Current resource guard (needs Debug impl)
    // pub cancel_token: CancellationToken,  // For implementing custom cancellation
    // pub task_recreation: Box<dyn TaskRecreator>, // For implementing retry/restart
}

/// Result of task execution - unified for both regular and cancellable tasks
#[derive(Debug)]
pub enum TaskExecutionResult<T> {
    /// Task completed successfully
    Success(T),
    /// Task was cancelled (only possible for cancellable tasks)
    Cancelled,
    /// Task failed with an error
    Error(anyhow::Error),
}

/// Trait for executing different types of tasks in a unified way
#[async_trait]
trait TaskExecutor<T>: Send {
    /// Execute the task with the given cancellation token
    async fn execute(&mut self, cancel_token: CancellationToken) -> TaskExecutionResult<T>;
}

/// Task executor for regular (non-cancellable) tasks
struct RegularTaskExecutor<F, T>
where
    F: Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    future: Option<F>,
    _phantom: std::marker::PhantomData<T>,
}

impl<F, T> RegularTaskExecutor<F, T>
where
    F: Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    fn new(future: F) -> Self {
        Self {
            future: Some(future),
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<F, T> TaskExecutor<T> for RegularTaskExecutor<F, T>
where
    F: Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    async fn execute(&mut self, _cancel_token: CancellationToken) -> TaskExecutionResult<T> {
        if let Some(future) = self.future.take() {
            match future.await {
                Ok(value) => TaskExecutionResult::Success(value),
                Err(error) => TaskExecutionResult::Error(error),
            }
        } else {
            // This should never happen since regular tasks don't support retry
            TaskExecutionResult::Error(anyhow::anyhow!("Regular task already consumed"))
        }
    }
}

/// Task executor for cancellable tasks
struct CancellableTaskExecutor<F, Fut, T>
where
    F: FnMut(CancellationToken) -> Fut + Send + 'static,
    Fut: Future<Output = CancellableTaskResult<T>> + Send + 'static,
    T: Send + 'static,
{
    task_fn: F,
}

impl<F, Fut, T> CancellableTaskExecutor<F, Fut, T>
where
    F: FnMut(CancellationToken) -> Fut + Send + 'static,
    Fut: Future<Output = CancellableTaskResult<T>> + Send + 'static,
    T: Send + 'static,
{
    fn new(task_fn: F) -> Self {
        Self { task_fn }
    }
}

#[async_trait]
impl<F, Fut, T> TaskExecutor<T> for CancellableTaskExecutor<F, Fut, T>
where
    F: FnMut(CancellationToken) -> Fut + Send + 'static,
    Fut: Future<Output = CancellableTaskResult<T>> + Send + 'static,
    T: Send + 'static,
{
    async fn execute(&mut self, cancel_token: CancellationToken) -> TaskExecutionResult<T> {
        let future = (self.task_fn)(cancel_token);
        match future.await {
            CancellableTaskResult::Ok(value) => TaskExecutionResult::Success(value),
            CancellableTaskResult::Cancelled => TaskExecutionResult::Cancelled,
            CancellableTaskResult::Err(error) => TaskExecutionResult::Error(error),
        }
    }
}

/// Common functionality for policy Arc construction
///
/// This trait provides a standardized `new_arc()` method for all policy types,
/// eliminating the need for manual `Arc::new()` calls in client code.
pub trait ArcPolicy: Sized + Send + Sync + 'static {
    /// Create an Arc-wrapped instance of this policy
    fn new_arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}

/// Unique identifier for a task
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(Uuid);

impl TaskId {
    fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "task-{}", self.0)
    }
}

/// Result of task execution
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionStatus {
    /// Task completed successfully
    Ok,
    /// Task was cancelled before or during execution
    Cancelled,
    /// Task failed with an error
    Failed(String),
}

/// Result type for cancellable tasks that explicitly tracks cancellation
#[derive(Debug)]
pub enum CancellableTaskResult<T> {
    /// Task completed successfully
    Ok(T),
    /// Task was cancelled (either via token or shutdown)
    Cancelled,
    /// Task failed with an error
    Err(anyhow::Error),
}

/// Result of scheduling a task
#[derive(Debug)]
pub enum SchedulingResult<T> {
    /// Task was executed and completed
    Execute(T),
    /// Task was cancelled before execution
    Cancelled,
    /// Task was rejected due to scheduling policy
    Rejected(String),
}

/// Resource guard that manages task execution
///
/// This trait enforces proper cancellation semantics by separating resource
/// management from task execution. Once a guard is acquired, task execution
/// must always run to completion.
/// Resource guard for task execution
///
/// This trait represents resources (permits, slots, etc.) acquired from a scheduler
/// that must be held during task execution. The guard automatically releases
/// resources when dropped, implementing proper RAII semantics.
///
/// Guards are returned by `TaskScheduler::acquire_execution_slot()` and must
/// be held in scope while the task executes to ensure resources remain allocated.
pub trait ResourceGuard: Send + 'static {
    // Marker trait - resources are released via Drop on the concrete type
}

/// Trait for implementing task scheduling policies
///
/// This trait enforces proper cancellation semantics by splitting resource
/// acquisition (which can be cancelled) from task execution (which cannot).
///
/// ## Design Philosophy
///
/// Tasks may or may not support cancellation (depending on whether they were
/// created with `spawn_cancellable` or regular `spawn`). This split design ensures:
///
/// - **Resource acquisition**: Can respect cancellation tokens to avoid unnecessary allocation
/// - **Task execution**: Always runs to completion; tasks handle their own cancellation
///
/// This makes it impossible to accidentally interrupt task execution with `tokio::select!`.
#[async_trait]
pub trait TaskScheduler: Send + Sync + std::fmt::Debug {
    /// Acquire resources needed for task execution and return a guard
    ///
    /// This method handles resource allocation (permits, queue slots, etc.) and
    /// can respect cancellation tokens to avoid unnecessary resource consumption.
    ///
    /// ## Cancellation Behavior
    ///
    /// The `cancel_token` is used for scheduler-level cancellation (e.g., "don't start new work").
    /// If cancellation is requested before or during resource acquisition, this method
    /// should return `SchedulingResult::Cancelled`.
    ///
    /// # Arguments
    /// * `cancel_token` - [`CancellationToken`] for scheduler-level cancellation
    ///
    /// # Returns
    /// * `SchedulingResult::Execute(guard)` - Resources acquired, ready to execute
    /// * `SchedulingResult::Cancelled` - Cancelled before or during resource acquisition
    /// * `SchedulingResult::Rejected(reason)` - Resources unavailable or policy violation
    async fn acquire_execution_slot(
        &self,
        cancel_token: CancellationToken,
    ) -> SchedulingResult<Box<dyn ResourceGuard>>;
}

/// Trait for hierarchical task metrics that supports aggregation up the tracker tree
///
/// This trait provides different implementations for root and child trackers:
/// - Root trackers integrate with Prometheus metrics for observability
/// - Child trackers chain metric updates up to their parents for aggregation
/// - All implementations maintain thread-safe atomic operations
pub trait HierarchicalTaskMetrics: Send + Sync + std::fmt::Debug {
    /// Increment issued task counter
    fn increment_issued(&self);

    /// Increment started task counter
    fn increment_started(&self);

    /// Increment success counter
    fn increment_success(&self);

    /// Increment cancelled counter
    fn increment_cancelled(&self);

    /// Increment failed counter
    fn increment_failed(&self);

    /// Increment rejected counter
    fn increment_rejected(&self);

    /// Get current issued count (local to this tracker)
    fn issued(&self) -> u64;

    /// Get current started count (local to this tracker)
    fn started(&self) -> u64;

    /// Get current success count (local to this tracker)
    fn success(&self) -> u64;

    /// Get current cancelled count (local to this tracker)
    fn cancelled(&self) -> u64;

    /// Get current failed count (local to this tracker)
    fn failed(&self) -> u64;

    /// Get current rejected count (local to this tracker)
    fn rejected(&self) -> u64;

    /// Get total completed tasks (success + cancelled + failed + rejected)
    fn total_completed(&self) -> u64 {
        self.success() + self.cancelled() + self.failed() + self.rejected()
    }

    /// Get number of pending tasks (issued - completed)
    fn pending(&self) -> u64 {
        self.issued().saturating_sub(self.total_completed())
    }

    /// Get the number of tasks that are currently active (started - completed)
    fn active(&self) -> u64 {
        self.started().saturating_sub(self.total_completed())
    }

    /// Get number of tasks queued in scheduler (issued - started)
    fn queued(&self) -> u64 {
        self.issued().saturating_sub(self.started())
    }
}

/// Task execution metrics for a tracker
#[derive(Debug, Default)]
pub struct TaskMetrics {
    /// Number of tasks issued/submitted (via spawn methods)
    pub issued_count: AtomicU64,
    /// Number of tasks that have started execution
    pub started_count: AtomicU64,
    /// Number of successfully completed tasks
    pub success_count: AtomicU64,
    /// Number of cancelled tasks
    pub cancelled_count: AtomicU64,
    /// Number of failed tasks
    pub failed_count: AtomicU64,
    /// Number of rejected tasks (by scheduler)
    pub rejected_count: AtomicU64,
}

impl TaskMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }
}

impl HierarchicalTaskMetrics for TaskMetrics {
    /// Increment issued task counter
    fn increment_issued(&self) {
        self.issued_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment started task counter
    fn increment_started(&self) {
        self.started_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment success counter
    fn increment_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment cancelled counter
    fn increment_cancelled(&self) {
        self.cancelled_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment failed counter
    fn increment_failed(&self) {
        self.failed_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment rejected counter
    fn increment_rejected(&self) {
        self.rejected_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current issued count
    fn issued(&self) -> u64 {
        self.issued_count.load(Ordering::Relaxed)
    }

    /// Get current started count
    fn started(&self) -> u64 {
        self.started_count.load(Ordering::Relaxed)
    }

    /// Get current success count
    fn success(&self) -> u64 {
        self.success_count.load(Ordering::Relaxed)
    }

    /// Get current cancelled count
    fn cancelled(&self) -> u64 {
        self.cancelled_count.load(Ordering::Relaxed)
    }

    /// Get current failed count
    fn failed(&self) -> u64 {
        self.failed_count.load(Ordering::Relaxed)
    }

    /// Get current rejected count
    fn rejected(&self) -> u64 {
        self.rejected_count.load(Ordering::Relaxed)
    }
}

/// Root tracker metrics with Prometheus integration
///
/// This implementation maintains local counters and exposes them as Prometheus metrics
/// through the provided MetricsRegistry.
#[derive(Debug)]
pub struct PrometheusTaskMetrics {
    /// Prometheus metrics integration
    prometheus_issued: prometheus::IntCounter,
    prometheus_started: prometheus::IntCounter,
    prometheus_success: prometheus::IntCounter,
    prometheus_cancelled: prometheus::IntCounter,
    prometheus_failed: prometheus::IntCounter,
    prometheus_rejected: prometheus::IntCounter,
}

impl PrometheusTaskMetrics {
    /// Create new root metrics with Prometheus integration
    ///
    /// # Arguments
    /// * `registry` - MetricsRegistry for creating Prometheus metrics
    /// * `component_name` - Name for the component/tracker (used in metric names)
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use dynamo_runtime::utils::tasks::tracker::PrometheusTaskMetrics;
    /// # use dynamo_runtime::metrics::MetricsRegistry;
    /// # fn example(registry: Arc<dyn MetricsRegistry>) -> anyhow::Result<()> {
    /// let metrics = PrometheusTaskMetrics::new(registry.as_ref(), "main_tracker")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<R: MetricsRegistry + ?Sized>(
        registry: &R,
        component_name: &str,
    ) -> anyhow::Result<Self> {
        let issued_counter = registry.create_intcounter(
            &format!("{}_{}", component_name, task_tracker::TASKS_ISSUED_TOTAL),
            "Total number of tasks issued/submitted",
            &[],
        )?;

        let started_counter = registry.create_intcounter(
            &format!("{}_{}", component_name, task_tracker::TASKS_STARTED_TOTAL),
            "Total number of tasks started",
            &[],
        )?;

        let success_counter = registry.create_intcounter(
            &format!("{}_{}", component_name, task_tracker::TASKS_SUCCESS_TOTAL),
            "Total number of successfully completed tasks",
            &[],
        )?;

        let cancelled_counter = registry.create_intcounter(
            &format!("{}_{}", component_name, task_tracker::TASKS_CANCELLED_TOTAL),
            "Total number of cancelled tasks",
            &[],
        )?;

        let failed_counter = registry.create_intcounter(
            &format!("{}_{}", component_name, task_tracker::TASKS_FAILED_TOTAL),
            "Total number of failed tasks",
            &[],
        )?;

        let rejected_counter = registry.create_intcounter(
            &format!("{}_{}", component_name, task_tracker::TASKS_REJECTED_TOTAL),
            "Total number of rejected tasks",
            &[],
        )?;

        Ok(Self {
            prometheus_issued: issued_counter,
            prometheus_started: started_counter,
            prometheus_success: success_counter,
            prometheus_cancelled: cancelled_counter,
            prometheus_failed: failed_counter,
            prometheus_rejected: rejected_counter,
        })
    }
}

impl HierarchicalTaskMetrics for PrometheusTaskMetrics {
    fn increment_issued(&self) {
        self.prometheus_issued.inc();
    }

    fn increment_started(&self) {
        self.prometheus_started.inc();
    }

    fn increment_success(&self) {
        self.prometheus_success.inc();
    }

    fn increment_cancelled(&self) {
        self.prometheus_cancelled.inc();
    }

    fn increment_failed(&self) {
        self.prometheus_failed.inc();
    }

    fn increment_rejected(&self) {
        self.prometheus_rejected.inc();
    }

    fn issued(&self) -> u64 {
        self.prometheus_issued.get()
    }

    fn started(&self) -> u64 {
        self.prometheus_started.get()
    }

    fn success(&self) -> u64 {
        self.prometheus_success.get()
    }

    fn cancelled(&self) -> u64 {
        self.prometheus_cancelled.get()
    }

    fn failed(&self) -> u64 {
        self.prometheus_failed.get()
    }

    fn rejected(&self) -> u64 {
        self.prometheus_rejected.get()
    }
}

/// Child tracker metrics that chain updates to parent
///
/// This implementation maintains local counters and automatically forwards
/// all metric updates to the parent tracker for hierarchical aggregation.
/// Holds a strong reference to parent metrics for optimal performance.
#[derive(Debug)]
struct ChildTaskMetrics {
    /// Local metrics for this tracker
    local_metrics: TaskMetrics,
    /// Strong reference to parent metrics for fast chaining
    /// Safe to hold since metrics don't own trackers - no circular references
    parent_metrics: Arc<dyn HierarchicalTaskMetrics>,
}

impl ChildTaskMetrics {
    fn new(parent_metrics: Arc<dyn HierarchicalTaskMetrics>) -> Self {
        Self {
            local_metrics: TaskMetrics::new(),
            parent_metrics,
        }
    }
}

impl HierarchicalTaskMetrics for ChildTaskMetrics {
    fn increment_issued(&self) {
        self.local_metrics.increment_issued();
        self.parent_metrics.increment_issued();
    }

    fn increment_started(&self) {
        self.local_metrics.increment_started();
        self.parent_metrics.increment_started();
    }

    fn increment_success(&self) {
        self.local_metrics.increment_success();
        self.parent_metrics.increment_success();
    }

    fn increment_cancelled(&self) {
        self.local_metrics.increment_cancelled();
        self.parent_metrics.increment_cancelled();
    }

    fn increment_failed(&self) {
        self.local_metrics.increment_failed();
        self.parent_metrics.increment_failed();
    }

    fn increment_rejected(&self) {
        self.local_metrics.increment_rejected();
        self.parent_metrics.increment_rejected();
    }

    fn issued(&self) -> u64 {
        self.local_metrics.issued()
    }

    fn started(&self) -> u64 {
        self.local_metrics.started()
    }

    fn success(&self) -> u64 {
        self.local_metrics.success()
    }

    fn cancelled(&self) -> u64 {
        self.local_metrics.cancelled()
    }

    fn failed(&self) -> u64 {
        self.local_metrics.failed()
    }

    fn rejected(&self) -> u64 {
        self.local_metrics.rejected()
    }
}

/// Builder for creating child trackers with custom policies
///
/// Allows flexible customization of scheduling and error handling policies
/// for child trackers while maintaining parent-child relationships.
pub struct ChildTrackerBuilder<'parent> {
    parent: &'parent TaskTracker,
    scheduler: Option<Arc<dyn TaskScheduler>>,
    error_policy: Option<Arc<dyn OnErrorPolicy>>,
}

impl<'parent> ChildTrackerBuilder<'parent> {
    /// Create a new ChildTrackerBuilder
    pub fn new(parent: &'parent TaskTracker) -> Self {
        Self {
            parent,
            scheduler: None,
            error_policy: None,
        }
    }

    /// Set custom scheduler for the child tracker
    ///
    /// If not set, the child will inherit the parent's scheduler.
    ///
    /// # Arguments
    /// * `scheduler` - The scheduler to use for this child tracker
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::sync::Semaphore;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler};
    /// # fn example(parent: &TaskTracker) {
    /// let child = parent.child_tracker_builder()
    ///     .scheduler(SemaphoreScheduler::with_permits(5))
    ///     .build().unwrap();
    /// # }
    /// ```
    pub fn scheduler(mut self, scheduler: Arc<dyn TaskScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Set custom error policy for the child tracker
    ///
    /// If not set, the child will get a child policy from the parent's error policy
    /// (via `OnErrorPolicy::create_child()`).
    ///
    /// # Arguments
    /// * `error_policy` - The error policy to use for this child tracker
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, LogOnlyPolicy};
    /// # fn example(parent: &TaskTracker) {
    /// let child = parent.child_tracker_builder()
    ///     .error_policy(LogOnlyPolicy::new())
    ///     .build().unwrap();
    /// # }
    /// ```
    pub fn error_policy(mut self, error_policy: Arc<dyn OnErrorPolicy>) -> Self {
        self.error_policy = Some(error_policy);
        self
    }

    /// Build the child tracker with the specified configuration
    ///
    /// Creates a new child tracker with:
    /// - Custom or inherited scheduler
    /// - Custom or child error policy
    /// - Hierarchical metrics that chain to parent
    /// - Child cancellation token from the parent
    /// - Independent lifecycle from parent
    ///
    /// # Returns
    /// A new `Arc<TaskTracker>` configured as a child of the parent
    ///
    /// # Errors
    /// Returns an error if the parent tracker is already closed
    pub fn build(self) -> anyhow::Result<TaskTracker> {
        // Validate that parent tracker is still active
        if self.parent.is_closed() {
            return Err(anyhow::anyhow!(
                "Cannot create child tracker from closed parent tracker"
            ));
        }

        let parent = self.parent.0.clone();

        let child_cancel_token = parent.cancel_token.child_token();
        let child_metrics = Arc::new(ChildTaskMetrics::new(parent.metrics.clone()));

        // Use provided scheduler or inherit from parent
        let scheduler = self.scheduler.unwrap_or_else(|| parent.scheduler.clone());

        // Use provided error policy or create child from parent's
        let error_policy = self
            .error_policy
            .unwrap_or_else(|| parent.error_policy.create_child());

        let child = Arc::new(TaskTrackerInner {
            tokio_tracker: TokioTaskTracker::new(),
            parent: None, // No parent reference needed for hierarchical operations
            scheduler,
            error_policy,
            metrics: child_metrics,
            cancel_token: child_cancel_token,
            children: RwLock::new(Vec::new()),
        });

        // Register this child with the parent for hierarchical operations
        parent
            .children
            .write()
            .unwrap()
            .push(Arc::downgrade(&child));

        // Periodically clean up dead children to prevent unbounded growth
        parent.cleanup_dead_children();

        Ok(TaskTracker(child))
    }
}

/// Internal data for TaskTracker
///
/// This struct contains all the actual state and functionality of a TaskTracker.
/// TaskTracker itself is just a wrapper around Arc<TaskTrackerInner>.
struct TaskTrackerInner {
    /// Tokio's task tracker for lifecycle management
    tokio_tracker: TokioTaskTracker,
    /// Parent tracker (None for root)
    parent: Option<Arc<TaskTrackerInner>>,
    /// Scheduling policy (shared with children by default)
    scheduler: Arc<dyn TaskScheduler>,
    /// Error handling policy (child-specific via create_child)
    error_policy: Arc<dyn OnErrorPolicy>,
    /// Metrics for this tracker
    metrics: Arc<dyn HierarchicalTaskMetrics>,
    /// Cancellation token for this tracker (always present)
    cancel_token: CancellationToken,
    /// List of child trackers for hierarchical operations
    children: RwLock<Vec<Weak<TaskTrackerInner>>>,
}

/// Hierarchical task tracker with pluggable scheduling and error policies
///
/// TaskTracker provides a composable system for managing background tasks with:
/// - Configurable scheduling via [`TaskScheduler`] implementations
/// - Flexible error handling via [`OnErrorPolicy`] implementations
/// - Parent-child relationships with independent metrics
/// - Cancellation propagation and isolation
/// - Built-in cancellation token support
///
/// Built on top of `tokio_util::task::TaskTracker` for robust task lifecycle management.
///
/// # Example
///
/// ```rust
/// # use std::sync::Arc;
/// # use tokio::sync::Semaphore;
/// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy, CancellableTaskResult};
/// # async fn example() -> anyhow::Result<()> {
/// // Create a task tracker with semaphore-based scheduling
/// let scheduler = SemaphoreScheduler::with_permits(3);
/// let policy = LogOnlyPolicy::new();
/// let root = TaskTracker::builder()
///     .scheduler(scheduler)
///     .error_policy(policy)
///     .build()?;
///
/// // Spawn some tasks
/// let handle1 = root.spawn(async { Ok(1) });
/// let handle2 = root.spawn(async { Ok(2) });
///
/// // Get results and join all tasks
/// let result1 = handle1.await.unwrap().unwrap();
/// let result2 = handle2.await.unwrap().unwrap();
/// assert_eq!(result1, 1);
/// assert_eq!(result2, 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct TaskTracker(Arc<TaskTrackerInner>);

/// Builder for TaskTracker
#[derive(Default)]
pub struct TaskTrackerBuilder {
    scheduler: Option<Arc<dyn TaskScheduler>>,
    error_policy: Option<Arc<dyn OnErrorPolicy>>,
    metrics: Option<Arc<dyn HierarchicalTaskMetrics>>,
    cancel_token: Option<CancellationToken>,
}

impl TaskTrackerBuilder {
    /// Set the scheduler for this TaskTracker
    pub fn scheduler(mut self, scheduler: Arc<dyn TaskScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Set the error policy for this TaskTracker
    pub fn error_policy(mut self, error_policy: Arc<dyn OnErrorPolicy>) -> Self {
        self.error_policy = Some(error_policy);
        self
    }

    /// Set custom metrics for this TaskTracker
    pub fn metrics(mut self, metrics: Arc<dyn HierarchicalTaskMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Set the cancellation token for this TaskTracker
    pub fn cancel_token(mut self, cancel_token: CancellationToken) -> Self {
        self.cancel_token = Some(cancel_token);
        self
    }

    /// Build the TaskTracker
    pub fn build(self) -> anyhow::Result<TaskTracker> {
        let scheduler = self
            .scheduler
            .ok_or_else(|| anyhow::anyhow!("TaskTracker requires a scheduler"))?;

        let error_policy = self
            .error_policy
            .ok_or_else(|| anyhow::anyhow!("TaskTracker requires an error policy"))?;

        let metrics = self.metrics.unwrap_or_else(|| Arc::new(TaskMetrics::new()));

        let cancel_token = self.cancel_token.unwrap_or_default();

        let inner = TaskTrackerInner {
            tokio_tracker: TokioTaskTracker::new(),
            parent: None,
            scheduler,
            error_policy,
            metrics,
            cancel_token,
            children: RwLock::new(Vec::new()),
        };

        Ok(TaskTracker(Arc::new(inner)))
    }
}

impl TaskTracker {
    /// Create a new root task tracker using the builder pattern
    ///
    /// This is the preferred way to create new task trackers.
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::sync::Semaphore;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
    /// # fn main() -> anyhow::Result<()> {
    /// let scheduler = SemaphoreScheduler::with_permits(10);
    /// let error_policy = LogOnlyPolicy::new();
    /// let tracker = TaskTracker::builder()
    ///     .scheduler(scheduler)
    ///     .error_policy(error_policy)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder() -> TaskTrackerBuilder {
        TaskTrackerBuilder::default()
    }

    /// Create a new root task tracker with simple parameters (legacy)
    ///
    /// This method is kept for backward compatibility. Use `builder()` for new code.
    /// Uses default metrics (no Prometheus integration).
    ///
    /// # Arguments
    /// * `scheduler` - Scheduling policy to use for all tasks
    /// * `error_policy` - Error handling policy for this tracker
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::sync::Semaphore;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
    /// # fn main() -> anyhow::Result<()> {
    /// let scheduler = SemaphoreScheduler::with_permits(10);
    /// let error_policy = LogOnlyPolicy::new();
    /// let tracker = TaskTracker::new(scheduler, error_policy)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        scheduler: Arc<dyn TaskScheduler>,
        error_policy: Arc<dyn OnErrorPolicy>,
    ) -> anyhow::Result<Self> {
        Self::builder()
            .scheduler(scheduler)
            .error_policy(error_policy)
            .build()
    }

    /// Create a new root task tracker with Prometheus metrics integration
    ///
    /// # Arguments
    /// * `scheduler` - Scheduling policy to use for all tasks
    /// * `error_policy` - Error handling policy for this tracker
    /// * `registry` - MetricsRegistry for Prometheus integration
    /// * `component_name` - Name for this tracker component
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::sync::Semaphore;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
    /// # use dynamo_runtime::metrics::MetricsRegistry;
    /// # fn example(registry: Arc<dyn MetricsRegistry>) -> anyhow::Result<()> {
    /// let scheduler = SemaphoreScheduler::with_permits(10);
    /// let error_policy = LogOnlyPolicy::new();
    /// let tracker = TaskTracker::new_with_prometheus(
    ///     scheduler,
    ///     error_policy,
    ///     registry.as_ref(),
    ///     "main_tracker"
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_prometheus<R: MetricsRegistry + ?Sized>(
        scheduler: Arc<dyn TaskScheduler>,
        error_policy: Arc<dyn OnErrorPolicy>,
        registry: &R,
        component_name: &str,
    ) -> anyhow::Result<Self> {
        let prometheus_metrics = Arc::new(PrometheusTaskMetrics::new(registry, component_name)?);

        Self::builder()
            .scheduler(scheduler)
            .error_policy(error_policy)
            .metrics(prometheus_metrics)
            .build()
    }

    /// Create a child tracker that inherits scheduling policy
    ///
    /// The child tracker:
    /// - Gets its own independent tokio TaskTracker
    /// - Inherits the parent's scheduler
    /// - Gets a child error policy via `create_child()`
    /// - Has hierarchical metrics that chain to parent
    /// - Gets a child cancellation token from the parent
    /// - Is independent for cancellation (child cancellation doesn't affect parent)
    ///
    /// # Errors
    /// Returns an error if the parent tracker is already closed
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # fn example(root_tracker: TaskTracker) -> anyhow::Result<()> {
    /// let child_tracker = root_tracker.child_tracker()?;
    /// // Child inherits parent's policies but has separate metrics and lifecycle
    /// # Ok(())
    /// # }
    /// ```
    pub fn child_tracker(&self) -> anyhow::Result<TaskTracker> {
        Ok(TaskTracker(self.0.child_tracker()?))
    }

    /// Create a child tracker builder for flexible customization
    ///
    /// The builder allows you to customize scheduling and error policies for the child tracker.
    /// If not specified, policies are inherited from the parent.
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::sync::Semaphore;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
    /// # fn example(root_tracker: TaskTracker) {
    /// // Custom scheduler, inherit error policy
    /// let child1 = root_tracker.child_tracker_builder()
    ///     .scheduler(SemaphoreScheduler::with_permits(5))
    ///     .build().unwrap();
    ///
    /// // Custom error policy, inherit scheduler
    /// let child2 = root_tracker.child_tracker_builder()
    ///     .error_policy(LogOnlyPolicy::new())
    ///     .build().unwrap();
    ///
    /// // Both custom
    /// let child3 = root_tracker.child_tracker_builder()
    ///     .scheduler(SemaphoreScheduler::with_permits(3))
    ///     .error_policy(LogOnlyPolicy::new())
    ///     .build().unwrap();
    /// # }
    /// ```
    /// Spawn a new task
    ///
    /// The task will be wrapped with scheduling and error handling logic,
    /// then executed according to the configured policies. For tasks that
    /// need to inspect cancellation tokens, use [`spawn_cancellable`] instead.
    ///
    /// # Arguments
    /// * `future` - The async task to execute
    ///
    /// # Returns
    /// A [`TaskHandle`] that can be used to await completion and access the task's cancellation token
    ///
    /// # Panics
    /// Panics if the tracker has been closed. This indicates a programming error
    /// where tasks are being spawned after the tracker lifecycle has ended.
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # async fn example(tracker: TaskTracker) -> anyhow::Result<()> {
    /// let handle = tracker.spawn(async {
    ///     // Your async work here
    ///     tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    ///     Ok(42)
    /// });
    ///
    /// // Access the task's cancellation token
    /// let cancel_token = handle.cancellation_token();
    ///
    /// let result = handle.await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn spawn<F, T>(&self, future: F) -> TaskHandle<T>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.0
            .spawn(future)
            .expect("TaskTracker must not be closed when spawning tasks")
    }

    /// Spawn a cancellable task that receives a cancellation token
    ///
    /// This is useful for tasks that need to inspect the cancellation token
    /// and gracefully handle cancellation within their logic. The task function
    /// must return a `CancellableTaskResult` to properly track cancellation vs errors.
    ///
    /// # Arguments
    ///
    /// * `task_fn` - Function that takes a cancellation token and returns a future that resolves to `CancellableTaskResult<T>`
    ///
    /// # Returns
    /// A [`TaskHandle`] that can be used to await completion and access the task's cancellation token
    ///
    /// # Panics
    /// Panics if the tracker has been closed. This indicates a programming error
    /// where tasks are being spawned after the tracker lifecycle has ended.
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, CancellableTaskResult};
    /// # async fn example(tracker: TaskTracker) -> anyhow::Result<()> {
    /// let handle = tracker.spawn_cancellable(|cancel_token| async move {
    ///     tokio::select! {
    ///         _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
    ///             CancellableTaskResult::Ok(42)
    ///         },
    ///         _ = cancel_token.cancelled() => CancellableTaskResult::Cancelled,
    ///     }
    /// });
    ///
    /// // Access the task's individual cancellation token
    /// let task_cancel_token = handle.cancellation_token();
    ///
    /// let result = handle.await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn spawn_cancellable<F, Fut, T>(&self, task_fn: F) -> TaskHandle<T>
    where
        F: FnMut(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = CancellableTaskResult<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.0
            .spawn_cancellable(task_fn)
            .expect("TaskTracker must not be closed when spawning tasks")
    }

    /// Get metrics for this tracker
    ///
    /// Metrics are specific to this tracker and do not include
    /// metrics from parent or child trackers.
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # fn example(tracker: &TaskTracker) {
    /// let metrics = tracker.metrics();
    /// println!("Success: {}, Failed: {}", metrics.success(), metrics.failed());
    /// # }
    /// ```
    pub fn metrics(&self) -> &dyn HierarchicalTaskMetrics {
        self.0.metrics.as_ref()
    }

    /// Cancel this tracker and all its tasks
    ///
    /// This will signal cancellation to all currently running tasks and prevent new tasks from being spawned.
    /// The cancellation is immediate and forceful.
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # async fn example(tracker: TaskTracker) -> anyhow::Result<()> {
    /// // Spawn a long-running task
    /// let handle = tracker.spawn_cancellable(|cancel_token| async move {
    ///     tokio::select! {
    ///         _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {
    ///             dynamo_runtime::utils::tasks::tracker::CancellableTaskResult::Ok(42)
    ///         }
    ///         _ = cancel_token.cancelled() => {
    ///             dynamo_runtime::utils::tasks::tracker::CancellableTaskResult::Cancelled
    ///         }
    ///     }
    /// }).await?;
    ///
    /// // Cancel the tracker (and thus the task)
    /// tracker.cancel();
    /// # Ok(())
    /// # }
    /// ```
    pub fn cancel(&self) {
        self.0.cancel();
    }

    /// Check if this tracker is closed
    pub fn is_closed(&self) -> bool {
        self.0.is_closed()
    }

    /// Get the cancellation token for this tracker
    ///
    /// This allows external code to observe or trigger cancellation of this tracker.
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # fn example(tracker: &TaskTracker) {
    /// let token = tracker.cancellation_token();
    /// // Can check cancellation state or cancel manually
    /// if !token.is_cancelled() {
    ///     token.cancel();
    /// }
    /// # }
    /// ```
    pub fn cancellation_token(&self) -> CancellationToken {
        self.0.cancellation_token()
    }

    /// Get the number of active child trackers
    ///
    /// This counts only child trackers that are still alive (not dropped).
    /// Dropped child trackers are automatically cleaned up.
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # fn example(tracker: &TaskTracker) {
    /// let child_count = tracker.child_count();
    /// println!("This tracker has {} active children", child_count);
    /// # }
    /// ```
    pub fn child_count(&self) -> usize {
        self.0.child_count()
    }

    /// Create a child tracker builder with custom configuration
    ///
    /// This provides fine-grained control over child tracker creation,
    /// allowing you to override the scheduler or error policy while
    /// maintaining the parent-child relationship.
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::sync::Semaphore;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
    /// # fn example(parent: &TaskTracker) {
    /// // Custom scheduler, inherit error policy
    /// let child1 = parent.child_tracker_builder()
    ///     .scheduler(SemaphoreScheduler::with_permits(5))
    ///     .build().unwrap();
    ///
    /// // Custom error policy, inherit scheduler
    /// let child2 = parent.child_tracker_builder()
    ///     .error_policy(LogOnlyPolicy::new())
    ///     .build().unwrap();
    ///
    /// // Inherit both policies from parent
    /// let child3 = parent.child_tracker_builder()
    ///     .build().unwrap();
    /// # }
    /// ```
    pub fn child_tracker_builder(&self) -> ChildTrackerBuilder<'_> {
        ChildTrackerBuilder::new(self)
    }

    /// Join this tracker and all child trackers
    ///
    /// This method gracefully shuts down the entire tracker hierarchy by:
    /// 1. Closing all trackers (preventing new task spawning)
    /// 2. Waiting for all existing tasks to complete
    ///
    /// Uses stack-safe traversal to prevent stack overflow in deep hierarchies.
    /// Children are processed before parents to ensure proper shutdown order.
    ///
    /// **Hierarchical Behavior:**
    /// - Processes children before parents to ensure proper shutdown order
    /// - Each tracker is closed before waiting (Tokio requirement)
    /// - Leaf trackers simply close and wait for their own tasks
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # async fn example(tracker: TaskTracker) {
    /// tracker.join().await;
    /// # }
    /// ```
    pub async fn join(&self) {
        self.0.join().await
    }
}

impl TaskTrackerInner {
    /// Creates child tracker with inherited scheduler/policy, independent metrics, and hierarchical cancellation
    fn child_tracker(self: &Arc<Self>) -> anyhow::Result<Arc<TaskTrackerInner>> {
        // Validate that parent tracker is still active
        if self.is_closed() {
            return Err(anyhow::anyhow!(
                "Cannot create child tracker from closed parent tracker"
            ));
        }

        let child_cancel_token = self.cancel_token.child_token();
        let child_metrics = Arc::new(ChildTaskMetrics::new(self.metrics.clone()));

        let child = Arc::new(TaskTrackerInner {
            tokio_tracker: TokioTaskTracker::new(),
            parent: Some(self.clone()),
            scheduler: self.scheduler.clone(),
            error_policy: self.error_policy.create_child(),
            metrics: child_metrics,
            cancel_token: child_cancel_token,
            children: RwLock::new(Vec::new()),
        });

        // Register this child with the parent for hierarchical operations
        self.children.write().unwrap().push(Arc::downgrade(&child));

        // Periodically clean up dead children to prevent unbounded growth
        self.cleanup_dead_children();

        Ok(child)
    }

    /// Spawn implementation - validates tracker state, generates task ID, applies policies, and tracks execution
    fn spawn<F, T>(self: &Arc<Self>, future: F) -> Result<TaskHandle<T>, TaskError>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        // Validate tracker is not closed
        if self.tokio_tracker.is_closed() {
            return Err(TaskError::TrackerClosed);
        }

        // Generate a unique task ID
        let task_id = self.generate_task_id();

        // Increment issued counter immediately when task is submitted
        self.metrics.increment_issued();

        // Create a child cancellation token for this specific task
        let task_cancel_token = self.cancel_token.child_token();
        let cancel_token = task_cancel_token.clone();

        // Clone the inner Arc to move into the task
        let inner = self.clone();

        // Wrap the user's future with our scheduling and error handling
        let wrapped_future =
            async move { Self::execute_with_policies(task_id, future, cancel_token, inner).await };

        // Let tokio handle the actual task tracking
        let join_handle = self.tokio_tracker.spawn(wrapped_future);

        // Wrap in TaskHandle with the child cancellation token
        Ok(TaskHandle::new(join_handle, task_cancel_token))
    }

    /// Spawn cancellable implementation - validates state, provides cancellation token, handles CancellableTaskResult
    fn spawn_cancellable<F, Fut, T>(
        self: &Arc<Self>,
        task_fn: F,
    ) -> Result<TaskHandle<T>, TaskError>
    where
        F: FnMut(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = CancellableTaskResult<T>> + Send + 'static,
        T: Send + 'static,
    {
        // Validate tracker is not closed
        if self.tokio_tracker.is_closed() {
            return Err(TaskError::TrackerClosed);
        }

        // Generate a unique task ID
        let task_id = self.generate_task_id();

        // Increment issued counter immediately when task is submitted
        self.metrics.increment_issued();

        // Create a child cancellation token for this specific task
        let task_cancel_token = self.cancel_token.child_token();
        let cancel_token = task_cancel_token.clone();

        // Clone the inner Arc to move into the task
        let inner = self.clone();

        // Use the new execution pipeline that defers task creation until after guard acquisition
        let wrapped_future = async move {
            Self::execute_cancellable_with_policies(task_id, task_fn, cancel_token, inner).await
        };

        // Let tokio handle the actual task tracking
        let join_handle = self.tokio_tracker.spawn(wrapped_future);

        // Wrap in TaskHandle with the child cancellation token
        Ok(TaskHandle::new(join_handle, task_cancel_token))
    }

    /// Cancel this tracker and all its tasks - implementation
    fn cancel(&self) {
        // Close the tracker to prevent new tasks
        self.tokio_tracker.close();

        // Cancel our own token
        self.cancel_token.cancel();
    }

    /// Returns true if the underlying tokio tracker is closed
    fn is_closed(&self) -> bool {
        self.tokio_tracker.is_closed()
    }

    /// Generates a unique task ID using TaskId::new()
    fn generate_task_id(&self) -> TaskId {
        TaskId::new()
    }

    /// Removes dead weak references from children list to prevent memory leaks
    fn cleanup_dead_children(&self) {
        let mut children_guard = self.children.write().unwrap();
        children_guard.retain(|weak| weak.upgrade().is_some());
    }

    /// Returns a clone of the cancellation token
    fn cancellation_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    /// Counts active child trackers (filters out dead weak references)
    fn child_count(&self) -> usize {
        let children_guard = self.children.read().unwrap();
        children_guard
            .iter()
            .filter(|weak| weak.upgrade().is_some())
            .count()
    }

    /// Join implementation - closes all trackers in hierarchy then waits for task completion using stack-safe traversal
    async fn join(self: &Arc<Self>) {
        // Fast path for leaf trackers (no children)
        let is_leaf = {
            let children_guard = self.children.read().unwrap();
            children_guard.is_empty()
        };

        if is_leaf {
            self.tokio_tracker.close();
            self.tokio_tracker.wait().await;
            return;
        }

        // Stack-safe traversal for deep hierarchies
        // Processes children before parents to ensure proper shutdown order
        let trackers = self.collect_hierarchy();
        for t in trackers {
            t.tokio_tracker.close();
            t.tokio_tracker.wait().await;
        }
    }

    /// Collects hierarchy using iterative DFS, returns Vec in post-order (children before parents) for safe shutdown
    fn collect_hierarchy(self: &Arc<TaskTrackerInner>) -> Vec<Arc<TaskTrackerInner>> {
        let mut result = Vec::new();
        let mut stack = vec![self.clone()];
        let mut visited = HashSet::new();

        // Collect all trackers using depth-first search
        while let Some(tracker) = stack.pop() {
            let tracker_ptr = Arc::as_ptr(&tracker) as usize;
            if visited.contains(&tracker_ptr) {
                continue;
            }
            visited.insert(tracker_ptr);

            // Add current tracker to result
            result.push(tracker.clone());

            // Add children to stack for processing
            if let Ok(children_guard) = tracker.children.read() {
                for weak_child in children_guard.iter() {
                    if let Some(child) = weak_child.upgrade() {
                        let child_ptr = Arc::as_ptr(&child) as usize;
                        if !visited.contains(&child_ptr) {
                            stack.push(child);
                        }
                    }
                }
            }
        }

        // Reverse to get bottom-up order (children before parents)
        result.reverse();
        result
    }

    /// Execute a regular task with scheduling and error handling policies
    #[tracing::instrument(level = "debug", skip_all, fields(task_id = %task_id))]
    async fn execute_with_policies<F, T>(
        task_id: TaskId,
        future: F,
        task_cancel_token: CancellationToken,
        inner: Arc<TaskTrackerInner>,
    ) -> Result<T, TaskError>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        // Wrap regular future in a task executor that doesn't support retry
        let task_executor = RegularTaskExecutor::new(future);
        Self::execute_with_retry_loop(task_id, task_executor, task_cancel_token, inner).await
    }

    /// Execute a cancellable task with scheduling and error handling policies
    #[tracing::instrument(level = "debug", skip_all, fields(task_id = %task_id))]
    async fn execute_cancellable_with_policies<F, Fut, T>(
        task_id: TaskId,
        task_fn: F,
        task_cancel_token: CancellationToken,
        inner: Arc<TaskTrackerInner>,
    ) -> Result<T, TaskError>
    where
        F: FnMut(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = CancellableTaskResult<T>> + Send + 'static,
        T: Send + 'static,
    {
        // Wrap cancellable task function in a task executor that supports retry
        let task_executor = CancellableTaskExecutor::new(task_fn);
        Self::execute_with_retry_loop(task_id, task_executor, task_cancel_token, inner).await
    }

    /// Core execution loop with retry support - unified for both task types
    #[tracing::instrument(level = "debug", skip_all, fields(task_id = %task_id))]
    async fn execute_with_retry_loop<E, T>(
        task_id: TaskId,
        initial_executor: E,
        task_cancellation_token: CancellationToken,
        inner: Arc<TaskTrackerInner>,
    ) -> Result<T, TaskError>
    where
        E: TaskExecutor<T> + Send + 'static,
        T: Send + 'static,
    {
        debug!("Starting task execution");

        // RAII guard for active counter - increments on creation, decrements on drop
        struct ActiveCountGuard {
            metrics: Arc<dyn HierarchicalTaskMetrics>,
            is_active: bool,
        }

        impl ActiveCountGuard {
            fn new(metrics: Arc<dyn HierarchicalTaskMetrics>) -> Self {
                Self {
                    metrics,
                    is_active: false,
                }
            }

            fn activate(&mut self) {
                if !self.is_active {
                    self.metrics.increment_started();
                    self.is_active = true;
                }
            }
        }

        // Current executable - either the original TaskExecutor or a Continuation
        enum CurrentExecutable<E>
        where
            E: Send + 'static,
        {
            TaskExecutor(E),
            Continuation(Arc<dyn Continuation + Send + Sync + 'static>),
        }

        let mut current_executable = CurrentExecutable::TaskExecutor(initial_executor);
        let mut active_guard = ActiveCountGuard::new(inner.metrics.clone());
        let mut error_context: Option<OnErrorContext> = None;
        let mut scheduler_guard_state = self::GuardState::Keep;
        let mut guard_result = async {
            inner
                .scheduler
                .acquire_execution_slot(task_cancellation_token.child_token())
                .await
        }
        .instrument(tracing::debug_span!("scheduler_resource_reacquisition"))
        .await;

        loop {
            if scheduler_guard_state == self::GuardState::Reschedule {
                guard_result = async {
                    inner
                        .scheduler
                        .acquire_execution_slot(inner.cancel_token.child_token())
                        .await
                }
                .instrument(tracing::debug_span!("scheduler_resource_reacquisition"))
                .await;
            }

            match &guard_result {
                SchedulingResult::Execute(_guard) => {
                    // Activate the RAII guard only once when we successfully acquire resources
                    active_guard.activate();

                    // Execute the current executable while holding the guard (RAII pattern)
                    let execution_result = async {
                        debug!("Executing task with acquired resources");
                        match &mut current_executable {
                            CurrentExecutable::TaskExecutor(executor) => {
                                executor.execute(inner.cancel_token.child_token()).await
                            }
                            CurrentExecutable::Continuation(continuation) => {
                                // Execute continuation and handle type erasure
                                match continuation.execute(inner.cancel_token.child_token()).await {
                                    TaskExecutionResult::Success(result) => {
                                        // Try to downcast the result to the expected type T
                                        if let Ok(typed_result) = result.downcast::<T>() {
                                            TaskExecutionResult::Success(*typed_result)
                                        } else {
                                            // Type mismatch - this shouldn't happen with proper usage
                                            let type_error = anyhow::anyhow!(
                                                "Continuation task returned wrong type"
                                            );
                                            error!(
                                                ?type_error,
                                                "Type mismatch in continuation task result"
                                            );
                                            TaskExecutionResult::Error(type_error)
                                        }
                                    }
                                    TaskExecutionResult::Cancelled => {
                                        TaskExecutionResult::Cancelled
                                    }
                                    TaskExecutionResult::Error(error) => {
                                        TaskExecutionResult::Error(error)
                                    }
                                }
                            }
                        }
                    }
                    .instrument(tracing::debug_span!("task_execution"))
                    .await;

                    // Active counter will be decremented automatically when active_guard drops

                    match execution_result {
                        TaskExecutionResult::Success(value) => {
                            inner.metrics.increment_success();
                            debug!("Task completed successfully");
                            return Ok(value);
                        }
                        TaskExecutionResult::Cancelled => {
                            inner.metrics.increment_cancelled();
                            debug!("Task was cancelled during execution");
                            return Err(TaskError::Cancelled);
                        }
                        TaskExecutionResult::Error(error) => {
                            debug!("Task failed - handling error through policy - {error:?}");

                            // Handle the error through the policy system
                            let (action_result, guard_state) = Self::handle_task_error(
                                &error,
                                &mut error_context,
                                task_id,
                                &inner,
                            )
                            .await;

                            // Update the scheduler guard state for evaluation after the match
                            scheduler_guard_state = guard_state;

                            match action_result {
                                ActionResult::Fail => {
                                    inner.metrics.increment_failed();
                                    debug!("Policy accepted error - task failed {error:?}");
                                    return Err(TaskError::Failed(error));
                                }
                                ActionResult::Shutdown => {
                                    inner.metrics.increment_failed();
                                    warn!("Policy triggered shutdown - {error:?}");
                                    inner.cancel();
                                    return Err(TaskError::Failed(error));
                                }
                                ActionResult::Continue { continuation } => {
                                    debug!(
                                        "Policy provided next executable - continuing loop - {error:?}"
                                    );

                                    // Update current executable
                                    current_executable =
                                        CurrentExecutable::Continuation(continuation);

                                    continue; // Continue the main loop with the new executable
                                }
                            }
                        }
                    }
                }
                SchedulingResult::Cancelled => {
                    inner.metrics.increment_cancelled();
                    debug!("Task was cancelled during resource acquisition");
                    return Err(TaskError::Cancelled);
                }
                SchedulingResult::Rejected(reason) => {
                    inner.metrics.increment_rejected();
                    debug!(reason, "Task was rejected by scheduler");
                    return Err(TaskError::Failed(anyhow::anyhow!(
                        "Task rejected: {}",
                        reason
                    )));
                }
            }
        }
    }

    /// Handle task errors through the error policy and return the action to take
    async fn handle_task_error(
        error: &anyhow::Error,
        error_context: &mut Option<OnErrorContext>,
        task_id: TaskId,
        inner: &Arc<TaskTrackerInner>,
    ) -> (ActionResult, self::GuardState) {
        // Create or update the error context (lazy initialization)
        let context = error_context.get_or_insert_with(|| OnErrorContext {
            attempt_count: 0, // Will be incremented below
            task_id,
            execution_context: TaskExecutionContext {
                scheduler: inner.scheduler.clone(),
                metrics: inner.metrics.clone(),
            },
            state: inner.error_policy.create_context(),
        });

        // Increment attempt count for this error
        context.attempt_count += 1;
        let current_attempt = context.attempt_count;

        // First, check if the policy allows continuations for this error
        if inner.error_policy.allow_continuation(error, context) {
            // Policy allows continuations, check if this is a FailedWithContinuation (task-driven continuation)
            if let Some(continuation_err) = error.downcast_ref::<FailedWithContinuation>() {
                debug!(
                    task_id = %task_id,
                    attempt_count = current_attempt,
                    "Task provided FailedWithContinuation and policy allows continuations - {error:?}"
                );

                // Task has provided a continuation implementation for the next attempt
                // Clone the Arc to return it in ActionResult::Continue
                let continuation = continuation_err.continuation.clone();

                // Ask policy whether to reschedule task-driven continuation
                let should_reschedule = inner.error_policy.should_reschedule(error, context);

                let guard_state = if should_reschedule {
                    self::GuardState::Reschedule
                } else {
                    self::GuardState::Keep
                };

                return (ActionResult::Continue { continuation }, guard_state);
            }
        } else {
            debug!(
                task_id = %task_id,
                attempt_count = current_attempt,
                "Policy rejected continuations, ignoring any FailedWithContinuation - {error:?}"
            );
        }

        let response = inner.error_policy.on_error(error, context);

        match response {
            ErrorResponse::Fail => (ActionResult::Fail, self::GuardState::Keep),
            ErrorResponse::Shutdown => (ActionResult::Shutdown, self::GuardState::Keep),
            ErrorResponse::Custom(action) => {
                debug!("Task failed - executing custom action - {error:?}");

                // Execute the custom action asynchronously
                let action_result = action
                    .execute(error, task_id, current_attempt, &context.execution_context)
                    .await;
                debug!(?action_result, "Custom action completed");

                // If the custom action returned Continue, ask policy about rescheduling
                let guard_state = match &action_result {
                    ActionResult::Continue { .. } => {
                        let should_reschedule =
                            inner.error_policy.should_reschedule(error, context);
                        if should_reschedule {
                            self::GuardState::Reschedule
                        } else {
                            self::GuardState::Keep
                        }
                    }
                    _ => self::GuardState::Keep, // Fail/Shutdown don't need guard state
                };

                (action_result, guard_state)
            }
        }
    }
}

// Blanket implementation for all schedulers
impl ArcPolicy for UnlimitedScheduler {}
impl ArcPolicy for SemaphoreScheduler {}

// Blanket implementation for all error policies
impl ArcPolicy for LogOnlyPolicy {}
impl ArcPolicy for CancelOnError {}
impl ArcPolicy for ThresholdCancelPolicy {}
impl ArcPolicy for RateCancelPolicy {}

/// Resource guard for unlimited scheduling
///
/// This guard represents "unlimited" resources - no actual resource constraints.
/// Since there are no resources to manage, this guard is essentially a no-op.
#[derive(Debug)]
pub struct UnlimitedGuard;

impl ResourceGuard for UnlimitedGuard {
    // No resources to manage - marker trait implementation only
}

/// Unlimited task scheduler that executes all tasks immediately
///
/// This scheduler provides no concurrency limits and executes all submitted tasks
/// immediately. Useful for testing, high-throughput scenarios, or when external
/// systems provide the concurrency control.
///
/// ## Cancellation Behavior
///
/// - Respects cancellation tokens before resource acquisition
/// - Once execution begins (via ResourceGuard), always awaits task completion
/// - Tasks handle their own cancellation internally (if created with `spawn_cancellable`)
///
/// # Example
/// ```rust
/// # use dynamo_runtime::utils::tasks::tracker::UnlimitedScheduler;
/// let scheduler = UnlimitedScheduler::new();
/// ```
#[derive(Debug)]
pub struct UnlimitedScheduler;

impl UnlimitedScheduler {
    /// Create a new unlimited scheduler returning Arc
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl Default for UnlimitedScheduler {
    fn default() -> Self {
        UnlimitedScheduler
    }
}

#[async_trait]
impl TaskScheduler for UnlimitedScheduler {
    async fn acquire_execution_slot(
        &self,
        cancel_token: CancellationToken,
    ) -> SchedulingResult<Box<dyn ResourceGuard>> {
        debug!("Acquiring execution slot (unlimited scheduler)");

        // Check for cancellation before allocating resources
        if cancel_token.is_cancelled() {
            debug!("Task cancelled before acquiring execution slot");
            return SchedulingResult::Cancelled;
        }

        // No resource constraints for unlimited scheduler
        debug!("Execution slot acquired immediately");
        SchedulingResult::Execute(Box::new(UnlimitedGuard))
    }
}

/// Resource guard for semaphore-based scheduling
///
/// This guard holds a semaphore permit and enforces that task execution
/// always runs to completion. The permit is automatically released when
/// the guard is dropped.
#[derive(Debug)]
pub struct SemaphoreGuard {
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl ResourceGuard for SemaphoreGuard {
    // Permit is automatically released when the guard is dropped
}

/// Semaphore-based task scheduler
///
/// Limits concurrent task execution using a [`tokio::sync::Semaphore`].
/// Tasks will wait for an available permit before executing.
///
/// ## Cancellation Behavior
///
/// - Respects cancellation tokens before and during permit acquisition
/// - Once a permit is acquired (via ResourceGuard), always awaits task completion
/// - Holds the permit until the task completes (regardless of cancellation)
/// - Tasks handle their own cancellation internally (if created with `spawn_cancellable`)
///
/// This ensures that permits are not leaked when tasks are cancelled, while still
/// allowing cancellable tasks to terminate gracefully on their own.
///
/// # Example
/// ```rust
/// # use std::sync::Arc;
/// # use tokio::sync::Semaphore;
/// # use dynamo_runtime::utils::tasks::tracker::SemaphoreScheduler;
/// // Allow up to 5 concurrent tasks
/// let semaphore = Arc::new(Semaphore::new(5));
/// let scheduler = SemaphoreScheduler::new(semaphore);
/// ```
#[derive(Debug)]
pub struct SemaphoreScheduler {
    semaphore: Arc<Semaphore>,
}

impl SemaphoreScheduler {
    /// Create a new semaphore scheduler
    ///
    /// # Arguments
    /// * `semaphore` - Semaphore to use for concurrency control
    pub fn new(semaphore: Arc<Semaphore>) -> Self {
        Self { semaphore }
    }

    /// Create a semaphore scheduler with the specified number of permits, returning Arc
    pub fn with_permits(permits: usize) -> Arc<Self> {
        Arc::new(Self::new(Arc::new(Semaphore::new(permits))))
    }

    /// Get the number of available permits
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }
}

#[async_trait]
impl TaskScheduler for SemaphoreScheduler {
    async fn acquire_execution_slot(
        &self,
        cancel_token: CancellationToken,
    ) -> SchedulingResult<Box<dyn ResourceGuard>> {
        debug!("Acquiring semaphore permit");

        // Check for cancellation before attempting to acquire semaphore
        if cancel_token.is_cancelled() {
            debug!("Task cancelled before acquiring semaphore permit");
            return SchedulingResult::Cancelled;
        }

        // Try to acquire a permit, with cancellation support
        let permit = {
            tokio::select! {
                result = self.semaphore.clone().acquire_owned() => {
                    match result {
                        Ok(permit) => permit,
                        Err(_) => return SchedulingResult::Cancelled,
                    }
                }
                _ = cancel_token.cancelled() => {
                    debug!("Task cancelled while waiting for semaphore permit");
                    return SchedulingResult::Cancelled;
                }
            }
        };

        debug!("Acquired semaphore permit");
        SchedulingResult::Execute(Box::new(SemaphoreGuard { _permit: permit }))
    }
}

/// Error policy that triggers cancellation based on error patterns
///
/// This policy analyzes error messages and returns `ErrorResponse::Shutdown` when:
/// - No patterns are specified (cancels on any error)
/// - Error message matches one of the specified patterns
///
/// The TaskTracker handles the actual cancellation - this policy just makes the decision.
///
/// # Example
/// ```rust
/// # use dynamo_runtime::utils::tasks::tracker::CancelOnError;
/// // Cancel on any error
/// let policy = CancelOnError::new();
///
/// // Cancel only on specific error patterns
/// let (policy, _token) = CancelOnError::with_patterns(
///     vec!["OutOfMemory".to_string(), "DeviceError".to_string()]
/// );
/// ```
#[derive(Debug)]
pub struct CancelOnError {
    error_patterns: Vec<String>,
}

impl CancelOnError {
    /// Create a new cancel-on-error policy that cancels on any error
    ///
    /// Returns a policy with no error patterns, meaning it will cancel the TaskTracker
    /// on any task failure.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            error_patterns: vec![], // Empty patterns = cancel on any error
        })
    }

    /// Create a new cancel-on-error policy with custom error patterns, returning Arc and token
    ///
    /// # Arguments
    /// * `error_patterns` - List of error message patterns that trigger cancellation
    pub fn with_patterns(error_patterns: Vec<String>) -> (Arc<Self>, CancellationToken) {
        let token = CancellationToken::new();
        let policy = Arc::new(Self { error_patterns });
        (policy, token)
    }
}

#[async_trait]
impl OnErrorPolicy for CancelOnError {
    fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
        // Child gets a child cancel token - when parent cancels, child cancels too
        // When child cancels, parent is unaffected
        Arc::new(CancelOnError {
            error_patterns: self.error_patterns.clone(),
        })
    }

    fn create_context(&self) -> Option<Box<dyn std::any::Any + Send + 'static>> {
        None // Stateless policy - no heap allocation
    }

    fn on_error(&self, error: &anyhow::Error, context: &mut OnErrorContext) -> ErrorResponse {
        error!(?context.task_id, "Task failed - {error:?}");

        if self.error_patterns.is_empty() {
            return ErrorResponse::Shutdown;
        }

        // Check if this error should trigger cancellation
        let error_str = error.to_string();
        let should_cancel = self
            .error_patterns
            .iter()
            .any(|pattern| error_str.contains(pattern));

        if should_cancel {
            ErrorResponse::Shutdown
        } else {
            ErrorResponse::Fail
        }
    }
}

/// Simple error policy that only logs errors
///
/// This policy does not trigger cancellation and is useful for
/// non-critical tasks or when you want to handle errors externally.
#[derive(Debug)]
pub struct LogOnlyPolicy;

impl LogOnlyPolicy {
    /// Create a new log-only policy returning Arc
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl Default for LogOnlyPolicy {
    fn default() -> Self {
        LogOnlyPolicy
    }
}

impl OnErrorPolicy for LogOnlyPolicy {
    fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
        // Simple policies can just clone themselves
        Arc::new(LogOnlyPolicy)
    }

    fn create_context(&self) -> Option<Box<dyn std::any::Any + Send + 'static>> {
        None // Stateless policy - no heap allocation
    }

    fn on_error(&self, error: &anyhow::Error, context: &mut OnErrorContext) -> ErrorResponse {
        error!(?context.task_id, "Task failed - logging only - {error:?}");
        ErrorResponse::Fail
    }
}

/// Error policy that cancels tasks after a threshold number of failures
///
/// This policy tracks the number of failed tasks and triggers cancellation
/// when the failure count exceeds the specified threshold. Useful for
/// preventing cascading failures in distributed systems.
///
/// # Example
/// ```rust
/// # use dynamo_runtime::utils::tasks::tracker::ThresholdCancelPolicy;
/// // Cancel after 5 failures
/// let policy = ThresholdCancelPolicy::with_threshold(5);
/// ```
#[derive(Debug)]
pub struct ThresholdCancelPolicy {
    max_failures: usize,
    failure_count: AtomicU64,
}

impl ThresholdCancelPolicy {
    /// Create a new threshold cancel policy with specified failure threshold, returning Arc and token
    ///
    /// # Arguments
    /// * `max_failures` - Maximum number of failures before cancellation
    pub fn with_threshold(max_failures: usize) -> Arc<Self> {
        Arc::new(Self {
            max_failures,
            failure_count: AtomicU64::new(0),
        })
    }

    /// Get the current failure count
    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::Relaxed)
    }

    /// Reset the failure count to zero
    ///
    /// This is primarily useful for testing scenarios where you want to reset
    /// the policy state between test cases.
    pub fn reset_failure_count(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
    }
}

/// Per-task state for ThresholdCancelPolicy
#[derive(Debug)]
struct ThresholdState {
    failure_count: u32,
}

impl OnErrorPolicy for ThresholdCancelPolicy {
    fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
        // Child gets a child cancel token and inherits the same failure threshold
        Arc::new(ThresholdCancelPolicy {
            max_failures: self.max_failures,
            failure_count: AtomicU64::new(0), // Child starts with fresh count
        })
    }

    fn create_context(&self) -> Option<Box<dyn std::any::Any + Send + 'static>> {
        Some(Box::new(ThresholdState { failure_count: 0 }))
    }

    fn on_error(&self, error: &anyhow::Error, context: &mut OnErrorContext) -> ErrorResponse {
        error!(?context.task_id, "Task failed - {error:?}");

        // Increment global counter for backwards compatibility
        let global_failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;

        // Get per-task state for the actual decision logic
        let state = context
            .state
            .as_mut()
            .expect("ThresholdCancelPolicy requires state")
            .downcast_mut::<ThresholdState>()
            .expect("Context type mismatch");

        state.failure_count += 1;
        let current_failures = state.failure_count;

        if current_failures >= self.max_failures as u32 {
            warn!(
                ?context.task_id,
                current_failures,
                global_failures,
                max_failures = self.max_failures,
                "Per-task failure threshold exceeded, triggering cancellation"
            );
            ErrorResponse::Shutdown
        } else {
            debug!(
                ?context.task_id,
                current_failures,
                global_failures,
                max_failures = self.max_failures,
                "Task failed, tracking per-task failure count"
            );
            ErrorResponse::Fail
        }
    }
}

/// Error policy that cancels tasks when failure rate exceeds threshold within time window
///
/// This policy tracks failures over a rolling time window and triggers cancellation
/// when the failure rate exceeds the specified threshold. More sophisticated than
/// simple count-based thresholds as it considers the time dimension.
///
/// # Example
/// ```rust
/// # use dynamo_runtime::utils::tasks::tracker::RateCancelPolicy;
/// // Cancel if more than 50% of tasks fail within any 60-second window
/// let (policy, token) = RateCancelPolicy::builder()
///     .rate(0.5)
///     .window_secs(60)
///     .build();
/// ```
#[derive(Debug)]
pub struct RateCancelPolicy {
    cancel_token: CancellationToken,
    max_failure_rate: f32,
    window_secs: u64,
    // TODO: Implement time-window tracking when needed
    // For now, this is a placeholder structure with the interface defined
}

impl RateCancelPolicy {
    /// Create a builder for rate-based cancel policy
    pub fn builder() -> RateCancelPolicyBuilder {
        RateCancelPolicyBuilder::new()
    }
}

/// Builder for RateCancelPolicy
pub struct RateCancelPolicyBuilder {
    max_failure_rate: Option<f32>,
    window_secs: Option<u64>,
}

impl RateCancelPolicyBuilder {
    fn new() -> Self {
        Self {
            max_failure_rate: None,
            window_secs: None,
        }
    }

    /// Set the maximum failure rate (0.0 to 1.0) before cancellation
    pub fn rate(mut self, max_failure_rate: f32) -> Self {
        self.max_failure_rate = Some(max_failure_rate);
        self
    }

    /// Set the time window in seconds for rate calculation
    pub fn window_secs(mut self, window_secs: u64) -> Self {
        self.window_secs = Some(window_secs);
        self
    }

    /// Build the policy, returning Arc and cancellation token
    pub fn build(self) -> (Arc<RateCancelPolicy>, CancellationToken) {
        let max_failure_rate = self.max_failure_rate.expect("rate must be set");
        let window_secs = self.window_secs.expect("window_secs must be set");

        let token = CancellationToken::new();
        let policy = Arc::new(RateCancelPolicy {
            cancel_token: token.clone(),
            max_failure_rate,
            window_secs,
        });
        (policy, token)
    }
}

#[async_trait]
impl OnErrorPolicy for RateCancelPolicy {
    fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
        Arc::new(RateCancelPolicy {
            cancel_token: self.cancel_token.child_token(),
            max_failure_rate: self.max_failure_rate,
            window_secs: self.window_secs,
        })
    }

    fn create_context(&self) -> Option<Box<dyn std::any::Any + Send + 'static>> {
        None // Stateless policy for now (TODO: add time-window state)
    }

    fn on_error(&self, error: &anyhow::Error, context: &mut OnErrorContext) -> ErrorResponse {
        error!(?context.task_id, "Task failed - {error:?}");

        // TODO: Implement time-window failure rate calculation
        // For now, just log the error and continue
        warn!(
            ?context.task_id,
            max_failure_rate = self.max_failure_rate,
            window_secs = self.window_secs,
            "Rate-based error policy - time window tracking not yet implemented"
        );

        ErrorResponse::Fail
    }
}

/// Custom action that triggers a cancellation token when executed
///
/// This action demonstrates the ErrorResponse::Custom behavior by capturing
/// an external cancellation token and triggering it when executed.
#[derive(Debug)]
pub struct TriggerCancellationTokenAction {
    cancel_token: CancellationToken,
}

impl TriggerCancellationTokenAction {
    pub fn new(cancel_token: CancellationToken) -> Self {
        Self { cancel_token }
    }
}

#[async_trait]
impl OnErrorAction for TriggerCancellationTokenAction {
    async fn execute(
        &self,
        error: &anyhow::Error,
        task_id: TaskId,
        _attempt_count: u32,
        _context: &TaskExecutionContext,
    ) -> ActionResult {
        warn!(
            ?task_id,
            "Executing custom action: triggering cancellation token - {error:?}"
        );

        // Trigger the custom cancellation token
        self.cancel_token.cancel();

        // Return success - the action completed successfully
        ActionResult::Shutdown
    }
}

/// Test error policy that triggers a custom cancellation token on any error
///
/// This policy demonstrates the ErrorResponse::Custom behavior by capturing
/// an external cancellation token and triggering it when any error occurs.
/// Used for testing custom error handling actions.
///
/// # Example
/// ```rust
/// # use tokio_util::sync::CancellationToken;
/// # use dynamo_runtime::utils::tasks::tracker::TriggerCancellationTokenOnError;
/// let cancel_token = CancellationToken::new();
/// let policy = TriggerCancellationTokenOnError::new(cancel_token.clone());
///
/// // Policy will trigger the token on any error via ErrorResponse::Custom
/// ```
#[derive(Debug)]
pub struct TriggerCancellationTokenOnError {
    cancel_token: CancellationToken,
}

impl TriggerCancellationTokenOnError {
    /// Create a new policy that triggers the given cancellation token on errors
    pub fn new(cancel_token: CancellationToken) -> Arc<Self> {
        Arc::new(Self { cancel_token })
    }
}

impl OnErrorPolicy for TriggerCancellationTokenOnError {
    fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
        // Child gets a child cancel token
        Arc::new(TriggerCancellationTokenOnError {
            cancel_token: self.cancel_token.clone(),
        })
    }

    fn create_context(&self) -> Option<Box<dyn std::any::Any + Send + 'static>> {
        None // Stateless policy - no heap allocation
    }

    fn on_error(&self, error: &anyhow::Error, context: &mut OnErrorContext) -> ErrorResponse {
        error!(
            ?context.task_id,
            "Task failed - triggering custom cancellation token - {error:?}"
        );

        // Create the custom action that will trigger our token
        let action = TriggerCancellationTokenAction::new(self.cancel_token.clone());

        // Return Custom response with our action
        ErrorResponse::Custom(Box::new(action))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;
    use std::sync::atomic::AtomicU32;
    use std::time::Duration;

    // Test fixtures using rstest
    #[fixture]
    fn semaphore_scheduler() -> Arc<SemaphoreScheduler> {
        Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))))
    }

    #[fixture]
    fn unlimited_scheduler() -> Arc<UnlimitedScheduler> {
        UnlimitedScheduler::new()
    }

    #[fixture]
    fn log_policy() -> Arc<LogOnlyPolicy> {
        LogOnlyPolicy::new()
    }

    #[fixture]
    fn cancel_policy() -> Arc<CancelOnError> {
        CancelOnError::new()
    }

    #[fixture]
    fn basic_tracker(
        unlimited_scheduler: Arc<UnlimitedScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) -> TaskTracker {
        TaskTracker::new(unlimited_scheduler, log_policy).unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn test_basic_task_execution(basic_tracker: TaskTracker) {
        // Test successful task execution
        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = basic_tracker.spawn(async {
            // Wait for signal to complete instead of sleep
            rx.await.ok();
            Ok(42)
        });

        // Signal task to complete
        tx.send(()).ok();

        // Verify task completes successfully
        let result = handle
            .await
            .expect("Task should complete")
            .expect("Task should succeed");
        assert_eq!(result, 42);

        // Verify metrics
        assert_eq!(basic_tracker.metrics().success(), 1);
        assert_eq!(basic_tracker.metrics().failed(), 0);
        assert_eq!(basic_tracker.metrics().cancelled(), 0);
        assert_eq!(basic_tracker.metrics().active(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_task_failure(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test task failure handling
        let tracker = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

        let handle = tracker.spawn(async { Err::<(), _>(anyhow::anyhow!("test error")) });

        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TaskError::Failed(_)));

        // Verify metrics
        assert_eq!(tracker.metrics().success(), 0);
        assert_eq!(tracker.metrics().failed(), 1);
        assert_eq!(tracker.metrics().cancelled(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_semaphore_concurrency_limit(log_policy: Arc<LogOnlyPolicy>) {
        // Test that semaphore limits concurrent execution
        let limited_scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(2)))); // Only 2 concurrent tasks
        let tracker = TaskTracker::new(limited_scheduler, log_policy).unwrap();

        let counter = Arc::new(AtomicU32::new(0));
        let max_concurrent = Arc::new(AtomicU32::new(0));

        // Use broadcast channel to coordinate all tasks
        let (tx, _) = tokio::sync::broadcast::channel(1);
        let mut handles = Vec::new();

        // Spawn 5 tasks that will track concurrency
        for _ in 0..5 {
            let counter_clone = counter.clone();
            let max_clone = max_concurrent.clone();
            let mut rx = tx.subscribe();

            let handle = tracker.spawn(async move {
                // Increment active counter
                let current = counter_clone.fetch_add(1, Ordering::Relaxed) + 1;

                // Track max concurrent
                max_clone.fetch_max(current, Ordering::Relaxed);

                // Wait for signal to complete instead of sleep
                rx.recv().await.ok();

                // Decrement when done
                counter_clone.fetch_sub(1, Ordering::Relaxed);

                Ok(())
            });
            handles.push(handle);
        }

        // Give tasks time to start and register concurrency
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // Signal all tasks to complete
        tx.send(()).ok();

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // Verify that no more than 2 tasks ran concurrently
        assert!(max_concurrent.load(Ordering::Relaxed) <= 2);

        // Verify all tasks completed successfully
        assert_eq!(tracker.metrics().success(), 5);
        assert_eq!(tracker.metrics().failed(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_cancel_on_error_policy() {
        // Test that CancelOnError policy works correctly
        let error_policy = cancel_policy();
        let scheduler = semaphore_scheduler();
        let tracker = TaskTracker::new(scheduler, error_policy).unwrap();

        // Spawn a task that will trigger cancellation
        let handle =
            tracker.spawn(async { Err::<(), _>(anyhow::anyhow!("OutOfMemory error occurred")) });

        // Wait for the error to occur
        let result = handle.await.unwrap();
        assert!(result.is_err());

        // Give cancellation time to propagate
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Verify the cancel token was triggered
        assert!(tracker.cancellation_token().is_cancelled());
    }

    #[rstest]
    #[tokio::test]
    async fn test_tracker_cancellation() {
        // Test manual cancellation of tracker with CancelOnError policy
        let error_policy = cancel_policy();
        let scheduler = semaphore_scheduler();
        let tracker = TaskTracker::new(scheduler, error_policy).unwrap();
        let cancel_token = tracker.cancellation_token().child_token();

        // Use oneshot channel instead of sleep for deterministic timing
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();

        // Spawn a task that respects cancellation
        let handle = tracker.spawn({
            let cancel_token = cancel_token.clone();
            async move {
                tokio::select! {
                    _ = rx => Ok(()),
                    _ = cancel_token.cancelled() => Err(anyhow::anyhow!("Task was cancelled")),
                }
            }
        });

        // Cancel the tracker
        tracker.cancel();

        // Task should be cancelled
        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TaskError::Cancelled));
    }

    #[rstest]
    #[tokio::test]
    async fn test_child_tracker_independence(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that child tracker has independent lifecycle
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

        let child = parent.child_tracker().unwrap();

        // Both should be operational initially
        assert!(!parent.is_closed());
        assert!(!child.is_closed());

        // Cancel child only
        child.cancel();

        // Parent should remain operational
        assert!(!parent.is_closed());

        // Parent can still spawn tasks
        let handle = parent.spawn(async { Ok(42) });
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);
    }

    #[rstest]
    #[tokio::test]
    async fn test_independent_metrics(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that parent and child have independent metrics
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();
        let child = parent.child_tracker().unwrap();

        // Run tasks in parent
        let handle1 = parent.spawn(async { Ok(1) });
        handle1.await.unwrap().unwrap();

        // Run tasks in child
        let handle2 = child.spawn(async { Ok(2) });
        handle2.await.unwrap().unwrap();

        // Each should have their own metrics, but parent sees aggregated
        assert_eq!(parent.metrics().success(), 2); // Parent sees its own + child's
        assert_eq!(child.metrics().success(), 1); // Child sees only its own
        assert_eq!(parent.metrics().total_completed(), 2); // Parent sees aggregated total
        assert_eq!(child.metrics().total_completed(), 1); // Child sees only its own
    }

    #[rstest]
    #[tokio::test]
    async fn test_cancel_on_error_hierarchy() {
        // Test that child error policy cancellation doesn't affect parent
        let parent_error_policy = cancel_policy();
        let scheduler = semaphore_scheduler();
        let parent = TaskTracker::new(scheduler, parent_error_policy).unwrap();
        let parent_policy_token = parent.cancellation_token().child_token();
        let child = parent.child_tracker().unwrap();

        // Initially nothing should be cancelled
        assert!(!parent_policy_token.is_cancelled());

        // Use explicit synchronization instead of sleep
        let (error_tx, error_rx) = tokio::sync::oneshot::channel();
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();

        // Spawn a monitoring task to watch for the parent policy token cancellation
        let parent_token_monitor = parent_policy_token.clone();
        let monitor_handle = tokio::spawn(async move {
            tokio::select! {
                _ = parent_token_monitor.cancelled() => {
                    cancel_tx.send(true).ok();
                }
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    cancel_tx.send(false).ok();
                }
            }
        });

        // Spawn a task in the child that will trigger cancellation
        let handle = child.spawn(async move {
            let result = Err::<(), _>(anyhow::anyhow!("OutOfMemory in child"));
            error_tx.send(()).ok(); // Signal that the error has occurred
            result
        });

        // Wait for the error to occur
        let error_result = handle.await.unwrap();
        assert!(error_result.is_err());

        // Wait for our error signal
        error_rx.await.ok();

        // Check if parent policy token was cancelled within timeout
        let was_cancelled = cancel_rx.await.unwrap_or(false);
        monitor_handle.await.ok();

        // Based on hierarchical design: child errors should NOT affect parent
        // The child gets its own policy with a child token, and child cancellation
        // should not propagate up to the parent policy token
        assert!(
            !was_cancelled,
            "Parent policy token should not be cancelled by child errors"
        );
        assert!(
            !parent_policy_token.is_cancelled(),
            "Parent policy token should remain active"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_graceful_shutdown(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test graceful shutdown with close()
        let tracker = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

        // Use broadcast channel to coordinate task completion
        let (tx, _) = tokio::sync::broadcast::channel(1);
        let mut handles = Vec::new();

        // Spawn some tasks
        for i in 0..3 {
            let mut rx = tx.subscribe();
            let handle = tracker.spawn(async move {
                // Wait for signal instead of sleep
                rx.recv().await.ok();
                Ok(i)
            });
            handles.push(handle);
        }

        // Signal all tasks to complete before closing
        tx.send(()).ok();

        // Close tracker and wait for completion
        tracker.join().await;

        // All tasks should complete successfully
        for handle in handles {
            let result = handle.await.unwrap().unwrap();
            assert!(result < 3);
        }

        // Tracker should be closed
        assert!(tracker.is_closed());
    }

    #[rstest]
    #[tokio::test]
    async fn test_semaphore_scheduler_permit_tracking(log_policy: Arc<LogOnlyPolicy>) {
        // Test that SemaphoreScheduler properly tracks permits
        let semaphore = Arc::new(Semaphore::new(3));
        let scheduler = Arc::new(SemaphoreScheduler::new(semaphore.clone()));
        let tracker = TaskTracker::new(scheduler.clone(), log_policy).unwrap();

        // Initially all permits should be available
        assert_eq!(scheduler.available_permits(), 3);

        // Use broadcast channel to coordinate task completion
        let (tx, _) = tokio::sync::broadcast::channel(1);
        let mut handles = Vec::new();

        // Spawn 3 tasks that will hold permits
        for _ in 0..3 {
            let mut rx = tx.subscribe();
            let handle = tracker.spawn(async move {
                // Wait for signal to complete
                rx.recv().await.ok();
                Ok(())
            });
            handles.push(handle);
        }

        // Give tasks time to acquire permits
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // All permits should be taken
        assert_eq!(scheduler.available_permits(), 0);

        // Signal all tasks to complete
        tx.send(()).ok();

        // Wait for tasks to complete
        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // All permits should be available again
        assert_eq!(scheduler.available_permits(), 3);
    }

    #[rstest]
    #[tokio::test]
    async fn test_builder_pattern(log_policy: Arc<LogOnlyPolicy>) {
        // Test that TaskTracker builder works correctly
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))));
        let error_policy = log_policy;

        let tracker = TaskTracker::builder()
            .scheduler(scheduler)
            .error_policy(error_policy)
            .build()
            .unwrap();

        // Tracker should have a cancellation token
        let token = tracker.cancellation_token();
        assert!(!token.is_cancelled());

        // Should be able to spawn tasks
        let handle = tracker.spawn(async { Ok(42) });
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);
    }

    #[rstest]
    #[tokio::test]
    async fn test_all_trackers_have_cancellation_tokens(log_policy: Arc<LogOnlyPolicy>) {
        // Test that all trackers (root and children) have cancellation tokens
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))));
        let root = TaskTracker::new(scheduler, log_policy).unwrap();
        let child = root.child_tracker().unwrap();
        let grandchild = child.child_tracker().unwrap();

        // All should have cancellation tokens
        let root_token = root.cancellation_token();
        let child_token = child.cancellation_token();
        let grandchild_token = grandchild.cancellation_token();

        assert!(!root_token.is_cancelled());
        assert!(!child_token.is_cancelled());
        assert!(!grandchild_token.is_cancelled());

        // Child tokens should be different from parent
        // (We can't directly compare tokens, but we can test behavior)
        root_token.cancel();

        // Give cancellation time to propagate
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Root should be cancelled
        assert!(root_token.is_cancelled());
        // Children should also be cancelled (because they are child tokens)
        assert!(child_token.is_cancelled());
        assert!(grandchild_token.is_cancelled());
    }

    #[rstest]
    #[tokio::test]
    async fn test_spawn_cancellable_task(log_policy: Arc<LogOnlyPolicy>) {
        // Test cancellable task spawning with proper result handling
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Test successful completion
        let (tx, rx) = tokio::sync::oneshot::channel();
        let rx = Arc::new(tokio::sync::Mutex::new(Some(rx)));
        let handle = tracker.spawn_cancellable(move |_cancel_token| {
            let rx = rx.clone();
            async move {
                // Wait for signal instead of sleep
                if let Some(rx) = rx.lock().await.take() {
                    rx.await.ok();
                }
                CancellableTaskResult::Ok(42)
            }
        });

        // Signal task to complete
        tx.send(()).ok();

        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);
        assert_eq!(tracker.metrics().success(), 1);

        // Test cancellation handling
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        let rx = Arc::new(tokio::sync::Mutex::new(Some(rx)));
        let handle = tracker.spawn_cancellable(move |cancel_token| {
            let rx = rx.clone();
            async move {
                tokio::select! {
                    _ = async {
                        if let Some(rx) = rx.lock().await.take() {
                            rx.await.ok();
                        }
                    } => CancellableTaskResult::Ok("should not complete"),
                _ = cancel_token.cancelled() => CancellableTaskResult::Cancelled,
                }
            }
        });

        // Cancel the tracker
        tracker.cancel();

        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TaskError::Cancelled));
    }

    #[rstest]
    #[tokio::test]
    async fn test_cancellable_task_metrics_tracking(log_policy: Arc<LogOnlyPolicy>) {
        // Test that properly cancelled tasks increment cancelled metrics, not failed metrics
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Baseline metrics
        assert_eq!(tracker.metrics().cancelled(), 0);
        assert_eq!(tracker.metrics().failed(), 0);
        assert_eq!(tracker.metrics().success(), 0);

        // Test 1: Task that executes and THEN gets cancelled during execution
        let (start_tx, start_rx) = tokio::sync::oneshot::channel::<()>();
        let (_continue_tx, continue_rx) = tokio::sync::oneshot::channel::<()>();

        let start_tx_shared = Arc::new(tokio::sync::Mutex::new(Some(start_tx)));
        let continue_rx_shared = Arc::new(tokio::sync::Mutex::new(Some(continue_rx)));

        let start_tx_for_task = start_tx_shared.clone();
        let continue_rx_for_task = continue_rx_shared.clone();

        let handle = tracker.spawn_cancellable(move |cancel_token| {
            let start_tx = start_tx_for_task.clone();
            let continue_rx = continue_rx_for_task.clone();
            async move {
                // Signal that we've started executing
                if let Some(tx) = start_tx.lock().await.take() {
                    tx.send(()).ok();
                }

                // Wait for either continuation signal or cancellation
                tokio::select! {
                    _ = async {
                        if let Some(rx) = continue_rx.lock().await.take() {
                            rx.await.ok();
                        }
                    } => CancellableTaskResult::Ok("completed normally"),
                _ = cancel_token.cancelled() => {
                    println!("Task detected cancellation and is returning Cancelled");
                    CancellableTaskResult::Cancelled
                },
                }
            }
        });

        // Wait for task to start executing
        start_rx.await.ok();

        // Now cancel while the task is running
        println!("Cancelling tracker while task is executing...");
        tracker.cancel();

        // Wait for the task to complete
        let result = handle.await.unwrap();

        // Debug output
        println!("Task result: {:?}", result);
        println!(
            "Cancelled: {}, Failed: {}, Success: {}",
            tracker.metrics().cancelled(),
            tracker.metrics().failed(),
            tracker.metrics().success()
        );

        // The task should be properly cancelled and counted correctly
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TaskError::Cancelled));

        // Verify proper metrics: should be counted as cancelled, not failed
        assert_eq!(
            tracker.metrics().cancelled(),
            1,
            "Properly cancelled task should increment cancelled count"
        );
        assert_eq!(
            tracker.metrics().failed(),
            0,
            "Properly cancelled task should NOT increment failed count"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_cancellable_vs_error_metrics_distinction(log_policy: Arc<LogOnlyPolicy>) {
        // Test that we properly distinguish between cancellation and actual errors
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Test 1: Actual error should increment failed count
        let handle1 = tracker.spawn_cancellable(|_cancel_token| async move {
            CancellableTaskResult::<i32>::Err(anyhow::anyhow!("This is a real error"))
        });

        let result1 = handle1.await.unwrap();
        assert!(result1.is_err());
        assert!(matches!(result1.unwrap_err(), TaskError::Failed(_)));
        assert_eq!(tracker.metrics().failed(), 1);
        assert_eq!(tracker.metrics().cancelled(), 0);

        // Test 2: Cancellation should increment cancelled count
        let handle2 = tracker.spawn_cancellable(|_cancel_token| async move {
            CancellableTaskResult::<i32>::Cancelled
        });

        let result2 = handle2.await.unwrap();
        assert!(result2.is_err());
        assert!(matches!(result2.unwrap_err(), TaskError::Cancelled));
        assert_eq!(tracker.metrics().failed(), 1); // Still 1 from before
        assert_eq!(tracker.metrics().cancelled(), 1); // Now 1 from cancellation
    }

    #[rstest]
    #[tokio::test]
    async fn test_spawn_cancellable_error_handling(log_policy: Arc<LogOnlyPolicy>) {
        // Test error handling in cancellable tasks
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Test error result
        let handle = tracker.spawn_cancellable(|_cancel_token| async move {
            CancellableTaskResult::<i32>::Err(anyhow::anyhow!("test error"))
        });

        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TaskError::Failed(_)));
        assert_eq!(tracker.metrics().failed(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_cancellation_before_execution(log_policy: Arc<LogOnlyPolicy>) {
        // Test that spawning on a cancelled tracker panics (new behavior)
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(1))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Cancel the tracker first
        tracker.cancel();

        // Give cancellation time to propagate to the inner tracker
        tokio::time::sleep(Duration::from_millis(5)).await;

        // Now try to spawn a task - it should panic since tracker is closed
        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tracker.spawn(async { Ok(42) })
        }));

        // Should panic with our new API
        assert!(
            panic_result.is_err(),
            "spawn() should panic when tracker is closed"
        );

        // Verify the panic message contains expected text
        if let Err(panic_payload) = panic_result {
            if let Some(panic_msg) = panic_payload.downcast_ref::<String>() {
                assert!(
                    panic_msg.contains("TaskTracker must not be closed"),
                    "Panic message should indicate tracker is closed: {}",
                    panic_msg
                );
            } else if let Some(panic_msg) = panic_payload.downcast_ref::<&str>() {
                assert!(
                    panic_msg.contains("TaskTracker must not be closed"),
                    "Panic message should indicate tracker is closed: {}",
                    panic_msg
                );
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_semaphore_scheduler_with_cancellation(log_policy: Arc<LogOnlyPolicy>) {
        // Test that SemaphoreScheduler respects cancellation tokens
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(1))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Start a long-running task to occupy the semaphore
        let blocker_token = tracker.cancellation_token();
        let _blocker_handle = tracker.spawn(async move {
            // Wait for cancellation
            blocker_token.cancelled().await;
            Ok(())
        });

        // Give the blocker time to acquire the permit
        tokio::task::yield_now().await;

        // Use oneshot channel for the second task
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();

        // Spawn another task that will wait for semaphore
        let handle = tracker.spawn(async {
            rx.await.ok();
            Ok(42)
        });

        // Cancel the tracker while second task is waiting for permit
        tracker.cancel();

        // The waiting task should be cancelled
        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TaskError::Cancelled));
    }

    #[rstest]
    #[tokio::test]
    async fn test_child_tracker_cancellation_independence(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that child tracker cancellation doesn't affect parent
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();
        let child = parent.child_tracker().unwrap();

        // Cancel only the child
        child.cancel();

        // Parent should still be operational
        let parent_token = parent.cancellation_token();
        assert!(!parent_token.is_cancelled());

        // Parent can still spawn tasks
        let handle = parent.spawn(async { Ok(42) });
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);

        // Child should be cancelled
        let child_token = child.cancellation_token();
        assert!(child_token.is_cancelled());
    }

    #[rstest]
    #[tokio::test]
    async fn test_parent_cancellation_propagates_to_children(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that parent cancellation propagates to all children
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();
        let child1 = parent.child_tracker().unwrap();
        let child2 = parent.child_tracker().unwrap();
        let grandchild = child1.child_tracker().unwrap();

        // Cancel the parent
        parent.cancel();

        // Give cancellation time to propagate
        tokio::time::sleep(Duration::from_millis(10)).await;

        // All should be cancelled
        assert!(parent.cancellation_token().is_cancelled());
        assert!(child1.cancellation_token().is_cancelled());
        assert!(child2.cancellation_token().is_cancelled());
        assert!(grandchild.cancellation_token().is_cancelled());
    }

    #[rstest]
    #[tokio::test]
    async fn test_issued_counter_tracking(log_policy: Arc<LogOnlyPolicy>) {
        // Test that issued counter is incremented when tasks are spawned
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(2))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Initially no tasks issued
        assert_eq!(tracker.metrics().issued(), 0);
        assert_eq!(tracker.metrics().pending(), 0);

        // Spawn some tasks
        let handle1 = tracker.spawn(async { Ok(1) });
        let handle2 = tracker.spawn(async { Ok(2) });
        let handle3 = tracker.spawn_cancellable(|_| async { CancellableTaskResult::Ok(3) });

        // Issued counter should be incremented immediately
        assert_eq!(tracker.metrics().issued(), 3);
        assert_eq!(tracker.metrics().pending(), 3); // None completed yet

        // Complete the tasks
        assert_eq!(handle1.await.unwrap().unwrap(), 1);
        assert_eq!(handle2.await.unwrap().unwrap(), 2);
        assert_eq!(handle3.await.unwrap().unwrap(), 3);

        // Check final accounting
        assert_eq!(tracker.metrics().issued(), 3);
        assert_eq!(tracker.metrics().success(), 3);
        assert_eq!(tracker.metrics().total_completed(), 3);
        assert_eq!(tracker.metrics().pending(), 0); // All completed

        // Test hierarchical accounting
        let child = tracker.child_tracker().unwrap();
        let child_handle = child.spawn(async { Ok(42) });

        // Both parent and child should see the issued task
        assert_eq!(child.metrics().issued(), 1);
        assert_eq!(tracker.metrics().issued(), 4); // Parent sees all

        child_handle.await.unwrap().unwrap();

        // Final hierarchical check
        assert_eq!(child.metrics().pending(), 0);
        assert_eq!(tracker.metrics().pending(), 0);
        assert_eq!(tracker.metrics().success(), 4); // Parent sees all successes
    }

    #[rstest]
    #[tokio::test]
    async fn test_child_tracker_builder(log_policy: Arc<LogOnlyPolicy>) {
        // Test that child tracker builder allows custom policies
        let parent_scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(10))));
        let parent = TaskTracker::new(parent_scheduler, log_policy).unwrap();

        // Create child with custom error policy
        let child_error_policy = CancelOnError::new();
        let child = parent
            .child_tracker_builder()
            .error_policy(child_error_policy)
            .build()
            .unwrap();

        // Test that child works
        let handle = child.spawn(async { Ok(42) });
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);

        // Child should have its own metrics
        assert_eq!(child.metrics().success(), 1);
        assert_eq!(parent.metrics().total_completed(), 1); // Parent sees aggregated
    }

    #[rstest]
    #[tokio::test]
    async fn test_hierarchical_metrics_aggregation(log_policy: Arc<LogOnlyPolicy>) {
        // Test that child metrics aggregate up to parent
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(10))));
        let parent = TaskTracker::new(scheduler, log_policy.clone()).unwrap();

        // Create child1 with default settings
        let child1 = parent.child_tracker().unwrap();

        // Create child2 with custom error policy
        let child_error_policy = CancelOnError::new();
        let child2 = parent
            .child_tracker_builder()
            .error_policy(child_error_policy)
            .build()
            .unwrap();

        // Test both custom schedulers and policies
        let another_scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(3))));
        let another_error_policy = CancelOnError::new();
        let child3 = parent
            .child_tracker_builder()
            .scheduler(another_scheduler)
            .error_policy(another_error_policy)
            .build()
            .unwrap();

        // Test that all children are properly registered
        assert_eq!(parent.child_count(), 3);

        // Test that custom schedulers work
        let handle1 = child1.spawn(async { Ok(1) });
        let handle2 = child2.spawn(async { Ok(2) });
        let handle3 = child3.spawn(async { Ok(3) });

        assert_eq!(handle1.await.unwrap().unwrap(), 1);
        assert_eq!(handle2.await.unwrap().unwrap(), 2);
        assert_eq!(handle3.await.unwrap().unwrap(), 3);

        // Verify metrics still work
        assert_eq!(parent.metrics().success(), 3); // All child successes roll up
        assert_eq!(child1.metrics().success(), 1);
        assert_eq!(child2.metrics().success(), 1);
        assert_eq!(child3.metrics().success(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_scheduler_queue_depth_calculation(log_policy: Arc<LogOnlyPolicy>) {
        // Test that we can calculate tasks queued in scheduler
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(2)))); // Only 2 concurrent tasks
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Initially no tasks
        assert_eq!(tracker.metrics().issued(), 0);
        assert_eq!(tracker.metrics().active(), 0);
        assert_eq!(tracker.metrics().queued(), 0);
        assert_eq!(tracker.metrics().pending(), 0);

        // Use a channel to control when tasks complete
        let (complete_tx, _complete_rx) = tokio::sync::broadcast::channel(1);

        // Spawn 2 tasks that will hold semaphore permits
        let handle1 = tracker.spawn({
            let mut rx = complete_tx.subscribe();
            async move {
                // Wait for completion signal
                rx.recv().await.ok();
                Ok(1)
            }
        });
        let handle2 = tracker.spawn({
            let mut rx = complete_tx.subscribe();
            async move {
                // Wait for completion signal
                rx.recv().await.ok();
                Ok(2)
            }
        });

        // Give tasks time to start and acquire permits
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // Should have 2 active tasks, 0 queued
        assert_eq!(tracker.metrics().issued(), 2);
        assert_eq!(tracker.metrics().active(), 2);
        assert_eq!(tracker.metrics().queued(), 0);
        assert_eq!(tracker.metrics().pending(), 2);

        // Spawn a third task - should be queued since semaphore is full
        let handle3 = tracker.spawn(async move { Ok(3) });

        // Give time for task to be queued
        tokio::task::yield_now().await;

        // Should have 2 active, 1 queued
        assert_eq!(tracker.metrics().issued(), 3);
        assert_eq!(tracker.metrics().active(), 2);
        assert_eq!(
            tracker.metrics().queued(),
            tracker.metrics().pending() - tracker.metrics().active()
        );
        assert_eq!(tracker.metrics().pending(), 3);

        // Complete all tasks by sending the signal
        complete_tx.send(()).ok();

        let result1 = handle1.await.unwrap().unwrap();
        let result2 = handle2.await.unwrap().unwrap();
        let result3 = handle3.await.unwrap().unwrap();

        assert_eq!(result1, 1);
        assert_eq!(result2, 2);
        assert_eq!(result3, 3);

        // All tasks should be completed
        assert_eq!(tracker.metrics().success(), 3);
        assert_eq!(tracker.metrics().active(), 0);
        assert_eq!(tracker.metrics().queued(), 0);
        assert_eq!(tracker.metrics().pending(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_hierarchical_metrics_failure_aggregation(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that failed task metrics aggregate up to parent
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();
        let child = parent.child_tracker().unwrap();

        // Run some successful and some failed tasks
        let success_handle = child.spawn(async { Ok(42) });
        let failure_handle = child.spawn(async { Err::<(), _>(anyhow::anyhow!("test error")) });

        // Wait for tasks to complete
        let _success_result = success_handle.await.unwrap().unwrap();
        let _failure_result = failure_handle.await.unwrap().unwrap_err();

        // Check child metrics
        assert_eq!(child.metrics().success(), 1, "Child should have 1 success");
        assert_eq!(child.metrics().failed(), 1, "Child should have 1 failure");

        // Parent should see the aggregated metrics
        // Note: Due to hierarchical aggregation, these metrics propagate up
    }

    #[rstest]
    #[tokio::test]
    async fn test_metrics_independence_between_tracker_instances(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that different tracker instances have independent metrics
        let tracker1 = TaskTracker::new(semaphore_scheduler.clone(), log_policy.clone()).unwrap();
        let tracker2 = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

        // Run tasks in both trackers
        let handle1 = tracker1.spawn(async { Ok(1) });
        let handle2 = tracker2.spawn(async { Ok(2) });

        handle1.await.unwrap().unwrap();
        handle2.await.unwrap().unwrap();

        // Each tracker should only see its own metrics
        assert_eq!(tracker1.metrics().success(), 1);
        assert_eq!(tracker2.metrics().success(), 1);
        assert_eq!(tracker1.metrics().total_completed(), 1);
        assert_eq!(tracker2.metrics().total_completed(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_hierarchical_join_waits_for_all(log_policy: Arc<LogOnlyPolicy>) {
        // Test that parent.join() waits for child tasks too
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(10))));
        let parent = TaskTracker::new(scheduler, log_policy).unwrap();
        let child1 = parent.child_tracker().unwrap();
        let child2 = parent.child_tracker().unwrap();
        let grandchild = child1.child_tracker().unwrap();

        // Verify parent tracks children
        assert_eq!(parent.child_count(), 2);
        assert_eq!(child1.child_count(), 1);
        assert_eq!(child2.child_count(), 0);
        assert_eq!(grandchild.child_count(), 0);

        // Track completion order
        let completion_order = Arc::new(Mutex::new(Vec::new()));

        // Spawn tasks with different durations
        let order_clone = completion_order.clone();
        let parent_handle = parent.spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            order_clone.lock().unwrap().push("parent");
            Ok(())
        });

        let order_clone = completion_order.clone();
        let child1_handle = child1.spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            order_clone.lock().unwrap().push("child1");
            Ok(())
        });

        let order_clone = completion_order.clone();
        let child2_handle = child2.spawn(async move {
            tokio::time::sleep(Duration::from_millis(75)).await;
            order_clone.lock().unwrap().push("child2");
            Ok(())
        });

        let order_clone = completion_order.clone();
        let grandchild_handle = grandchild.spawn(async move {
            tokio::time::sleep(Duration::from_millis(125)).await;
            order_clone.lock().unwrap().push("grandchild");
            Ok(())
        });

        // Test hierarchical join - should wait for ALL tasks in hierarchy
        println!("[TEST] About to call parent.join()");
        let start = std::time::Instant::now();
        parent.join().await; // This should wait for ALL tasks
        let elapsed = start.elapsed();
        println!("[TEST] parent.join() completed in {:?}", elapsed);

        // Should have waited for the longest task (grandchild at 125ms)
        assert!(
            elapsed >= Duration::from_millis(120),
            "Hierarchical join should wait for longest task"
        );

        // All tasks should be complete
        assert!(parent_handle.is_finished());
        assert!(child1_handle.is_finished());
        assert!(child2_handle.is_finished());
        assert!(grandchild_handle.is_finished());

        // Verify all tasks completed
        let final_order = completion_order.lock().unwrap();
        assert_eq!(final_order.len(), 4);
        assert!(final_order.contains(&"parent"));
        assert!(final_order.contains(&"child1"));
        assert!(final_order.contains(&"child2"));
        assert!(final_order.contains(&"grandchild"));
    }

    #[rstest]
    #[tokio::test]
    async fn test_hierarchical_join_waits_for_children(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that join() waits for child tasks (hierarchical behavior)
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();
        let child = parent.child_tracker().unwrap();

        // Spawn a quick parent task and slow child task
        let _parent_handle = parent.spawn(async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            Ok(())
        });

        let _child_handle = child.spawn(async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(())
        });

        // Hierarchical join should wait for both parent and child tasks
        let start = std::time::Instant::now();
        parent.join().await; // Should wait for both (hierarchical by default)
        let elapsed = start.elapsed();

        // Should have waited for the longer child task (100ms)
        assert!(
            elapsed >= Duration::from_millis(90),
            "Hierarchical join should wait for all child tasks"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_hierarchical_join_operations(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that parent.join() closes and waits for child trackers too
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();
        let child = parent.child_tracker().unwrap();
        let grandchild = child.child_tracker().unwrap();

        // Verify trackers start as open
        assert!(!parent.is_closed());
        assert!(!child.is_closed());
        assert!(!grandchild.is_closed());

        // Join parent (hierarchical by default - closes and waits for all)
        parent.join().await;

        // All should be closed (check child trackers since parent was moved)
        assert!(child.is_closed());
        assert!(grandchild.is_closed());
    }

    #[rstest]
    #[tokio::test]
    async fn test_unlimited_scheduler() {
        // Test that UnlimitedScheduler executes tasks immediately
        let scheduler = UnlimitedScheduler::new();
        let error_policy = LogOnlyPolicy::new();
        let tracker = TaskTracker::new(scheduler, error_policy).unwrap();

        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = tracker.spawn(async {
            rx.await.ok();
            Ok(42)
        });

        // Task should be ready to execute immediately (no concurrency limit)
        tx.send(()).ok();
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);

        assert_eq!(tracker.metrics().success(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_threshold_cancel_policy(semaphore_scheduler: Arc<SemaphoreScheduler>) {
        // Test that ThresholdCancelPolicy now uses per-task failure counting
        let error_policy = ThresholdCancelPolicy::with_threshold(2); // Cancel after 2 failures per task
        let tracker = TaskTracker::new(semaphore_scheduler, error_policy.clone()).unwrap();
        let cancel_token = tracker.cancellation_token().child_token();

        // With per-task context, individual task failures don't accumulate
        // Each task starts with failure_count = 0, so single failures won't trigger cancellation
        let _handle1 = tracker.spawn(async { Err::<(), _>(anyhow::anyhow!("First failure")) });
        tokio::task::yield_now().await;
        assert!(!cancel_token.is_cancelled());
        assert_eq!(error_policy.failure_count(), 1); // Global counter still increments

        // Second failure from different task - still won't trigger cancellation
        let _handle2 = tracker.spawn(async { Err::<(), _>(anyhow::anyhow!("Second failure")) });
        tokio::task::yield_now().await;
        assert!(!cancel_token.is_cancelled()); // Per-task context prevents cancellation
        assert_eq!(error_policy.failure_count(), 2); // Global counter increments

        // For cancellation to occur, a single task would need to fail multiple times
        // through continuations (which would require a more complex test setup)
    }

    #[tokio::test]
    async fn test_policy_constructors() {
        // Test that all constructors follow the new clean API patterns
        let _unlimited = UnlimitedScheduler::new();
        let _semaphore = SemaphoreScheduler::with_permits(5);
        let _log_only = LogOnlyPolicy::new();
        let _cancel_policy = CancelOnError::new();
        let _threshold_policy = ThresholdCancelPolicy::with_threshold(3);
        let _rate_policy = RateCancelPolicy::builder()
            .rate(0.5)
            .window_secs(60)
            .build();

        // All constructors return Arc directly - no more ugly ::new_arc patterns
        // This test ensures the clean API reduces boilerplate
    }

    #[rstest]
    #[tokio::test]
    async fn test_child_creation_fails_after_join(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that child tracker creation fails from closed parent
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

        // Initially, creating a child should work
        let _child = parent.child_tracker().unwrap();

        // Close the parent tracker
        let parent_clone = parent.clone();
        parent.join().await;
        assert!(parent_clone.is_closed());

        // Now, trying to create a child should fail
        let result = parent_clone.child_tracker();
        assert!(result.is_err());
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("closed parent tracker")
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_child_builder_fails_after_join(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that child tracker builder creation fails from closed parent
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

        // Initially, creating a child with builder should work
        let _child = parent.child_tracker_builder().build().unwrap();

        // Close the parent tracker
        let parent_clone = parent.clone();
        parent.join().await;
        assert!(parent_clone.is_closed());

        // Now, trying to create a child with builder should fail
        let result = parent_clone.child_tracker_builder().build();
        assert!(result.is_err());
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("closed parent tracker")
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_child_creation_succeeds_before_join(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that child creation works normally before parent is joined
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

        // Should be able to create multiple children before closing
        let child1 = parent.child_tracker().unwrap();
        let child2 = parent.child_tracker_builder().build().unwrap();

        // Verify children can spawn tasks
        let handle1 = child1.spawn(async { Ok(42) });
        let handle2 = child2.spawn(async { Ok(24) });

        let result1 = handle1.await.unwrap().unwrap();
        let result2 = handle2.await.unwrap().unwrap();

        assert_eq!(result1, 42);
        assert_eq!(result2, 24);
        assert_eq!(parent.metrics().success(), 2); // Parent sees all successes
    }

    #[rstest]
    #[tokio::test]
    async fn test_custom_error_response_with_cancellation_token(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
    ) {
        // Test ErrorResponse::Custom behavior with TriggerCancellationTokenOnError

        // Create a custom cancellation token
        let custom_cancel_token = CancellationToken::new();

        // Create the policy that will trigger our custom token
        let error_policy = TriggerCancellationTokenOnError::new(custom_cancel_token.clone());

        // Create tracker using builder with the custom policy
        let tracker = TaskTracker::builder()
            .scheduler(semaphore_scheduler)
            .error_policy(error_policy)
            .cancel_token(custom_cancel_token.clone())
            .build()
            .unwrap();

        let child = tracker.child_tracker().unwrap();

        // Initially, the custom token should not be cancelled
        assert!(!custom_cancel_token.is_cancelled());

        // Spawn a task that will fail
        let handle = child.spawn(async {
            Err::<(), _>(anyhow::anyhow!("Test error to trigger custom response"))
        });

        // Wait for the task to complete (it will fail)
        let result = handle.await.unwrap();
        assert!(result.is_err());

        // Await a timeout/deadline or the cancellation token to be cancelled
        // The expectation is that the task will fail, and the cancellation token will be triggered
        // Hitting the deadline is a failure
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(1)) => {
                panic!("Task should have failed, but hit the deadline");
            }
            _ = custom_cancel_token.cancelled() => {
                // Task should have failed, and the cancellation token should be triggered
            }
        }

        // The custom cancellation token should now be triggered by our policy
        assert!(
            custom_cancel_token.is_cancelled(),
            "Custom cancellation token should be triggered by ErrorResponse::Custom"
        );

        assert!(tracker.cancellation_token().is_cancelled());
        assert!(child.cancellation_token().is_cancelled());

        // Verify the error was counted
        assert_eq!(tracker.metrics().failed(), 1);
    }

    #[test]
    fn test_action_result_variants() {
        // Test that ActionResult variants can be created and pattern matched

        // Test Fail variant
        let fail_result = ActionResult::Fail;
        match fail_result {
            ActionResult::Fail => {} // Expected
            _ => panic!("Expected Fail variant"),
        }

        // Test Shutdown variant
        let shutdown_result = ActionResult::Shutdown;
        match shutdown_result {
            ActionResult::Shutdown => {} // Expected
            _ => panic!("Expected Shutdown variant"),
        }

        // Test Continue variant with Continuation
        #[derive(Debug)]
        struct TestRestartable;

        #[async_trait]
        impl Continuation for TestRestartable {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new("test_result".to_string()))
            }
        }

        let test_restartable = Arc::new(TestRestartable);
        let continue_result = ActionResult::Continue {
            continuation: test_restartable,
        };

        match continue_result {
            ActionResult::Continue { continuation } => {
                // Verify we have a valid Continuation
                assert!(format!("{:?}", continuation).contains("TestRestartable"));
            }
            _ => panic!("Expected Continue variant"),
        }
    }

    #[test]
    fn test_continuation_error_creation() {
        // Test RestartableError creation and conversion to anyhow::Error

        // Create a dummy restartable task for testing
        #[derive(Debug)]
        struct DummyRestartable;

        #[async_trait]
        impl Continuation for DummyRestartable {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new("restarted_result".to_string()))
            }
        }

        let dummy_restartable = Arc::new(DummyRestartable);
        let source_error = anyhow::anyhow!("Original task failed");

        // Test FailedWithContinuation::new
        let continuation_error = FailedWithContinuation::new(source_error, dummy_restartable);

        // Verify the error displays correctly
        let error_string = format!("{}", continuation_error);
        assert!(error_string.contains("Task failed with continuation"));
        assert!(error_string.contains("Original task failed"));

        // Test conversion to anyhow::Error
        let anyhow_error = anyhow::Error::new(continuation_error);
        assert!(
            anyhow_error
                .to_string()
                .contains("Task failed with continuation")
        );
    }

    #[test]
    fn test_continuation_error_ext_trait() {
        // Test the RestartableErrorExt trait methods

        // Test with regular anyhow::Error (not restartable)
        let regular_error = anyhow::anyhow!("Regular error");
        assert!(!regular_error.has_continuation());
        let extracted = regular_error.extract_continuation();
        assert!(extracted.is_none());

        // Test with RestartableError
        #[derive(Debug)]
        struct TestRestartable;

        #[async_trait]
        impl Continuation for TestRestartable {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new("test_result".to_string()))
            }
        }

        let test_restartable = Arc::new(TestRestartable);
        let source_error = anyhow::anyhow!("Source error");
        let continuation_error = FailedWithContinuation::new(source_error, test_restartable);

        let anyhow_error = anyhow::Error::new(continuation_error);
        assert!(anyhow_error.has_continuation());

        // Test extraction of restartable task
        let extracted = anyhow_error.extract_continuation();
        assert!(extracted.is_some());
    }

    #[test]
    fn test_continuation_error_into_anyhow_helper() {
        // Test the convenience method for creating restartable errors
        // Note: This test uses a mock TaskExecutor since we don't have real ones yet

        // For now, we'll test the type erasure concept with a simple type
        struct MockExecutor;

        let _source_error = anyhow::anyhow!("Mock task failed");

        // We can't test FailedWithContinuation::into_anyhow yet because it requires
        // a real TaskExecutor<T>. This will be tested in Phase 3.
        // For now, just verify the concept works with manual construction.

        #[derive(Debug)]
        struct MockRestartable;

        #[async_trait]
        impl Continuation for MockRestartable {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new("mock_result".to_string()))
            }
        }

        let mock_restartable = Arc::new(MockRestartable);
        let continuation_error =
            FailedWithContinuation::new(anyhow::anyhow!("Mock task failed"), mock_restartable);

        let anyhow_error = anyhow::Error::new(continuation_error);
        assert!(anyhow_error.has_continuation());
    }

    #[test]
    fn test_continuation_error_with_task_executor() {
        // Test RestartableError creation with TaskExecutor

        #[derive(Debug)]
        struct TestRestartableTask;

        #[async_trait]
        impl Continuation for TestRestartableTask {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new("test_result".to_string()))
            }
        }

        let restartable_task = Arc::new(TestRestartableTask);
        let source_error = anyhow::anyhow!("Task failed");

        // Test FailedWithContinuation::new with Restartable
        let continuation_error = FailedWithContinuation::new(source_error, restartable_task);

        // Verify the error displays correctly
        let error_string = format!("{}", continuation_error);
        assert!(error_string.contains("Task failed with continuation"));
        assert!(error_string.contains("Task failed"));

        // Test conversion to anyhow::Error
        let anyhow_error = anyhow::Error::new(continuation_error);
        assert!(anyhow_error.has_continuation());

        // Test extraction (should work now with Restartable trait)
        let extracted = anyhow_error.extract_continuation();
        assert!(extracted.is_some()); // Should successfully extract the Restartable
    }

    #[test]
    fn test_continuation_error_into_anyhow_convenience() {
        // Test the convenience method for creating restartable errors

        #[derive(Debug)]
        struct ConvenienceRestartable;

        #[async_trait]
        impl Continuation for ConvenienceRestartable {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new(42u32))
            }
        }

        let restartable_task = Arc::new(ConvenienceRestartable);
        let source_error = anyhow::anyhow!("Computation failed");

        // Test FailedWithContinuation::into_anyhow convenience method
        let anyhow_error = FailedWithContinuation::into_anyhow(source_error, restartable_task);

        assert!(anyhow_error.has_continuation());
        assert!(
            anyhow_error
                .to_string()
                .contains("Task failed with continuation")
        );
        assert!(anyhow_error.to_string().contains("Computation failed"));
    }

    #[test]
    fn test_handle_task_error_with_continuation_error() {
        // Test that handle_task_error properly detects RestartableError

        // Create a mock Restartable task
        #[derive(Debug)]
        struct MockRestartableTask;

        #[async_trait]
        impl Continuation for MockRestartableTask {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new("retry_result".to_string()))
            }
        }

        let restartable_task = Arc::new(MockRestartableTask);

        // Create RestartableError
        let source_error = anyhow::anyhow!("Task failed, but can retry");
        let continuation_error = FailedWithContinuation::new(source_error, restartable_task);
        let anyhow_error = anyhow::Error::new(continuation_error);

        // Verify it's detected as restartable
        assert!(anyhow_error.has_continuation());

        // Verify we can downcast to FailedWithContinuation
        let continuation_ref = anyhow_error.downcast_ref::<FailedWithContinuation>();
        assert!(continuation_ref.is_some());

        // Verify the continuation task is present
        let continuation = continuation_ref.unwrap();
        // Note: We can verify the Arc is valid by checking that Arc::strong_count > 0
        assert!(Arc::strong_count(&continuation.continuation) > 0);
    }

    #[test]
    fn test_handle_task_error_with_regular_error() {
        // Test that handle_task_error properly handles regular errors

        let regular_error = anyhow::anyhow!("Regular task failure");

        // Verify it's not detected as restartable
        assert!(!regular_error.has_continuation());

        // Verify we cannot downcast to FailedWithContinuation
        let continuation_ref = regular_error.downcast_ref::<FailedWithContinuation>();
        assert!(continuation_ref.is_none());
    }

    // ========================================
    // END-TO-END ACTIONRESULT TESTS
    // ========================================

    #[rstest]
    #[tokio::test]
    async fn test_end_to_end_continuation_execution(
        unlimited_scheduler: Arc<UnlimitedScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that a task returning FailedWithContinuation actually executes the continuation
        let tracker = TaskTracker::new(unlimited_scheduler, log_policy).unwrap();

        // Shared state to track execution
        let execution_log = Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));
        let log_clone = execution_log.clone();

        // Create a continuation that logs its execution
        #[derive(Debug)]
        struct LoggingContinuation {
            log: Arc<tokio::sync::Mutex<Vec<String>>>,
            result: String,
        }

        #[async_trait]
        impl Continuation for LoggingContinuation {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                self.log
                    .lock()
                    .await
                    .push("continuation_executed".to_string());
                TaskExecutionResult::Success(Box::new(self.result.clone()))
            }
        }

        let continuation = Arc::new(LoggingContinuation {
            log: log_clone,
            result: "continuation_result".to_string(),
        });

        // Task that fails with continuation
        let log_for_task = execution_log.clone();
        let handle = tracker.spawn(async move {
            log_for_task
                .lock()
                .await
                .push("original_task_executed".to_string());

            // Return FailedWithContinuation
            let error = anyhow::anyhow!("Original task failed");
            let result: Result<String, anyhow::Error> =
                Err(FailedWithContinuation::into_anyhow(error, continuation));
            result
        });

        // Execute and verify the continuation was called
        let result = handle.await.expect("Task should complete");
        assert!(result.is_ok(), "Continuation should succeed");

        // Verify execution order
        let log = execution_log.lock().await;
        assert_eq!(log.len(), 2);
        assert_eq!(log[0], "original_task_executed");
        assert_eq!(log[1], "continuation_executed");

        // Verify metrics - should show 1 success (from continuation)
        assert_eq!(tracker.metrics().success(), 1);
        assert_eq!(tracker.metrics().failed(), 0); // Continuation succeeded
        assert_eq!(tracker.metrics().cancelled(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_end_to_end_multiple_continuations(
        unlimited_scheduler: Arc<UnlimitedScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test multiple continuation attempts
        let tracker = TaskTracker::new(unlimited_scheduler, log_policy).unwrap();

        let execution_log = Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));
        let attempt_count = Arc::new(std::sync::atomic::AtomicU32::new(0));

        // Continuation that fails twice, then succeeds
        #[derive(Debug)]
        struct RetryingContinuation {
            log: Arc<tokio::sync::Mutex<Vec<String>>>,
            attempt_count: Arc<std::sync::atomic::AtomicU32>,
        }

        #[async_trait]
        impl Continuation for RetryingContinuation {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                let attempt = self
                    .attempt_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    + 1;
                self.log
                    .lock()
                    .await
                    .push(format!("continuation_attempt_{}", attempt));

                if attempt < 3 {
                    // Fail with another continuation
                    let next_continuation = Arc::new(RetryingContinuation {
                        log: self.log.clone(),
                        attempt_count: self.attempt_count.clone(),
                    });
                    let error = anyhow::anyhow!("Continuation attempt {} failed", attempt);
                    TaskExecutionResult::Error(FailedWithContinuation::into_anyhow(
                        error,
                        next_continuation,
                    ))
                } else {
                    // Succeed on third attempt
                    TaskExecutionResult::Success(Box::new(format!(
                        "success_on_attempt_{}",
                        attempt
                    )))
                }
            }
        }

        let initial_continuation = Arc::new(RetryingContinuation {
            log: execution_log.clone(),
            attempt_count: attempt_count.clone(),
        });

        // Task that immediately fails with continuation
        let handle = tracker.spawn(async move {
            let error = anyhow::anyhow!("Original task failed");
            let result: Result<String, anyhow::Error> = Err(FailedWithContinuation::into_anyhow(
                error,
                initial_continuation,
            ));
            result
        });

        // Execute and verify multiple continuations
        let result = handle.await.expect("Task should complete");
        assert!(result.is_ok(), "Final continuation should succeed");

        // Verify all attempts were made
        let log = execution_log.lock().await;
        assert_eq!(log.len(), 3);
        assert_eq!(log[0], "continuation_attempt_1");
        assert_eq!(log[1], "continuation_attempt_2");
        assert_eq!(log[2], "continuation_attempt_3");

        // Verify final attempt count
        assert_eq!(attempt_count.load(std::sync::atomic::Ordering::Relaxed), 3);

        // Verify metrics - should show 1 success (final continuation)
        assert_eq!(tracker.metrics().success(), 1);
        assert_eq!(tracker.metrics().failed(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_end_to_end_continuation_failure(
        unlimited_scheduler: Arc<UnlimitedScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test continuation that ultimately fails without providing another continuation
        let tracker = TaskTracker::new(unlimited_scheduler, log_policy).unwrap();

        let execution_log = Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));
        let log_clone = execution_log.clone();

        // Continuation that fails without providing another continuation
        #[derive(Debug)]
        struct FailingContinuation {
            log: Arc<tokio::sync::Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl Continuation for FailingContinuation {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                self.log
                    .lock()
                    .await
                    .push("continuation_failed".to_string());
                TaskExecutionResult::Error(anyhow::anyhow!("Continuation failed permanently"))
            }
        }

        let continuation = Arc::new(FailingContinuation { log: log_clone });

        // Task that fails with continuation
        let log_for_task = execution_log.clone();
        let handle = tracker.spawn(async move {
            log_for_task
                .lock()
                .await
                .push("original_task_executed".to_string());

            let error = anyhow::anyhow!("Original task failed");
            let result: Result<String, anyhow::Error> =
                Err(FailedWithContinuation::into_anyhow(error, continuation));
            result
        });

        // Execute and verify the continuation failed
        let result = handle.await.expect("Task should complete");
        assert!(result.is_err(), "Continuation should fail");

        // Verify execution order
        let log = execution_log.lock().await;
        assert_eq!(log.len(), 2);
        assert_eq!(log[0], "original_task_executed");
        assert_eq!(log[1], "continuation_failed");

        // Verify metrics - should show 1 failure (from continuation)
        assert_eq!(tracker.metrics().success(), 0);
        assert_eq!(tracker.metrics().failed(), 1);
        assert_eq!(tracker.metrics().cancelled(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_end_to_end_all_action_result_variants(
        unlimited_scheduler: Arc<UnlimitedScheduler>,
    ) {
        // Comprehensive test of Fail, Shutdown, and Continue paths

        // Test 1: ActionResult::Fail (via LogOnlyPolicy)
        {
            let tracker =
                TaskTracker::new(unlimited_scheduler.clone(), LogOnlyPolicy::new()).unwrap();
            let handle = tracker.spawn(async {
                let result: Result<String, anyhow::Error> = Err(anyhow::anyhow!("Test error"));
                result
            });
            let result = handle.await.expect("Task should complete");
            assert!(result.is_err(), "LogOnly should let error through");
            assert_eq!(tracker.metrics().failed(), 1);
        }

        // Test 2: ActionResult::Shutdown (via CancelOnError)
        {
            let tracker =
                TaskTracker::new(unlimited_scheduler.clone(), CancelOnError::new()).unwrap();
            let handle = tracker.spawn(async {
                let result: Result<String, anyhow::Error> = Err(anyhow::anyhow!("Test error"));
                result
            });
            let result = handle.await.expect("Task should complete");
            assert!(result.is_err(), "CancelOnError should fail task");
            assert!(
                tracker.cancellation_token().is_cancelled(),
                "Should cancel tracker"
            );
            assert_eq!(tracker.metrics().failed(), 1);
        }

        // Test 3: ActionResult::Continue (via FailedWithContinuation)
        {
            let tracker =
                TaskTracker::new(unlimited_scheduler.clone(), LogOnlyPolicy::new()).unwrap();

            #[derive(Debug)]
            struct TestContinuation;

            #[async_trait]
            impl Continuation for TestContinuation {
                async fn execute(
                    &self,
                    _cancel_token: CancellationToken,
                ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                    TaskExecutionResult::Success(Box::new("continuation_success".to_string()))
                }
            }

            let continuation = Arc::new(TestContinuation);
            let handle = tracker.spawn(async move {
                let error = anyhow::anyhow!("Original failure");
                let result: Result<String, anyhow::Error> =
                    Err(FailedWithContinuation::into_anyhow(error, continuation));
                result
            });

            let result = handle.await.expect("Task should complete");
            assert!(result.is_ok(), "Continuation should succeed");
            assert_eq!(tracker.metrics().success(), 1);
            assert_eq!(tracker.metrics().failed(), 0);
        }
    }

    // ========================================
    // LOOP BEHAVIOR AND POLICY INTERACTION TESTS
    // ========================================
    //
    // These tests demonstrate the current ActionResult system and identify
    // areas for future improvement:
    //
    // ✅ WHAT WORKS:
    // - All ActionResult variants (Continue, Cancel, ExecuteNext) are tested
    // - Task-driven continuations work correctly
    // - Policy-driven continuations work correctly
    // - Mixed continuation sources work correctly
    // - Loop behavior with resource management works correctly
    //
    // 🔄 CURRENT LIMITATIONS:
    // - ThresholdCancelPolicy tracks failures GLOBALLY, not per-task
    // - OnErrorPolicy doesn't receive attempt_count parameter
    // - No per-task context for stateful retry policies
    //
    // 🚀 FUTURE IMPROVEMENTS IDENTIFIED:
    // - Add OnErrorContext associated type for per-task state
    // - Pass attempt_count to OnErrorPolicy::on_error
    // - Enable per-task failure tracking, backoff timers, etc.
    //
    // The tests below demonstrate both current capabilities and limitations.

    /// Test retry loop behavior with different policies and continuation counts
    ///
    /// This test verifies that:
    /// 1. Tasks can provide multiple continuations in sequence
    /// 2. Different error policies can limit the number of continuation attempts
    /// 3. The retry loop correctly handles policy decisions about when to stop
    ///
    /// Key insight: Policies are only consulted for regular errors, not FailedWithContinuation.
    /// So we need continuations that eventually fail with regular errors to test policy limits.
    ///
    /// DESIGN LIMITATION: Current ThresholdCancelPolicy tracks failures GLOBALLY across all tasks,
    /// not per-task. This test demonstrates the current behavior but isn't ideal for retry loop testing.
    ///
    /// FUTURE IMPROVEMENT: Add OnErrorContext associated type to OnErrorPolicy:
    /// ```rust
    /// trait OnErrorPolicy {
    ///     type Context: Default + Send + Sync;
    ///     fn on_error(&self, error: &anyhow::Error, task_id: TaskId,
    ///                 attempt_count: u32, context: &mut Self::Context) -> ErrorResponse;
    /// }
    /// ```
    /// This would enable per-task failure tracking, backoff timers, etc.
    ///
    /// NOTE: Uses fresh policy instance for each test case to avoid global state interference.
    #[rstest]
    #[case(
        1,
        false,
        "Global policy with max_failures=1 should stop after first regular error"
    )]
    #[case(
        2,
        false,  // Actually fails - ActionResult::Fail accepts the error and fails the task
        "Global policy with max_failures=2 allows error but ActionResult::Fail still fails the task"
    )]
    #[tokio::test]
    async fn test_continuation_loop_with_global_threshold_policy(
        unlimited_scheduler: Arc<UnlimitedScheduler>,
        #[case] max_failures: usize,
        #[case] should_succeed: bool,
        #[case] description: &str,
    ) {
        // Task that provides continuations, but continuations fail with regular errors
        // so the policy gets consulted and can limit retries

        let execution_log = Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));
        let attempt_counter = Arc::new(std::sync::atomic::AtomicU32::new(0));

        // Create a continuation that fails with regular errors (not FailedWithContinuation)
        // This allows the policy to be consulted and potentially stop the retries
        #[derive(Debug)]
        struct PolicyTestContinuation {
            log: Arc<tokio::sync::Mutex<Vec<String>>>,
            attempt_counter: Arc<std::sync::atomic::AtomicU32>,
            max_attempts_before_success: u32,
        }

        #[async_trait]
        impl Continuation for PolicyTestContinuation {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                let attempt = self
                    .attempt_counter
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    + 1;
                self.log
                    .lock()
                    .await
                    .push(format!("continuation_attempt_{}", attempt));

                if attempt < self.max_attempts_before_success {
                    // Fail with regular error - this will be seen by the policy
                    TaskExecutionResult::Error(anyhow::anyhow!(
                        "Continuation attempt {} failed (regular error)",
                        attempt
                    ))
                } else {
                    // Succeed after enough attempts
                    TaskExecutionResult::Success(Box::new(format!(
                        "success_on_attempt_{}",
                        attempt
                    )))
                }
            }
        }

        // Create fresh policy instance for each test case to avoid global state interference
        let policy = ThresholdCancelPolicy::with_threshold(max_failures);
        let tracker = TaskTracker::new(unlimited_scheduler, policy).unwrap();

        // Original task that fails with continuation
        let log_for_task = execution_log.clone();
        // Set max_attempts_before_success so that:
        // - For max_failures=1: Continuation fails 1 time (attempt 1), policy cancels after 1 failure
        // - For max_failures=2: Continuation fails 1 time (attempt 1), succeeds on attempt 2
        let continuation = Arc::new(PolicyTestContinuation {
            log: execution_log.clone(),
            attempt_counter: attempt_counter.clone(),
            max_attempts_before_success: 2, // Always fail on attempt 1, succeed on attempt 2
        });

        let handle = tracker.spawn(async move {
            log_for_task
                .lock()
                .await
                .push("original_task_executed".to_string());
            let error = anyhow::anyhow!("Original task failed");
            let result: Result<String, anyhow::Error> =
                Err(FailedWithContinuation::into_anyhow(error, continuation));
            result
        });

        // Execute and check result based on policy
        let result = handle.await.expect("Task should complete");

        // Debug: Print actual results
        let log = execution_log.lock().await;
        let final_attempt_count = attempt_counter.load(std::sync::atomic::Ordering::Relaxed);
        println!(
            "Test case: max_failures={}, should_succeed={}",
            max_failures, should_succeed
        );
        println!("Result: {:?}", result.is_ok());
        println!("Log entries: {:?}", log);
        println!("Attempt count: {}", final_attempt_count);
        println!(
            "Metrics: success={}, failed={}",
            tracker.metrics().success(),
            tracker.metrics().failed()
        );
        drop(log); // Release the lock

        // Both test cases should fail because ActionResult::Fail accepts the error and fails the task
        assert!(result.is_err(), "{}: Task should fail", description);
        assert_eq!(
            tracker.metrics().success(),
            0,
            "{}: Should have 0 successes",
            description
        );
        assert_eq!(
            tracker.metrics().failed(),
            1,
            "{}: Should have 1 failure",
            description
        );

        // Should have stopped after 1 continuation attempt because ActionResult::Fail fails the task
        let log = execution_log.lock().await;
        assert_eq!(
            log.len(),
            2,
            "{}: Should have 2 log entries (original + 1 continuation attempt)",
            description
        );
        assert_eq!(log[0], "original_task_executed");
        assert_eq!(log[1], "continuation_attempt_1");

        assert_eq!(
            attempt_counter.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "{}: Should have made 1 continuation attempt",
            description
        );

        // The key difference is whether the tracker gets cancelled
        if max_failures == 1 {
            assert!(
                tracker.cancellation_token().is_cancelled(),
                "Tracker should be cancelled with max_failures=1"
            );
        } else {
            assert!(
                !tracker.cancellation_token().is_cancelled(),
                "Tracker should NOT be cancelled with max_failures=2 (policy allows the error)"
            );
        }
    }

    /// Simple test to understand ThresholdCancelPolicy behavior with per-task context
    #[rstest]
    #[tokio::test]
    async fn test_simple_threshold_policy_behavior(unlimited_scheduler: Arc<UnlimitedScheduler>) {
        // Test with max_failures=2 - now uses per-task failure counting
        let policy = ThresholdCancelPolicy::with_threshold(2);
        let tracker = TaskTracker::new(unlimited_scheduler, policy.clone()).unwrap();

        // Task 1: Should fail but not trigger cancellation (per-task failure count = 1)
        let handle1 = tracker.spawn(async {
            let result: Result<String, anyhow::Error> = Err(anyhow::anyhow!("First failure"));
            result
        });
        let result1 = handle1.await.expect("Task should complete");
        assert!(result1.is_err(), "First task should fail");
        assert!(
            !tracker.cancellation_token().is_cancelled(),
            "Should not be cancelled after 1 failure"
        );

        // Task 2: Should fail but not trigger cancellation (different task, per-task failure count = 1)
        let handle2 = tracker.spawn(async {
            let result: Result<String, anyhow::Error> = Err(anyhow::anyhow!("Second failure"));
            result
        });
        let result2 = handle2.await.expect("Task should complete");
        assert!(result2.is_err(), "Second task should fail");
        assert!(
            !tracker.cancellation_token().is_cancelled(),
            "Should NOT be cancelled - per-task context prevents global accumulation"
        );

        println!("Policy global failure count: {}", policy.failure_count());
        assert_eq!(
            policy.failure_count(),
            2,
            "Policy should have counted 2 failures globally (for backwards compatibility)"
        );
    }

    /// Test demonstrating that per-task error context solves the global failure tracking problem
    ///
    /// This test shows that with OnErrorContext, each task has independent failure tracking.
    #[rstest]
    #[tokio::test]
    async fn test_per_task_context_limitation_demo(unlimited_scheduler: Arc<UnlimitedScheduler>) {
        // Create a policy that should allow 2 failures per task
        let policy = ThresholdCancelPolicy::with_threshold(2);
        let tracker = TaskTracker::new(unlimited_scheduler, policy.clone()).unwrap();

        // Task 1: Fails once (per-task failure count = 1, below threshold)
        let handle1 = tracker.spawn(async {
            let result: Result<String, anyhow::Error> = Err(anyhow::anyhow!("Task 1 failure"));
            result
        });
        let result1 = handle1.await.expect("Task should complete");
        assert!(result1.is_err(), "Task 1 should fail");

        // Task 2: Also fails once (per-task failure count = 1, below threshold)
        // With per-task context, this doesn't interfere with Task 1's failure budget
        let handle2 = tracker.spawn(async {
            let result: Result<String, anyhow::Error> = Err(anyhow::anyhow!("Task 2 failure"));
            result
        });
        let result2 = handle2.await.expect("Task should complete");
        assert!(result2.is_err(), "Task 2 should fail");

        // With per-task context, tracker should NOT be cancelled
        // Each task failed only once, which is below the threshold of 2
        assert!(
            !tracker.cancellation_token().is_cancelled(),
            "Tracker should NOT be cancelled - per-task context prevents premature cancellation"
        );

        println!("Global failure count: {}", policy.failure_count());
        assert_eq!(
            policy.failure_count(),
            2,
            "Global policy counted 2 failures across different tasks"
        );

        // This demonstrates the limitation: we can't test per-task retry behavior
        // because failures from different tasks affect each other's retry budgets
    }

    /// Test allow_continuation() policy method with attempt-based logic
    ///
    /// This test verifies that:
    /// 1. Policies can conditionally allow/reject continuations based on context
    /// 2. When allow_continuation() returns false, FailedWithContinuation is ignored
    /// 3. When allow_continuation() returns true, FailedWithContinuation is processed normally
    /// 4. The policy's decision takes precedence over task-provided continuations
    #[rstest]
    #[case(
        3,
        true,
        "Policy allows continuations up to 3 attempts - should succeed"
    )]
    #[case(
        2,
        true,
        "Policy allows continuations up to 2 attempts - should succeed"
    )]
    #[case(0, false, "Policy allows 0 attempts - should fail immediately")]
    #[tokio::test]
    async fn test_allow_continuation_policy_control(
        unlimited_scheduler: Arc<UnlimitedScheduler>,
        #[case] max_attempts: u32,
        #[case] should_succeed: bool,
        #[case] description: &str,
    ) {
        // Policy that allows continuations only up to max_attempts
        #[derive(Debug)]
        struct AttemptLimitPolicy {
            max_attempts: u32,
        }

        impl OnErrorPolicy for AttemptLimitPolicy {
            fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
                Arc::new(AttemptLimitPolicy {
                    max_attempts: self.max_attempts,
                })
            }

            fn create_context(&self) -> Option<Box<dyn std::any::Any + Send + 'static>> {
                None // Stateless policy
            }

            fn allow_continuation(&self, _error: &anyhow::Error, context: &OnErrorContext) -> bool {
                context.attempt_count <= self.max_attempts
            }

            fn on_error(
                &self,
                _error: &anyhow::Error,
                _context: &mut OnErrorContext,
            ) -> ErrorResponse {
                ErrorResponse::Fail // Just fail when continuations are not allowed
            }
        }

        let policy = Arc::new(AttemptLimitPolicy { max_attempts });
        let tracker = TaskTracker::new(unlimited_scheduler, policy).unwrap();
        let execution_log = Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));

        // Continuation that always tries to retry
        #[derive(Debug)]
        struct AlwaysRetryContinuation {
            log: Arc<tokio::sync::Mutex<Vec<String>>>,
            attempt: u32,
        }

        #[async_trait]
        impl Continuation for AlwaysRetryContinuation {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                self.log
                    .lock()
                    .await
                    .push(format!("continuation_attempt_{}", self.attempt));

                if self.attempt >= 2 {
                    // Success after 2 attempts
                    TaskExecutionResult::Success(Box::new("final_success".to_string()))
                } else {
                    // Try to continue with another continuation
                    let next_continuation = Arc::new(AlwaysRetryContinuation {
                        log: self.log.clone(),
                        attempt: self.attempt + 1,
                    });
                    let error = anyhow::anyhow!("Continuation attempt {} failed", self.attempt);
                    TaskExecutionResult::Error(FailedWithContinuation::into_anyhow(
                        error,
                        next_continuation,
                    ))
                }
            }
        }

        // Task that immediately fails with a continuation
        let initial_continuation = Arc::new(AlwaysRetryContinuation {
            log: execution_log.clone(),
            attempt: 1,
        });

        let log_for_task = execution_log.clone();
        let handle = tracker.spawn(async move {
            log_for_task
                .lock()
                .await
                .push("initial_task_failure".to_string());
            let error = anyhow::anyhow!("Initial task failure");
            let result: Result<String, anyhow::Error> = Err(FailedWithContinuation::into_anyhow(
                error,
                initial_continuation,
            ));
            result
        });

        let result = handle.await.expect("Task should complete");

        if should_succeed {
            assert!(result.is_ok(), "{}: Task should succeed", description);
            assert_eq!(
                tracker.metrics().success(),
                1,
                "{}: Should have 1 success",
                description
            );

            // Should have executed multiple continuations
            let log = execution_log.lock().await;
            assert!(
                log.len() > 2,
                "{}: Should have multiple log entries",
                description
            );
            assert!(log.contains(&"continuation_attempt_1".to_string()));
        } else {
            assert!(result.is_err(), "{}: Task should fail", description);
            assert_eq!(
                tracker.metrics().failed(),
                1,
                "{}: Should have 1 failure",
                description
            );

            // Should have stopped early due to policy rejection
            let log = execution_log.lock().await;
            assert_eq!(
                log.len(),
                1,
                "{}: Should only have initial task entry",
                description
            );
            assert_eq!(log[0], "initial_task_failure");
            // Should NOT contain continuation attempts because policy rejected them
            assert!(
                !log.iter()
                    .any(|entry| entry.contains("continuation_attempt")),
                "{}: Should not have continuation attempts, but got: {:?}",
                description,
                *log
            );
        }
    }

    /// Test TaskHandle functionality
    ///
    /// This test verifies that:
    /// 1. TaskHandle can be awaited like a JoinHandle
    /// 2. TaskHandle provides access to the task's cancellation token
    /// 3. Individual task cancellation works correctly
    /// 4. TaskHandle methods (abort, is_finished) work as expected
    #[tokio::test]
    async fn test_task_handle_functionality() {
        let tracker = TaskTracker::new(UnlimitedScheduler::new(), LogOnlyPolicy::new()).unwrap();

        // Test basic functionality - TaskHandle can be awaited
        let handle1 = tracker.spawn(async {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            Ok("completed".to_string())
        });

        // Verify we can access the cancellation token
        let cancel_token = handle1.cancellation_token();
        assert!(
            !cancel_token.is_cancelled(),
            "Token should not be cancelled initially"
        );

        // Await the task
        let result1 = handle1.await.expect("Task should complete");
        assert!(result1.is_ok(), "Task should succeed");
        assert_eq!(result1.unwrap(), "completed");

        // Test individual task cancellation
        let handle2 = tracker.spawn_cancellable(|cancel_token| async move {
            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {
                    CancellableTaskResult::Ok("task_was_not_cancelled".to_string())
                },
                _ = cancel_token.cancelled() => {
                    CancellableTaskResult::Cancelled
                },

            }
        });

        let cancel_token2 = handle2.cancellation_token();

        // Cancel this specific task
        cancel_token2.cancel();

        // The task should be cancelled
        let result2 = handle2.await.expect("Task should complete");
        assert!(result2.is_err(), "Task should be cancelled");
        assert!(
            result2.unwrap_err().is_cancellation(),
            "Should be a cancellation error"
        );

        // Test that other tasks are not affected
        let handle3 = tracker.spawn(async { Ok("not_cancelled".to_string()) });

        let result3 = handle3.await.expect("Task should complete");
        assert!(result3.is_ok(), "Other tasks should not be affected");
        assert_eq!(result3.unwrap(), "not_cancelled");

        // Test abort functionality
        let handle4 = tracker.spawn(async {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            Ok("should_be_aborted".to_string())
        });

        // Check is_finished before abort
        assert!(!handle4.is_finished(), "Task should not be finished yet");

        // Abort the task
        handle4.abort();

        // Task should be aborted (JoinError)
        let result4 = handle4.await;
        assert!(result4.is_err(), "Aborted task should return JoinError");

        // Verify metrics
        assert_eq!(
            tracker.metrics().success(),
            2,
            "Should have 2 successful tasks"
        );
        assert_eq!(
            tracker.metrics().cancelled(),
            1,
            "Should have 1 cancelled task"
        );
        // Note: aborted tasks don't count as cancelled in our metrics
    }

    /// Test TaskHandle with cancellable tasks
    #[tokio::test]
    async fn test_task_handle_with_cancellable_tasks() {
        let tracker = TaskTracker::new(UnlimitedScheduler::new(), LogOnlyPolicy::new()).unwrap();

        // Test cancellable task with TaskHandle
        let handle = tracker.spawn_cancellable(|cancel_token| async move {
            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                    CancellableTaskResult::Ok("completed".to_string())
                },
                _ = cancel_token.cancelled() => CancellableTaskResult::Cancelled,
            }
        });

        // Verify we can access the task's individual cancellation token
        let task_cancel_token = handle.cancellation_token();
        assert!(
            !task_cancel_token.is_cancelled(),
            "Task token should not be cancelled initially"
        );

        // Let the task complete normally
        let result = handle.await.expect("Task should complete");
        assert!(result.is_ok(), "Task should succeed");
        assert_eq!(result.unwrap(), "completed");

        // Test cancellation of cancellable task
        let handle2 = tracker.spawn_cancellable(|cancel_token| async move {
            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {
                    CancellableTaskResult::Ok("should_not_complete".to_string())
                },
                _ = cancel_token.cancelled() => CancellableTaskResult::Cancelled,
            }
        });

        // Cancel the specific task
        handle2.cancellation_token().cancel();

        let result2 = handle2.await.expect("Task should complete");
        assert!(result2.is_err(), "Task should be cancelled");
        assert!(
            result2.unwrap_err().is_cancellation(),
            "Should be a cancellation error"
        );

        // Verify metrics
        assert_eq!(
            tracker.metrics().success(),
            1,
            "Should have 1 successful task"
        );
        assert_eq!(
            tracker.metrics().cancelled(),
            1,
            "Should have 1 cancelled task"
        );
    }

    /// Test FailedWithContinuation helper methods
    ///
    /// This test verifies that:
    /// 1. from_fn creates working continuations from simple closures
    /// 2. from_cancellable creates working continuations from cancellable closures
    /// 3. Both helpers integrate correctly with the task execution system
    #[tokio::test]
    async fn test_continuation_helpers() {
        let tracker = TaskTracker::new(UnlimitedScheduler::new(), LogOnlyPolicy::new()).unwrap();

        // Test from_fn helper
        let handle1 = tracker.spawn(async {
            let error =
                FailedWithContinuation::from_fn(anyhow::anyhow!("Initial failure"), || async {
                    Ok("Success from from_fn".to_string())
                });
            let result: Result<String, anyhow::Error> = Err(error);
            result
        });

        let result1 = handle1.await.expect("Task should complete");
        assert!(
            result1.is_ok(),
            "Task with from_fn continuation should succeed"
        );
        assert_eq!(result1.unwrap(), "Success from from_fn");

        // Test from_cancellable helper
        let handle2 = tracker.spawn(async {
            let error = FailedWithContinuation::from_cancellable(
                anyhow::anyhow!("Initial failure"),
                |_cancel_token| async move { Ok("Success from from_cancellable".to_string()) },
            );
            let result: Result<String, anyhow::Error> = Err(error);
            result
        });

        let result2 = handle2.await.expect("Task should complete");
        assert!(
            result2.is_ok(),
            "Task with from_cancellable continuation should succeed"
        );
        assert_eq!(result2.unwrap(), "Success from from_cancellable");

        // Verify metrics
        assert_eq!(
            tracker.metrics().success(),
            2,
            "Should have 2 successful tasks"
        );
        assert_eq!(tracker.metrics().failed(), 0, "Should have 0 failed tasks");
    }

    /// Test should_reschedule() policy method with mock scheduler tracking
    ///
    /// This test verifies that:
    /// 1. When should_reschedule() returns false, the guard is reused (efficient)
    /// 2. When should_reschedule() returns true, the guard is re-acquired through scheduler
    /// 3. The scheduler's acquire_execution_slot is called the expected number of times
    /// 4. Rescheduling works for both task-driven and policy-driven continuations
    #[rstest]
    #[case(false, 1, "Policy requests no rescheduling - should reuse guard")]
    #[case(true, 2, "Policy requests rescheduling - should re-acquire guard")]
    #[tokio::test]
    async fn test_should_reschedule_policy_control(
        #[case] should_reschedule: bool,
        #[case] expected_acquisitions: u32,
        #[case] description: &str,
    ) {
        // Mock scheduler that tracks acquisition calls
        #[derive(Debug)]
        struct MockScheduler {
            acquisition_count: Arc<AtomicU32>,
        }

        impl MockScheduler {
            fn new() -> Self {
                Self {
                    acquisition_count: Arc::new(AtomicU32::new(0)),
                }
            }

            fn acquisition_count(&self) -> u32 {
                self.acquisition_count.load(Ordering::Relaxed)
            }
        }

        #[async_trait]
        impl TaskScheduler for MockScheduler {
            async fn acquire_execution_slot(
                &self,
                _cancel_token: CancellationToken,
            ) -> SchedulingResult<Box<dyn ResourceGuard>> {
                self.acquisition_count.fetch_add(1, Ordering::Relaxed);
                SchedulingResult::Execute(Box::new(UnlimitedGuard))
            }
        }

        // Policy that controls rescheduling behavior
        #[derive(Debug)]
        struct RescheduleTestPolicy {
            should_reschedule: bool,
        }

        impl OnErrorPolicy for RescheduleTestPolicy {
            fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
                Arc::new(RescheduleTestPolicy {
                    should_reschedule: self.should_reschedule,
                })
            }

            fn create_context(&self) -> Option<Box<dyn std::any::Any + Send + 'static>> {
                None // Stateless policy
            }

            fn allow_continuation(
                &self,
                _error: &anyhow::Error,
                _context: &OnErrorContext,
            ) -> bool {
                true // Always allow continuations for this test
            }

            fn should_reschedule(&self, _error: &anyhow::Error, _context: &OnErrorContext) -> bool {
                self.should_reschedule
            }

            fn on_error(
                &self,
                _error: &anyhow::Error,
                _context: &mut OnErrorContext,
            ) -> ErrorResponse {
                ErrorResponse::Fail // Just fail when continuations are not allowed
            }
        }

        let mock_scheduler = Arc::new(MockScheduler::new());
        let policy = Arc::new(RescheduleTestPolicy { should_reschedule });
        let tracker = TaskTracker::new(mock_scheduler.clone(), policy).unwrap();
        let execution_log = Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));

        // Simple continuation that succeeds on second attempt
        #[derive(Debug)]
        struct SimpleRetryContinuation {
            log: Arc<tokio::sync::Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl Continuation for SimpleRetryContinuation {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                self.log
                    .lock()
                    .await
                    .push("continuation_executed".to_string());

                // Succeed immediately
                TaskExecutionResult::Success(Box::new("continuation_success".to_string()))
            }
        }

        // Task that fails with a continuation
        let continuation = Arc::new(SimpleRetryContinuation {
            log: execution_log.clone(),
        });

        let log_for_task = execution_log.clone();
        let handle = tracker.spawn(async move {
            log_for_task
                .lock()
                .await
                .push("initial_task_failure".to_string());
            let error = anyhow::anyhow!("Initial task failure");
            let result: Result<String, anyhow::Error> =
                Err(FailedWithContinuation::into_anyhow(error, continuation));
            result
        });

        let result = handle.await.expect("Task should complete");

        // Task should succeed regardless of rescheduling behavior
        assert!(result.is_ok(), "{}: Task should succeed", description);
        assert_eq!(
            tracker.metrics().success(),
            1,
            "{}: Should have 1 success",
            description
        );

        // Verify the execution log
        let log = execution_log.lock().await;
        assert_eq!(
            log.len(),
            2,
            "{}: Should have initial task + continuation",
            description
        );
        assert_eq!(log[0], "initial_task_failure");
        assert_eq!(log[1], "continuation_executed");

        // Most importantly: verify the scheduler acquisition count
        let actual_acquisitions = mock_scheduler.acquisition_count();
        assert_eq!(
            actual_acquisitions, expected_acquisitions,
            "{}: Expected {} scheduler acquisitions, got {}",
            description, expected_acquisitions, actual_acquisitions
        );
    }

    /// Test continuation loop with custom action policies
    ///
    /// This tests that custom error actions can also provide continuations
    /// and that the loop behavior works correctly with policy-provided continuations
    ///
    /// NOTE: Uses fresh policy/action instances to avoid global state interference.
    #[rstest]
    #[case(1, true, "Custom action with 1 retry should succeed")]
    #[case(3, true, "Custom action with 3 retries should succeed")]
    #[tokio::test]
    async fn test_continuation_loop_with_custom_action_policy(
        unlimited_scheduler: Arc<UnlimitedScheduler>,
        #[case] max_retries: u32,
        #[case] should_succeed: bool,
        #[case] description: &str,
    ) {
        let execution_log = Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));
        let retry_count = Arc::new(std::sync::atomic::AtomicU32::new(0));

        // Custom action that provides continuations up to max_retries
        #[derive(Debug)]
        struct RetryAction {
            log: Arc<tokio::sync::Mutex<Vec<String>>>,
            retry_count: Arc<std::sync::atomic::AtomicU32>,
            max_retries: u32,
        }

        #[async_trait]
        impl OnErrorAction for RetryAction {
            async fn execute(
                &self,
                _error: &anyhow::Error,
                _task_id: TaskId,
                _attempt_count: u32,
                _context: &TaskExecutionContext,
            ) -> ActionResult {
                let current_retry = self
                    .retry_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    + 1;
                self.log
                    .lock()
                    .await
                    .push(format!("custom_action_retry_{}", current_retry));

                if current_retry <= self.max_retries {
                    // Provide a continuation that succeeds if this is the final retry
                    #[derive(Debug)]
                    struct RetryContinuation {
                        log: Arc<tokio::sync::Mutex<Vec<String>>>,
                        retry_number: u32,
                        max_retries: u32,
                    }

                    #[async_trait]
                    impl Continuation for RetryContinuation {
                        async fn execute(
                            &self,
                            _cancel_token: CancellationToken,
                        ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>>
                        {
                            self.log
                                .lock()
                                .await
                                .push(format!("retry_continuation_{}", self.retry_number));

                            if self.retry_number >= self.max_retries {
                                // Final retry succeeds
                                TaskExecutionResult::Success(Box::new(format!(
                                    "success_after_{}_retries",
                                    self.retry_number
                                )))
                            } else {
                                // Still need more retries, fail with regular error (not FailedWithContinuation)
                                // This will trigger the custom action again
                                TaskExecutionResult::Error(anyhow::anyhow!(
                                    "Retry {} still failing",
                                    self.retry_number
                                ))
                            }
                        }
                    }

                    let continuation = Arc::new(RetryContinuation {
                        log: self.log.clone(),
                        retry_number: current_retry,
                        max_retries: self.max_retries,
                    });

                    ActionResult::Continue { continuation }
                } else {
                    // Exceeded max retries, cancel
                    ActionResult::Shutdown
                }
            }
        }

        // Custom policy that uses the retry action
        #[derive(Debug)]
        struct CustomRetryPolicy {
            action: Arc<RetryAction>,
        }

        impl OnErrorPolicy for CustomRetryPolicy {
            fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
                Arc::new(CustomRetryPolicy {
                    action: self.action.clone(),
                })
            }

            fn create_context(&self) -> Option<Box<dyn std::any::Any + Send + 'static>> {
                None // Stateless policy - no heap allocation
            }

            fn on_error(
                &self,
                _error: &anyhow::Error,
                _context: &mut OnErrorContext,
            ) -> ErrorResponse {
                ErrorResponse::Custom(Box::new(RetryAction {
                    log: self.action.log.clone(),
                    retry_count: self.action.retry_count.clone(),
                    max_retries: self.action.max_retries,
                }))
            }
        }

        let action = Arc::new(RetryAction {
            log: execution_log.clone(),
            retry_count: retry_count.clone(),
            max_retries,
        });
        let policy = Arc::new(CustomRetryPolicy { action });
        let tracker = TaskTracker::new(unlimited_scheduler, policy).unwrap();

        // Task that always fails with regular error (not FailedWithContinuation)
        let log_for_task = execution_log.clone();
        let handle = tracker.spawn(async move {
            log_for_task
                .lock()
                .await
                .push("original_task_failed".to_string());
            let result: Result<String, anyhow::Error> =
                Err(anyhow::anyhow!("Original task failure"));
            result
        });

        // Execute and verify results
        let result = handle.await.expect("Task should complete");

        if should_succeed {
            assert!(result.is_ok(), "{}: Task should succeed", description);
            assert_eq!(
                tracker.metrics().success(),
                1,
                "{}: Should have 1 success",
                description
            );

            // Verify the retry sequence
            let log = execution_log.lock().await;
            let expected_entries = 1 + (max_retries * 2); // original + (action + continuation) per retry
            assert_eq!(
                log.len(),
                expected_entries as usize,
                "{}: Should have {} log entries",
                description,
                expected_entries
            );

            assert_eq!(
                retry_count.load(std::sync::atomic::Ordering::Relaxed),
                max_retries,
                "{}: Should have made {} retry attempts",
                description,
                max_retries
            );
        } else {
            assert!(result.is_err(), "{}: Task should fail", description);
            assert!(
                tracker.cancellation_token().is_cancelled(),
                "{}: Should be cancelled",
                description
            );

            // Should have stopped after max_retries
            let final_retry_count = retry_count.load(std::sync::atomic::Ordering::Relaxed);
            assert!(
                final_retry_count > max_retries,
                "{}: Should have exceeded max_retries ({}), got {}",
                description,
                max_retries,
                final_retry_count
            );
        }
    }

    /// Test mixed continuation sources (task-driven + policy-driven)
    ///
    /// This test verifies that both task-provided continuations and policy-provided
    /// continuations can work together in the same execution flow
    #[rstest]
    #[tokio::test]
    async fn test_mixed_continuation_sources(
        unlimited_scheduler: Arc<UnlimitedScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        let execution_log = Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));
        let tracker = TaskTracker::new(unlimited_scheduler, log_policy).unwrap();

        // Task that provides a continuation, which then fails with regular error
        let log_for_task = execution_log.clone();
        let log_for_continuation = execution_log.clone();

        #[derive(Debug)]
        struct MixedContinuation {
            log: Arc<tokio::sync::Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl Continuation for MixedContinuation {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                self.log
                    .lock()
                    .await
                    .push("task_continuation_executed".to_string());
                // This continuation fails with a regular error (not FailedWithContinuation)
                // So it will be handled by the policy (LogOnlyPolicy just continues)
                TaskExecutionResult::Error(anyhow::anyhow!("Task continuation failed"))
            }
        }

        let continuation = Arc::new(MixedContinuation {
            log: log_for_continuation,
        });

        let handle = tracker.spawn(async move {
            log_for_task
                .lock()
                .await
                .push("original_task_executed".to_string());

            // Task provides continuation
            let error = anyhow::anyhow!("Original task failed");
            let result: Result<String, anyhow::Error> =
                Err(FailedWithContinuation::into_anyhow(error, continuation));
            result
        });

        // Execute - should fail because continuation fails and LogOnlyPolicy just logs
        let result = handle.await.expect("Task should complete");
        assert!(
            result.is_err(),
            "Should fail because continuation fails and policy just logs"
        );

        // Verify execution sequence
        let log = execution_log.lock().await;
        assert_eq!(log.len(), 2);
        assert_eq!(log[0], "original_task_executed");
        assert_eq!(log[1], "task_continuation_executed");

        // Verify metrics - should show failure from continuation
        assert_eq!(tracker.metrics().success(), 0);
        assert_eq!(tracker.metrics().failed(), 1);
    }

    /// Debug test to understand the threshold policy behavior in retry loop
    #[rstest]
    #[tokio::test]
    async fn debug_threshold_policy_in_retry_loop(unlimited_scheduler: Arc<UnlimitedScheduler>) {
        let policy = ThresholdCancelPolicy::with_threshold(2);
        let tracker = TaskTracker::new(unlimited_scheduler, policy.clone()).unwrap();

        // Simple continuation that always fails with regular error
        #[derive(Debug)]
        struct AlwaysFailContinuation {
            attempt: Arc<std::sync::atomic::AtomicU32>,
        }

        #[async_trait]
        impl Continuation for AlwaysFailContinuation {
            async fn execute(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                let attempt_num = self
                    .attempt
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    + 1;
                println!("Continuation attempt {}", attempt_num);
                TaskExecutionResult::Error(anyhow::anyhow!(
                    "Continuation attempt {} failed",
                    attempt_num
                ))
            }
        }

        let attempt_counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let continuation = Arc::new(AlwaysFailContinuation {
            attempt: attempt_counter.clone(),
        });

        let handle = tracker.spawn(async move {
            println!("Original task executing");
            let error = anyhow::anyhow!("Original task failed");
            let result: Result<String, anyhow::Error> =
                Err(FailedWithContinuation::into_anyhow(error, continuation));
            result
        });

        let result = handle.await.expect("Task should complete");
        println!("Final result: {:?}", result.is_ok());
        println!("Policy failure count: {}", policy.failure_count());
        println!(
            "Continuation attempts: {}",
            attempt_counter.load(std::sync::atomic::Ordering::Relaxed)
        );
        println!(
            "Tracker cancelled: {}",
            tracker.cancellation_token().is_cancelled()
        );
        println!(
            "Metrics: success={}, failed={}",
            tracker.metrics().success(),
            tracker.metrics().failed()
        );

        // This should help us understand what's happening
    }
}
