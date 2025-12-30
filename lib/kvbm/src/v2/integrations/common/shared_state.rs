// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared state trait for scheduler-connector communication.
//!
//! This module defines a minimal, extensible trait for bidirectional communication
//! between the scheduler and connector. Both components can operate independently
//! without this shared state - it is completely optional.

use std::any::Any;

/// Minimal trait for scheduler-connector shared state.
///
/// This trait is intentionally minimal and uses `Any` for maximum flexibility.
/// Extend as use cases emerge. Both the scheduler and connector hold
/// `Option<Arc<Mutex<dyn SchedulerConnectorState>>>` - when None, they operate
/// independently.
///
/// # Example
///
/// ```ignore
/// use std::any::Any;
///
/// struct MySharedState {
///     // Your shared state fields
/// }
///
/// impl SchedulerConnectorState for MySharedState {
///     fn as_any(&self) -> &dyn Any {
///         self
///     }
///
///     fn as_any_mut(&mut self) -> &mut dyn Any {
///         self
///     }
/// }
/// ```
pub trait SchedulerConnectorState: Send + Sync + 'static {
    /// Convert to Any for downcasting to concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Convert to mutable Any for downcasting to concrete type.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}



