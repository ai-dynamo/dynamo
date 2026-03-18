// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runnable Module.
//!
//! This module provides a way to run a task in a runtime.
//!

use std::{
    pin::Pin,
    task::{Context, Poll},
};

pub use anyhow::{Error, Result};
pub use async_trait::async_trait;
pub use tokio::task::JoinHandle;
pub use tokio_util::sync::CancellationToken;

#[async_trait]
pub trait ExecutionHandle {
    fn is_finished(&self) -> bool;
    fn is_cancelled(&self) -> bool;
    fn cancel(&self);
    fn cancellation_token(&self) -> CancellationToken;
    fn handle(self) -> JoinHandle<Result<()>>;
}
