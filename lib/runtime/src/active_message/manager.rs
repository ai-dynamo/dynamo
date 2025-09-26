// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::broadcast;

use super::client::ActiveMessageClient;
use super::handler::{ActiveMessageHandler, HandlerEvent, HandlerId};
use crate::utils::tasks::tracker::TaskTracker;

#[derive(Default)]
pub struct HandlerConfig {
    pub task_tracker: Option<TaskTracker>,
}

impl HandlerConfig {
    pub fn with_task_tracker(mut self, tracker: TaskTracker) -> Self {
        self.task_tracker = Some(tracker);
        self
    }
}

#[async_trait]
pub trait ActiveMessageManager: Send + Sync {
    fn client(&self) -> Arc<dyn ActiveMessageClient>;

    async fn register_handler(
        &self,
        handler: Arc<dyn ActiveMessageHandler>,
        config: Option<HandlerConfig>,
    ) -> Result<()>;

    async fn deregister_handler(&self, name: &str) -> Result<()>;

    async fn list_handlers(&self) -> Vec<HandlerId>;

    fn handler_events(&self) -> broadcast::Receiver<HandlerEvent>;

    async fn shutdown(&self) -> Result<()>;
}
