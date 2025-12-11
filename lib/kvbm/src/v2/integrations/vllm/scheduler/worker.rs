// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::KvbmVllmConfig;
use crate::v2::physical::manager::{LayoutHandle, TransportManager};

/// Object implementing the vLLM Connector API for the Worker
#[allow(dead_code)]
pub struct SchedulerWorker {
    config: KvbmVllmConfig,
    manager: TransportManager,
    device_handle: Option<LayoutHandle>,
    host_handle: Option<LayoutHandle>,
    disk_handle: Option<LayoutHandle>,
}

impl SchedulerWorker {
    pub fn new(config: KvbmVllmConfig) -> Self {
        Self {
            config,
            manager: TransportManager::builder().build().unwrap(),
            device_handle: None,
            host_handle: None,
            disk_handle: None,
        }
    }
}
