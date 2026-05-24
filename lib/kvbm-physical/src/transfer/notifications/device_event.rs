// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Device event polling-based completion checker (multi-backend).
//!
//! This replaces the CUDA-specific CudaEventChecker with a backend-agnostic
//! implementation that works with any DeviceEvent (CUDA, XPU).

use anyhow::Result;

use crate::device::DeviceEvent;
use super::CompletionChecker;

/// Completion checker that polls device event status (supports CUDA, XPU).
pub struct DeviceEventChecker {
    event: DeviceEvent,
}

impl DeviceEventChecker {
    pub fn new(event: DeviceEvent) -> Self {
        Self { event }
    }
}

impl CompletionChecker for DeviceEventChecker {
    fn is_complete(&self) -> Result<bool> {
        self.event.is_complete()
    }
}
