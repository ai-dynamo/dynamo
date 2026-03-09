// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Level Zero event polling-based completion checker.

use anyhow::Result;
use syclrc::{DriverError, ZeEvent};

use super::CompletionChecker;

/// Completion checker that polls a Level Zero event status.
///
/// After the executor appends all memcpy operations to an immediate command list,
/// it signals a [`ZeEvent`]. This checker polls that event using
/// [`ZeEvent::query_status`] — returning `true` once the GPU has signalled it.
pub struct ZeEventChecker {
    event: ZeEvent,
}

impl ZeEventChecker {
    pub fn new(event: ZeEvent) -> Self {
        Self { event }
    }
}

impl CompletionChecker for ZeEventChecker {
    fn is_complete(&self) -> Result<bool> {
        // ZeEvent::query_status returns:
        //   Ok(true)  — event has been signalled (transfer complete)
        //   Ok(false) — ZE_RESULT_NOT_READY (still pending)
        //   Err(e)    — unexpected driver error
        match self.event.query_status() {
            Ok(signalled) => Ok(signalled),
            Err(DriverError(code)) => Err(anyhow::anyhow!(
                "Level Zero event query failed: ze_result_t = 0x{:x}",
                code as u32
            )),
        }
    }
}
