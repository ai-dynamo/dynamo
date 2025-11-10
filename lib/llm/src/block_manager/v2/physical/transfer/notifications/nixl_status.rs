// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL status polling-based completion checker.

use anyhow::{Result, anyhow};
use nixl_sys::{Agent as NixlAgent, XferRequest};

use super::CompletionChecker;

/// Completion checker that polls NIXL transfer status.
pub struct NixlStatusChecker {
    agent: NixlAgent,
    xfer_req: XferRequest,
}

impl NixlStatusChecker {
    pub fn new(agent: NixlAgent, xfer_req: XferRequest) -> Self {
        Self { agent, xfer_req }
    }
}

impl CompletionChecker for NixlStatusChecker {
    fn is_complete(&self) -> Result<bool> {
        // get_xfer_status returns XferStatus enum:
        // - XferStatus::Success means transfer is complete
        // - XferStatus::InProgress means still pending
        match self.agent.get_xfer_status(&self.xfer_req) {
            Ok(status) => Ok(status.is_success()),
            Err(e) => Err(anyhow!("NIXL transfer status check failed: {}", e)),
        }
    }
}
