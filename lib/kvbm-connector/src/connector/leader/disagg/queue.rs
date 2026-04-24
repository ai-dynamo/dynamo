// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Queue abstraction for dispatching decode-created remote-prefill requests.

use anyhow::Result;
use kvbm_disagg_protocol::RemotePrefillRequest;

pub trait RemotePrefillQueue: Send + Sync {
    fn enqueue(&self, request: RemotePrefillRequest) -> Result<()>;
}
