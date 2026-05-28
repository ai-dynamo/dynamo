// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo AM subscriber for KV-cache event ingestion.
//!
//! Full implementation lands in a follow-up MR.

use std::sync::Arc;

use anyhow::Result;
use velo::Messenger;

/// Full implementation lands in a follow-up MR.
pub fn register(_messenger: &Arc<Messenger>) -> Result<()> {
    Ok(())
}
