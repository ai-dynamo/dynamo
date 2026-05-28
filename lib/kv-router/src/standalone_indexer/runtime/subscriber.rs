// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo AM event ingestion handlers (DEP-980, upcoming PR).

use std::sync::Arc;
use anyhow::Result;
use velo::Messenger;
use super::RuntimeRegistry;

/// Full implementation lands in DEP-980.
pub fn register(_messenger: &Arc<Messenger>, _registry: Arc<RuntimeRegistry>) -> Result<()> {
    Ok(())
}