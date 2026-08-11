// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The empty catalog used when a custom image does not replace Dynamo's catalog slot.

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyRegistry, WorkerSelectionPolicyRegistryError,
};

/// Register policies linked into this image.
///
/// Custom catalogs replace this crate and register their own factories. The default catalog is
/// intentionally empty so `default` always selects Dynamo's built-in worker selector.
pub fn register(
    _registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    Ok(())
}
