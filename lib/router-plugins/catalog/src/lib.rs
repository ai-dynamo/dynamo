// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The empty router-plugin catalog used when a custom image does not replace Dynamo's catalog
//! slot.

use dynamo_kv_router::scheduling::{RequestClassifierRegistry, RequestClassifierRegistryError};
use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyRegistry, WorkerSelectionPolicyRegistryError,
};

/// Register worker-selection policies linked into this image.
///
/// Custom catalogs replace this crate and register their own factories. The default registration
/// is intentionally empty so `default` always selects Dynamo's built-in worker selector.
pub fn register(
    _registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    Ok(())
}

/// Register request classifiers linked into this image.
///
/// The default registration is intentionally empty, so omitting `request_classifier` keeps
/// Dynamo's pass-through behavior.
pub fn register_request_classifiers(
    _registry: &mut RequestClassifierRegistry,
) -> Result<(), RequestClassifierRegistryError> {
    Ok(())
}
