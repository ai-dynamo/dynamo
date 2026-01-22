// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared validation helpers for discovery system configuration.

use validator::ValidationError;

/// Validate that a cluster identifier is non-empty and free of surrounding whitespace.
pub(crate) fn validate_cluster_id(cluster_id: &str) -> Result<(), ValidationError> {
    if cluster_id.trim().is_empty() {
        let mut err = ValidationError::new("cluster_id_empty");
        err.add_param("value".into(), &cluster_id);
        return Err(err);
    }

    // Reject cluster IDs with leading or trailing whitespace
    if cluster_id.trim() != cluster_id {
        let mut err = ValidationError::new("cluster_id_has_whitespace");
        err.add_param("value".into(), &cluster_id);
        return Err(err);
    }

    Ok(())
}
