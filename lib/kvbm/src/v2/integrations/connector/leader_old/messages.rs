// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Message structures for leader-worker communication.

use anyhow::Result;
use derive_builder::Builder;
use dynamo_nova::events::EventHandle;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::{Validate, ValidationError};

use super::{G1, G2, G3};

use crate::integrations::connector::leader::data::BlocksView;
use crate::physical::manager::LayoutHandle;
use crate::v2::logical::blocks::BlockId;

/// Request to onboard KV cache blocks from G2/G3 to G1.
///
/// Sent by leader to worker cohort to initiate onboarding operation.
/// Workers will transfer blocks from their local CPU/disk (G2/G3) layouts
/// to GPU (G1) layout.
#[derive(Debug, Clone, Serialize, Deserialize, Builder, Validate)]
#[builder(pattern = "owned", build_fn(private, name = "build_private"))]
#[validate(schema(function = "validate_onboard_request"))]
pub struct KvbmOnboardRequest {
    /// Request identifier
    pub request_id: String,

    /// Operation identifier
    #[builder(default = "Uuid::new_v4()", setter(skip))]
    pub operation_id: Uuid,

    // Source blocks (G2 - CPU/host tier)
    pub g2_layout: Vec<LayoutHandle>,
    pub g2_block_ids: BlocksView<G2>,

    // Source blocks (G3 - disk tier, optional)
    #[builder(default)]
    pub g3_layout: Option<Vec<LayoutHandle>>,
    #[builder(default)]
    pub g3_block_ids: Option<BlocksView<G3>>,

    // Bounce buffer blocks for two-hop transfers G3 -> Bounce -> G1
    #[builder(default)]
    pub bounce_layout: Option<Vec<LayoutHandle>>,
    #[builder(default)]
    pub bounce_block_ids: Option<Vec<BlockId>>,

    // Destination blocks (G1 - GPU tier)
    pub g1_layout: Vec<LayoutHandle>,
    pub g1_block_ids: BlocksView<G1>,

    // Completion events (one per worker rank)
    pub completion_events: Vec<EventHandle>,
}

/// Custom validation for KvbmOnboardRequest
fn validate_onboard_request(request: &KvbmOnboardRequest) -> Result<(), ValidationError> {
    // Validate that sum of G2 and G3 block IDs equals G1 block IDs
    let g2_count = request.g2_block_ids.len();
    let g3_count = request.g3_block_ids.as_ref().map(|v| v.len()).unwrap_or(0);
    let total_source = g2_count + g3_count;
    let g1_count = request.g1_block_ids.len();

    if total_source != g1_count {
        let mut error = ValidationError::new("block_count_mismatch");
        error.message = Some(
            format!(
                "Sum of G2 ({}) and G3 ({}) block IDs ({}) must equal G1 block IDs ({})",
                g2_count, g3_count, total_source, g1_count
            )
            .into(),
        );
        return Err(error);
    }

    // Validate that all layout vectors and completion events have matching lengths
    // These represent per-worker-rank data structures
    let num_ranks = request.completion_events.len();

    if request.g2_layout.len() != num_ranks {
        let mut error = ValidationError::new("g2_layout_size_mismatch");
        error.message = Some(
            format!(
                "G2 layout count ({}) must match completion events count ({})",
                request.g2_layout.len(),
                num_ranks
            )
            .into(),
        );
        return Err(error);
    }

    if request.g1_layout.len() != num_ranks {
        let mut error = ValidationError::new("g1_layout_size_mismatch");
        error.message = Some(
            format!(
                "G1 layout count ({}) must match completion events count ({})",
                request.g1_layout.len(),
                num_ranks
            )
            .into(),
        );
        return Err(error);
    }

    // Validate optional G3 layout if present
    if let Some(g3_layout) = &request.g3_layout {
        if g3_layout.len() != num_ranks {
            let mut error = ValidationError::new("g3_layout_size_mismatch");
            error.message = Some(
                format!(
                    "G3 layout count ({}) must match completion events count ({})",
                    g3_layout.len(),
                    num_ranks
                )
                .into(),
            );
            return Err(error);
        }
    }

    // Validate optional bounce layout if present
    if let Some(bounce_layout) = &request.bounce_layout {
        if bounce_layout.len() != num_ranks {
            let mut error = ValidationError::new("bounce_layout_size_mismatch");
            error.message = Some(
                format!(
                    "Bounce layout count ({}) must match completion events count ({})",
                    bounce_layout.len(),
                    num_ranks
                )
                .into(),
            );
            return Err(error);
        }
    }

    Ok(())
}

impl KvbmOnboardRequestBuilder {
    pub fn build(self) -> Result<KvbmOnboardRequest> {
        let request = self.build_private()?;
        request.validate()?;
        Ok(request)
    }
}

impl KvbmOnboardRequest {
    pub fn builder() -> KvbmOnboardRequestBuilder {
        KvbmOnboardRequestBuilder::default()
    }
}
