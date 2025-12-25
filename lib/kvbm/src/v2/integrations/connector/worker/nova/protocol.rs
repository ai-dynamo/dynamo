// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::BlockId;

/// Message sent by leader to workers when onboarding completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnboardCompleteMessage {
    pub request_id: String,
}

/// Message sent by leader to workers when offloading completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffloadCompleteMessage {
    pub request_id: String,
}

/// Message sent by leader to workers when onboarding fails for specific blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedOnboardMessage {
    pub request_id: String,
    pub block_ids: Vec<BlockId>,
}
