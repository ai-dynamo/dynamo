// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::model::ModelDeploymentCard;

/// Helper struct for backend engines to register runtime data during initialization
pub struct RuntimeConfigBuilder {
    card: ModelDeploymentCard,
}

impl RuntimeConfigBuilder {
    /// Create a new RuntimeConfigBuilder from a ModelDeploymentCard
    pub fn new(card: ModelDeploymentCard) -> Self {
        Self { card }
    }

    /// Register the total number of KV blocks
    pub fn with_total_kv_blocks(mut self, total_blocks: u64) -> Self {
        self.card.register_total_kv_blocks(total_blocks);
        self
    }

    /// Build the final ModelDeploymentCard with all registered runtime data
    pub fn build(self) -> ModelDeploymentCard {
        self.card
    }
}
