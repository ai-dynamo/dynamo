// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::MODEL_ROOT_PATH;

#[derive(Debug, Clone)]
pub struct ModelNetworkName(String);

impl ModelNetworkName {
    pub fn new() -> Self {
        ModelNetworkName(format!("{MODEL_ROOT_PATH}/{}", uuid::Uuid::new_v4()))
    }
}

impl Default for ModelNetworkName {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ModelNetworkName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
