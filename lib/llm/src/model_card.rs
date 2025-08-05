// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod create;
pub mod model;
pub mod runtime_config;
pub use model::ModelDeploymentCard;

/// Identify model deployment cards in the key-value store
pub const ROOT_PATH: &str = "mdc";
