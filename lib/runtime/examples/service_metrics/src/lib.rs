// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

pub const DEFAULT_NAMESPACE: &str = "dynamo";

#[derive(Serialize, Deserialize)]
// Dummy Stats object to demonstrate how to attach a custom stats handler
pub struct MyStats {
    pub val: u32,
}
