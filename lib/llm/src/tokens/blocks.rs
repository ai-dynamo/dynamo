// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type GlobalHash = u64;

/// Represents an active block beign built
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum UniqueBlock {
    /// Block identified by UUID
    PartialBlock(Uuid),
    /// Block identified by hash
    FullBlock(GlobalHash),
}

impl Default for UniqueBlock {
    fn default() -> Self {
        // Generate a random UUID when default is used
        Self::PartialBlock(Uuid::new_v4())
    }
}
