// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;

use crate::protocols::WorkerWithDpRank;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchIntent {
    Any,
    Exact(WorkerWithDpRank),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum QueueAdmissionConfig {
    SessionAware {},
}
