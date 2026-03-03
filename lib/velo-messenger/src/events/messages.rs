// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire message types for the distributed event system.

use serde::{Deserialize, Serialize};
use velo_common::InstanceId;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct EventSubscribeMessage {
    pub handle: u128,
    pub subscriber_worker: u64,
    pub subscriber_instance: InstanceId,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct EventCompletionMessage {
    pub handle: u128,
    pub poisoned: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct EventTriggerRequestMessage {
    pub handle: u128,
    pub requester_worker: u64,
    pub requester_instance: InstanceId,
    pub poisoned: Option<String>,
    pub response_id: Option<u128>,
}
