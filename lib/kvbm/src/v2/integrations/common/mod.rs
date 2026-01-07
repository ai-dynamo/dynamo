// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Common types shared between the scheduler and connector modules.
//!
//! This module contains types that are used by both the scheduler (G1 block management)
//! and the connector (G2+ offloading), allowing them to communicate without tight coupling.

mod block_assignments;
mod output;
mod request;
mod shared_state;

pub use block_assignments::{
    AssignedBlock, AssignedBlockId, BlockAssignmentOps, BlockAssignmentStorage,
    KvbmSequenceHashProvider, UnassignedBlock,
};
pub use output::{CachedRequestData, NewRequestData, SchedulerOutput};
pub use request::{Request, RequestMetadata};
pub use shared_state::SchedulerConnectorState;
