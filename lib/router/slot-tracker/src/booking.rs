// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The booking the scheduler hands the slot tracker after a pick: which request landed on
//! which rank, fenced by the attempt the tracker minted.

use dynamo_router_protocol::attempt::AttemptId;
use dynamo_router_protocol::worker::WorkerWithDpRank;

/// One request's live booking on one worker rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerBookingDescriptor {
    /// The request the booking belongs to.
    pub request_id: String,
    /// The rank it was booked on.
    pub worker: WorkerWithDpRank,
    /// The attempt that owns the booking; later lifecycle steps are fenced to it.
    pub attempt_id: AttemptId,
}
