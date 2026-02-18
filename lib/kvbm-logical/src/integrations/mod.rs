// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Composition layer wiring together [`BlockSequence`](crate::BlockSequence),
//! [`LogicalBlockAssignments`](crate::LogicalBlockAssignments), and
//! [`BlockManager`](crate::BlockManager) into higher-level request lifecycle types.

mod request;
mod scheduled;

pub use request::RequestSequence;
pub use scheduled::{
    ApplyError, DecodeOutcome, SchedulableSequence, ScheduleError, SequenceEvent, SequenceState,
};
