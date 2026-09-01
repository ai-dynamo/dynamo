// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable Dynamo command/lifecycle protocol for the generalized engine.

use uuid::Uuid;

use crate::common::handoff::{HandoffId, HandoffTransferTiming};
use crate::common::protocols::DirectRequest;

pub enum SchedulerCommand {
    Submit(DirectRequest),
    CancelRequest {
        request_id: Uuid,
    },
    SubmitHandoffPrefill {
        handoff_id: HandoffId,
        request: DirectRequest,
    },
    ReleaseSource {
        handoff_id: HandoffId,
    },
    CancelSource {
        handoff_id: HandoffId,
    },
    ReserveDestination {
        handoff_id: HandoffId,
        request: DirectRequest,
    },
    ActivateDestination {
        handoff_id: HandoffId,
    },
    CancelDestination {
        handoff_id: HandoffId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulerCommandResult {
    Submitted(Uuid),
    DestinationAccepted { request_id: Uuid },
    Applied,
    Noop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SchedulerLifecycleEvent {
    SourceHeld {
        handoff_id: HandoffId,
        request_id: Uuid,
        transfer_timing: HandoffTransferTiming,
    },
    DestinationReserved {
        handoff_id: HandoffId,
        request_id: Uuid,
        transferable_prompt_tokens: usize,
    },
}

impl SchedulerLifecycleEvent {
    pub fn handoff_id(&self) -> HandoffId {
        match *self {
            Self::SourceHeld { handoff_id, .. } | Self::DestinationReserved { handoff_id, .. } => {
                handoff_id
            }
        }
    }
}

#[derive(Debug)]
pub struct SchedulerCommandEffects {
    pub result: SchedulerCommandResult,
    pub lifecycle_events: Vec<SchedulerLifecycleEvent>,
    pub kv_events: Vec<dynamo_kv_router::protocols::RouterEvent>,
}
