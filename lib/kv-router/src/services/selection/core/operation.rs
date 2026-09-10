// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The one selection operation every host runs. Wire handlers and embedding
//! hosts build a [`SelectionOperation`] and consume a [`SelectionOutcome`].

use std::collections::HashSet;

use dynamo_tokens::SequenceHash;

use crate::identity::RoutingPartitionId;
use crate::kv_hints::KvHint;
use crate::protocols::{RoutingConstraints, WorkerAffinityTarget, WorkerId, WorkerWithDpRank};
use crate::scheduling::config::RouterConfigOverride;
use crate::scheduling::{AdvisoryWorkerLoad, QueueRejection, SchedulingResponse, SessionContext};

use super::super::input::PromptView;

/// Every input to one selection. Fields are independent: `session_context`
/// feeds worker selection, `session` says what the core does with the session
/// table, `affinity_target` and `pinned_worker` are explicit steering.
pub struct SelectionOperation<'a> {
    pub key: RoutingPartitionId,
    pub prompt: PromptView<'a>,
    pub router_config_override: Option<RouterConfigOverride>,
    pub expected_output_tokens: Option<u32>,
    pub priority_jump: f64,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
    pub session_context: Option<SessionContext>,
    pub session: SessionBinding,
    pub affinity_target: Option<WorkerAffinityTarget>,
    pub pinned_worker: Option<WorkerWithDpRank>,
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    pub routing_constraints: RoutingConstraints,
    pub admission: SelectionAdmission,
}

pub enum SelectionAdmission {
    /// Queue admission without a booking; a `request_id` caches the inputs for
    /// a later `create_reservation` replay.
    Query { request_id: Option<String> },
    /// Queue admission with a booking recorded under `selection_id`.
    Book { selection_id: String },
    /// Skip queue admission and report the chosen worker's load; a
    /// `request_id` caches the inputs like `Query`.
    Advisory { request_id: Option<String> },
}

impl SelectionAdmission {
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::Query { request_id } | Self::Advisory { request_id } => request_id.as_deref(),
            Self::Book { selection_id } => Some(selection_id),
        }
    }

    pub fn is_booking(&self) -> bool {
        matches!(self, Self::Book { .. })
    }
}

/// What the core does with the partition's session table for this request.
pub enum SessionBinding {
    None,
    /// Hold the session, steer to its worker, and bind it to the worker booked.
    Managed {
        session_id: String,
    },
    /// Steer to the session's worker without holding or binding it.
    Query {
        session_id: String,
    },
}

// Every caller matches this immediately; boxing the common variant to shrink
// the rare one would put an allocation on the hot path.
#[allow(clippy::large_enum_variant)]
pub enum SelectionOutcome {
    Selected(Selected),
    QueueRejected { rejection: QueueRejection },
}

/// A completed selection. For `Book`, the reservation is already installed.
#[must_use]
pub struct Selected {
    pub key: RoutingPartitionId,
    pub response: SchedulingResponse,
    pub advisory_load: Option<AdvisoryWorkerLoad>,
    /// The chosen worker's KV capacity as known when it was selected.
    pub total_kv_blocks: Option<u64>,
    pub endpoint: String,
    pub block_size: u32,
    pub isl_tokens: usize,
    /// The hashes the booking tracks; `Book` only.
    pub sequence_hashes: Option<Vec<SequenceHash>>,
    pub track_prefill_tokens: bool,
    pub effective_prefill_tokens: usize,
    pub kv_hint: Option<KvHint>,
}
