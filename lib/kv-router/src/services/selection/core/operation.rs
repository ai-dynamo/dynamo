// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The one selection operation every host runs. Wire handlers and embedding
//! hosts build a [`SelectionOperation`] and consume a [`SelectionOutcome`].

use std::collections::HashSet;
use std::time::Duration;

use dynamo_tokens::SequenceHash;

use crate::identity::RoutingPartitionId;
use crate::kv_hints::KvHint;
use crate::protocols::{
    LocalBlockHash, RoutingConstraints, SharedCacheHits, WorkerAffinityTarget, WorkerId,
    WorkerWithDpRank,
};
use crate::scheduling::config::RouterConfigOverride;
use crate::scheduling::queue::BookingHandle;
use crate::scheduling::{AdvisoryWorkerLoad, QueueRejection, SchedulingResponse, SessionContext};

use super::super::error::SelectionError;
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
    /// Send the prompt's tracking hashes to the scheduler so the booking's
    /// blocks count as active on its worker.
    pub track_active_blocks: bool,
    /// Return the prompt's public block hashes for the host to record.
    pub return_routing_hashes: bool,
    /// Cache the inputs under this id for a later `create_reservation` replay
    /// (unbooked admissions only).
    pub replay_id: Option<String>,
}

pub enum SelectionAdmission {
    /// Queue admission without a booking.
    Query { request_id: Option<String> },
    /// Queue admission with a booking the core records as a reservation under
    /// `selection_id` and releases through its lifecycle calls.
    Book { selection_id: String },
    /// Queue admission with a booking the caller owns: its booking handle is
    /// returned armed in [`Selected::booking`].
    Lease { request_id: String },
    /// Skip queue admission and report the chosen worker's load.
    Advisory { request_id: Option<String> },
}

impl SelectionAdmission {
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::Query { request_id } | Self::Advisory { request_id } => request_id.as_deref(),
            Self::Book { selection_id }
            | Self::Lease {
                request_id: selection_id,
            } => Some(selection_id),
        }
    }

    pub fn is_booking(&self) -> bool {
        matches!(self, Self::Book { .. } | Self::Lease { .. })
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

/// One selection's result together with how long its lookups took. The
/// timings ride with errors too, so a host can account a shared-cache failure
/// even when scheduling then fails.
pub struct SelectionRun {
    pub result: Result<SelectionOutcome, SelectionError>,
    /// `None` when selection failed before the lookups started.
    pub lookup: Option<LookupTimings>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LookupTimings {
    pub block_hashing: Duration,
    pub seq_hashing: Duration,
    /// Wall time of the index and shared-cache lookups together.
    pub lookups: Duration,
    pub indexer: Duration,
    /// `None` when no shared cache was queried.
    pub shared_cache: Option<Duration>,
    pub shared_cache_error: bool,
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
    /// `None` only for `Lease` admission when the worker left the catalog
    /// after it was booked: the host dispatches by worker id and lets the
    /// transport report the departure, which is what drives migration.
    pub endpoint: Option<String>,
    pub block_size: u32,
    pub isl_tokens: usize,
    /// The hashes the booking tracks; `Book` only.
    pub sequence_hashes: Option<Vec<SequenceHash>>,
    pub track_prefill_tokens: bool,
    pub effective_prefill_tokens: usize,
    pub kv_hint: Option<KvHint>,
    pub routing_hashes: Option<Vec<LocalBlockHash>>,
    pub shared_cache_hits: Option<SharedCacheHits>,
    /// The booking's handle; `Lease` admission only. Dropping it frees the
    /// booking, `commit` hands it to the caller's own cleanup.
    pub booking: Option<BookingHandle>,
}
