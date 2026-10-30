// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replication wire between session-affinity tables: one binding event per published
//! binding. Both ends are affinity tables; the mesh that carries it is a `dynamo-kv-router`
//! adapter.

use serde::{Deserialize, Serialize};

use dynamo_router_protocol::identity::RoutingPartitionId;

use crate::{AffinityTarget, AffinityVersion};

/// One replicated session binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffinityBindingEvent {
    #[serde(flatten)]
    pub partition: RoutingPartitionId,
    pub session_id: String,
    pub worker_id: u64,
    pub dp_rank: Option<u32>,
    pub sequence: u64,
    pub writer_id: u64,
}

impl AffinityBindingEvent {
    /// The bound worker and optional rank this event carries.
    pub fn target(&self) -> AffinityTarget {
        AffinityTarget::new(self.worker_id, self.dp_rank)
    }

    /// The binding's replication version.
    pub fn version(&self) -> AffinityVersion {
        AffinityVersion {
            sequence: self.sequence,
            writer_id: self.writer_id,
        }
    }
}
