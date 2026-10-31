// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an in-tree host names from the session-affinity stage: the table and its
//! configuration, the replication sink it implements, the versioned target it forwards, and the
//! limits it validates against. Every item is re-exported from the crate root; this list is the
//! host contract (hosts inventory v2, section 3).

pub use crate::{
    AffinityReplicaSink, AffinityTarget, AffinityVersion, MAX_SESSION_AFFINITY_ID_BYTES,
    MAX_SESSION_AFFINITY_TTL_SECS, ReplicaApplyOutcome, SessionAffinityConfig, SessionAffinityMode,
    WeakSessionAffinity,
};
