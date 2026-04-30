// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-owned conditional-disaggregation session protocol and helpers.
//!
//! Hub-visible request metadata lives in `kvbm-disagg-protocol`. The
//! [`session`] submodule owns the symmetric `Session` API + its production
//! ([`session::VeloSession`]) and test ([`session::MockSession`]) impls.
//!
//! This module also re-exports the small data types used by the engine's
//! RDMA primitive (`InstanceLeader::pull_remote_block_sets`) and by
//! `VeloSession::pull`: [`RemoteBlockRef`] and [`RemoteBlockSet`], plus
//! [`SessionEndpoint`] from `kvbm-disagg-protocol`.

pub mod session;

use kvbm_logical::SequenceHash;
use serde::{Deserialize, Serialize};

use crate::{BlockId, LogicalLayoutHandle};

pub use kvbm_disagg_protocol::SessionEndpoint;

/// Serializable reference to a block made available through a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteBlockRef {
    pub block_id: BlockId,
    pub sequence_hash: SequenceHash,
}

/// A set of blocks that all share the same remote logical source layout.
///
/// Workers resolve `(peer instance, worker rank, source_layout)` through
/// imported `SerializedLayout` metadata before executing the transfer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteBlockSet {
    pub source_layout: LogicalLayoutHandle,
    pub blocks: Vec<RemoteBlockRef>,
}
