// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dashmap::DashMap;
use derive_builder::Builder;
use dynamo_nova::am::Nova;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};

use std::sync::Arc;

use crate::{
    logical::{BlockId, manager::BlockManager},
    v2::{
        InstanceId, SequenceHash,
        integrations::{G2, G3},
        logical::blocks::ImmutableBlock,
    },
};

use super::{SessionId, Worker};

pub trait Leader: Send + Sync {
    fn find_matches(&self, sequence_hashes: &[SequenceHash]) -> Result<FindMatchesTask> {
        self.find_matches_with_options(sequence_hashes, FindOptions::default())
    }

    fn find_matches_with_options(
        &self,
        sequence_hashes: &[SequenceHash],
        options: FindOptions,
    ) -> Result<FindMatchesTask>;
}

#[derive(Default)]
pub struct FindOptions {}

pub struct FindMatchesTask {}

impl FindMatchesResult {
    pub fn status(&self) -> FindStatus {
        todo!()
    }
}

pub enum FindStatus {
    Searching(SessionId),
    Staging(StagingDetails),
    Done(SessionId),
}

pub struct StagingDetails {
    pub session_id: SessionId,
    pub matched: usize,

    /// Number of blocks matched in any local backend
    pub local_matches: usize,

    /// Number of blocks staging from g3 -> g2
    pub local_staging: usize,

    /// Number of blocks inflight from g3 -> g2
    pub local_inflight: usize,

    /// Number of blocks matched in any remote backend
    pub remote_matches: usize,

    /// Number of blocks staging from a remote backend to local g2
    pub remote_staging: usize,
}

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(private, name = "build_private"))]
pub struct InstanceLeader {
    rt: tokio::runtime::Handle,
    nova: Arc<Nova>,
    g2_manager: BlockManager<G2>,
    g3_manager: Option<BlockManager<G3>>,
    workers: Vec<Arc<dyn Worker>>,
}

impl InstanceLeader {}

impl Leader for InstanceLeader {
    fn find_matches_with_options(
        &self,
        sequence_hashes: &[SequenceHash],
        options: FindOptions,
    ) -> Result<FindMatchesTask> {
        // acquire permits from the onboarding budget
        // here we could fail if the options tell us not to queue
        // if we queue, we start a session

        let g2_matches = self.g2_manager.match_blocks(sequence_hashes);

        let remaining_sequence_hashes = &sequence_hashes[g2_matches.len()..];

        let g3_matches = if let Some(g3_manager) = &self.g3_manager {
            g3_manager.match_blocks(remaining_sequence_hashes)
        } else {
            Vec::new()
        };

        if g3_matches.is_empty() {
            !todo!(
                "return FindMatchesTask::Done - this arm does not have a future assocated with it"
            )
        }

        // todo: create an onboarding session for the g2 -> g3 transfers
        // let the connector api determine

        todo!()
    }
}
