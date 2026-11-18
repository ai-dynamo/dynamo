// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-owned node placement, separate from Dynamo worker identities.

use std::collections::{BTreeMap, BTreeSet};

use dynamo_backend_common::DynamoError;
use serde::{Deserialize, Serialize};

use crate::{client, proto as pb};

pub(crate) const NODE_METADATA_KEY: &str = "trtllm_node";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct NodeMetadata {
    pub version: u32,
    pub engine_id: String,
    pub node_id: String,
    pub leader: bool,
    pub node_count: u32,
    pub kv_block_size: u32,
    pub local_dp_ranks: Vec<u32>,
    pub source_owners: BTreeMap<u32, String>,
}
impl NodeMetadata {
    pub fn from_info(info: Option<&pb::ServerInfo>) -> Result<Option<Self>, DynamoError> {
        let Some(info) = info else { return Ok(None) };
        let extra = dynamo_sidecar_common::struct_to_json(
            info.extra.clone().unwrap_or_default(),
            "TensorRT-LLM",
            "server metadata",
        )?;
        let Some(value) = extra.get(NODE_METADATA_KEY) else {
            return Ok(None);
        };
        let node: Self = serde_json::from_value(value.clone())
            .map_err(|error| client::protocol_error(format!("Invalid node metadata: {error}")))?;
        let size = info
            .parallelism
            .as_ref()
            .and_then(|p| p.data_parallel_size)
            .unwrap_or(1);
        node.validate(size)?;
        Ok(Some(node))
    }

    fn validate(&self, dp_size: u32) -> Result<(), DynamoError> {
        let local: BTreeSet<_> = self.local_dp_ranks.iter().copied().collect();
        let expected: BTreeSet<_> = self
            .source_owners
            .iter()
            .filter_map(|(&rank, node)| (node == &self.node_id).then_some(rank))
            .collect();
        if self.version != 1
            || self.engine_id.trim().is_empty()
            || self.node_id.trim().is_empty()
            || self.node_count == 0
            || (!self.leader && self.node_count < 2)
            || (!self.source_owners.is_empty() && self.kv_block_size == 0)
            || local.len() != self.local_dp_ranks.len()
            || local != expected
            || self
                .source_owners
                .iter()
                .any(|(&rank, node)| rank >= dp_size || node.trim().is_empty())
        {
            return Err(client::protocol_error(
                "Invalid node identity or KV source ownership",
            ));
        }
        Ok(())
    }

    pub fn select_local_sources(
        &self,
        sources: &mut Vec<pb::KvEventSource>,
    ) -> Result<(), DynamoError> {
        let expected: BTreeSet<_> = if self.leader {
            self.source_owners.keys().copied().collect()
        } else {
            self.local_dp_ranks.iter().copied().collect()
        };
        let actual: BTreeSet<_> = sources
            .iter()
            .filter_map(|s| s.data_parallel_rank)
            .collect();
        if actual != expected || actual.len() != sources.len() {
            return Err(client::protocol_error(
                "KV sources disagree with authoritative node ownership",
            ));
        }
        sources.retain(|source| {
            source
                .data_parallel_rank
                .is_some_and(|rank| self.local_dp_ranks.contains(&rank))
        });
        Ok(())
    }

    pub fn matches_leader(&self, leader: &Self) -> bool {
        leader.leader
            && leader.engine_id == self.engine_id
            && leader.node_count == self.node_count
            && leader.source_owners == self.source_owners
            && leader.kv_block_size == self.kv_block_size
            && leader.node_id != self.node_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(leader: bool) -> NodeMetadata {
        NodeMetadata {
            version: 1,
            engine_id: "incarnation-a".into(),
            node_id: if leader { "leader" } else { "follower" }.into(),
            leader,
            node_count: 2,
            kv_block_size: 32,
            local_dp_ranks: vec![if leader { 0 } else { 1 }],
            source_owners: [(0, "leader".into()), (1, "follower".into())].into(),
        }
    }

    #[test]
    fn authoritative_ownership_filters_leader_but_rejects_foreign_follower_source() {
        let leader = node(true);
        let follower = node(false);
        let mut sources: Vec<_> = [0, 1]
            .into_iter()
            .map(|rank| pb::KvEventSource {
                data_parallel_rank: Some(rank),
                ..Default::default()
            })
            .collect();
        assert!(follower.select_local_sources(&mut sources.clone()).is_err());
        leader.select_local_sources(&mut sources).unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].data_parallel_rank, Some(0));
        let mut missing = Vec::new();
        assert!(follower.select_local_sources(&mut missing).is_err());
        let mut empty = follower;
        empty.source_owners.clear();
        empty.local_dp_ranks.clear();
        empty.select_local_sources(&mut missing).unwrap();
    }

    #[test]
    fn reused_address_cannot_attach_to_new_engine_incarnation() {
        let follower = node(false);
        let mut leader = node(true);
        assert!(follower.matches_leader(&leader));
        leader.engine_id = "incarnation-b".into();
        assert!(!follower.matches_leader(&leader));
        leader = node(true);
        leader.source_owners.insert(1, "leader".into());
        assert!(!follower.matches_leader(&leader));
    }

    #[test]
    fn metadata_distinguishes_legacy_from_invalid_or_overlapping_ownership() {
        assert!(
            NodeMetadata::from_info(Some(&pb::ServerInfo::default()))
                .unwrap()
                .is_none()
        );
        let mut node = node(false);
        let info = |node: &NodeMetadata| pb::ServerInfo {
            extra: Some(
                dynamo_sidecar_common::json_to_struct(
                    serde_json::json!({ NODE_METADATA_KEY: node }),
                    "node",
                )
                .unwrap(),
            ),
            parallelism: Some(pb::ParallelismInfo {
                data_parallel_size: Some(2),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            NodeMetadata::from_info(Some(&info(&node))).unwrap(),
            Some(node.clone())
        );
        node.local_dp_ranks.push(0);
        assert!(NodeMetadata::from_info(Some(&info(&node))).is_err());
        node.local_dp_ranks = vec![1, 1];
        assert!(NodeMetadata::from_info(Some(&info(&node))).is_err());
    }
}
