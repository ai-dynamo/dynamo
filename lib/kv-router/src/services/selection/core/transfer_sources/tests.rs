// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::identity::{
    CacheOwnerId, CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    StableDpSlotId,
};
use crate::protocols::{
    ExternalSequenceBlockHash, ResidencyOwner, ResidencyProjection, ResidencyRoutingSnapshot,
    RouterHintSourceMetadata,
};
use crate::services::selection::PromptRequest;

fn config() -> SelectionWorkerConfig {
    SelectionWorkerConfig {
        endpoint: "http://donor:8000".to_string(),
        data_parallel_start_rank: 0,
        data_parallel_size: 2,
        max_num_batched_tokens: None,
        total_kv_blocks: None,
        stable_routing_id: None,
        is_eagle: None,
        taints: Default::default(),
        topology_domains: Default::default(),
        kv_transfer_domain: None,
        kv_transfer_enforcement: None,
        kv_transfer_preferred_weight: None,
        router_hint_worker_type: Some("decode".to_string()),
        router_hint_transfer_domain: Some("certified-layout-v1".to_string()),
        router_hint_source_control_endpoints: HashMap::from([
            (0, "tcp://donor:9000".to_string()),
            (1, "tcp://donor:9001".to_string()),
        ]),
        kv_event_source_mode: None,
    }
}

fn query(prompt: &PromptRequest) -> TransferSourceQuery<'_> {
    TransferSourceQuery {
        model_name: "default",
        routing_group: "default",
        prompt: prompt.view(),
        target: WorkerWithDpRank::new(99, 0),
        target_worker_type: "prefill",
        target_transfer_domain: Some("certified-layout-v1"),
        target_block_size: 4,
        target_is_eagle: false,
        minimum_prefix_blocks: 1,
        deadline: Instant::now() + Duration::from_secs(1),
    }
}

fn candidates(source: KvTransferCandidateSource) -> KvTransferCandidates {
    KvTransferCandidates {
        block_hashes: vec![101, 102, 103]
            .into_iter()
            .map(ExternalSequenceBlockHash)
            .collect(),
        owner_prefix_blocks: vec![(source, 3)],
        routing_snapshot: None,
    }
}

#[test]
fn selects_source_dp_endpoint_and_preserves_complete_root_chain() {
    let prompt = PromptRequest::default();
    let configs = HashMap::from([(7, config())]);
    let source = WorkerWithDpRank::new(7, 1).into();
    let result =
        transfer_source_for_target(&configs, &query(&prompt), &candidates(source)).unwrap();
    assert_eq!(result.source, source);
    assert_eq!(result.payload.source_control_endpoint, "tcp://donor:9001");
    assert_eq!(
        result.payload.block_hashes,
        vec![
            ExternalSequenceBlockHash(101),
            ExternalSequenceBlockHash(102),
            ExternalSequenceBlockHash(103)
        ]
    );
}

#[test]
fn filters_unknown_layout_wrong_role_state_agent_endpoint_and_dp_rank() {
    let prompt = PromptRequest::default();
    let candidates = candidates(WorkerWithDpRank::new(7, 1).into());
    let mut variants = Vec::new();
    let mut unknown = config();
    unknown.router_hint_transfer_domain = None;
    variants.push(unknown);
    let mut other = config();
    other.router_hint_transfer_domain = Some("other-layout".to_string());
    variants.push(other);
    let mut empty = config();
    empty.router_hint_transfer_domain = Some(" ".to_string());
    variants.push(empty);
    let mut role = config();
    role.router_hint_worker_type = Some("aggregated".to_string());
    variants.push(role);
    let mut state_agent = config();
    state_agent.kv_event_source_mode = Some("state_agent_v2".to_string());
    variants.push(state_agent);
    let mut endpoint = config();
    endpoint
        .router_hint_source_control_endpoints
        .insert(1, " ".to_string());
    variants.push(endpoint);
    let mut wrong_dp = config();
    wrong_dp.data_parallel_size = 1;
    variants.push(wrong_dp);
    let mut eagle = config();
    eagle.is_eagle = Some(true);
    variants.push(eagle);
    for source in variants {
        assert!(
            transfer_source_for_target(&HashMap::from([(7, source)]), &query(&prompt), &candidates)
                .is_none()
        );
    }
    let configs = HashMap::from([(7, config())]);
    let mut target = query(&prompt);
    target.target = WorkerWithDpRank::new(7, 1);
    assert!(transfer_source_for_target(&configs, &target, &candidates).is_none());
    target = query(&prompt);
    target.minimum_prefix_blocks = 3;
    assert!(transfer_source_for_target(&configs, &target, &candidates).is_none());
}

#[test]
fn persistent_owner_uses_authoritative_attachment_and_rejects_unknown_domain() {
    let prompt = PromptRequest::default();
    let owner = CacheOwnerId::new(
        PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                RoutingScopeId::new([2; 16], IdentitySource::Explicit),
            ),
            DcId::new(3),
        ),
        StableDpSlotId::new([4; 16], IdentitySource::Explicit),
    );
    let mut source = config();
    source.kv_event_source_mode = Some("state_agent_v2".to_string());
    let configs = HashMap::from([(7, source)]);
    let candidate_source =
        KvTransferCandidateSource::CacheOwner(ResidencyOwner::cache_owner(owner).compact_key());
    for attached in [
        Some(WorkerWithDpRank::new(7, 1)),
        None,
        Some(WorkerWithDpRank::new(8, 1)),
    ] {
        let mut candidates = candidates(candidate_source);
        candidates.routing_snapshot = Some(Arc::new(ResidencyRoutingSnapshot::new(
            ResidencyProjection::default(),
            [(
                owner,
                RouterHintSourceMetadata {
                    source_control_endpoint: "tcp://persistent:9001".to_string(),
                    worker_type: "decode".to_string(),
                },
                attached,
            )],
        )));
        let result = transfer_source_for_target(&configs, &query(&prompt), &candidates);
        assert_eq!(
            result.is_some(),
            attached == Some(WorkerWithDpRank::new(7, 1))
        );
        if let Some(result) = result {
            assert_eq!(
                result.payload.source_control_endpoint,
                "tcp://persistent:9001"
            );
        }
    }
}
