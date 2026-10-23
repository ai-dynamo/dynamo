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
        router_hint_gpu_transfer: None,
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
        memory_kind: crate::services::selection::TransferSourceMemoryKind::Host,
        target_gpu_direct: false,
        target_transfer_domain: Some("certified-layout-v1"),
        target_topology_domains: None,
        target_kv_transfer_domain: None,
        target_kv_transfer_enforcement: None,
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

fn owner_id() -> CacheOwnerId {
    CacheOwnerId::new(
        PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                RoutingScopeId::new([2; 16], IdentitySource::Explicit),
            ),
            DcId::new(3),
        ),
        StableDpSlotId::new([4; 16], IdentitySource::Explicit),
    )
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
    let owner = owner_id();
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
                    topology_domains: HashMap::new(),
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

#[test]
fn required_topology_is_independent_of_certified_transfer_domain() {
    let prompt = PromptRequest::default();
    let target_domains = HashMap::from([("zone".to_string(), "near".to_string())]);
    let empty_domains = HashMap::new();
    let mut target = query(&prompt);
    target.target_kv_transfer_domain = Some("zone");
    let candidate = candidates(WorkerWithDpRank::new(7, 0).into());
    for enforcement in [None, Some(KvTransferEnforcement::Required)] {
        target.target_kv_transfer_enforcement = enforcement;
        for target_topology in [None, Some(&empty_domains), Some(&target_domains)] {
            target.target_topology_domains = target_topology;
            for island in [None, Some("far"), Some("near")] {
                let mut donor = config();
                if let Some(island) = island {
                    donor
                        .topology_domains
                        .insert("zone".to_string(), island.to_string());
                }
                let result =
                    transfer_source_for_target(&HashMap::from([(7, donor)]), &target, &candidate);
                assert_eq!(
                    result.is_some(),
                    target_topology == Some(&target_domains) && island == Some("near")
                );
            }
        }
    }
    // A topology match never certifies a different model/layout namespace.
    let mut donor = config();
    donor.topology_domains = target_domains.clone();
    donor.router_hint_transfer_domain = Some("another-layout".to_string());
    assert!(
        transfer_source_for_target(&HashMap::from([(7, donor)]), &target, &candidate).is_none()
    );
}

#[test]
fn preferred_topology_ranks_shorter_near_prefix_before_longer_far_prefix() {
    let prompt = PromptRequest::default();
    let domains = HashMap::from([("zone".to_string(), "near".to_string())]);
    let mut target = query(&prompt);
    target.target_topology_domains = Some(&domains);
    target.target_kv_transfer_domain = Some("zone");
    target.target_kv_transfer_enforcement = Some(KvTransferEnforcement::Preferred);
    let near = WorkerWithDpRank::new(7, 0).into();
    let far = WorkerWithDpRank::new(8, 0).into();
    let mut near_config = config();
    near_config.topology_domains = domains.clone();
    let mut far_config = config();
    far_config
        .topology_domains
        .insert("zone".to_string(), "far".to_string());
    let mut configs = HashMap::from([(7, near_config), (8, far_config)]);
    let candidates = KvTransferCandidates {
        block_hashes: (101..=104).map(ExternalSequenceBlockHash).collect(),
        owner_prefix_blocks: vec![(near, 3), (far, 4)],
        routing_snapshot: None,
    };
    let result = transfer_source_for_target(&configs, &target, &candidates).unwrap();
    assert_eq!(result.source, near);
    assert_eq!(result.payload.block_hashes.len(), 3);
    // An incompatible same-island donor cannot suppress the preferred fallback.
    configs.get_mut(&7).unwrap().router_hint_transfer_domain = None;
    assert_eq!(
        transfer_source_for_target(&configs, &target, &candidates)
            .unwrap()
            .source,
        far
    );
    target.target_kv_transfer_enforcement = Some(KvTransferEnforcement::Required);
    assert!(transfer_source_for_target(&configs, &target, &candidates).is_none());
}

#[test]
fn persistent_topology_overrides_attachment_and_only_empty_metadata_inherits() {
    let prompt = PromptRequest::default();
    let domains = HashMap::from([("zone".to_string(), "near".to_string())]);
    let mut target = query(&prompt);
    target.target_topology_domains = Some(&domains);
    target.target_kv_transfer_domain = Some("zone");
    let owner = owner_id();
    let source =
        KvTransferCandidateSource::CacheOwner(ResidencyOwner::cache_owner(owner).compact_key());
    for (attached_island, owner_domains, expected) in [
        ("near", HashMap::new(), true),
        ("far", domains.clone(), true),
        (
            "near",
            HashMap::from([("zone".to_string(), "far".to_string())]),
            false,
        ),
        (
            "near",
            HashMap::from([("rack".to_string(), "near".to_string())]),
            false,
        ),
    ] {
        let mut donor = config();
        donor.kv_event_source_mode = Some("state_agent_v2".to_string());
        donor
            .topology_domains
            .insert("zone".to_string(), attached_island.to_string());
        let mut candidates = candidates(source);
        candidates.routing_snapshot = Some(Arc::new(ResidencyRoutingSnapshot::new(
            ResidencyProjection::default(),
            [(
                owner,
                RouterHintSourceMetadata {
                    source_control_endpoint: "tcp://owner:9000".to_string(),
                    worker_type: "decode".to_string(),
                    topology_domains: owner_domains,
                },
                Some(WorkerWithDpRank::new(7, 0)),
            )],
        )));
        assert_eq!(
            transfer_source_for_target(&HashMap::from([(7, donor)]), &target, &candidates)
                .is_some(),
            expected
        );
    }
}

fn gpu_config() -> SelectionWorkerConfig {
    let mut config = config();
    config.router_hint_worker_type = None;
    config.router_hint_source_control_endpoints.clear();
    config.router_hint_gpu_transfer = Some(crate::protocols::KvGpuTransferWorkerConfig {
        enabled: true,
        worker_type: "decode".into(),
        source_control_endpoints: HashMap::from([(1, "tcp://gpu-only:9011".into())]),
    });
    config
}

fn gpu_query(prompt: &PromptRequest) -> TransferSourceQuery<'_> {
    let mut query = query(prompt);
    query.memory_kind = TransferSourceMemoryKind::Gpu;
    query.target_gpu_direct = true;
    query
}

#[test]
fn gpu_donor_requires_separate_source_target_capabilities_and_dp_endpoint() {
    let prompt = PromptRequest::default();
    let candidates = candidates(WorkerWithDpRank::new(7, 1).into());
    let gpu = HashMap::from([(7, gpu_config())]);
    let result = transfer_source_for_target(&gpu, &gpu_query(&prompt), &candidates).unwrap();
    assert_eq!(result.memory_kind, TransferSourceMemoryKind::Gpu);
    assert_eq!(
        result.payload.source_control_endpoint,
        "tcp://gpu-only:9011"
    );
    assert_eq!(result.payload.block_hashes.len(), 3);
    // A CPU endpoint can never certify GPU memory, and GPU-only cards cannot
    // be silently interpreted as legacy CPU donors.
    assert!(
        transfer_source_for_target(
            &HashMap::from([(7, config())]),
            &gpu_query(&prompt),
            &candidates
        )
        .is_none()
    );
    assert!(transfer_source_for_target(&gpu, &query(&prompt), &candidates).is_none());
    let mut old_target = gpu_query(&prompt);
    old_target.target_gpu_direct = false;
    assert!(transfer_source_for_target(&gpu, &old_target, &candidates).is_none());
    for change in 0..5 {
        let mut config = gpu_config();
        match change {
            0 => config.router_hint_gpu_transfer.as_mut().unwrap().enabled = false,
            1 => config
                .router_hint_gpu_transfer
                .as_mut()
                .unwrap()
                .source_control_endpoints
                .clear(),
            2 => {
                config
                    .router_hint_gpu_transfer
                    .as_mut()
                    .unwrap()
                    .worker_type = "prefill".into()
            }
            3 => config.router_hint_transfer_domain = Some("foreign-layout".into()),
            _ => config.data_parallel_size = 1,
        }
        assert!(
            transfer_source_for_target(
                &HashMap::from([(7, config)]),
                &gpu_query(&prompt),
                &candidates
            )
            .is_none()
        );
    }
}
