// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::topology::proto::{
    GetSnapshotRequest, TopologyPayload, topology_service_server::TopologyService,
};
use crate::topology::{consumer::Consumer, payload::decode_payload};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use dynamo_runtime::discovery::{Discovery, DiscoveryQuery, DiscoverySpec, KVStoreDiscovery};
use dynamo_runtime::storage::kv;
use prost::Message;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tonic::{Code, Request};

fn encoded(snapshot: Snapshot, version: u32) -> String {
    STANDARD.encode(
        TopologyPayload {
            schema_version: version,
            snapshot: Some(snapshot),
        }
        .encode_to_vec(),
    )
}
fn spec(identity: Identity, payload: String) -> DiscoverySpec {
    DiscoverySpec::LpxTopology {
        namespace: "test".into(),
        deployment: identity.deployment,
        allocation: identity.allocation,
        build: identity.build,
        payload,
    }
}

#[test]
fn extension_requires_supported_version_and_complete_valid_snapshot() {
    assert_eq!(decode_payload(&encoded(snapshot(), 1)).unwrap(), snapshot());
    for version in [0, 2, u32::MAX] {
        assert_eq!(
            decode_payload(&encoded(snapshot(), version))
                .unwrap_err()
                .code(),
            Code::FailedPrecondition
        );
    }
    for malformed in ["not-base64", "", "AAAA"] {
        assert_eq!(
            decode_payload(malformed).unwrap_err().code(),
            Code::FailedPrecondition
        );
    }
    let mut invalid = snapshot();
    invalid.partitions.push(invalid.partitions[0].clone());
    assert_eq!(
        decode_payload(&encoded(invalid, 1)).unwrap_err().code(),
        Code::FailedPrecondition
    );
}

#[tokio::test]
async fn published_snapshot_needs_no_agent_and_replacements_do_not_mutate_returned_map() {
    let discovery = Arc::new(KVStoreDiscovery::new(
        kv::Manager::memory(),
        CancellationToken::new(),
    ));
    let consumer = Consumer::new(discovery.clone(), "test".into(), CancellationToken::new());
    let request = || {
        Request::new(GetSnapshotRequest {
            expected: Some(identity()),
        })
    };
    assert_eq!(
        consumer.get_snapshot(request()).await.unwrap_err().code(),
        Code::Unavailable
    );
    discovery
        .register(spec(identity(), encoded(snapshot(), 1)))
        .await
        .unwrap();
    // No Agent server, inference endpoint, or model card exists in this test.
    let pinned = consumer
        .get_snapshot(request())
        .await
        .unwrap()
        .into_inner()
        .snapshot
        .unwrap();
    assert_eq!(pinned, snapshot());
    assert!(
        discovery
            .list(DiscoveryQuery::AllEndpoints)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        discovery
            .list(DiscoveryQuery::AllModels)
            .await
            .unwrap()
            .is_empty()
    );
    let mut changed = snapshot();
    changed.partitions[0].partition_id = 77;
    let published = discovery
        .register(spec(identity(), encoded(changed.clone(), 1)))
        .await
        .unwrap();
    assert_eq!(
        consumer
            .get_snapshot(request())
            .await
            .unwrap()
            .into_inner()
            .snapshot
            .unwrap(),
        changed
    );
    discovery.unregister(published).await.unwrap();
    assert_eq!(
        consumer.get_snapshot(request()).await.unwrap_err().code(),
        Code::Unavailable
    );
    assert_eq!(pinned, snapshot());
}

#[tokio::test]
async fn wrong_build_or_payload_identity_fails_without_retry() {
    let discovery = Arc::new(KVStoreDiscovery::new(
        kv::Manager::memory(),
        CancellationToken::new(),
    ));
    let consumer = Consumer::new(discovery.clone(), "test".into(), CancellationToken::new());
    let request = || {
        Request::new(GetSnapshotRequest {
            expected: Some(identity()),
        })
    };
    let mut wrong = identity();
    wrong.build = "another-build".into();
    discovery
        .register(spec(wrong, encoded(snapshot(), 1)))
        .await
        .unwrap();
    assert_eq!(
        consumer.get_snapshot(request()).await.unwrap_err().code(),
        Code::FailedPrecondition
    );
    let mut wrong_snapshot = snapshot();
    wrong_snapshot.identity.as_mut().unwrap().allocation = "another-replica".into();
    discovery
        .register(spec(identity(), encoded(wrong_snapshot, 1)))
        .await
        .unwrap();
    assert_eq!(
        consumer.get_snapshot(request()).await.unwrap_err().code(),
        Code::FailedPrecondition
    );
}

#[test]
fn multiple_authorities_fail_immediately_even_if_their_snapshots_agree() {
    let first = spec(identity(), encoded(snapshot(), 1)).into_instance(1);
    let second = spec(identity(), encoded(snapshot(), 1)).into_instance(2);
    let error =
        crate::topology::consumer::select_snapshot(&identity(), vec![first, second]).unwrap_err();
    assert_eq!(error.code(), Code::FailedPrecondition);
    assert!(error.message().contains("ambiguous"));
}
