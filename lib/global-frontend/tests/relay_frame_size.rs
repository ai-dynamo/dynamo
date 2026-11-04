// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_global_common::grpc::MAX_POOL_VIEW_FRAME_BYTES;
use dynamo_global_common::{PoolKey, grpc};
use dynamo_global_frontend::{PoolSnapshot, RelayGatedPoolCatalog, serve_relay_views};
use dynamo_global_router::Pool;
use dynamo_llm::pool_relay::wan::grpc::protocol as relay;
use prost::Message;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
mod common;

#[tokio::test]
async fn combined_subscription_state_fits_receiving_frame_limit() {
    tokio::time::timeout(Duration::from_secs(20), async {
        let raw = common::state(true, 1, 1, 1);
        let mut catalog = relay::KvPoolCatalogUpdate::decode(raw.catalog).unwrap();
        let mut readiness = relay::ServingReadinessUpdate::decode(raw.readiness).unwrap();
        let mut serving = relay::ServingLoadUpdate::decode(raw.serving_load).unwrap();
        let mut kv = relay::KvPoolLoadUpdate::decode(raw.kv_load).unwrap();
        let descriptor = catalog.snapshot.as_ref().unwrap().pools[0].clone();
        let member = readiness.entries[0].members[0].clone();
        let serving_pool = serving.pools[0].clone();
        let kv_pool = kv.pools[0].clone();
        catalog.snapshot.as_mut().unwrap().pools.clear();
        readiness.entries[0].members.clear();
        serving.pools.clear();
        kv.pools.clear();
        for index in 1..=8000_u64 {
            let mut descriptor = descriptor.clone();
            descriptor
                .producer
                .as_mut()
                .unwrap()
                .pool_id
                .as_mut()
                .unwrap()
                .indexer_domain
                .as_mut()
                .unwrap()
                .routing_scope
                .as_mut()
                .unwrap()
                .digest = (index as u128).to_le_bytes().to_vec().into();
            descriptor.serving_endpoint.as_mut().unwrap().component =
                format!("worker-{index:05}-{}", "a".repeat(48));
            relay::validate_pool_descriptor(&descriptor).unwrap();
            let mut member = member.clone();
            member.endpoint = descriptor.serving_endpoint.clone();
            member.pool_id = descriptor.producer.as_ref().unwrap().pool_id.clone();
            let mut serving_pool = serving_pool.clone();
            serving_pool.producer = descriptor.producer.clone();
            let mut kv_pool = kv_pool.clone();
            kv_pool.producer = descriptor.producer.clone();
            catalog.snapshot.as_mut().unwrap().pools.push(descriptor);
            readiness.entries[0].members.push(member);
            serving.pools.push(serving_pool);
            kv.pools.push(kv_pool);
        }
        relay::validate_topology_entry(&readiness.entries[0]).unwrap();
        let lengths = [
            catalog.encoded_len(),
            readiness.encoded_len(),
            serving.encoded_len(),
            kv.encoded_len(),
        ];
        assert!(
            lengths.iter().all(|size| *size < 4 * 1024 * 1024),
            "{lengths:?}"
        );
        let state = grpc::PoolState {
            catalog: catalog.encode_to_vec().into(),
            readiness: readiness.encode_to_vec().into(),
            serving_load: serving.encode_to_vec().into(),
            kv_load: kv.encode_to_vec().into(),
        };
        assert!(state.encoded_len() > 4 * 1024 * 1024, "{lengths:?}");
        assert!(state.encoded_len() + 32 <= MAX_POOL_VIEW_FRAME_BYTES);
        let key = PoolKey::new("east", "dynamo", "blue").unwrap();
        let (_sender, discovery) = watch::channel(Arc::new(PoolSnapshot::new(vec![Pool::new(
            &key,
            "qwen",
            "east",
            vec!["http://pool:8000".parse().unwrap()],
        )])));
        let cancel = CancellationToken::new();
        let gated = Arc::new(RelayGatedPoolCatalog::new(discovery, cancel.clone()));
        let mut current = gated.subscribe();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(serve_relay_views(
            listener,
            gated,
            cancel.clone(),
        )));
        let mut client = grpc::pool_view_receiver_client::PoolViewReceiverClient::connect(format!(
            "http://{address}"
        ))
        .await
        .unwrap();
        use grpc::pool_view_frame::Event;
        let frames = vec![
            grpc::PoolViewFrame {
                event: Some(Event::Hello(grpc::Hello {
                    pool_key: Some(grpc::PoolKey {
                        cluster_id: "east".into(),
                        namespace: "dynamo".into(),
                        dgd_name: "blue".into(),
                    }),
                    logical_service: "blue".into(),
                    replica_id: "1".into(),
                    incarnation: 1,
                })),
            },
            grpc::PoolViewFrame {
                event: Some(Event::Snapshot(grpc::Snapshot {
                    revision: 0,
                    pool_present: true,
                    state: Some(state),
                })),
            },
        ];
        let (sender, receiver) = tokio::sync::mpsc::channel(2);
        let publishing = tokio::spawn(async move {
            client
                .publish(tokio_stream::wrappers::ReceiverStream::new(receiver))
                .await
        });
        for frame in frames {
            sender.send(frame).await.unwrap();
        }
        while current.borrow().pools().is_empty() {
            current.changed().await.unwrap();
        }
        assert_eq!(current.borrow().pools().len(), 1);
        drop(sender);
        assert_eq!(
            publishing
                .await
                .unwrap()
                .unwrap()
                .into_inner()
                .accepted_revision,
            0
        );
        while !current.borrow().pools().is_empty() {
            current.changed().await.unwrap();
        }
        cancel.cancel();
        drop(server);
    })
    .await
    .unwrap();
}
