// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
#[macro_use]
#[path = "common/dynamodb.rs"]
mod dynamodb_support;

use std::sync::Arc;
use std::time::Duration;

use dynamo_global_common::{Lease, RouterRegistration};
use dynamo_global_discovery_client::{GlobalDiscoveryClient, RegistrationStatus, RouterHeartbeat};
use dynamo_global_discovery_service::backends::{memory::MemoryPoolStore, routers::RouterStore};
use dynamo_global_discovery_service::pool_discovery::{Directory, RouterRegistrar};
use dynamo_global_discovery_service::{ManualClock, PoolDiscoveryBackend};
use reqwest::StatusCode;
use tokio_util::sync::CancellationToken;

fn registration(id: &str) -> RouterRegistration {
    RouterRegistration {
        replica: id.into(),
        url: "http://127.0.0.1:50051/".parse().unwrap(),
    }
}

async fn start(store: Arc<RouterStore>) -> (GlobalDiscoveryClient, String, CancellationToken) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let cancel = CancellationToken::new();
    let pools = Arc::new(MemoryPoolStore::new(Lease::new(1, 3).unwrap()));
    tokio::spawn(dynamo_global_discovery_service::serve(
        listener,
        PoolDiscoveryBackend {
            pools: pools.clone(),
            routers: store.clone(),
            registrar: Some(pools),
            router_registrar: Some(store),
        },
        None,
        cancel.clone(),
    ));
    (
        GlobalDiscoveryClient::new(base.parse().unwrap(), Duration::from_secs(2)).unwrap(),
        base,
        cancel,
    )
}

async fn lifecycle(store: Arc<RouterStore>, clock: Arc<ManualClock>) {
    let (client, base, cancel) = start(store.clone()).await;
    let first = registration("region/pod-uid-1");
    assert_eq!(
        client.register_router(&first).await.unwrap(),
        Lease::new(1, 3).unwrap()
    );
    let listing = client.router_listing().await.unwrap();
    assert!(!listing.complete);
    assert_eq!(listing.routers[0].replica.0, first.replica);
    clock.advance(Duration::from_secs(2));
    let _ = client.register_router(&first).await.unwrap();
    clock.advance(Duration::from_secs(2));
    assert_eq!(
        client.routers().await.unwrap().len(),
        1,
        "heartbeat extends original lease"
    );
    let mut conflict = first.clone();
    conflict.url = "http://127.0.0.2:50051".parse().unwrap();
    assert!(client.register_router(&conflict).await.is_err());
    assert_eq!(client.routers().await.unwrap()[0].url, first.url);
    for body in [
        serde_json::json!({"replica":"", "url":"http://127.0.0.1:50051"}),
        serde_json::json!({"replica":"x", "url":"http://0.0.0.0:50051"}),
        serde_json::json!({"replica":"x", "url":"file:///etc/passwd"}),
    ] {
        let response = reqwest::Client::new()
            .post(format!("{base}/routers"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }
    // A replacement incarnation can overlap while the previous lease drains.
    let _ = client
        .register_router(&registration("region/pod-uid-2"))
        .await
        .unwrap();
    assert_eq!(client.routers().await.unwrap().len(), 2);
    clock.advance(Duration::from_secs(6));
    let listing = client.router_listing().await.unwrap();
    assert!(listing.complete);
    assert!(listing.routers.is_empty());
    assert!(
        client.register_router(&conflict).await.is_err(),
        "expiry cannot reassign an incarnation ID"
    );
    cancel.cancel();
}

#[tokio::test]
async fn memory_http_registration_renewal_conflict_and_expiry() {
    let clock = Arc::new(ManualClock::new(1_800_000_000_000));
    lifecycle(
        Arc::new(RouterStore::memory_with_clock(
            Lease::new(1, 3).unwrap(),
            clock.clone(),
        )),
        clock,
    )
    .await;
}

#[tokio::test]
async fn router_heartbeat_registers_over_http_and_stops_on_cancel() {
    let clock = Arc::new(ManualClock::new(1_800_000_000_000));
    let store = Arc::new(RouterStore::memory_with_clock(
        Lease::new(1, 3).unwrap(),
        clock.clone(),
    ));
    let (client, _, server_cancel) = start(store).await;
    let cancel = CancellationToken::new();
    let heartbeat =
        RouterHeartbeat::spawn(client.clone(), registration("region/uid"), cancel.clone());
    tokio::time::timeout(Duration::from_secs(2), async {
        while !matches!(heartbeat.status(), RegistrationStatus::Registered(_)) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(client.routers().await.unwrap().len(), 1);
    cancel.cancel();
    drop(heartbeat);
    clock.advance(Duration::from_secs(4));
    assert!(client.routers().await.unwrap().is_empty());
    server_cancel.cancel();
}

#[tokio::test]
async fn dynamodb_http_registration_survives_service_restart_and_expires() {
    let endpoint = skip_without_dynamodb!();
    let fixture = dynamodb_support::Fixture::new(&endpoint).await;
    let pool_store = fixture.store_with_lease(Lease::new(1, 3).unwrap());
    let store = Arc::new(pool_store.router_store());
    lifecycle(store, fixture.clock.clone()).await;
    let first = pool_store.router_store();
    let restarted_registration = registration("restart/uid");
    let _ = first
        .register_router(restarted_registration.clone())
        .await
        .unwrap();
    drop(first);
    let restarted = fixture
        .store_with_lease(Lease::new(1, 3).unwrap())
        .router_store();
    assert_eq!(
        restarted.list().await.unwrap().items[0].replica.0,
        restarted_registration.replica
    );
    let _ = restarted
        .register_router(restarted_registration)
        .await
        .unwrap();
    // Simulate a row replicated from another region: local writers must not
    // take ownership, and reads retain it conservatively only within grace.
    fixture
        .client
        .update_item()
        .table_name(&fixture.table)
        .key(
            "pk",
            aws_sdk_dynamodb::types::AttributeValue::S("ROUTERS".into()),
        )
        .key(
            "sk",
            aws_sdk_dynamodb::types::AttributeValue::S("restart/uid".into()),
        )
        .update_expression("SET writer_region = :region")
        .expression_attribute_values(
            ":region",
            aws_sdk_dynamodb::types::AttributeValue::S("us-west-2".into()),
        )
        .send()
        .await
        .unwrap();
    assert!(
        restarted
            .register_router(registration("restart/uid"))
            .await
            .is_err()
    );
    fixture.clock.advance(Duration::from_secs(6));
    let remote = restarted.list().await.unwrap();
    assert_eq!(remote.items.len(), 1);
    assert_eq!(
        remote.completeness,
        dynamo_global_discovery_service::pool_discovery::Completeness::Estimated
    );
    fixture.clock.advance(Duration::from_secs(60));
    assert!(restarted.list().await.unwrap().items.is_empty());
    fixture
        .client
        .delete_table()
        .table_name(&fixture.table)
        .send()
        .await
        .unwrap();
    assert!(
        restarted.list().await.is_err(),
        "storage failure is not an empty directory"
    );
    fixture.clock.advance(Duration::from_secs(3600));
    dynamodb_support::create_table(&fixture.client, &fixture.table).await;
    let recovered = restarted.list().await.unwrap();
    assert!(recovered.items.is_empty());
    assert_eq!(
        recovered.completeness,
        dynamo_global_discovery_service::pool_discovery::Completeness::Estimated,
        "first recovery read must allow a full lease for routers to renew after a long outage"
    );
    fixture.clock.advance(Duration::from_secs(4));
    assert_eq!(
        restarted.list().await.unwrap().completeness,
        dynamo_global_discovery_service::pool_discovery::Completeness::Exact
    );
    fixture
        .client
        .delete_table()
        .table_name(&fixture.table)
        .send()
        .await
        .unwrap();
}
