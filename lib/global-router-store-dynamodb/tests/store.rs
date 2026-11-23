// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real SDK and conditional-expression tests, against DynamoDB Local. CI does
//! not start DynamoDB Local, so these are `#[ignore]`d there; run them by hand
//! after changing this crate:
//!
//! ```text
//! docker run --rm -d -p 18000:8000 amazon/dynamodb-local:3.0.0
//! DYNAMODB_LOCAL_ENDPOINT=http://127.0.0.1:18000 \
//!   cargo test -p dynamo-global-router-store-dynamodb --test store -- --ignored
//! ```
//!
//! Each test creates and deletes its own table, and refuses any endpoint other
//! than localhost. These tests do not emulate or claim to verify MREC
//! replication.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aws_sdk_dynamodb::{
    Client,
    config::{BehaviorVersion, Credentials, Region, retry::RetryConfig, timeout::TimeoutConfig},
    types::{
        AttributeDefinition, AttributeValue, BillingMode, KeySchemaElement, KeyType,
        ScalarAttributeType,
    },
};
use dynamo_global_common::PoolKey;
use dynamo_global_router::affinity::{
    AffinityBinding, AffinityConfig, AffinityError, AffinityKey, AffinityStore, Expected,
    SessionAffinity, StoreBackedSessionAffinity, WriteOutcome,
};
use dynamo_global_router::{
    AffinitySelector, InFlightTracker, Pool, PoolSelector, PoolSignals, RoutingRequest,
};
use dynamo_global_router_store_dynamodb::DynamoDbAffinityStore;

struct Table {
    client: Client,
    name: String,
}

impl Table {
    async fn create() -> Self {
        let endpoint = std::env::var("DYNAMODB_LOCAL_ENDPOINT")
            .expect("set DYNAMODB_LOCAL_ENDPOINT to DynamoDB Local");
        assert!(
            endpoint.starts_with("http://127.0.0.1:") || endpoint.starts_with("http://localhost:"),
            "these tests create/delete tables and must target localhost"
        );
        let client = Client::from_conf(
            aws_sdk_dynamodb::Config::builder()
                .behavior_version(BehaviorVersion::latest())
                .region(Region::new("us-east-2"))
                .credentials_provider(Credentials::new(
                    "local",
                    "local",
                    None,
                    None,
                    "local-tests",
                ))
                .endpoint_url(endpoint)
                .retry_config(RetryConfig::standard().with_max_attempts(1))
                .timeout_config(
                    TimeoutConfig::builder()
                        .operation_timeout(Duration::from_secs(5))
                        .build(),
                )
                .build(),
        );
        let name = format!("affinity-test-{}", uuid::Uuid::new_v4());
        client
            .create_table()
            .table_name(&name)
            .billing_mode(BillingMode::PayPerRequest)
            .attribute_definitions(
                AttributeDefinition::builder()
                    .attribute_name("itemHashKey")
                    .attribute_type(ScalarAttributeType::S)
                    .build()
                    .unwrap(),
            )
            .attribute_definitions(
                AttributeDefinition::builder()
                    .attribute_name("sortKey")
                    .attribute_type(ScalarAttributeType::S)
                    .build()
                    .unwrap(),
            )
            .key_schema(
                KeySchemaElement::builder()
                    .attribute_name("itemHashKey")
                    .key_type(KeyType::Hash)
                    .build()
                    .unwrap(),
            )
            .key_schema(
                KeySchemaElement::builder()
                    .attribute_name("sortKey")
                    .key_type(KeyType::Range)
                    .build()
                    .unwrap(),
            )
            .send()
            .await
            .unwrap();
        Self { client, name }
    }

    fn store(&self) -> Arc<DynamoDbAffinityStore> {
        Arc::new(DynamoDbAffinityStore::new(self.client.clone(), self.name.clone(), true).unwrap())
    }

    async fn delete(self) {
        self.client
            .delete_table()
            .table_name(self.name)
            .send()
            .await
            .unwrap();
    }
}

fn key() -> AffinityKey {
    AffinityKey {
        scope: "test-deployment".into(),
        tenant: "anonymous".into(),
        policy: "headroom-v1".into(),
        model: "qwen".into(),
        session: "s".into(),
    }
}

fn pool(region: &str) -> Pool {
    Pool::new(
        &PoolKey::new(region, "dynamo", "qwen").unwrap(),
        "qwen",
        region,
        vec![format!("http://{region}:8000").parse().unwrap()],
    )
}

fn binding(region: &str, revision: &str) -> AffinityBinding {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    AffinityBinding {
        pool: pool(region).id,
        revision: revision.into(),
        updated_at: now,
        expires_at: now + 1800,
    }
}

#[tokio::test]
#[ignore = "requires DynamoDB Local at DYNAMODB_LOCAL_ENDPOINT"]
async fn conditional_writes_expiration_and_schema_match_the_store_contract() {
    let table = Table::create().await;
    let a = table.store();
    let b = table.store();
    let key = key();
    assert_eq!(a.read(&key).await.unwrap(), None);
    let east = binding("east", "a");
    let west = binding("west", "b");
    let (left, right) = tokio::join!(
        a.compare_exchange(&key, &Expected::Absent, &east),
        b.compare_exchange(&key, &Expected::Absent, &west)
    );
    let outcomes = [left.unwrap(), right.unwrap()];
    assert_eq!(
        outcomes
            .iter()
            .filter(|&&o| o == WriteOutcome::Applied)
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|&&o| o == WriteOutcome::Conflict)
            .count(),
        1
    );
    let winner = a.read(&key).await.unwrap().unwrap();
    assert!(winner == east || winner == west);
    let expired = AffinityBinding {
        updated_at: 1,
        expires_at: 2,
        revision: "expired".into(),
        ..winner.clone()
    };
    assert_eq!(
        b.compare_exchange(&key, &Expected::Revision(winner.revision.clone()), &expired)
            .await
            .unwrap(),
        WriteOutcome::Applied
    );
    assert_eq!(
        a.compare_exchange(&key, &Expected::Revision(winner.revision.clone()), &winner)
            .await
            .unwrap(),
        WriteOutcome::Conflict
    );
    assert_eq!(
        a.compare_exchange(&key, &Expected::Absent, &winner)
            .await
            .unwrap(),
        WriteOutcome::Conflict
    );
    let tracker = StoreBackedSessionAffinity::new(a.clone(), Duration::from_secs(1800)).unwrap();
    let observed = tracker.lookup(&key).await.unwrap();
    assert!(observed.pool.is_none());
    assert_eq!(observed.expected, Expected::Revision("expired".into()));
    assert_eq!(
        tracker.bind(&key, &observed, &west.pool).await.unwrap(),
        WriteOutcome::Applied
    );
    assert_eq!(b.read(&key).await.unwrap().unwrap().pool, west.pool);

    for field in 0..5 {
        let mut other = key.clone();
        match field {
            0 => other.scope = format!("{}x", other.scope).into(),
            1 => other.tenant.push('x'),
            2 => other.policy = format!("{}x", other.policy).into(),
            3 => other.model.push('x'),
            _ => other.session.push('x'),
        }
        assert!(a.read(&other).await.unwrap().is_none());
    }
    // Scan is test-only: inspect the actual wire schema without exposing it
    // through the domain store interface.
    let rows = table
        .client
        .scan()
        .table_name(&table.name)
        .consistent_read(true)
        .send()
        .await
        .unwrap();
    let mut item = rows.items.unwrap().pop().unwrap();
    assert!(
        item["itemHashKey"]
            .as_s()
            .unwrap()
            .starts_with("AFFINITY#v1#")
    );
    assert!(item["sortKey"].as_s().unwrap().starts_with("MODEL#"));
    assert_eq!(item["schemaVersion"], AttributeValue::N("1".into()));
    let expires: u64 = item["expiresAt"].as_n().unwrap().parse().unwrap();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // The idle deadline is rounded up to the next whole second, so a write
    // at 10.4s expires at 1811, never at 1810: a binding is never cut short
    // of the idle window it was given.
    assert!(
        (now + 1790..=now + 1801).contains(&expires),
        "TTL must be epoch seconds"
    );
    item.insert("schemaVersion".into(), AttributeValue::N("999".into()));
    table
        .client
        .put_item()
        .table_name(&table.name)
        .set_item(Some(item))
        .send()
        .await
        .unwrap();
    assert!(matches!(
        a.read(&key).await,
        Err(AffinityError::InvalidRecord(_))
    ));
    // That row is now unreachable by any conditional write: its revision
    // cannot be read, and `Absent` never matches a present item. Without a
    // repair path the session stays fail-open until the table is fixed by
    // hand, and TTL cannot clear it when `expiresAt` is what is corrupt.
    let observed = tracker.lookup(&key).await.unwrap();
    assert!(observed.pool.is_none());
    assert_eq!(observed.expected, Expected::Unreadable);
    assert_eq!(
        tracker.bind(&key, &observed, &east.pool).await.unwrap(),
        WriteOutcome::Applied
    );
    // Readable again, and bound normally from here, so a mixed-version
    // rollout recovers as soon as an old reader serves that session.
    let repaired = a.read(&key).await.unwrap().unwrap();
    assert_eq!(repaired.pool, east.pool);
    assert_eq!(
        tracker.lookup(&key).await.unwrap().expected,
        Expected::Revision(repaired.revision)
    );
    table.delete().await;
}

fn selector(store: Arc<dyn AffinityStore>, region: &str) -> AffinitySelector {
    AffinitySelector::new(
        Arc::new(PoolSignals::new(Arc::new(InFlightTracker::new()))),
        region.into(),
        2.0,
        Arc::new(StoreBackedSessionAffinity::new(store, Duration::from_secs(1800)).unwrap()),
        AffinityConfig {
            scope: "test-deployment".into(),
            timeout: Duration::from_secs(5),
            ..Default::default()
        },
    )
}

#[tokio::test]
#[ignore = "requires DynamoDB Local at DYNAMODB_LOCAL_ENDPOINT"]
async fn independent_selectors_and_restarts_reuse_persisted_bindings() {
    let table = Table::create().await;
    let east = pool("east");
    let west = pool("west");
    let candidates = [&east, &west];
    let request = RoutingRequest::new("qwen").with_session(Some("s"));
    {
        let first = selector(table.store(), "east");
        assert_eq!(
            first.select(&request, &candidates).await.unwrap().pool.id,
            east.id
        );
    }
    let restarted = selector(table.store(), "west");
    assert_eq!(
        restarted
            .select(&request, &candidates)
            .await
            .unwrap()
            .pool
            .id,
        east.id
    );
    assert_eq!(
        restarted.select(&request, &[&west]).await.unwrap().pool.id,
        west.id
    );
    assert_eq!(
        table.store().read(&key()).await.unwrap().unwrap().pool,
        west.id
    );
    table.delete().await;
}
