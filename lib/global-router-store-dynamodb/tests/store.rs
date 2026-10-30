// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real SDK/conditional-expression tests. Run with an ephemeral DynamoDB Local:
//! DYNAMODB_LOCAL_ENDPOINT=http://127.0.0.1:18000 cargo test -p
//! dynamo-global-router-store-dynamodb --test store -- --ignored
//! These tests do not emulate or claim to verify MREC replication.

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
    AffinityBinding, AffinityConfig, AffinityError, AffinityKey, AffinityStore, SessionAffinity,
    StoreBackedSessionAffinity, WriteOutcome,
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
        a.compare_exchange(&key, None, &east),
        b.compare_exchange(&key, None, &west)
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
        b.compare_exchange(&key, Some(&winner.revision), &expired)
            .await
            .unwrap(),
        WriteOutcome::Applied
    );
    assert_eq!(
        a.compare_exchange(&key, Some(&winner.revision), &winner)
            .await
            .unwrap(),
        WriteOutcome::Conflict
    );
    assert_eq!(
        a.compare_exchange(&key, None, &winner).await.unwrap(),
        WriteOutcome::Conflict
    );
    let tracker = StoreBackedSessionAffinity::new(a.clone(), Duration::from_secs(1800)).unwrap();
    let observed = tracker.lookup(&key).await.unwrap();
    assert!(observed.pool.is_none());
    assert_eq!(observed.revision.as_deref(), Some("expired"));
    assert_eq!(
        tracker.bind(&key, &observed, &west.pool).await.unwrap(),
        WriteOutcome::Applied
    );
    assert_eq!(b.read(&key).await.unwrap().unwrap().pool, west.pool);

    for field in 0..4 {
        let mut other = key.clone();
        match field {
            0 => other.scope.push('x'),
            1 => other.policy.push('x'),
            2 => other.model.push('x'),
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
    assert!(
        (now + 1790..=now + 1800).contains(&expires),
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
    // New tracker/store instance, with a different local-region preference.
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
