// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helpers for the tests that run against dynamodb-local. A test binary
//! includes this with
//!
//! ```ignore
//! mod common;
//! #[macro_use]
//! #[path = "common/dynamodb.rs"]
//! mod dynamodb_support;
//! ```
//!
//! It is separate from `common/mod.rs` so the memory conformance suite does
//! not depend on the DynamoDB store or the AWS SDK.

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aws_sdk_dynamodb::Client;
use aws_sdk_dynamodb::config::retry::RetryConfig;
use aws_sdk_dynamodb::config::{BehaviorVersion, Credentials, Region};
use aws_sdk_dynamodb::types::{
    AttributeDefinition, BillingMode, KeySchemaElement, KeyType, ScalarAttributeType,
    TimeToLiveSpecification,
};
use dynamo_global_discovery_service::pool_discovery::Lease;
use dynamo_global_discovery_service::{DynamoDbConfig, DynamoDbPoolStore, ManualClock};
use url::Url;

use crate::common::{ClockHandle, lease};

/// Where dynamodb-local listens, from `GDS_TEST_DYNAMODB_ENDPOINT`
/// (`scripts/dynamodb-local.sh up` prints it). `None` when it is unset or
/// empty.
pub fn dynamodb_endpoint() -> Option<Url> {
    let value = std::env::var("GDS_TEST_DYNAMODB_ENDPOINT").ok()?;
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    Some(Url::parse(value).unwrap_or_else(|error| {
        panic!("GDS_TEST_DYNAMODB_ENDPOINT={value:?} is not a URL: {error}")
    }))
}

/// Evaluates to the dynamodb-local endpoint, or prints why and returns from
/// the test when `GDS_TEST_DYNAMODB_ENDPOINT` is not set.
macro_rules! skip_without_dynamodb {
    () => {
        match $crate::dynamodb_support::dynamodb_endpoint() {
            Some(endpoint) => endpoint,
            None => {
                eprintln!(
                    "skipping: GDS_TEST_DYNAMODB_ENDPOINT is not set \
                     (scripts/dynamodb-local.sh up prints it)"
                );
                return;
            }
        }
    };
}

/// A client for dynamodb-local. It ignores credentials, so these are dummies.
pub fn test_client(endpoint: &Url) -> Client {
    client_with(endpoint.as_str(), RetryConfig::standard())
}

/// A client that does not retry, for tests of an endpoint that is down.
pub fn test_client_without_retries(endpoint: &str) -> Client {
    client_with(endpoint, RetryConfig::disabled())
}

fn client_with(endpoint: &str, retry: RetryConfig) -> Client {
    let config = aws_sdk_dynamodb::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .endpoint_url(endpoint.trim_end_matches('/'))
        .credentials_provider(Credentials::new("test", "test", None, None, "gds-tests"))
        .retry_config(retry)
        .build();
    Client::from_conf(config)
}

/// A table name no other test, and no earlier run against the same
/// dynamodb-local, uses.
pub fn fresh_table_name() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "gds-test-{}-{nanos}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

/// The skew every store under test allows when it checks a lease.
pub const SKEW: Duration = Duration::from_secs(2);
/// How often a store under test polls for the watch.
pub const POLL: Duration = Duration::from_millis(200);

pub fn config(table: &str, endpoint: &Url) -> DynamoDbConfig {
    DynamoDbConfig {
        table: table.to_owned(),
        region: Some("us-east-1".to_owned()),
        endpoint: Some(endpoint.clone()),
        poll_interval: POLL,
        lease_skew: SKEW,
        ensure_table: false,
    }
}

/// Creates the table of the design (`docs/dynamodb-store.md`, "Table
/// definition") with TTL on `expires_at`, as `ensure_table` does.
pub async fn create_table(client: &Client, table: &str) {
    let attribute = |name: &str| {
        AttributeDefinition::builder()
            .attribute_name(name)
            .attribute_type(ScalarAttributeType::S)
            .build()
            .unwrap()
    };
    let key = |name: &str, key_type: KeyType| {
        KeySchemaElement::builder()
            .attribute_name(name)
            .key_type(key_type)
            .build()
            .unwrap()
    };
    client
        .create_table()
        .table_name(table)
        .attribute_definitions(attribute("pk"))
        .attribute_definitions(attribute("sk"))
        .key_schema(key("pk", KeyType::Hash))
        .key_schema(key("sk", KeyType::Range))
        .billing_mode(BillingMode::PayPerRequest)
        .send()
        .await
        .unwrap_or_else(|error| panic!("cannot create {table}: {error:?}"));
    client
        .update_time_to_live()
        .table_name(table)
        .time_to_live_specification(
            TimeToLiveSpecification::builder()
                .enabled(true)
                .attribute_name("expires_at")
                .build()
                .unwrap(),
        )
        .send()
        .await
        .unwrap_or_else(|error| panic!("cannot enable TTL on {table}: {error:?}"));
}

/// A fresh table, a clock the tests move, and stores over them.
pub struct Fixture {
    pub endpoint: Url,
    pub client: Client,
    pub table: String,
    pub clock: Arc<ManualClock>,
}

impl Fixture {
    /// Creates a fresh table. The clock starts at the real time, so the TTL
    /// that dynamodb-local may enforce is never in the past.
    pub async fn new(endpoint: &Url) -> Self {
        let client = test_client(endpoint);
        let table = fresh_table_name();
        create_table(&client, &table).await;
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        Self {
            endpoint: endpoint.clone(),
            client,
            table,
            clock: Arc::new(ManualClock::new(u64::try_from(now_ms).unwrap())),
        }
    }

    /// A store over the table, on the fixture's clock, with the standard lease.
    pub fn store(&self) -> Arc<DynamoDbPoolStore> {
        self.store_with_lease(lease())
    }

    /// Another store over the same table: another replica of the service.
    pub fn store_with_lease(&self, lease: Lease) -> Arc<DynamoDbPoolStore> {
        DynamoDbPoolStore::with_client(
            lease,
            config(&self.table, &self.endpoint),
            self.client.clone(),
            self.clock.clone(),
        )
    }

    /// How the conformance cases move this fixture's clock. This is the one
    /// place that calls `ManualClock::advance`.
    pub fn clock_handle(&self) -> ClockHandle {
        let clock = Arc::clone(&self.clock);
        ClockHandle::Manual(Arc::new(move |by| clock.advance(by)))
    }

    /// Advances the fixture's clock.
    pub fn advance(&self, by: Duration) {
        self.clock.advance(by);
    }

    /// The fixture's clock now, in unix milliseconds.
    pub fn now_ms(&self) -> u64 {
        use dynamo_global_discovery_service::Clock;
        self.clock.now_ms()
    }
}
