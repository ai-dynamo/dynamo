// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leased router membership on the same trusted network as pool registration.
//! An incarnation keeps one endpoint and writes through one DynamoDB region.
//! Global Tables do not provide cross-region conditional-write fencing: IDs
//! must be unique across incarnations and regions.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use aws_sdk_dynamodb::{Client, types::AttributeValue};
use dynamo_global_common::{RouterRegistration, RouterReplica, RouterReplicaId};
use tokio_util::sync::CancellationToken;

use super::clock::{Clock, SystemClock, millis};
use super::dynamodb::DynamoDbConfig;
use crate::pool_discovery::{
    Completeness, Directory, DiscoveryError, Lease, RouterRegistrar, Snapshot, WatchEvent,
    WatchStream,
};

const PARTITION: &str = "ROUTERS";
const RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

#[derive(Clone)]
struct Stored {
    registration: RouterRegistration,
    renewed_ms: u64,
    lease_ms: u64,
    region: String,
}

#[derive(Clone)]
enum Storage {
    Memory(Arc<Mutex<HashMap<String, Stored>>>),
    DynamoDb {
        client: Client,
        table: String,
        region: String,
    },
}

/// A registration directory. HTTP callers read a fresh, paginated snapshot;
/// a failed page fails the whole read, never publishing an empty success.
#[derive(Clone)]
pub struct RouterStore {
    storage: Storage,
    lease: Lease,
    clock: Arc<dyn Clock>,
    skew: Duration,
    grace: Duration,
    poll_interval: Duration,
    completeness: Arc<Mutex<CompletenessState>>,
}

struct CompletenessState {
    complete_at: u64,
    failed: bool,
}

impl RouterStore {
    pub fn memory(lease: Lease) -> Self {
        Self::memory_with_clock(lease, Arc::new(SystemClock))
    }

    pub fn memory_with_clock(lease: Lease, clock: Arc<dyn Clock>) -> Self {
        Self::new(
            Storage::Memory(Arc::default()),
            lease,
            clock,
            Duration::ZERO,
            Duration::ZERO,
            lease.heartbeat_interval(),
        )
    }

    /// Shares a checked pool store's client, table and lease configuration.
    pub fn dynamodb(
        lease: Lease,
        config: &DynamoDbConfig,
        client: Client,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let region = client
            .config()
            .region()
            .map(ToString::to_string)
            .unwrap_or_default();
        Self::new(
            Storage::DynamoDb {
                client,
                table: config.table.clone(),
                region,
            },
            lease,
            clock,
            config.lease_skew,
            config.replication_grace,
            config.poll_interval,
        )
    }

    fn new(
        storage: Storage,
        lease: Lease,
        clock: Arc<dyn Clock>,
        skew: Duration,
        grace: Duration,
        poll_interval: Duration,
    ) -> Self {
        let complete_at = clock.now_ms().saturating_add(millis(lease.duration()));
        Self {
            storage,
            lease,
            clock,
            skew,
            grace,
            poll_interval: poll_interval.max(Duration::from_millis(10)),
            completeness: Arc::new(Mutex::new(CompletenessState {
                complete_at,
                failed: false,
            })),
        }
    }

    fn uncertain(&self) {
        self.completeness
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .failed = true;
    }

    fn is_complete(&self, read_started: u64) -> bool {
        let mut state = self
            .completeness
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if state.failed {
            state.failed = false;
            state.complete_at = self
                .clock
                .now_ms()
                .saturating_add(millis(self.lease.duration()));
        }
        read_started >= state.complete_at
    }

    async fn read(&self) -> Result<Vec<Stored>, DiscoveryError> {
        let (client, table) = match &self.storage {
            Storage::DynamoDb { client, table, .. } => (client, table),
            Storage::Memory(rows) => {
                let now = self.clock.now_ms();
                let mut rows = rows.lock().unwrap_or_else(PoisonError::into_inner);
                rows.retain(|_, row| row.renewed_ms.saturating_add(millis(RETENTION)) >= now);
                return Ok(rows.values().cloned().collect());
            }
        };
        let mut rows = Vec::new();
        let mut after = None;
        loop {
            let page = client
                .query()
                .table_name(table)
                .consistent_read(true)
                .key_condition_expression("pk = :pk")
                .expression_attribute_values(":pk", AttributeValue::S(PARTITION.into()))
                .set_exclusive_start_key(after)
                .send()
                .await
                .map_err(|e| DiscoveryError::Unavailable(e.into()))?;
            for item in page.items() {
                rows.push(decode(item).map_err(DiscoveryError::Unavailable)?);
            }
            after = page.last_evaluated_key;
            if after.as_ref().is_none_or(HashMap::is_empty) {
                break;
            }
        }
        Ok(rows)
    }
}

#[async_trait]
impl RouterRegistrar for RouterStore {
    async fn register_router(
        &self,
        registration: RouterRegistration,
    ) -> Result<Lease, DiscoveryError> {
        registration
            .validate()
            .map_err(|e| DiscoveryError::Rejected(e.to_string()))?;
        let now = self.clock.now_ms();
        let lease_ms = millis(self.lease.duration());
        match &self.storage {
            Storage::Memory(rows) => {
                let mut rows = rows.lock().unwrap_or_else(PoisonError::into_inner);
                if rows
                    .get(&registration.replica)
                    .is_some_and(|row| row.registration.url != registration.url)
                {
                    return Err(DiscoveryError::Rejected(
                        "replica ID already belongs to another endpoint; use a new incarnation ID"
                            .into(),
                    ));
                }
                rows.insert(
                    registration.replica.clone(),
                    Stored {
                        registration,
                        renewed_ms: now,
                        lease_ms,
                        region: String::new(),
                    },
                );
            }
            Storage::DynamoDb {
                client,
                table,
                region,
            } => {
                let result = client.update_item().table_name(table)
                    .key("pk", AttributeValue::S(PARTITION.into()))
                    .key("sk", AttributeValue::S(registration.replica.clone()))
                    .update_expression("SET #url = :url, renewed_at_ms = :now, lease_ms = :lease, writer_region = :region, expires_at = :ttl")
                    .condition_expression("attribute_not_exists(pk) OR (#url = :url AND writer_region = :region AND renewed_at_ms <= :now)")
                    .expression_attribute_names("#url", "url")
                    .expression_attribute_values(":url", AttributeValue::S(registration.url.to_string()))
                    .expression_attribute_values(":now", AttributeValue::N(now.to_string()))
                    .expression_attribute_values(":lease", AttributeValue::N(lease_ms.to_string()))
                    .expression_attribute_values(":region", AttributeValue::S(region.clone()))
                    .expression_attribute_values(":ttl", AttributeValue::N((now.saturating_add(millis(RETENTION)) / 1000).to_string()))
                    .send().await;
                if let Err(error) = result {
                    if error
                        .as_service_error()
                        .is_some_and(|e| e.is_conditional_check_failed_exception())
                    {
                        return Err(DiscoveryError::Rejected("replica ID has another endpoint or writer region, or the heartbeat clock moved backwards".into()));
                    }
                    self.uncertain();
                    return Err(DiscoveryError::Unavailable(error.into()));
                }
            }
        }
        Ok(self.lease)
    }
}

#[async_trait]
impl Directory<RouterReplica, RouterReplicaId> for RouterStore {
    async fn list(&self) -> Result<Snapshot<RouterReplica>, DiscoveryError> {
        let read_started = self.clock.now_ms();
        let rows = match self.read().await {
            Ok(rows) => rows,
            Err(error) => {
                self.uncertain();
                return Err(error);
            }
        };
        let now = read_started;
        let mut complete = self.is_complete(read_started);
        let region = match &self.storage {
            Storage::DynamoDb { region, .. } => region.as_str(),
            _ => "",
        };
        let mut items = Vec::new();
        for row in rows {
            let expires = row
                .renewed_ms
                .saturating_add(row.lease_ms)
                .saturating_add(millis(self.skew));
            let grace = if row.region != region {
                millis(self.grace)
            } else {
                0
            };
            if now <= expires.saturating_add(grace) {
                complete &= now <= expires;
                items.push(RouterReplica {
                    replica: RouterReplicaId(row.registration.replica),
                    url: row.registration.url,
                });
            }
        }
        items.sort_by(|a, b| a.replica.0.cmp(&b.replica.0));
        Ok(Snapshot {
            items,
            completeness: if complete {
                Completeness::Exact
            } else {
                Completeness::Estimated
            },
        })
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<RouterReplica, RouterReplicaId>, DiscoveryError> {
        let store = self.clone();
        let cancel = cancel_token.unwrap_or_default();
        Ok(Box::pin(async_stream::stream! {
            let mut first = true;
            loop {
                let Some(snapshot) = cancel.run_until_cancelled(store.list()).await else { break; };
                if first && let Ok(snapshot) = &snapshot {
                    for item in &snapshot.items {
                        yield Ok(WatchEvent::Upserted(item.clone()));
                    }
                    first = false;
                }
                yield snapshot.map(WatchEvent::Resync);
                if cancel.run_until_cancelled(tokio::time::sleep(store.poll_interval)).await.is_none() { break; }
            }
        }))
    }
}

fn decode(item: &HashMap<String, AttributeValue>) -> anyhow::Result<Stored> {
    let string = |key: &str| {
        item.get(key)
            .and_then(|v| v.as_s().ok())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("router item missing string {key}"))
    };
    let number = |key: &str| -> anyhow::Result<u64> {
        Ok(item
            .get(key)
            .and_then(|v| v.as_n().ok())
            .ok_or_else(|| anyhow::anyhow!("router item missing number {key}"))?
            .parse()?)
    };
    let registration = RouterRegistration {
        replica: string("sk")?,
        url: string("url")?.parse()?,
    };
    registration.validate()?;
    Ok(Stored {
        registration,
        renewed_ms: number("renewed_at_ms")?,
        lease_ms: number("lease_ms")?,
        region: string("writer_region")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::clock::ManualClock;

    #[test]
    fn slow_recovery_read_waits_a_full_lease_after_completion() {
        let clock = Arc::new(ManualClock::new(1000));
        let store = RouterStore::memory_with_clock(Lease::new(1, 3).unwrap(), clock.clone());
        store.uncertain();
        clock.advance(Duration::from_secs(100));
        let began = clock.now_ms();
        clock.advance(Duration::from_secs(10));
        assert!(!store.is_complete(began));
        assert!(!store.is_complete(clock.now_ms()));
        clock.advance(Duration::from_secs(3));
        assert!(store.is_complete(clock.now_ms()));
    }
}
