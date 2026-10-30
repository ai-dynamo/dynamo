// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The DynamoDB backend: registrations in one table that every replica
//! shares, in one region or, through a Global Table, in several. Design:
//! `lib/global-discovery-service/docs/dynamodb-store.md`.
//!
//! Each pool is one item ([`keys`]) holding its registration, when it was
//! last renewed, and the lease its writer granted ([`codec`]). A reader
//! decides on its own clock which pools are live, so the store never expires
//! anything; DynamoDB's TTL removes items nobody has renewed for a week. Every
//! replica reads the same items, so lists are `Exact` from the start. Router
//! replicas still come from configuration.

pub mod codec;
pub mod keys;
pub mod table;

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use crate::pool_discovery::{
    Completeness, Directory, DiscoveryError, Lease, PoolEvent, PoolListing, PoolRegistrar,
    PoolRegistration, Snapshot, WatchEvent, WatchStream,
};
use anyhow::bail;
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_dynamodb::Client;
use aws_sdk_dynamodb::config::Region;
use aws_sdk_dynamodb::config::timeout::TimeoutConfig;
use aws_sdk_dynamodb::types::AttributeValue;
use dynamo_global_common::PoolKey;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use url::Url;

use super::clock::{Clock, SystemClock};
use super::memory::{listing, sorted};
use super::stream::{EVENT_BUFFER, watch_stream};
use codec::Encoded;

/// How long the client waits to connect to DynamoDB.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
/// How long one try of a request may take. A table that stops answering
/// shows up as errors, not as requests that never end.
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a request may take with the SDK's retries.
const OPERATION_TIMEOUT: Duration = Duration::from_secs(15);
/// The shortest poll interval, so a zero interval does not poll back to back.
const MIN_POLL_INTERVAL: Duration = Duration::from_millis(10);
/// The longest wait between polls while they fail.
const MAX_POLL_BACKOFF: Duration = Duration::from_secs(30);

/// Where the table is, and how the store reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynamoDbConfig {
    pub table: String,
    /// The region of the table replica to use. `None` takes the SDK's
    /// (`AWS_REGION`, then the shared config).
    pub region: Option<String>,
    /// Another endpoint than the region's, for example dynamodb-local.
    pub endpoint: Option<Url>,
    /// How often a watch reads the table again. At least 10 ms.
    pub poll_interval: Duration,
    /// Clock skew between replicas that a lease allows.
    pub lease_skew: Duration,
    /// Create the table in [`DynamoDbPoolStore::connect`] if it does not
    /// exist.
    pub ensure_table: bool,
}

impl DynamoDbConfig {
    /// `table`, with the service's defaults: a 2 s poll, 2 s of skew, and no
    /// table creation.
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            region: None,
            endpoint: None,
            poll_interval: Duration::from_secs(2),
            lease_skew: Duration::from_secs(2),
            ensure_table: false,
        }
    }
}

/// Pool registrations in a DynamoDB table. Any number of stores, in any
/// number of processes and regions, may share it.
///
/// Any caller may register any pool: V1 assumes a trusted network, and
/// several PoolRelay replicas may register the same pool. The last write
/// wins.
pub struct DynamoDbPoolStore {
    lease: Lease,
    poll_interval: Duration,
    shared: Arc<Shared>,
    polling: OnceLock<()>,
    stop: CancellationToken,
}

/// What the store and its poller share. The poller holds this, not the
/// store, so dropping the store stops it.
struct Shared {
    client: Client,
    table: String,
    clock: Arc<dyn Clock>,
    lease_skew: Duration,
    watched: Mutex<Watched>,
    /// One refresh at a time, so each applies a newer read than the last.
    refreshing: tokio::sync::Mutex<()>,
    /// Sent while holding the `watched` lock, so a subscriber that snapshots
    /// and subscribes under it misses no event and sees none twice.
    events: broadcast::Sender<PoolEvent>,
}

/// The pools this replica's watchers know: the live ones as of the last
/// poll, and the ones this replica registered since.
#[derive(Debug, Default)]
struct Watched {
    /// `None` until the first poll.
    pools: Option<HashMap<PoolKey, Seen>>,
    /// Counts this replica's registrations, so a poll can tell the ones it
    /// may have read too early to see.
    writes: u64,
}

#[derive(Clone, Debug)]
struct Seen {
    listing: PoolListing,
    hash: String,
    /// `writes` when this replica last registered the pool, or 0 if a poll
    /// set it.
    written: u64,
}

impl Watched {
    fn snapshot(&self) -> Option<Snapshot<PoolListing>> {
        let pools = self.pools.as_ref()?;
        Some(Snapshot {
            items: pools.values().map(|seen| seen.listing.clone()).collect(),
            completeness: Completeness::Exact,
        })
    }

    /// Replaces the pools with the live ones a poll read, and returns the
    /// events that turn one view into the other. `began` is `writes` when
    /// the poll began: this replica's registrations since may be missing from
    /// what it read, so they stay until the next poll.
    fn apply_poll(&mut self, mut live: HashMap<PoolKey, Seen>, began: u64) -> Vec<PoolEvent> {
        let before = self.pools.take().unwrap_or_default();
        for (key, seen) in &before {
            if seen.written > began {
                live.insert(key.clone(), seen.clone());
            }
        }
        let mut events: Vec<PoolEvent> = before
            .keys()
            .filter(|key| !live.contains_key(*key))
            .map(|key| WatchEvent::Removed(key.clone()))
            .collect();
        events.extend(
            live.iter()
                .filter(|(key, seen)| before.get(*key).is_none_or(|old| old.hash != seen.hash))
                .map(|(_, seen)| WatchEvent::Upserted(seen.listing.clone())),
        );
        self.pools = Some(live);
        events
    }

    /// Records a registration this replica stored. Returns the `Upserted`
    /// that watchers need, unless they already have the pool as it is.
    fn apply_write(&mut self, listing: PoolListing, hash: String) -> Option<PoolEvent> {
        self.writes += 1;
        let pools = self.pools.as_mut()?;
        let changed = pools.get(&listing.key).is_none_or(|seen| seen.hash != hash);
        let event = changed.then(|| WatchEvent::Upserted(listing.clone()));
        pools.insert(
            listing.key.clone(),
            Seen {
                listing,
                hash,
                written: self.writes,
            },
        );
        event
    }
}

impl DynamoDbPoolStore {
    /// Builds the SDK client from the default chain plus `config`, optionally
    /// creates the table, and checks it with DescribeTable.
    pub async fn connect(lease: Lease, config: DynamoDbConfig) -> anyhow::Result<Arc<Self>> {
        let client = client(&config).await?;
        if config.ensure_table {
            table::ensure_table(&client, &config.table).await?;
        }
        table::describe(&client, &config.table).await?;
        Ok(Self::with_client(
            lease,
            config,
            client,
            Arc::new(SystemClock),
        ))
    }

    /// For tests: a store over an existing client and clock. It neither
    /// creates nor checks the table.
    pub fn with_client(
        lease: Lease,
        config: DynamoDbConfig,
        client: Client,
        clock: Arc<dyn Clock>,
    ) -> Arc<Self> {
        Arc::new(Self {
            lease,
            poll_interval: config.poll_interval.max(MIN_POLL_INTERVAL),
            shared: Arc::new(Shared {
                client,
                table: config.table,
                clock,
                lease_skew: config.lease_skew,
                watched: Mutex::default(),
                refreshing: tokio::sync::Mutex::new(()),
                events: broadcast::channel(EVENT_BUFFER).0,
            }),
            polling: OnceLock::new(),
            stop: CancellationToken::new(),
        })
    }

    /// Every trait method runs in a Tokio runtime, so the spawn is safe.
    fn ensure_polling(&self) {
        self.polling.get_or_init(|| {
            tokio::spawn(poll(
                Arc::clone(&self.shared),
                self.poll_interval,
                self.stop.clone(),
            ));
        });
    }
}

/// The SDK's default chain for credentials and region, with `config`'s
/// region and endpoint, and timeouts.
async fn client(config: &DynamoDbConfig) -> anyhow::Result<Client> {
    let timeouts = TimeoutConfig::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .operation_attempt_timeout(ATTEMPT_TIMEOUT)
        .operation_timeout(OPERATION_TIMEOUT)
        .build();
    let mut loader = aws_config::defaults(BehaviorVersion::latest()).timeout_config(timeouts);
    if let Some(region) = &config.region {
        loader = loader.region(Region::new(region.clone()));
    }
    let sdk = loader.load().await;
    if sdk.region().is_none() {
        bail!(
            "no AWS region for DynamoDB table {}: set --dynamodb-region or AWS_REGION",
            config.table
        );
    }
    let mut builder = aws_sdk_dynamodb::config::Builder::from(&sdk);
    if let Some(endpoint) = &config.endpoint {
        builder = builder.endpoint_url(endpoint.as_str().trim_end_matches('/'));
    }
    Ok(Client::from_conf(builder.build()))
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Watched> {
        self.watched.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn unavailable<E>(&self, doing: &str, error: E) -> DiscoveryError
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        DiscoveryError::Unavailable(
            anyhow::Error::new(error).context(format!("{doing} DynamoDB table {}", self.table)),
        )
    }

    /// Stores a registration. The whole item is last writer wins.
    async fn write(
        &self,
        encoded: &Encoded,
        now_ms: u64,
        lease_s: u64,
    ) -> Result<(), DiscoveryError> {
        let update = codec::update(encoded, now_ms, lease_s);
        self.client
            .update_item()
            .table_name(&self.table)
            .key(keys::PK, AttributeValue::S(keys::POOLS.to_owned()))
            .key(keys::SK, AttributeValue::S(encoded.sort_key.clone()))
            .update_expression(update.expression)
            .set_expression_attribute_names(Some(update.names))
            .set_expression_attribute_values(Some(update.values))
            .send()
            .await
            .map_err(|error| self.unavailable("write to", error))?;
        Ok(())
    }

    /// Every pool's item, live or not, read page by page. An item that does
    /// not decode is skipped, so one bad write cannot hide every pool.
    async fn query(&self) -> Result<Vec<codec::Stored>, DiscoveryError> {
        let mut stored = Vec::new();
        let mut start = None;
        loop {
            let page = self
                .client
                .query()
                .table_name(&self.table)
                .key_condition_expression("#pk = :pk")
                .expression_attribute_names("#pk", keys::PK)
                .expression_attribute_values(":pk", AttributeValue::S(keys::POOLS.to_owned()))
                .set_exclusive_start_key(start)
                .send()
                .await
                .map_err(|error| self.unavailable("query", error))?;
            for item in page.items() {
                match codec::decode(item) {
                    Ok(item) => stored.push(item),
                    Err(error) => tracing::warn!(
                        table = %self.table,
                        error = %format!("{error:#}"),
                        "skipping a pool item that does not decode"
                    ),
                }
            }
            match page.last_evaluated_key {
                Some(last) if !last.is_empty() => start = Some(last),
                _ => return Ok(stored),
            }
        }
    }

    /// The live pools, on this replica's clock.
    async fn live(&self) -> Result<Vec<codec::Stored>, DiscoveryError> {
        let mut stored = self.query().await?;
        let now_ms = self.clock.now_ms();
        stored.retain(|item| item.is_live(self.lease_skew, now_ms));
        Ok(stored)
    }

    /// Reads the table and tells watchers what changed since the last read.
    async fn refresh(&self) -> Result<(), DiscoveryError> {
        let _one_at_a_time = self.refreshing.lock().await;
        let began = self.lock().writes;
        let live = self
            .live()
            .await?
            .into_iter()
            .map(|item| {
                let seen = Seen {
                    listing: listing(&item.registration),
                    hash: item.hash,
                    written: 0,
                };
                (item.registration.key, seen)
            })
            .collect();
        let mut watched = self.lock();
        for event in watched.apply_poll(live, began) {
            // No subscribers is not an error.
            let _ = self.events.send(event);
        }
        Ok(())
    }
}

/// Refreshes every `every` until `stop`. A failed refresh tells watchers
/// nothing, and the next waits twice as long, up to [`MAX_POLL_BACKOFF`].
async fn poll(shared: Arc<Shared>, every: Duration, stop: CancellationToken) {
    let mut failures: u32 = 0;
    loop {
        let refreshed = tokio::select! {
            biased;
            () = stop.cancelled() => return,
            refreshed = async {
                tokio::time::sleep(backoff(every, failures)).await;
                shared.refresh().await
            } => refreshed,
        };
        match refreshed {
            Ok(()) if failures > 0 => {
                tracing::info!(table = %shared.table, failures, "reading the DynamoDB table works again");
                failures = 0;
            }
            Ok(()) => {}
            Err(error) => {
                failures = failures.saturating_add(1);
                if failures == 1 {
                    tracing::warn!(?error, table = %shared.table, "cannot poll the DynamoDB table; retrying with backoff");
                } else {
                    tracing::debug!(?error, table = %shared.table, failures, "cannot poll the DynamoDB table");
                }
            }
        }
    }
}

/// `every`, doubled for each failure in a row, up to [`MAX_POLL_BACKOFF`] or
/// `every` if that is longer.
fn backoff(every: Duration, failures: u32) -> Duration {
    let factor = 1_u32.checked_shl(failures).unwrap_or(u32::MAX);
    every
        .saturating_mul(factor)
        .min(MAX_POLL_BACKOFF.max(every))
}

impl fmt::Debug for DynamoDbPoolStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamoDbPoolStore")
            .field("table", &self.shared.table)
            .field("lease", &self.lease)
            .field("poll_interval", &self.poll_interval)
            .field("lease_skew", &self.shared.lease_skew)
            .finish_non_exhaustive()
    }
}

impl Drop for DynamoDbPoolStore {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

#[async_trait]
impl PoolRegistrar for DynamoDbPoolStore {
    /// Stores the registration with this replica's time and lease. An
    /// unchanged pool only renews its lease. A pool that this replica's
    /// watchers do not have as it is now gets an `Upserted` at once; other
    /// replicas' watchers learn of it at their next poll. Lists in another
    /// order are not a change.
    async fn register(&self, registration: PoolRegistration) -> Result<Lease, DiscoveryError> {
        let registration = sorted(registration);
        let encoded = codec::encode(&registration).map_err(|error| {
            DiscoveryError::Unavailable(
                anyhow::Error::new(error).context("encode the registration"),
            )
        })?;
        let now_ms = self.shared.clock.now_ms();
        self.shared
            .write(&encoded, now_ms, self.lease.duration().as_secs())
            .await?;
        let mut watched = self.shared.lock();
        if let Some(event) = watched.apply_write(listing(&registration), encoded.hash) {
            // No subscribers is not an error.
            let _ = self.shared.events.send(event);
        }
        Ok(self.lease)
    }
}

#[async_trait]
impl Directory<PoolListing, PoolKey> for DynamoDbPoolStore {
    /// The live pools in the table, `Exact`. Fails while the table cannot be
    /// read.
    async fn list(&self) -> Result<Snapshot<PoolListing>, DiscoveryError> {
        let live = self.shared.live().await?;
        Ok(Snapshot {
            items: live
                .iter()
                .map(|item| listing(&item.registration))
                .collect(),
            completeness: Completeness::Exact,
        })
    }

    /// Reads the table, then follows it by polling every
    /// [`DynamoDbConfig::poll_interval`], and this replica's registrations at
    /// once. Fails if the first read does; later failed polls leave the
    /// stream open and quiet until a poll succeeds.
    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<PoolListing, PoolKey>, DiscoveryError> {
        self.shared.refresh().await?;
        self.ensure_polling();
        let shared = Arc::clone(&self.shared);
        Ok(watch_stream(
            move || {
                let watched = shared.lock();
                (shared.events.subscribe(), watched.snapshot())
            },
            cancel_token.unwrap_or_default(),
            self.stop.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::net::TcpListener;

    use aws_sdk_dynamodb::config::Credentials;
    use aws_sdk_dynamodb::config::retry::RetryConfig;
    use dynamo_global_common::PoolLocation;

    use super::*;
    use crate::backends::clock::ManualClock;

    fn key(dgd: &str) -> PoolKey {
        PoolKey::new("west", "dynamo", dgd).unwrap()
    }

    fn pool(dgd: &str, model: &str) -> PoolListing {
        PoolListing {
            key: key(dgd),
            location: PoolLocation {
                region: "us-west-2".to_owned(),
                availability_zone: None,
                datacenter: None,
            },
            model: Some(model.to_owned()),
            capabilities: Vec::new(),
            frontends: Vec::new(),
            capacity: Vec::new(),
            relay_identities: Vec::new(),
        }
    }

    /// A poll's view of `pools`, as `(dgd, model)`, with the model as hash.
    fn polled(pools: &[(&str, &str)]) -> HashMap<PoolKey, Seen> {
        pools
            .iter()
            .map(|(dgd, model)| {
                let seen = Seen {
                    listing: pool(dgd, model),
                    hash: (*model).to_owned(),
                    written: 0,
                };
                (key(dgd), seen)
            })
            .collect()
    }

    /// Events as `+dgd/model` and `-dgd`, in any order.
    fn described(events: &[PoolEvent]) -> HashSet<String> {
        events
            .iter()
            .map(|event| match event {
                WatchEvent::Upserted(listing) => format!(
                    "+{}/{}",
                    listing.key.dgd_name(),
                    listing.model.as_deref().unwrap_or_default()
                ),
                WatchEvent::Removed(key) => format!("-{}", key.dgd_name()),
                WatchEvent::Resync(_) => "resync".to_owned(),
            })
            .collect()
    }

    fn set(events: &[&str]) -> HashSet<String> {
        events.iter().map(|event| (*event).to_owned()).collect()
    }

    #[test]
    fn a_poll_publishes_what_changed_since_the_last() {
        let mut watched = Watched::default();
        assert!(watched.snapshot().is_none());
        let first = watched.apply_poll(polled(&[("a", "m1"), ("b", "m1"), ("c", "m1")]), 0);
        assert_eq!(described(&first), set(&["+a/m1", "+b/m1", "+c/m1"]));

        // `a` is renewed, `b` changes, `c` expires or goes away, `d` is new.
        let second = watched.apply_poll(polled(&[("a", "m1"), ("b", "m2"), ("d", "m1")]), 0);
        assert_eq!(described(&second), set(&["+b/m2", "-c", "+d/m1"]));
        let snapshot = watched.snapshot().unwrap();
        assert_eq!(snapshot.completeness, Completeness::Exact);
        let mut listed: Vec<String> = snapshot
            .items
            .iter()
            .map(|listing| listing.key.dgd_name().to_owned())
            .collect();
        listed.sort();
        assert_eq!(listed, ["a", "b", "d"]);

        // The same view again publishes nothing.
        let third = watched.apply_poll(polled(&[("a", "m1"), ("b", "m2"), ("d", "m1")]), 0);
        assert!(third.is_empty());
    }

    #[test]
    fn a_registration_is_published_unless_watchers_have_it() {
        let mut watched = Watched::default();
        // Before the first poll nobody watches, so nothing is published.
        assert!(
            watched
                .apply_write(pool("a", "m1"), "m1".to_owned())
                .is_none()
        );
        let _ = watched.apply_poll(polled(&[("a", "m1")]), watched.writes);

        assert!(
            watched
                .apply_write(pool("a", "m1"), "m1".to_owned())
                .is_none()
        );
        let changed = watched.apply_write(pool("a", "m2"), "m2".to_owned());
        assert_eq!(described(&Vec::from_iter(changed)), set(&["+a/m2"]));
        let added = watched.apply_write(pool("b", "m1"), "m1".to_owned());
        assert_eq!(described(&Vec::from_iter(added)), set(&["+b/m1"]));
    }

    #[test]
    fn a_poll_does_not_undo_a_registration_it_may_have_missed() {
        let mut watched = Watched::default();
        let _ = watched.apply_poll(polled(&[("a", "m1")]), 0);
        // A poll begins, then this replica registers a change and a new pool,
        // which the poll's read may predate.
        let began = watched.writes;
        let _ = watched.apply_write(pool("a", "m2"), "m2".to_owned());
        let _ = watched.apply_write(pool("b", "m1"), "m1".to_owned());
        let stale = watched.apply_poll(polled(&[("a", "m1")]), began);
        assert!(stale.is_empty(), "{:?}", described(&stale));

        // The next poll began after them, so it decides, whatever it reads:
        // here another replica changed `a` again, and `b` expired.
        let next = watched.apply_poll(polled(&[("a", "m3")]), watched.writes);
        assert_eq!(described(&next), set(&["+a/m3", "-b"]));
    }

    #[test]
    fn failed_polls_back_off() {
        let every = Duration::from_secs(2);
        assert_eq!(backoff(every, 0), every);
        assert_eq!(backoff(every, 1), Duration::from_secs(4));
        assert_eq!(backoff(every, 3), Duration::from_secs(16));
        assert_eq!(backoff(every, 4), MAX_POLL_BACKOFF);
        assert_eq!(backoff(every, u32::MAX), MAX_POLL_BACKOFF);
        // An interval above the cap is kept.
        let long = Duration::from_secs(60);
        assert_eq!(backoff(long, 5), long);
    }

    /// A store whose client talks to a closed local port, with no retries.
    fn unreachable_store() -> Arc<DynamoDbPoolStore> {
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let config = aws_sdk_dynamodb::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new("us-west-2"))
            .endpoint_url(format!("http://127.0.0.1:{port}"))
            .credentials_provider(Credentials::new("test", "test", None, None, "test"))
            .retry_config(RetryConfig::disabled())
            .build();
        DynamoDbPoolStore::with_client(
            Lease::new(5, 15).unwrap(),
            DynamoDbConfig::new("pools"),
            Client::from_conf(config),
            Arc::new(ManualClock::new(1_000)),
        )
    }

    #[tokio::test]
    async fn an_unreachable_table_is_unavailable() {
        let store = unreachable_store();
        let DiscoveryError::Unavailable(error) = store.list().await.unwrap_err();
        assert!(
            format!("{error:#}").contains("query DynamoDB table pools"),
            "{error:#}"
        );
        let registration = PoolRegistration {
            key: key("a"),
            location: pool("a", "m1").location,
            model: "m1".to_owned(),
            capabilities: Vec::new(),
            frontends: Vec::new(),
            capacity: Vec::new(),
        };
        let DiscoveryError::Unavailable(error) = store.register(registration).await.unwrap_err();
        assert!(
            format!("{error:#}").contains("write to DynamoDB table pools"),
            "{error:#}"
        );
        assert!(store.list_and_watch(None).await.is_err());
        assert!(format!("{store:?}").contains("pools"));
    }
}
