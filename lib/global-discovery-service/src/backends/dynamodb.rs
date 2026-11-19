// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The DynamoDB backend: registrations in one table that every replica
//! shares, in one region or, through a Global Table, in several. Design:
//! `lib/global-discovery-service/docs/dynamodb-store.md`.
//!
//! Each pool is one item ([`keys`]) holding its registration, when it was
//! last renewed, and the lease its writer granted ([`codec`]). A reader
//! decides on its own clock which pools are live, so the store never expires
//! anything; DynamoDB's TTL removes items nobody has renewed for a week.
//!
//! Every replica reads the same items, so a new replica lists every pool at
//! once. Its lists are still `Estimated` for one lease after it starts, as
//! with the memory store: while a rollout moves the service to this table,
//! from memory or from another table, PoolRelays that still register with the
//! old replicas are missing from it. After a failed call to the table, lists
//! are `Estimated` until a lease has passed since reads recovered: no
//! PoolRelay could renew meanwhile, so items may have lapsed. While polls
//! fail, watchers are told the view is `Estimated`, and pools drop out of it
//! as their leases lapse on this replica's clock.
//!
//! With a Global Table, another region's writes arrive late when replication
//! lags. Each item records the region it was written through, and a pool
//! written through another region stays listed for
//! [`DynamoDbConfig::replication_grace`] past its lease, while lists are
//! `Estimated`, rather than being dropped as expired while it still
//! heartbeats there.
//!
//! The store keeps pool registrations for every discovery mode. `router_store()`
//! shares this table and client with the separately partitioned router directory.

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
use thiserror::Error;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use url::Url;

use super::clock::{Clock, SystemClock, millis};
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
    /// With a Global Table, how long past its lease and the skew a pool
    /// stays listed when its last registration was written through another
    /// region's replica: that region's later renewals may still be on their
    /// way. Lists are `Estimated` while such a pool is listed. Zero drops it
    /// when its lease lapses.
    pub replication_grace: Duration,
    /// Create the table in [`DynamoDbPoolStore::connect`] if it does not
    /// exist.
    pub ensure_table: bool,
}

impl DynamoDbConfig {
    /// `table`, with the service's defaults: a 2 s poll, 2 s of skew, 60 s
    /// of replication grace, and no table creation.
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            region: None,
            endpoint: None,
            poll_interval: Duration::from_secs(2),
            lease_skew: Duration::from_secs(2),
            replication_grace: Duration::from_secs(60),
            ensure_table: false,
        }
    }
}

/// Pool registrations in a DynamoDB table. Any number of stores, in any
/// number of processes and regions, may share it.
///
/// Any caller may register any pool: V1 assumes a trusted network, and
/// several PoolRelay replicas may register the same pool. The last write
/// wins, so they must send the same registration, lists in any order. The
/// store warns about a pool whose registration keeps switching between two
/// versions.
pub struct DynamoDbPoolStore {
    lease: Lease,
    poll_interval: Duration,
    shared: Arc<Shared>,
    polling: OnceLock<()>,
    stop: CancellationToken,
}

/// What a read of the table lists.
struct Read {
    /// Each pool, with the last Unix millisecond at which it is listed
    /// unless renewed.
    pools: Vec<(codec::Stored, u64)>,
    /// A pool past its lease is listed because another region's renewal may
    /// be late, so the list may be wrong.
    on_trust: bool,
}

/// What the store and its poller share. The poller holds this, not the
/// store, so dropping the store stops it.
struct Shared {
    client: Client,
    table: String,
    /// The region of the table replica this store reads and writes, which it
    /// records in each registration. `None` if the client has none.
    region: Option<String>,
    clock: Arc<dyn Clock>,
    lease_skew: Duration,
    /// [`DynamoDbConfig::replication_grace`].
    replication_grace: Duration,
    /// The lease the store grants, in milliseconds. Lists are `Estimated` for
    /// this long after the store was made, and after reads recover.
    lease_ms: u64,
    watched: Mutex<Watched>,
    /// One refresh at a time, so each applies a newer read than the last.
    refreshing: tokio::sync::Mutex<()>,
    /// Sent while holding the `watched` lock, so a subscriber that snapshots
    /// and subscribes under it misses no event and sees none twice.
    events: broadcast::Sender<PoolEvent>,
}

/// The pools this replica's watchers know: the live ones as of the last
/// poll, and the ones this replica registered since. Also whether this
/// replica's lists are complete, which `list` shares.
#[derive(Debug, Default)]
struct Watched {
    /// `None` until the first poll.
    pools: Option<HashMap<PoolKey, Seen>>,
    /// Counts this replica's registrations, so a poll can tell the ones it
    /// may have read too early to see.
    writes: u64,
    /// Watchers have been told that the view is complete.
    exact: bool,
    /// While reads of the table succeed, lists are `Exact` from this time
    /// on, in Unix milliseconds on the store's clock: one lease after the
    /// store was made, or after reads recovered from a failed call.
    complete_at_ms: u64,
    /// A call to the table failed, and no read has succeeded since.
    failing: bool,
    /// Pools whose registration switches back and forth.
    flaps: Flaps,
}

/// Changes in a row back to the version before that make a pool's
/// registration flap. One is a rollback, as of a model changed and changed
/// back.
const FLAPPING_AFTER: u32 = 2;

/// Notices a pool whose registration keeps switching between two versions,
/// as when its PoolRelays send different registrations and each write
/// replaces the other's. Watchers then see the pool change at almost every
/// poll.
#[derive(Debug, Default)]
struct Flaps {
    pools: HashMap<PoolKey, Flap>,
}

#[derive(Debug)]
struct Flap {
    /// The pool's version, by hash, before its last change.
    before: String,
    /// Changes in a row that went back to the version before.
    returns: u32,
    /// The warning has been logged.
    warned: bool,
}

impl Flaps {
    /// Records that `key` changed from version `from` to `to`. True once per
    /// pool, when its changes have gone back to the version before
    /// [`FLAPPING_AFTER`] times in a row.
    fn changed(&mut self, key: &PoolKey, from: &str, to: &str) -> bool {
        let flap = self.pools.entry(key.clone()).or_insert_with(|| Flap {
            before: String::new(),
            returns: 0,
            warned: false,
        });
        flap.returns = if flap.before == to {
            flap.returns.saturating_add(1)
        } else {
            0
        };
        from.clone_into(&mut flap.before);
        if flap.returns < FLAPPING_AFTER || flap.warned {
            return false;
        }
        flap.warned = true;
        true
    }

    /// Forgets a pool that is gone, so it starts over if it comes back.
    fn forget(&mut self, key: &PoolKey) {
        self.pools.remove(key);
    }
}

#[derive(Clone, Debug)]
struct Seen {
    listing: PoolListing,
    hash: String,
    /// `writes` when this replica last registered the pool, or 0 if a poll
    /// set it.
    written: u64,
    /// The last Unix millisecond at which the view lists the pool unless it
    /// is renewed: its renewal plus its lease and the skew, and for a pool
    /// written through another region, the replication grace too.
    live_until_ms: u64,
    /// The last Unix millisecond its own lease and the skew cover. Earlier
    /// than `live_until_ms` only for a pool from another region, which is
    /// listed on trust in between.
    lease_until_ms: u64,
}

impl Watched {
    fn snapshot(&self) -> Option<Snapshot<PoolListing>> {
        let pools = self.pools.as_ref()?;
        Some(Snapshot {
            items: pools.values().map(|seen| seen.listing.clone()).collect(),
            completeness: if self.exact {
                Completeness::Exact
            } else {
                Completeness::Estimated
            },
        })
    }

    /// What `list` serves from the view at `now_ms`; `None` before the first
    /// poll. A pool whose lease has lapsed since the last poll is left out
    /// at once, as a read of the table would. The list is `Exact` only while
    /// watchers were told so, no call to the table has failed since, no pool
    /// is listed past its own lease, and no pool has lapsed since the last
    /// poll, which may have missed its renewal.
    fn listed(&self, now_ms: u64) -> Option<Snapshot<PoolListing>> {
        let pools = self.pools.as_ref()?;
        let mut on_trust = false;
        let items = pools
            .values()
            .filter(|seen| seen.live_until_ms >= now_ms)
            .map(|seen| {
                on_trust |= seen.lease_until_ms < now_ms;
                seen.listing.clone()
            })
            .collect();
        // Renewals through other replicas show only at the next poll, so a
        // pool that lapsed since the last one may still be live.
        let lapsed = pools.values().any(|seen| seen.live_until_ms < now_ms);
        let exact = self.exact && !self.failing && !on_trust && !lapsed;
        Some(Snapshot {
            items,
            completeness: if exact {
                Completeness::Exact
            } else {
                Completeness::Estimated
            },
        })
    }

    /// Records a call to the table that failed.
    fn failed(&mut self) {
        self.failing = true;
    }

    /// Records a read of the table that began at `read_at_ms` and succeeded,
    /// and returns how complete a list from it is. After a failed call, lists
    /// stay `Estimated` for one more lease: PoolRelays could not renew while
    /// the table could not be reached, so their items may have lapsed until
    /// they register again.
    fn read(&mut self, read_at_ms: u64, lease_ms: u64) -> Completeness {
        if self.failing {
            self.failing = false;
            self.complete_at_ms = self.complete_at_ms.max(read_at_ms.saturating_add(lease_ms));
        }
        if read_at_ms >= self.complete_at_ms {
            Completeness::Exact
        } else {
            Completeness::Estimated
        }
    }

    /// Sets whether the view is complete. On a change, once there is a view,
    /// returns the `Resync` that tells watchers.
    fn set_completeness(&mut self, completeness: Completeness) -> Option<PoolEvent> {
        let exact = completeness == Completeness::Exact;
        if self.pools.is_none() || self.exact == exact {
            return None;
        }
        self.exact = exact;
        self.snapshot().map(WatchEvent::Resync)
    }

    /// Drops the pools whose leases have lapsed at `now_ms`, with a `Removed`
    /// for each. While the table cannot be read, no renewal shows, and the
    /// view would otherwise keep every pool for as long as reads fail.
    fn expire(&mut self, now_ms: u64) -> Vec<PoolEvent> {
        let Some(pools) = self.pools.as_mut() else {
            return Vec::new();
        };
        let mut removed = Vec::new();
        pools.retain(|key, seen| {
            let live = seen.live_until_ms >= now_ms;
            if !live {
                removed.push(WatchEvent::Removed(key.clone()));
            }
            live
        });
        removed
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
        let mut events = Vec::new();
        for key in before.keys().filter(|key| !live.contains_key(*key)) {
            self.flaps.forget(key);
            events.push(WatchEvent::Removed(key.clone()));
        }
        for (key, seen) in &live {
            match before.get(key) {
                Some(old) if old.hash == seen.hash => {}
                Some(old) => {
                    self.changed(key, &old.hash, &seen.hash);
                    events.push(WatchEvent::Upserted(seen.listing.clone()));
                }
                None => events.push(WatchEvent::Upserted(seen.listing.clone())),
            }
        }
        self.pools = Some(live);
        events
    }

    /// Records that a pool's registration changed, and warns, once, about a
    /// pool whose registration keeps switching between two versions.
    fn changed(&mut self, key: &PoolKey, from: &str, to: &str) {
        if self.flaps.changed(key, from, to) {
            tracing::warn!(
                ?key,
                "this pool's registration keeps switching between two versions, so watchers \
                 see it change at almost every poll. Its PoolRelays send different registrations, \
                 and each write replaces the other's: every PoolRelay of a pool must send the same \
                 model, capabilities, capacity, location and frontends."
            );
        }
    }

    /// Records a registration this replica stored, live until
    /// `live_until_ms`. Returns the `Upserted` that watchers need, unless they
    /// already have the pool as it is.
    fn apply_write(
        &mut self,
        listing: PoolListing,
        hash: String,
        live_until_ms: u64,
    ) -> Option<PoolEvent> {
        self.writes += 1;
        let before = self
            .pools
            .as_ref()?
            .get(&listing.key)
            .map(|seen| seen.hash.clone());
        if let Some(before) = &before
            && *before != hash
        {
            self.changed(&listing.key, before, &hash);
        }
        let changed = before.is_none_or(|before| before != hash);
        let event = changed.then(|| WatchEvent::Upserted(listing.clone()));
        let pools = self.pools.as_mut()?;
        pools.insert(
            listing.key.clone(),
            Seen {
                listing,
                hash,
                written: self.writes,
                live_until_ms,
                lease_until_ms: live_until_ms,
            },
        );
        event
    }
}

/// Why [`DynamoDbPoolStore::connect`] failed.
#[derive(Debug, Error)]
pub enum ConnectError {
    /// The configuration cannot work, as when it gives no AWS region. Trying
    /// again does not help.
    #[error(transparent)]
    Config(anyhow::Error),
    /// The table could not be checked or created: DynamoDB or STS did not
    /// answer, access was denied, or the table is missing or not usable.
    /// Trying again may help once that is fixed.
    #[error(transparent)]
    Table(anyhow::Error),
}

impl DynamoDbPoolStore {
    /// Builds the SDK client from the default chain plus `config`, optionally
    /// creates the table, checks it with DescribeTable, and checks that it
    /// may read and write it, so that a store that cannot is never served.
    pub async fn connect(lease: Lease, config: DynamoDbConfig) -> Result<Arc<Self>, ConnectError> {
        let client = client(&config).await.map_err(ConnectError::Config)?;
        if config.ensure_table {
            table::ensure_table(&client, &config.table)
                .await
                .map_err(ConnectError::Table)?;
        }
        table::describe(&client, &config.table, config.endpoint.as_ref())
            .await
            .map_err(ConnectError::Table)?;
        table::check_access(&client, &config.table)
            .await
            .map_err(ConnectError::Table)?;
        Ok(Self::with_client(
            lease,
            config,
            client,
            Arc::new(SystemClock),
        ))
    }

    /// For tests: a store over an existing client and clock. It neither
    /// creates nor checks the table. Its lists are `Estimated` until a lease
    /// has passed on `clock`.
    pub fn with_client(
        lease: Lease,
        config: DynamoDbConfig,
        client: Client,
        clock: Arc<dyn Clock>,
    ) -> Arc<Self> {
        let lease_ms = millis(lease.duration());
        let watched = Watched {
            complete_at_ms: clock.now_ms().saturating_add(lease_ms),
            ..Watched::default()
        };
        let region = client.config().region().map(ToString::to_string);
        Arc::new(Self {
            lease,
            poll_interval: config.poll_interval.max(MIN_POLL_INTERVAL),
            shared: Arc::new(Shared {
                client,
                table: config.table,
                region,
                clock,
                lease_skew: config.lease_skew,
                replication_grace: config.replication_grace,
                lease_ms,
                watched: Mutex::new(watched),
                refreshing: tokio::sync::Mutex::new(()),
                events: broadcast::channel(EVENT_BUFFER).0,
            }),
            polling: OnceLock::new(),
            stop: CancellationToken::new(),
        })
    }

    /// A router directory sharing this pool store's checked table and client.
    pub fn router_store(&self) -> super::routers::RouterStore {
        let config = DynamoDbConfig {
            table: self.shared.table.clone(),
            region: self.shared.region.clone(),
            endpoint: None,
            poll_interval: self.poll_interval,
            lease_skew: self.shared.lease_skew,
            replication_grace: self.shared.replication_grace,
            ensure_table: false,
        };
        super::routers::RouterStore::dynamodb(
            self.lease,
            &config,
            self.shared.client.clone(),
            self.shared.clock.clone(),
        )
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
        let update = codec::update(encoded, now_ms, lease_s, self.region.as_deref());
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
    ///
    /// The reads are strongly consistent, so a poll sees every write that
    /// this region's replica of the table took before the poll began. An
    /// eventually consistent read could return an item older than one this
    /// replica already has, and make watchers see the pool change back and
    /// forth. Other regions' writes still arrive by replication.
    async fn query(&self) -> Result<Vec<codec::Stored>, DiscoveryError> {
        let mut stored = Vec::new();
        let mut start = None;
        loop {
            let page = self
                .client
                .query()
                .table_name(&self.table)
                .consistent_read(true)
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

    /// The pools to list, on this replica's clock. A failed read is recorded,
    /// so that lists stay `Estimated` until a lease after reads recover.
    async fn live(&self) -> Result<Read, DiscoveryError> {
        let stored = match self.query().await {
            Ok(stored) => stored,
            Err(error) => {
                self.lock().failed();
                return Err(error);
            }
        };
        Ok(self.select(stored, self.clock.now_ms()))
    }

    /// The pools that items list at `now_ms`: those whose lease and skew
    /// have not run out, and, on trust, those written through another
    /// region's replica that are still within the replication grace.
    fn select(&self, stored: Vec<codec::Stored>, now_ms: u64) -> Read {
        let mut on_trust = false;
        let pools = stored
            .into_iter()
            .filter_map(|item| {
                let listed_until_ms = self.listed_until_ms(&item);
                if listed_until_ms < now_ms {
                    return None;
                }
                on_trust |= !item.is_live(self.lease_skew, now_ms);
                Some((item, listed_until_ms))
            })
            .collect();
        Read { pools, on_trust }
    }

    /// The last Unix millisecond at which `item`'s pool is listed unless it
    /// is renewed: when its lease and the skew run out, or, if it was
    /// written through another region's replica of a Global Table, the
    /// replication grace later, since that region's renewals may be late.
    fn listed_until_ms(&self, item: &codec::Stored) -> u64 {
        let live_until_ms = item.live_until_ms(self.lease_skew);
        let from_another_region = matches!(
            (&item.written_in, &self.region),
            (Some(writer), Some(reader)) if writer != reader
        );
        if from_another_region {
            live_until_ms.saturating_add(millis(self.replication_grace))
        } else {
            live_until_ms
        }
    }

    /// Reads the table and tells watchers what changed since the last read,
    /// and when the view becomes complete or stops being complete.
    async fn refresh(&self) -> Result<(), DiscoveryError> {
        let _one_at_a_time = self.refreshing.lock().await;
        let began = self.lock().writes;
        // Before the read: a read that began before the lists became complete
        // may miss a PoolRelay that wrote meanwhile.
        let read_at_ms = self.clock.now_ms();
        let read = self.live().await?;
        let on_trust = read.on_trust;
        let live = read
            .pools
            .into_iter()
            .map(|(item, listed_until_ms)| {
                let seen = Seen {
                    listing: listing(&item.registration),
                    live_until_ms: listed_until_ms,
                    lease_until_ms: item.live_until_ms(self.lease_skew),
                    hash: item.hash,
                    written: 0,
                };
                (item.registration.key, seen)
            })
            .collect();
        let mut watched = self.lock();
        let completeness = match watched.read(read_at_ms, self.lease_ms) {
            _ if on_trust => Completeness::Estimated,
            completeness => completeness,
        };
        let mut events = watched.apply_poll(live, began);
        // After the poll's changes, so a `Resync` holds the new view.
        events.extend(watched.set_completeness(completeness));
        for event in events {
            // No subscribers is not an error.
            let _ = self.events.send(event);
        }
        Ok(())
    }

    /// What `list` serves now from the view; `None` before the first poll.
    fn listed(&self) -> Option<Snapshot<PoolListing>> {
        let now_ms = self.clock.now_ms();
        self.lock().listed(now_ms)
    }

    /// After a failed poll: drops the pools whose leases have lapsed, and
    /// tells watchers that the view is no longer complete. Nobody can renew
    /// or confirm a pool while the table cannot be reached.
    fn outage(&self) {
        let now_ms = self.clock.now_ms();
        let mut watched = self.lock();
        let mut events = watched.expire(now_ms);
        events.extend(watched.set_completeness(Completeness::Estimated));
        for event in events {
            // No subscribers is not an error.
            let _ = self.events.send(event);
        }
    }
}

/// Refreshes every `every` until `stop`. After a failed refresh, the view
/// drops the pools whose leases have lapsed and stops being complete, and
/// the next refresh waits twice as long, up to [`MAX_POLL_BACKOFF`].
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
                shared.outage();
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
        // DynamoDB would refuse the write with a `ValidationException` on
        // every heartbeat, which the PoolRelay would see as an outage.
        if encoded.sort_key.len() > keys::MAX_SORT_KEY_BYTES {
            return Err(DiscoveryError::Rejected(format!(
                "the pool key is {} bytes as a DynamoDB sort key, and DynamoDB allows at most {}",
                encoded.sort_key.len(),
                keys::MAX_SORT_KEY_BYTES
            )));
        }
        let now_ms = self.shared.clock.now_ms();
        let lease_s = self.lease.duration().as_secs();
        if let Err(error) = self.shared.write(&encoded, now_ms, lease_s).await {
            // PoolRelays may not be able to renew, so their items may lapse.
            self.shared.lock().failed();
            return Err(error);
        }
        let live_until_ms = codec::live_until_ms(now_ms, lease_s, self.shared.lease_skew);
        let mut watched = self.shared.lock();
        if let Some(event) =
            watched.apply_write(listing(&registration), encoded.hash, live_until_ms)
        {
            // No subscribers is not an error.
            let _ = self.shared.events.send(event);
        }
        Ok(self.lease)
    }
}

#[async_trait]
impl Directory<PoolListing, PoolKey> for DynamoDbPoolStore {
    /// The live pools, from this replica's view of the table, which a poll
    /// every [`DynamoDbConfig::poll_interval`] keeps current and this
    /// replica's registrations update at once. The first call reads the
    /// table and starts the polls, so a request costs no read of its own. A
    /// pool registered through another replica shows within a poll interval;
    /// one whose lease lapses is left out at once. `Estimated` until a lease
    /// has passed since the store was made, after a failed call to the table
    /// until a lease has passed since reads recovered, and while a pool is
    /// listed on trust; `Exact` otherwise. Fails only if the first read
    /// does.
    async fn list(&self) -> Result<Snapshot<PoolListing>, DiscoveryError> {
        self.ensure_polling();
        if let Some(snapshot) = self.shared.listed() {
            return Ok(snapshot);
        }
        // Before the first poll: read the table now, which starts the view.
        self.shared.refresh().await?;
        self.shared.listed().ok_or_else(|| {
            DiscoveryError::Unavailable(anyhow::anyhow!(
                "no view of DynamoDB table {} after reading it",
                self.shared.table
            ))
        })
    }

    /// Reads the table, then follows it by polling every
    /// [`DynamoDbConfig::poll_interval`], and this replica's registrations at
    /// once. A `Resync` tells watchers each time the view becomes complete or
    /// stops being complete, as `list` decides. While polls fail, the view
    /// drops pools as their leases lapse. Fails if the first read does; later
    /// failed polls leave the stream open.
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
                    live_until_ms: u64::MAX,
                    lease_until_ms: u64::MAX,
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
        assert_eq!(snapshot.completeness, Completeness::Estimated);
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
    fn watchers_learn_when_the_view_turns_complete_and_back() {
        let mut watched = Watched::default();
        // Before the first poll there is no view to tell about.
        assert!(watched.set_completeness(Completeness::Exact).is_none());
        let _ = watched.apply_poll(polled(&[("a", "m1")]), 0);
        assert_eq!(
            watched.snapshot().unwrap().completeness,
            Completeness::Estimated
        );

        let Some(WatchEvent::Resync(complete)) = watched.set_completeness(Completeness::Exact)
        else {
            panic!("expected a Resync");
        };
        assert_eq!(complete.completeness, Completeness::Exact);
        assert_eq!(complete.items.len(), 1);
        // Only a change is published.
        assert!(watched.set_completeness(Completeness::Exact).is_none());

        let Some(WatchEvent::Resync(partial)) = watched.set_completeness(Completeness::Estimated)
        else {
            panic!("expected a Resync");
        };
        assert_eq!(partial.completeness, Completeness::Estimated);
        assert_eq!(partial.items.len(), 1);
    }

    #[test]
    fn while_the_table_cannot_be_read_lapsed_pools_drop_out() {
        let mut watched = Watched::default();
        let mut view = polled(&[("a", "m1"), ("b", "m1")]);
        view.get_mut(&key("a")).unwrap().live_until_ms = 17_000;
        view.get_mut(&key("b")).unwrap().live_until_ms = 30_000;
        let _ = watched.apply_poll(view, 0);
        // At 17 s `a` is still live; just after, nothing can have renewed it.
        assert!(watched.expire(17_000).is_empty());
        assert_eq!(described(&watched.expire(17_001)), set(&["-a"]));
        assert!(watched.expire(17_001).is_empty());
        let listed: Vec<_> = watched.snapshot().unwrap().items;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].key, key("b"));
    }

    #[test]
    fn a_registration_is_published_unless_watchers_have_it() {
        let mut watched = Watched::default();
        // Before the first poll nobody watches, so nothing is published.
        assert!(
            watched
                .apply_write(pool("a", "m1"), "m1".to_owned(), u64::MAX)
                .is_none()
        );
        let _ = watched.apply_poll(polled(&[("a", "m1")]), watched.writes);

        assert!(
            watched
                .apply_write(pool("a", "m1"), "m1".to_owned(), u64::MAX)
                .is_none()
        );
        let changed = watched.apply_write(pool("a", "m2"), "m2".to_owned(), u64::MAX);
        assert_eq!(described(&Vec::from_iter(changed)), set(&["+a/m2"]));
        let added = watched.apply_write(pool("b", "m1"), "m1".to_owned(), u64::MAX);
        assert_eq!(described(&Vec::from_iter(added)), set(&["+b/m1"]));
    }

    #[test]
    fn a_poll_does_not_undo_a_registration_it_may_have_missed() {
        let mut watched = Watched::default();
        let _ = watched.apply_poll(polled(&[("a", "m1")]), 0);
        // A poll begins, then this replica registers a change and a new pool,
        // which the poll's read may predate.
        let began = watched.writes;
        let _ = watched.apply_write(pool("a", "m2"), "m2".to_owned(), u64::MAX);
        let _ = watched.apply_write(pool("b", "m1"), "m1".to_owned(), u64::MAX);
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

    /// DynamoDB refuses a sort key over 1,024 bytes on every write. The store
    /// says so before writing, so the PoolRelay learns why instead of seeing
    /// an outage.
    #[tokio::test]
    async fn a_pool_key_too_long_for_dynamodb_is_rejected() {
        let store = unreachable_store();
        let registration = |dgd: &str| PoolRegistration {
            key: key(dgd),
            location: pool(dgd, "m1").location,
            model: "m1".to_owned(),
            capabilities: Vec::new(),
            frontends: Vec::new(),
            capacity: Vec::new(),
        };
        // The sort key is `west#dynamo#` and the name.
        let longest = "d".repeat(keys::MAX_SORT_KEY_BYTES - "west#dynamo#".len());
        let Err(DiscoveryError::Unavailable(_)) = store.register(registration(&longest)).await
        else {
            panic!("a key at the limit should reach the table, which is down");
        };
        let longer = format!("{longest}d");
        let Err(DiscoveryError::Rejected(reason)) = store.register(registration(&longer)).await
        else {
            panic!("expected the key to be rejected");
        };
        assert!(reason.contains("1025 bytes"), "{reason}");
    }

    #[test]
    fn a_pool_that_keeps_switching_versions_is_noticed_once() {
        let mut flaps = Flaps::default();
        let a = key("a");
        // A change and its rollback are not flapping.
        assert!(!flaps.changed(&a, "m1", "m2"));
        assert!(!flaps.changed(&a, "m2", "m1"));
        // Going back to the version before once more is.
        assert!(flaps.changed(&a, "m1", "m2"));
        // It is noticed once.
        assert!(!flaps.changed(&a, "m2", "m1"));

        // Changes through ever new versions are not flapping.
        let b = key("b");
        for (from, to) in [("m1", "m2"), ("m2", "m3"), ("m3", "m4")] {
            assert!(!flaps.changed(&b, from, to));
        }

        // A pool that goes away and comes back starts over.
        flaps.forget(&a);
        assert!(!flaps.changed(&a, "m1", "m2"));
        assert!(!flaps.changed(&a, "m2", "m1"));
    }

    /// Two PoolRelays that send different registrations for one pool make
    /// every poll see a change. Each is still published, and the view notes
    /// the flapping.
    #[test]
    fn polls_that_switch_a_pool_between_two_versions_are_noticed() {
        let mut watched = Watched::default();
        let _ = watched.apply_poll(polled(&[("a", "m1")]), 0);
        for model in ["m2", "m1", "m2"] {
            let events = watched.apply_poll(polled(&[("a", model)]), 0);
            assert_eq!(described(&events), set(&[format!("+a/{model}").as_str()]));
        }
        assert!(watched.flaps.pools[&key("a")].warned);
        // It is forgotten once it goes.
        let _ = watched.apply_poll(polled(&[]), 0);
        assert!(watched.flaps.pools.is_empty());
    }

    /// `list` answers from the view between polls, as a read of the table at
    /// that moment would: lapsed pools are left out at once, and the list
    /// says when it may be wrong.
    #[test]
    fn a_list_from_the_view_leaves_out_lapsed_pools_at_once() {
        let mut watched = Watched::default();
        assert!(watched.listed(0).is_none());
        // `here` lapses at 17 s. `there` came through another region: its
        // lease lapses at 17 s, but it is listed until 77 s.
        let mut view = polled(&[("here", "m1"), ("there", "m1")]);
        for (dgd, lease_until_ms, live_until_ms) in
            [("here", 17_000, 17_000), ("there", 17_000, 77_000)]
        {
            let seen = view.get_mut(&key(dgd)).unwrap();
            seen.lease_until_ms = lease_until_ms;
            seen.live_until_ms = live_until_ms;
        }
        let _ = watched.apply_poll(view, 0);
        let _ = watched.set_completeness(Completeness::Exact);
        let names = |snapshot: &Snapshot<PoolListing>| {
            let mut names: Vec<String> = snapshot
                .items
                .iter()
                .map(|listing| listing.key.dgd_name().to_owned())
                .collect();
            names.sort();
            names
        };

        let listed = watched.listed(17_000).unwrap();
        assert_eq!(names(&listed), ["here", "there"]);
        assert_eq!(listed.completeness, Completeness::Exact);
        // Just after, `here` is gone, and `there` is listed on trust.
        let listed = watched.listed(17_001).unwrap();
        assert_eq!(names(&listed), ["there"]);
        assert_eq!(listed.completeness, Completeness::Estimated);
        // A failed call makes the list incomplete at once, before any poll.
        watched.failed();
        let listed = watched.listed(17_000).unwrap();
        assert_eq!(listed.completeness, Completeness::Estimated);
    }

    /// Renewals through other replicas reach the view only through polls.
    /// While a poll hangs, a pool can lapse in the view although it still
    /// renews elsewhere, so a list that leaves it out is not complete.
    #[test]
    fn a_pool_that_lapses_between_polls_makes_the_list_estimated() {
        let mut watched = Watched::default();
        let mut view = polled(&[("renewing", "m1"), ("other", "m1")]);
        let seen = view.get_mut(&key("renewing")).unwrap();
        seen.live_until_ms = 17_000;
        seen.lease_until_ms = 17_000;
        let _ = watched.apply_poll(view, 0);
        let _ = watched.set_completeness(Completeness::Exact);
        assert_eq!(
            watched.listed(17_000).unwrap().completeness,
            Completeness::Exact
        );
        let listed = watched.listed(17_001).unwrap();
        assert_eq!(listed.items.len(), 1);
        assert_eq!(listed.completeness, Completeness::Estimated);
    }

    /// While a rollout moves the service to a table, PoolRelays that still
    /// register with the old replicas are missing from it. After the table
    /// could not be reached, their items may have lapsed until they register
    /// again.
    #[test]
    fn lists_are_estimated_for_a_lease_after_start_and_after_reads_recover() {
        const LEASE_MS: u64 = 15_000;
        // A store made at 1 s.
        let mut watched = Watched {
            complete_at_ms: 16_000,
            ..Watched::default()
        };
        assert_eq!(watched.read(1_000, LEASE_MS), Completeness::Estimated);
        assert_eq!(watched.read(15_999, LEASE_MS), Completeness::Estimated);
        assert_eq!(watched.read(16_000, LEASE_MS), Completeness::Exact);

        // A call fails, and reads work again at 40 s.
        watched.failed();
        assert_eq!(watched.read(40_000, LEASE_MS), Completeness::Estimated);
        assert_eq!(watched.read(54_999, LEASE_MS), Completeness::Estimated);
        assert_eq!(watched.read(55_000, LEASE_MS), Completeness::Exact);
    }

    /// With a Global Table, another region's renewals may arrive late, so a
    /// pool written through another region's replica stays listed, on trust,
    /// for the replication grace after its lease.
    #[tokio::test]
    async fn another_regions_pool_stays_listed_on_trust_for_the_grace() {
        // This store reads the us-west-2 replica, with a 2 s skew and 60 s of
        // grace. A pool renewed at 0 with a 15 s lease is live until 17 s.
        let store = unreachable_store();
        let stored = |dgd: &str, written_in: Option<&str>| codec::Stored {
            registration: PoolRegistration {
                key: key(dgd),
                location: pool(dgd, "m1").location,
                model: "m1".to_owned(),
                capabilities: Vec::new(),
                frontends: Vec::new(),
                capacity: Vec::new(),
            },
            hash: "m1".to_owned(),
            renewed_at_ms: 0,
            lease_s: 15,
            written_in: written_in.map(str::to_owned),
        };
        let items = || {
            vec![
                stored("here", Some("us-west-2")),
                stored("unknown", None),
                stored("there", Some("eu-west-1")),
            ]
        };
        let listed = |read: &Read| {
            let mut names: Vec<String> = read
                .pools
                .iter()
                .map(|(item, _)| item.registration.key.dgd_name().to_owned())
                .collect();
            names.sort();
            names
        };

        let read = store.shared.select(items(), 17_000);
        assert_eq!(listed(&read), ["here", "there", "unknown"]);
        assert!(!read.on_trust);

        // Past the lease, only the other region's pool stays, on trust.
        let read = store.shared.select(items(), 17_001);
        assert_eq!(listed(&read), ["there"]);
        assert!(read.on_trust);
        assert_eq!(read.pools[0].1, 77_000);

        // Past the grace, it goes too.
        let read = store.shared.select(items(), 77_001);
        assert!(read.pools.is_empty());
        assert!(!read.on_trust);
    }

    /// A failed read or write is recorded, so lists stay `Estimated` for a
    /// lease after reads recover.
    #[tokio::test]
    async fn a_failed_call_is_recorded() {
        let store = unreachable_store();
        assert!(!store.shared.lock().failing);
        assert!(store.list().await.is_err());
        assert!(store.shared.lock().failing);
    }

    #[tokio::test]
    async fn an_unreachable_table_is_unavailable() {
        let store = unreachable_store();
        let Err(DiscoveryError::Unavailable(error)) = store.list().await else {
            panic!("expected the table to be unavailable");
        };
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
        let Err(DiscoveryError::Unavailable(error)) = store.register(registration).await else {
            panic!("expected the table to be unavailable");
        };
        assert!(
            format!("{error:#}").contains("write to DynamoDB table pools"),
            "{error:#}"
        );
        assert!(store.list_and_watch(None).await.is_err());
        assert!(format!("{store:?}").contains("pools"));
    }
}
