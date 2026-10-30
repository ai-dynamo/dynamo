// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The DynamoDB backend: registrations in one DynamoDB table that every
//! service instance reads, such as a Global Table with a replica in each
//! region. Instances need not ask each other, and a restarted instance lists
//! every live pool as soon as it has read the table, so its lists are `Exact`
//! from its first read.
//!
//! Each instance reads the table every [`REFRESH_EVERY`], from its region's
//! replica with strongly consistent reads, and lists the registrations whose
//! leases have not ended. A registration made in another region arrives with
//! replication, usually within a second. Each lease ends at a time the
//! writing instance's clock set, so the lease's slack over the heartbeat
//! interval, 10 s with the defaults, covers both that delay and the clock
//! skew between instances. While reads fail, the last list stays, marked
//! `Estimated`.
//!
//! The table has the partition key `pk` and the sort key `sk`, both strings.
//! Each pool is one item:
//! - `pk` is `POOL`, and `sk` is the pool key as
//!   `<cluster_id>#<namespace>#<dgd_name>`, with `%` and `#` escaped in each
//!   part;
//! - `registration` holds the registration as JSON, with its lists sorted;
//! - `lease_ends_ms` is when its lease ends, in Unix milliseconds;
//! - `expires_at`, the time-to-live attribute, is an hour later, in Unix
//!   seconds, so that DynamoDB deletes the items of pools that stopped
//!   registering.
//!
//! ```bash
//! aws dynamodb create-table --table-name global-discovery-registrations \
//!     --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
//!     --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
//!     --billing-mode PAY_PER_REQUEST
//! aws dynamodb update-time-to-live --table-name global-discovery-registrations \
//!     --time-to-live-specification Enabled=true,AttributeName=expires_at
//! ```
//!
//! The service needs `dynamodb:PutItem` and `dynamodb:Query` on the table.
//! Every instance reads every item once a second, so reads grow with the
//! number of pools and of instances: 100 pools of about 1 KB take about 25
//! read units a second per instance.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, anyhow, bail};
use async_trait::async_trait;
use aws_config::timeout::TimeoutConfig;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_dynamodb::Client;
use aws_sdk_dynamodb::error::DisplayErrorContext;
use aws_sdk_dynamodb::types::AttributeValue;
use dynamo_global_common::PoolKey;
use tokio::sync::broadcast;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use url::Url;

use super::memory::{listing, sorted};
use super::stream::{EVENT_BUFFER, watch_stream};
use crate::pool_discovery::{
    Completeness, Directory, DiscoveryError, Lease, PoolEvent, PoolListing, PoolRegistrar,
    PoolRegistration, Snapshot, WatchEvent, WatchStream,
};

/// How often each instance reads the table. Watchers, such as the MCS
/// backend's join, drop a pool at the first read after its lease ended.
pub const REFRESH_EVERY: Duration = Duration::from_secs(1);

/// How long a pool that stopped registering stays in the table after its
/// lease ended, before DynamoDB's time to live deletes it.
const KEEP_AFTER_LEASE: Duration = Duration::from_secs(3600);

/// How long one call to DynamoDB may take, retries included. A registration
/// waits for its write.
const CALL_TIMEOUT: Duration = Duration::from_secs(5);

/// The partition key of every pool item.
const POOLS: &str = "POOL";

const PK: &str = "pk";
const SK: &str = "sk";
const REGISTRATION: &str = "registration";
const LEASE_ENDS_MS: &str = "lease_ends_ms";
const EXPIRES_AT: &str = "expires_at";

/// Where the registrations table is.
#[derive(Clone, Debug)]
pub struct TableLocation {
    pub name: String,
    /// `None` takes the region from the AWS environment, such as
    /// `AWS_REGION`.
    pub region: Option<String>,
    /// Another endpoint than AWS's, such as DynamoDB Local's.
    pub endpoint_url: Option<Url>,
}

/// Pool registrations in a DynamoDB table. A pool expires the lease duration
/// after its last registration; reads list it until the next read of the
/// table after that, which comes every [`REFRESH_EVERY`] while reads work.
/// While they fail, the last list stays, marked `Estimated`.
///
/// Any caller may register any pool: V1 assumes a trusted network, and
/// several PoolRelay replicas may register the same pool.
pub struct DynamoDbPoolStore {
    lease: Lease,
    table: Arc<dyn Table>,
    view: Arc<Mutex<View>>,
    /// Sent while holding the `view` lock, so a subscriber that snapshots
    /// and subscribes under it misses no event and sees none twice.
    events: broadcast::Sender<PoolEvent>,
    refreshing: OnceLock<()>,
    stop: CancellationToken,
}

impl DynamoDbPoolStore {
    /// Credentials, and the region unless `table` names one, come from the
    /// AWS environment, as for the AWS CLI: for example a web identity token
    /// on EKS. The service bounds the lease, since the store adds it to the
    /// current time.
    pub async fn connect(table: &TableLocation, lease: Lease) -> anyhow::Result<Self> {
        let mut loader = aws_config::defaults(BehaviorVersion::latest());
        if let Some(region) = &table.region {
            loader = loader.region(Region::new(region.clone()));
        }
        let shared = loader.load().await;
        if shared.region().is_none() {
            bail!("no AWS region for DynamoDB: name one, or set AWS_REGION");
        }
        let mut config = aws_sdk_dynamodb::config::Builder::from(&shared).timeout_config(
            TimeoutConfig::builder()
                .operation_timeout(CALL_TIMEOUT)
                .build(),
        );
        if let Some(url) = &table.endpoint_url {
            config = config.endpoint_url(url.as_str());
        }
        let client = Client::from_conf(config.build());
        Ok(Self::with_table(
            Arc::new(SdkTable::new(client, table.name.clone())),
            lease,
        ))
    }

    fn with_table(table: Arc<dyn Table>, lease: Lease) -> Self {
        Self {
            lease,
            table,
            view: Arc::default(),
            events: broadcast::channel(EVENT_BUFFER).0,
            refreshing: OnceLock::new(),
            stop: CancellationToken::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, View> {
        self.view.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Every trait method runs in a Tokio runtime, so the spawn is safe.
    fn ensure_refreshing(&self) {
        self.refreshing.get_or_init(|| {
            tokio::spawn(refresh(
                Arc::clone(&self.table),
                Arc::clone(&self.view),
                self.events.clone(),
                self.stop.clone(),
            ));
        });
    }
}

impl fmt::Debug for DynamoDbPoolStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamoDbPoolStore")
            .field("lease", &self.lease)
            .finish_non_exhaustive()
    }
}

impl Drop for DynamoDbPoolStore {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

/// The table as of the latest read.
#[derive(Default)]
struct View {
    /// The registrations whose leases had not ended, with their lists
    /// sorted.
    registrations: HashMap<PoolKey, PoolRegistration>,
    /// A read has listed the whole table.
    synced: bool,
    /// The latest read failed, so the list may be stale.
    stale: bool,
}

impl View {
    /// `None` until a read has listed the whole table.
    fn snapshot(&self) -> Option<Snapshot<PoolListing>> {
        self.synced.then(|| Snapshot {
            items: self.registrations.values().map(listing).collect(),
            completeness: if self.stale {
                Completeness::Estimated
            } else {
                Completeness::Exact
            },
        })
    }

    /// Takes the live registrations from a read of the table, and publishes
    /// what changed. The first read,
    /// and the first after failures, publish one `Resync`, since watchers
    /// then need the whole list.
    fn apply(
        &mut self,
        live: HashMap<PoolKey, PoolRegistration>,
        events: &broadcast::Sender<PoolEvent>,
    ) {
        let previous = std::mem::replace(&mut self.registrations, live);
        let resync = !self.synced || self.stale;
        self.synced = true;
        self.stale = false;
        if resync {
            if let Some(snapshot) = self.snapshot() {
                // No subscribers is not an error.
                let _ = events.send(WatchEvent::Resync(snapshot));
            }
            return;
        }
        for key in previous.keys() {
            if !self.registrations.contains_key(key) {
                let _ = events.send(WatchEvent::Removed(key.clone()));
            }
        }
        for (key, registration) in &self.registrations {
            if previous.get(key) != Some(registration) {
                let _ = events.send(WatchEvent::Upserted(listing(registration)));
            }
        }
    }

    /// Keeps the last list, and tells watchers it may be stale.
    fn fail(&mut self, events: &broadcast::Sender<PoolEvent>) {
        let was_exact = self.synced && !self.stale;
        self.stale = true;
        if was_exact && let Some(snapshot) = self.snapshot() {
            // No subscribers is not an error.
            let _ = events.send(WatchEvent::Resync(snapshot));
        }
    }
}

/// The registrations whose leases have not ended at `now_ms`, by pool, with
/// their lists sorted.
fn live(stored: Vec<Stored>, now_ms: u64) -> HashMap<PoolKey, PoolRegistration> {
    stored
        .into_iter()
        .filter(|stored| stored.lease_ends_ms > now_ms)
        .map(|stored| (stored.registration.key.clone(), sorted(stored.registration)))
        .collect()
}

/// Reads the table every [`REFRESH_EVERY`] and applies each read to the
/// view. A failure is logged when a run of them starts and when it ends.
async fn refresh(
    table: Arc<dyn Table>,
    view: Arc<Mutex<View>>,
    events: broadcast::Sender<PoolEvent>,
    stop: CancellationToken,
) {
    let mut ticks = tokio::time::interval(REFRESH_EVERY);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut failing = false;
    loop {
        tokio::select! {
            biased;
            () = stop.cancelled() => return,
            _ = ticks.tick() => {}
        }
        let read = tokio::select! {
            biased;
            () = stop.cancelled() => return,
            read = table.read_all() => read,
        };
        // Sorted before the lock, which `list` and new watchers wait for.
        let read = read.map(|stored| live(stored, unix_ms_now()));
        let mut view = view.lock().unwrap_or_else(PoisonError::into_inner);
        match read {
            Ok(live) => {
                if failing {
                    tracing::info!("the registrations table can be read again");
                    failing = false;
                }
                view.apply(live, &events);
            }
            Err(error) => {
                if failing {
                    tracing::debug!(?error, "the registrations table still cannot be read");
                } else {
                    tracing::warn!(
                        ?error,
                        "cannot read the registrations table; keeping the last list, marked \
                         incomplete"
                    );
                    failing = true;
                }
                view.fail(&events);
            }
        }
    }
}

/// Now, in Unix milliseconds. Leases end at times that every instance can
/// read, so they come from the wall clock, not from `Instant`.
fn unix_ms_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| {
            u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
        })
}

#[async_trait]
impl PoolRegistrar for DynamoDbPoolStore {
    /// Writes the registration with the end of its lease. It is listed from
    /// the next read of the table. Lists in another order, for example from
    /// another PoolRelay replica, are not a change.
    async fn register(&self, registration: PoolRegistration) -> Result<Lease, DiscoveryError> {
        let registration = sorted(registration);
        let lease_ms = u64::try_from(self.lease.duration().as_millis()).unwrap_or(u64::MAX);
        let lease_ends_ms = unix_ms_now().saturating_add(lease_ms);
        self.table
            .put(&registration, lease_ends_ms)
            .await
            .map_err(DiscoveryError::Unavailable)?;
        // After the write, so the first read already lists it.
        self.ensure_refreshing();
        Ok(self.lease)
    }
}

#[async_trait]
impl Directory<PoolListing, PoolKey> for DynamoDbPoolStore {
    /// From the latest read. Before the first one, one read of the table, so
    /// a caller never waits for the refresh to start.
    async fn list(&self) -> Result<Snapshot<PoolListing>, DiscoveryError> {
        self.ensure_refreshing();
        let snapshot = self.lock().snapshot();
        if let Some(snapshot) = snapshot {
            return Ok(snapshot);
        }
        let stored = self
            .table
            .read_all()
            .await
            .map_err(DiscoveryError::Unavailable)?;
        Ok(Snapshot {
            items: live(stored, unix_ms_now()).values().map(listing).collect(),
            completeness: Completeness::Exact,
        })
    }

    /// Starts at the first read of the whole table, then follows the reads.
    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<PoolListing, PoolKey>, DiscoveryError> {
        self.ensure_refreshing();
        let view = Arc::clone(&self.view);
        let events = self.events.clone();
        Ok(watch_stream(
            move || {
                let view = view.lock().unwrap_or_else(PoisonError::into_inner);
                (events.subscribe(), view.snapshot())
            },
            cancel_token.unwrap_or_default(),
            self.stop.clone(),
        ))
    }
}

/// The store's calls to the table, so that tests can replace DynamoDB.
#[async_trait]
trait Table: Send + Sync + 'static {
    /// Stores a registration whose lease ends at `lease_ends_ms`.
    async fn put(&self, registration: &PoolRegistration, lease_ends_ms: u64) -> anyhow::Result<()>;

    /// Every stored registration, with the end of its lease.
    async fn read_all(&self) -> anyhow::Result<Vec<Stored>>;
}

/// A registration as stored.
#[derive(Clone, Debug, PartialEq)]
struct Stored {
    registration: PoolRegistration,
    /// Unix milliseconds.
    lease_ends_ms: u64,
}

/// The table, through the AWS SDK.
struct SdkTable {
    client: Client,
    name: String,
    /// The sort keys of items that did not parse, so each is logged once.
    unparsed: Mutex<HashSet<String>>,
}

impl SdkTable {
    fn new(client: Client, name: String) -> Self {
        Self {
            client,
            name,
            unparsed: Mutex::default(),
        }
    }
}

#[async_trait]
impl Table for SdkTable {
    async fn put(&self, registration: &PoolRegistration, lease_ends_ms: u64) -> anyhow::Result<()> {
        self.client
            .put_item()
            .table_name(&self.name)
            .set_item(Some(item(registration, lease_ends_ms)?))
            .send()
            .await
            .map_err(|error| {
                anyhow!(
                    "write to DynamoDB table {}: {}",
                    self.name,
                    DisplayErrorContext(&error)
                )
            })?;
        Ok(())
    }

    /// Every page of the pools partition. An item that does not parse is
    /// skipped, so that one bad item does not hide every pool.
    async fn read_all(&self) -> anyhow::Result<Vec<Stored>> {
        let mut stored = Vec::new();
        let mut start = None;
        loop {
            let page = self
                .client
                .query()
                .table_name(&self.name)
                .key_condition_expression("pk = :pools")
                .expression_attribute_values(":pools", AttributeValue::S(POOLS.to_owned()))
                .consistent_read(true)
                .set_exclusive_start_key(start)
                .send()
                .await
                .map_err(|error| {
                    anyhow!(
                        "read DynamoDB table {}: {}",
                        self.name,
                        DisplayErrorContext(&error)
                    )
                })?;
            for item in page.items() {
                match stored_from(item) {
                    Ok(item) => stored.push(item),
                    Err(error) => {
                        let sk = match item.get(SK) {
                            Some(AttributeValue::S(sk)) => sk.clone(),
                            _ => String::new(),
                        };
                        let mut unparsed =
                            self.unparsed.lock().unwrap_or_else(PoisonError::into_inner);
                        if unparsed.insert(sk.clone()) {
                            tracing::warn!(%sk, ?error, "skipping a registration item that does not parse");
                        }
                    }
                }
            }
            start = page.last_evaluated_key().cloned();
            if start.is_none() {
                return Ok(stored);
            }
        }
    }
}

/// The item for a registration whose lease ends at `lease_ends_ms`.
fn item(
    registration: &PoolRegistration,
    lease_ends_ms: u64,
) -> anyhow::Result<HashMap<String, AttributeValue>> {
    let expires_at = (lease_ends_ms / 1000).saturating_add(KEEP_AFTER_LEASE.as_secs());
    Ok(HashMap::from([
        (PK.to_owned(), AttributeValue::S(POOLS.to_owned())),
        (
            SK.to_owned(),
            AttributeValue::S(sort_key(&registration.key)),
        ),
        (
            REGISTRATION.to_owned(),
            AttributeValue::S(serde_json::to_string(registration)?),
        ),
        (
            LEASE_ENDS_MS.to_owned(),
            AttributeValue::N(lease_ends_ms.to_string()),
        ),
        (
            EXPIRES_AT.to_owned(),
            AttributeValue::N(expires_at.to_string()),
        ),
    ]))
}

/// The registration and lease end an item holds.
fn stored_from(item: &HashMap<String, AttributeValue>) -> anyhow::Result<Stored> {
    let Some(AttributeValue::S(json)) = item.get(REGISTRATION) else {
        bail!("no `{REGISTRATION}` string");
    };
    let Some(AttributeValue::N(lease_ends_ms)) = item.get(LEASE_ENDS_MS) else {
        bail!("no `{LEASE_ENDS_MS}` number");
    };
    Ok(Stored {
        registration: serde_json::from_str(json).context("parse the registration")?,
        lease_ends_ms: lease_ends_ms.parse().context("parse the lease end")?,
    })
}

/// `<cluster_id>#<namespace>#<dgd_name>`, with `%` and `#` escaped in each
/// part, so that no two pools share a sort key.
fn sort_key(key: &PoolKey) -> String {
    [key.cluster_id(), key.namespace(), key.dgd_name()]
        .map(|part| part.replace('%', "%25").replace('#', "%23"))
        .join("#")
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use dynamo_global_common::PoolLocation;
    use futures::{FutureExt, StreamExt};

    use super::*;

    fn lease() -> Lease {
        Lease::new(5, 15).unwrap()
    }

    fn key(dgd: &str) -> PoolKey {
        PoolKey::new("west", "dynamo", dgd).unwrap()
    }

    fn registration(dgd: &str, model: &str) -> PoolRegistration {
        PoolRegistration {
            key: key(dgd),
            location: PoolLocation {
                region: "us-west-2".to_owned(),
                availability_zone: None,
                datacenter: None,
            },
            model: model.to_owned(),
            capabilities: vec!["prefill".to_owned(), "decode".to_owned()],
            frontends: vec![
                Url::parse("http://10.43.1.3:8000/").unwrap(),
                Url::parse("http://10.43.1.2:8000/").unwrap(),
            ],
            capacity: Vec::new(),
        }
    }

    /// A table in memory that can be made to fail.
    #[derive(Default)]
    struct FakeTable {
        items: Mutex<HashMap<PoolKey, Stored>>,
        down: AtomicBool,
    }

    impl FakeTable {
        fn set_down(&self, down: bool) {
            self.down.store(down, Ordering::Relaxed);
        }

        /// Ends the lease of `dgd`'s registration.
        fn end_lease(&self, dgd: &str) {
            self.items
                .lock()
                .unwrap()
                .get_mut(&key(dgd))
                .unwrap()
                .lease_ends_ms = 0;
        }
    }

    #[async_trait]
    impl Table for FakeTable {
        async fn put(
            &self,
            registration: &PoolRegistration,
            lease_ends_ms: u64,
        ) -> anyhow::Result<()> {
            if self.down.load(Ordering::Relaxed) {
                bail!("down");
            }
            self.items.lock().unwrap().insert(
                registration.key.clone(),
                Stored {
                    registration: registration.clone(),
                    lease_ends_ms,
                },
            );
            Ok(())
        }

        async fn read_all(&self) -> anyhow::Result<Vec<Stored>> {
            if self.down.load(Ordering::Relaxed) {
                bail!("down");
            }
            Ok(self.items.lock().unwrap().values().cloned().collect())
        }
    }

    fn store() -> (Arc<FakeTable>, DynamoDbPoolStore) {
        let table = Arc::new(FakeTable::default());
        let store = DynamoDbPoolStore::with_table(table.clone(), lease());
        (table, store)
    }

    fn models(snapshot: &Snapshot<PoolListing>) -> Vec<&str> {
        let mut models: Vec<&str> = snapshot
            .items
            .iter()
            .filter_map(|listing| listing.model.as_deref())
            .collect();
        models.sort_unstable();
        models
    }

    #[tokio::test(start_paused = true)]
    async fn a_registration_is_listed_from_the_table() {
        let (table, store) = store();
        assert_eq!(
            store.register(registration("a", "m1")).await.unwrap(),
            lease()
        );
        // Before the first refresh, `list` reads the table itself.
        let listed = store.list().await.unwrap();
        assert_eq!(listed.completeness, Completeness::Exact);
        assert_eq!(models(&listed), ["m1"]);
        // The stored lists are sorted, and the lease ends a lease from now.
        let stored = table.read_all().await.unwrap();
        assert_eq!(
            stored[0].registration.frontends,
            [
                Url::parse("http://10.43.1.2:8000/").unwrap(),
                Url::parse("http://10.43.1.3:8000/").unwrap(),
            ]
        );
        let lease_ms = 15_000;
        assert!(stored[0].lease_ends_ms > unix_ms_now() + lease_ms - 1_000);
    }

    #[tokio::test(start_paused = true)]
    async fn a_registration_whose_lease_ended_is_not_listed() {
        let (table, store) = store();
        let _ = store.register(registration("a", "m1")).await.unwrap();
        table.end_lease("a");
        assert!(store.list().await.unwrap().items.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn the_watch_starts_at_the_first_read_and_follows_the_table() {
        let (table, store) = store();
        let _ = store.register(registration("a", "m1")).await.unwrap();
        let mut watch = store.list_and_watch(None).await.unwrap();
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Upserted(listing))) if listing.key == key("a"))
        );
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Resync(snapshot))) if snapshot.completeness == Completeness::Exact)
        );

        let _ = store.register(registration("b", "m2")).await.unwrap();
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Upserted(listing))) if listing.key == key("b"))
        );
        let _ = store.register(registration("b", "m3")).await.unwrap();
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Upserted(listing))) if listing.model.as_deref() == Some("m3"))
        );
        table.end_lease("a");
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Removed(gone))) if gone == key("a"))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_heartbeat_with_reordered_lists_is_not_a_change() {
        let (_table, store) = store();
        let _ = store.register(registration("a", "m1")).await.unwrap();
        let mut watch = store.list_and_watch(None).await.unwrap();
        // The first read: an `Upserted` and a `Resync`.
        watch.next().await.unwrap().unwrap();
        watch.next().await.unwrap().unwrap();
        let mut reordered = registration("a", "m1");
        reordered.capabilities.reverse();
        reordered.frontends.reverse();
        let _ = store.register(reordered).await.unwrap();
        tokio::time::sleep(REFRESH_EVERY * 3).await;
        assert!(watch.next().now_or_never().is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn a_failed_read_keeps_the_last_list_as_estimated() {
        let (table, store) = store();
        let _ = store.register(registration("a", "m1")).await.unwrap();
        let mut watch = store.list_and_watch(None).await.unwrap();
        watch.next().await.unwrap().unwrap();
        watch.next().await.unwrap().unwrap();

        table.set_down(true);
        let Some(Ok(WatchEvent::Resync(stale))) = watch.next().await else {
            panic!("expected a Resync");
        };
        assert_eq!(stale.completeness, Completeness::Estimated);
        assert_eq!(models(&stale), ["m1"]);
        let listed = store.list().await.unwrap();
        assert_eq!(listed.completeness, Completeness::Estimated);
        assert_eq!(models(&listed), ["m1"]);
        // A registration needs the table.
        assert!(matches!(
            store.register(registration("b", "m2")).await,
            Err(DiscoveryError::Unavailable(_))
        ));

        table.set_down(false);
        let Some(Ok(WatchEvent::Resync(fresh))) = watch.next().await else {
            panic!("expected a Resync");
        };
        assert_eq!(fresh.completeness, Completeness::Exact);
    }

    #[tokio::test(start_paused = true)]
    async fn an_unreadable_table_before_the_first_read_is_unavailable() {
        let (table, store) = store();
        table.set_down(true);
        assert!(matches!(
            store.list().await,
            Err(DiscoveryError::Unavailable(_))
        ));
    }

    #[test]
    fn an_item_holds_the_registration_and_its_lease() {
        let registration = registration("a", "m1");
        let item = item(&registration, 1_700_000_000_123).unwrap();
        assert_eq!(item[PK], AttributeValue::S("POOL".to_owned()));
        assert_eq!(item[SK], AttributeValue::S("west#dynamo#a".to_owned()));
        assert_eq!(
            item[EXPIRES_AT],
            AttributeValue::N((1_700_000_000 + 3600).to_string())
        );
        assert_eq!(
            stored_from(&item).unwrap(),
            Stored {
                registration,
                lease_ends_ms: 1_700_000_000_123,
            }
        );
    }

    #[test]
    fn an_item_without_its_attributes_does_not_parse() {
        let mut item = item(&registration("a", "m1"), 1).unwrap();
        item.insert(LEASE_ENDS_MS.to_owned(), AttributeValue::S("1".to_owned()));
        assert!(stored_from(&item).is_err());
        item.remove(REGISTRATION);
        assert!(stored_from(&item).is_err());
    }

    #[test]
    fn sort_keys_escape_their_separator() {
        let one = PoolKey::new("a#b", "c", "d").unwrap();
        let two = PoolKey::new("a", "b#c", "d").unwrap();
        assert_ne!(sort_key(&one), sort_key(&two));
        assert_eq!(sort_key(&one), "a%23b#c#d");
        let percent = PoolKey::new("a%23b", "c", "d").unwrap();
        assert_eq!(sort_key(&percent), "a%2523b#c#d");
    }

    /// Against DynamoDB Local, so it runs only when asked for:
    ///
    /// ```bash
    /// docker run --rm -d -p 8000:8000 amazon/dynamodb-local
    /// DYNAMODB_LOCAL_URL=http://localhost:8000 cargo test -p dynamo-global-discovery-service \
    ///     --features dynamodb dynamodb_local -- --ignored
    /// ```
    #[tokio::test]
    #[ignore = "needs DynamoDB Local: set DYNAMODB_LOCAL_URL and pass --ignored"]
    async fn dynamodb_local_lists_registrations_until_their_leases_end() {
        use aws_sdk_dynamodb::config::Credentials;
        use aws_sdk_dynamodb::types::{
            AttributeDefinition, BillingMode, KeySchemaElement, KeyType, ScalarAttributeType,
        };

        let url =
            std::env::var("DYNAMODB_LOCAL_URL").expect("DYNAMODB_LOCAL_URL names DynamoDB Local");
        let config = aws_sdk_dynamodb::config::Builder::new()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new("us-west-2"))
            .endpoint_url(url)
            .credentials_provider(Credentials::new("local", "local", None, None, "test"))
            .build();
        let client = Client::from_conf(config);
        let name = format!("registrations-{}", unix_ms_now());
        let attribute = |name: &str| {
            AttributeDefinition::builder()
                .attribute_name(name)
                .attribute_type(ScalarAttributeType::S)
                .build()
                .unwrap()
        };
        let key_part = |name: &str, key_type| {
            KeySchemaElement::builder()
                .attribute_name(name)
                .key_type(key_type)
                .build()
                .unwrap()
        };
        client
            .create_table()
            .table_name(&name)
            .attribute_definitions(attribute(PK))
            .attribute_definitions(attribute(SK))
            .key_schema(key_part(PK, KeyType::Hash))
            .key_schema(key_part(SK, KeyType::Range))
            .billing_mode(BillingMode::PayPerRequest)
            .send()
            .await
            .unwrap();
        // An item that does not parse is skipped.
        client
            .put_item()
            .table_name(&name)
            .item(PK, AttributeValue::S(POOLS.to_owned()))
            .item(SK, AttributeValue::S("bad".to_owned()))
            .item(REGISTRATION, AttributeValue::S("{".to_owned()))
            .send()
            .await
            .unwrap();

        let store = DynamoDbPoolStore::with_table(
            Arc::new(SdkTable::new(client.clone(), name.clone())),
            Lease::new(1, 2).unwrap(),
        );
        let _ = store.register(registration("a", "m1")).await.unwrap();
        let listed = store.list().await.unwrap();
        assert_eq!(listed.completeness, Completeness::Exact);
        assert_eq!(models(&listed), ["m1"]);

        let mut watch = store.list_and_watch(None).await.unwrap();
        async fn next(
            watch: &mut WatchStream<PoolListing, PoolKey>,
        ) -> Option<Result<PoolEvent, DiscoveryError>> {
            tokio::time::timeout(Duration::from_secs(10), watch.next())
                .await
                .expect("an event within 10 s")
        }
        assert!(
            matches!(next(&mut watch).await, Some(Ok(WatchEvent::Upserted(listing))) if listing.key == key("a"))
        );
        assert!(matches!(
            next(&mut watch).await,
            Some(Ok(WatchEvent::Resync(_)))
        ));
        // No heartbeat: the 2 s lease ends, and the next read drops the pool.
        assert!(
            matches!(next(&mut watch).await, Some(Ok(WatchEvent::Removed(gone))) if gone == key("a"))
        );

        client
            .delete_table()
            .table_name(&name)
            .send()
            .await
            .unwrap();
    }
}
