// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use dynamo_global_common::{PoolId, PoolKey};
use dynamo_global_router::affinity::{
    AffinityBinding, AffinityConfig, AffinityError, AffinityKey, AffinitySnapshot, AffinityStore,
    BreakerConfig, CachedAffinityStore, Expected, InMemoryAffinityStore, SessionAffinity,
    StoreBackedSessionAffinity, WriteOutcome, tenant_id,
};
use dynamo_global_router::{
    AffinitySelector, InFlightTracker, Pool, PoolSelector, PoolSignals, RoutingRequest,
};

fn key() -> AffinityKey {
    AffinityKey {
        scope: "test".into(),
        tenant: "anonymous".into(),
        policy: "headroom-v1".into(),
        model: "qwen".into(),
        session: "s".into(),
    }
}

fn pool(region: &str, model: &str) -> Pool {
    Pool::new(
        &PoolKey::new(region, "dynamo", model).unwrap(),
        model,
        region,
        vec![format!("http://{region}:8000").parse().unwrap()],
    )
}

fn tracker(store: Arc<dyn AffinityStore>) -> Arc<StoreBackedSessionAffinity> {
    Arc::new(StoreBackedSessionAffinity::new(store, Duration::from_secs(1800)).unwrap())
}

fn selector(store: Arc<dyn AffinityStore>, region: &str, scope: &str) -> AffinitySelector {
    selector_with(store, region, scope, |_| ())
}

fn selector_with(
    store: Arc<dyn AffinityStore>,
    region: &str,
    scope: &str,
    adjust: impl FnOnce(&mut AffinityConfig),
) -> AffinitySelector {
    let mut config = AffinityConfig {
        scope: scope.into(),
        ..Default::default()
    };
    adjust(&mut config);
    AffinitySelector::new(
        Arc::new(PoolSignals::new(Arc::new(InFlightTracker::new()))),
        region.into(),
        2.0,
        tracker(store),
        config,
    )
}

async fn pick(
    selector: &AffinitySelector,
    pools: &[Pool],
    model: &str,
    session: Option<&str>,
) -> PoolId {
    let candidates = pools
        .iter()
        .filter(|pool| pool.model == model)
        .collect::<Vec<_>>();
    selector
        .select(
            &RoutingRequest::new(model).with_session(session),
            &candidates,
        )
        .await
        .unwrap()
        .pool
        .id
        .clone()
}

fn binding(pool: &PoolId, revision: &str, updated_at: u64, expires_at: u64) -> AffinityBinding {
    AffinityBinding {
        pool: pool.clone(),
        revision: revision.into(),
        updated_at,
        expires_at,
    }
}

#[tokio::test]
async fn selectors_share_pins_but_isolate_scope_model_and_policy() {
    let store = Arc::new(InMemoryAffinityStore::default());
    let east = selector(store.clone(), "east", "test");
    let west = selector(store.clone(), "west", "test");
    let isolated = selector(store.clone(), "west", "other-deployment");
    let pools = [
        pool("east", "qwen"),
        pool("west", "qwen"),
        pool("east", "llama"),
        pool("west", "llama"),
    ];
    assert_eq!(pick(&east, &pools, "qwen", Some("s")).await, pools[0].id);
    assert_eq!(pick(&west, &pools, "qwen", Some("s")).await, pools[0].id);
    assert_eq!(
        pick(&isolated, &pools, "qwen", Some("s")).await,
        pools[1].id
    );
    assert_eq!(pick(&west, &pools, "llama", Some("s")).await, pools[3].id);
    // A different policy namespace is a different binding: this selector
    // pins the same session independently rather than inheriting east's pin.
    let other_policy = selector_with(store.clone(), "west", "test", |config| {
        config.policy = "new-policy".into();
    });
    assert_eq!(
        pick(&other_policy, &pools, "qwen", Some("s")).await,
        pools[1].id
    );
    assert_eq!(pick(&west, &pools, "qwen", Some("s")).await, pools[0].id);
    assert_eq!(
        pick(&other_policy, &pools, "qwen", Some("s")).await,
        pools[1].id
    );
    assert_eq!(store.len(), 4);
}

#[tokio::test]
async fn removed_pool_is_replaced_and_requests_without_sessions_do_not_pin() {
    let store = Arc::new(InMemoryAffinityStore::default());
    let router = selector(store.clone(), "east", "test");
    let pools = [pool("east", "qwen"), pool("west", "qwen")];
    pick(&router, &pools, "qwen", None).await;
    assert!(store.is_empty());
    pick(&router, &pools, "qwen", Some("s")).await;
    assert_eq!(
        pick(&router, &pools[1..], "qwen", Some("s")).await,
        pools[1].id
    );
    assert_eq!(store.read(&key()).await.unwrap().unwrap().pool, pools[1].id);
}

#[tokio::test]
async fn expired_present_record_is_replaced_before_sweep() {
    let store = Arc::new(InMemoryAffinityStore::default());
    let old = pool("east", "qwen").id;
    let new = pool("west", "qwen").id;
    let expired = binding(&old, "expired", 0, 1);
    store
        .compare_exchange(&key(), &Expected::Absent, &expired)
        .await
        .unwrap();
    let affinity = tracker(store.clone());
    let observed = affinity.lookup(&key()).await.unwrap();
    assert_eq!(observed.pool, None);
    assert_eq!(observed.expected, Expected::Revision("expired".into()));
    assert_eq!(
        affinity.bind(&key(), &observed, &new).await.unwrap(),
        WriteOutcome::Applied
    );
    assert_eq!(affinity.lookup(&key()).await.unwrap().pool, Some(new));
    let updated = store.read(&key()).await.unwrap().unwrap();
    store.sweep(updated.expires_at - 1);
    assert_eq!(store.len(), 1);
    store.sweep(updated.expires_at);
    assert!(store.is_empty());
    // An old request cannot recreate an item after cleanup under its old revision.
    assert_eq!(
        store
            .compare_exchange(&key(), &Expected::Revision("expired".into()), &expired)
            .await
            .unwrap(),
        WriteOutcome::Conflict
    );
}

#[tokio::test]
async fn concurrent_creates_have_one_winner_and_stale_revisions_cannot_overwrite() {
    let store = Arc::new(InMemoryAffinityStore::default());
    let a = tracker(store.clone());
    let b = tracker(store.clone());
    let key = key();
    let observed_a = a.lookup(&key).await.unwrap();
    let observed_b = b.lookup(&key).await.unwrap();
    let east = pool("east", "qwen").id;
    let west = pool("west", "qwen").id;
    let (a_result, b_result) = tokio::join!(
        a.bind(&key, &observed_a, &east),
        b.bind(&key, &observed_b, &west)
    );
    let results = [a_result.unwrap(), b_result.unwrap()];
    assert_eq!(
        results
            .iter()
            .filter(|&&r| r == WriteOutcome::Applied)
            .count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|&&r| r == WriteOutcome::Conflict)
            .count(),
        1
    );
    let stale = a.lookup(&key).await.unwrap();
    a.bind(&key, &stale, &west).await.unwrap();
    assert_eq!(
        b.bind(&key, &stale, &east).await.unwrap(),
        WriteOutcome::Conflict
    );
    assert_eq!(a.lookup(&key).await.unwrap().pool, Some(west));
}

#[tokio::test]
async fn activity_refreshes_idle_expiration_and_changes_revision() {
    let store = Arc::new(InMemoryAffinityStore::default());
    let affinity = tracker(store.clone());
    let pool = pool("east", "qwen").id;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let old = binding(&pool, "old", now - 10, now + 60);
    store
        .compare_exchange(&key(), &Expected::Absent, &old)
        .await
        .unwrap();
    let observed = affinity.lookup(&key()).await.unwrap();
    affinity.bind(&key(), &observed, &pool).await.unwrap();
    let updated = store.read(&key()).await.unwrap().unwrap();
    assert_eq!(updated.pool, old.pool);
    assert_ne!(updated.revision, old.revision);
    assert!(updated.expires_at > old.expires_at);
}

enum Failure {
    Read,
    ReadHang,
    Write,
    CommitThenHang,
    Conflict,
}

struct FaultStore {
    inner: InMemoryAffinityStore,
    failure: Failure,
    reads: AtomicUsize,
    writes: AtomicUsize,
}

impl FaultStore {
    fn new(failure: Failure) -> Arc<Self> {
        Arc::new(Self {
            inner: InMemoryAffinityStore::default(),
            failure,
            reads: AtomicUsize::new(0),
            writes: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl AffinityStore for FaultStore {
    async fn read(&self, key: &AffinityKey) -> Result<Option<AffinityBinding>, AffinityError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        match self.failure {
            Failure::Read => Err(AffinityError::Backend("injected read failure".into())),
            Failure::ReadHang => std::future::pending().await,
            _ => self.inner.read(key).await,
        }
    }

    async fn compare_exchange(
        &self,
        key: &AffinityKey,
        expected: &Expected,
        next: &AffinityBinding,
    ) -> Result<WriteOutcome, AffinityError> {
        self.writes.fetch_add(1, Ordering::Relaxed);
        match self.failure {
            Failure::Write => Err(AffinityError::Backend("injected write failure".into())),
            Failure::CommitThenHang => {
                self.inner.compare_exchange(key, expected, next).await?;
                std::future::pending().await
            }
            Failure::Conflict => Ok(WriteOutcome::Conflict),
            _ => panic!("a failed lookup must never attempt a write"),
        }
    }
}

#[tokio::test]
async fn failed_lookup_falls_back_without_writing_or_clobbering_a_binding() {
    let store = FaultStore::new(Failure::Read);
    let router = selector(store.clone(), "east", "test");
    let pools = [pool("east", "qwen"), pool("west", "qwen")];
    let original = binding(&pools[1].id, "keep-me", 1, u64::MAX);
    store
        .inner
        .compare_exchange(&key(), &Expected::Absent, &original)
        .await
        .unwrap();
    assert_eq!(pick(&router, &pools, "qwen", Some("s")).await, pools[0].id);
    assert_eq!(store.writes.load(Ordering::Relaxed), 0);
    assert_eq!(store.inner.read(&key()).await.unwrap(), Some(original));
    pick(&router, &pools, "qwen", None).await;
    assert_eq!(store.reads.load(Ordering::Relaxed), 1);
    assert!(
        router
            .select(&RoutingRequest::new("qwen").with_session(Some("s")), &[])
            .await
            .is_none()
    );
    assert_eq!(store.reads.load(Ordering::Relaxed), 1);
}

#[tokio::test(start_paused = true)]
async fn stalled_lookup_obeys_total_budget_and_never_writes() {
    let store = FaultStore::new(Failure::ReadHang);
    let router = selector(store.clone(), "east", "test");
    let start = tokio::time::Instant::now();
    let pools = [pool("east", "qwen")];
    assert_eq!(pick(&router, &pools, "qwen", Some("s")).await, pools[0].id);
    assert_eq!(start.elapsed(), AffinityConfig::default().timeout);
    assert_eq!(store.writes.load(Ordering::Relaxed), 0);
}

#[tokio::test(start_paused = true)]
async fn failed_or_ambiguous_write_keeps_selected_pool_without_rollback() {
    for failure in [Failure::Write, Failure::CommitThenHang] {
        let store = FaultStore::new(failure);
        let router = selector(store.clone(), "east", "test");
        let pools = [pool("east", "qwen"), pool("west", "qwen")];
        let original = binding(&pools[1].id, "original", 1, u64::MAX);
        store
            .inner
            .compare_exchange(&key(), &Expected::Absent, &original)
            .await
            .unwrap();
        assert_eq!(pick(&router, &pools, "qwen", Some("s")).await, pools[1].id);
        assert_eq!(store.writes.load(Ordering::Relaxed), 1);
        assert_eq!(
            store.inner.read(&key()).await.unwrap().unwrap().pool,
            pools[1].id
        );
    }
}

#[tokio::test]
async fn repeated_conflicts_have_a_fixed_retry_bound() {
    let store = FaultStore::new(Failure::Conflict);
    let router = selector(store.clone(), "east", "test");
    pick(&router, &[pool("east", "qwen")], "qwen", Some("s")).await;
    assert_eq!(store.reads.load(Ordering::Relaxed), 2);
    assert_eq!(store.writes.load(Ordering::Relaxed), 2);
}

struct FixedAffinity(PoolId);

#[async_trait]
impl SessionAffinity for FixedAffinity {
    async fn lookup(&self, _: &AffinityKey) -> Result<AffinitySnapshot, AffinityError> {
        Ok(AffinitySnapshot {
            pool: Some(self.0.clone()),
            expected: Expected::Absent,
        })
    }
    async fn bind(
        &self,
        _: &AffinityKey,
        _: &AffinitySnapshot,
        _: &PoolId,
    ) -> Result<WriteOutcome, AffinityError> {
        Ok(WriteOutcome::Applied)
    }
}

#[tokio::test]
async fn affinity_mechanism_can_be_replaced_independently_of_storage() {
    let pools = [pool("east", "qwen"), pool("west", "qwen")];
    let router = AffinitySelector::new(
        Arc::new(PoolSignals::new(Arc::new(InFlightTracker::new()))),
        "east".into(),
        2.0,
        Arc::new(FixedAffinity(pools[1].id.clone())),
        AffinityConfig::default(),
    );
    assert_eq!(pick(&router, &pools, "qwen", Some("s")).await, pools[1].id);
}

#[tokio::test]
async fn two_tenants_reusing_one_session_id_do_not_share_a_binding() {
    let store = Arc::new(InMemoryAffinityStore::default());
    let router = selector(store.clone(), "east", "test");
    let pools = [pool("east", "qwen"), pool("west", "qwen")];
    let candidates = pools.iter().collect::<Vec<_>>();
    let pick_as = async |credential: Option<&str>, candidates: &[&Pool]| {
        let tenant = tenant_id(credential);
        let request = RoutingRequest::new("qwen")
            .with_session(Some("s"))
            .with_tenant(&tenant);
        router
            .select(&request, candidates)
            .await
            .unwrap()
            .pool
            .id
            .clone()
    };
    // One tenant's session pins to east and follows that pin.
    assert_eq!(
        pick_as(Some("Bearer alice"), &candidates).await,
        pools[0].id
    );
    assert_eq!(
        pick_as(Some("Bearer alice"), &candidates[1..]).await,
        pools[1].id
    );
    assert_eq!(
        pick_as(Some("Bearer alice"), &candidates).await,
        pools[1].id
    );
    // Another tenant reusing the same session ID is placed fresh, in its own
    // binding, and does not inherit or disturb the first tenant's pin.
    assert_eq!(pick_as(Some("Bearer bob"), &candidates).await, pools[0].id);
    assert_eq!(
        pick_as(Some("Bearer alice"), &candidates).await,
        pools[1].id
    );
    // A request with no credential is a third namespace, not either tenant's.
    assert_eq!(pick_as(None, &candidates).await, pools[0].id);
    assert_eq!(store.len(), 3);
}

/// Fails every read until `recovered` is set, counting what reached it.
struct Flaky {
    inner: InMemoryAffinityStore,
    recovered: AtomicBool,
    reads: AtomicUsize,
}

impl Flaky {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: InMemoryAffinityStore::default(),
            recovered: AtomicBool::new(false),
            reads: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl AffinityStore for Flaky {
    async fn read(&self, key: &AffinityKey) -> Result<Option<AffinityBinding>, AffinityError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if self.recovered.load(Ordering::Relaxed) {
            return self.inner.read(key).await;
        }
        Err(AffinityError::Backend("injected outage".into()))
    }

    async fn compare_exchange(
        &self,
        key: &AffinityKey,
        expected: &Expected,
        binding: &AffinityBinding,
    ) -> Result<WriteOutcome, AffinityError> {
        self.inner.compare_exchange(key, expected, binding).await
    }
}

#[tokio::test(start_paused = true)]
async fn a_failing_store_is_dropped_from_the_request_path_until_a_cooldown_passes() {
    let store = Flaky::new();
    let router = selector_with(store.clone(), "east", "test", |config| {
        config.breaker = BreakerConfig {
            failures: 3,
            cooldown: Duration::from_secs(5),
        };
    });
    let pools = [pool("east", "qwen")];

    // The breaker is closed, so each request still pays for a lookup.
    for _ in 0..3 {
        pick(&router, &pools, "qwen", Some("s")).await;
    }
    assert_eq!(store.reads.load(Ordering::Relaxed), 3);

    // Open: requests route on headroom alone and stop loading the backend.
    for _ in 0..10 {
        assert_eq!(pick(&router, &pools, "qwen", Some("s")).await, pools[0].id);
    }
    assert_eq!(store.reads.load(Ordering::Relaxed), 3);

    // After the cooldown exactly one request probes, and it still fails.
    tokio::time::advance(Duration::from_secs(6)).await;
    for _ in 0..4 {
        pick(&router, &pools, "qwen", Some("s")).await;
    }
    assert_eq!(store.reads.load(Ordering::Relaxed), 4);

    // Once the store recovers, the next probe closes the breaker and every
    // request consults storage again.
    store.recovered.store(true, Ordering::Relaxed);
    tokio::time::advance(Duration::from_secs(6)).await;
    for _ in 0..3 {
        pick(&router, &pools, "qwen", Some("s")).await;
    }
    assert_eq!(store.reads.load(Ordering::Relaxed), 7);
}

/// Holds one row that no reader can decode until it is overwritten.
struct Corrupt(InMemoryAffinityStore);

#[async_trait]
impl AffinityStore for Corrupt {
    async fn read(&self, key: &AffinityKey) -> Result<Option<AffinityBinding>, AffinityError> {
        match self.0.read(key).await? {
            Some(binding) => Ok(Some(binding)),
            None => Err(AffinityError::InvalidRecord(
                "unsupported schemaVersion".into(),
            )),
        }
    }

    async fn compare_exchange(
        &self,
        key: &AffinityKey,
        expected: &Expected,
        binding: &AffinityBinding,
    ) -> Result<WriteOutcome, AffinityError> {
        self.0.compare_exchange(key, expected, binding).await
    }
}

#[tokio::test]
async fn an_undecodable_record_is_repaired_rather_than_stranding_the_session() {
    let store = Arc::new(Corrupt(InMemoryAffinityStore::default()));
    let router = selector(store.clone(), "east", "test");
    let pools = [pool("east", "qwen"), pool("west", "qwen")];

    // The row cannot be read, so the request routes on headroom...
    assert_eq!(pick(&router, &pools, "qwen", Some("s")).await, pools[0].id);
    // ...and the same request replaced it with a row this build can read,
    // rather than leaving the session fail-open until someone repairs the
    // table by hand.
    let repaired = store.0.read(&key()).await.unwrap().unwrap();
    assert_eq!(repaired.pool, pools[0].id);
    // From here the session is bound normally again.
    assert_eq!(
        pick(&router, &pools[1..], "qwen", Some("s")).await,
        pools[1].id
    );
    assert_eq!(pick(&router, &pools, "qwen", Some("s")).await, pools[1].id);
    assert_eq!(store.0.len(), 1);
}

#[tokio::test]
async fn a_cached_store_serves_repeat_requests_without_re_reading() {
    let store = Flaky::new();
    store.recovered.store(true, Ordering::Relaxed);
    let cache =
        Arc::new(CachedAffinityStore::new(store.clone(), Duration::from_secs(60), 1024).unwrap());
    let router = selector(cache, "east", "test");
    let pools = [pool("east", "qwen"), pool("west", "qwen")];
    for _ in 0..5 {
        assert_eq!(pick(&router, &pools, "qwen", Some("s")).await, pools[0].id);
    }
    // One read on the first request; every later lookup came from memory,
    // and each applied write refreshed the entry in place.
    assert_eq!(store.reads.load(Ordering::Relaxed), 1);
    assert_eq!(
        store.inner.read(&key()).await.unwrap().unwrap().pool,
        pools[0].id
    );
}

#[tokio::test]
async fn a_cache_that_missed_another_replicas_write_conflicts_once_then_agrees() {
    let backing = Arc::new(InMemoryAffinityStore::default());
    let cache =
        Arc::new(CachedAffinityStore::new(backing.clone(), Duration::from_secs(60), 1024).unwrap());
    let cached = selector(cache.clone(), "east", "test");
    let direct = selector(backing.clone(), "east", "test");
    let pools = [pool("east", "qwen"), pool("west", "qwen")];

    assert_eq!(pick(&cached, &pools, "qwen", Some("s")).await, pools[0].id);
    // Another replica moves the session while the cache still holds the old
    // revision.
    assert_eq!(
        pick(&direct, &pools[1..], "qwen", Some("s")).await,
        pools[1].id
    );
    // The stale entry loses its conditional write; the retry reads through
    // and lands on the binding that actually exists.
    assert_eq!(pick(&cached, &pools, "qwen", Some("s")).await, pools[1].id);
    assert_eq!(
        backing.read(&key()).await.unwrap().unwrap().pool,
        pools[1].id
    );
}

#[tokio::test(start_paused = true)]
async fn a_probe_whose_client_left_does_not_wedge_the_breaker_open() {
    let store = Flaky::new();
    let router = Arc::new(selector_with(store.clone(), "east", "test", |config| {
        config.breaker = BreakerConfig {
            failures: 1,
            cooldown: Duration::from_secs(5),
        };
    }));
    let pools = Arc::new([pool("east", "qwen")]);
    pick(&router, pools.as_ref(), "qwen", Some("s")).await;
    assert_eq!(store.reads.load(Ordering::Relaxed), 1);

    // After the cooldown, a request claims the probe and then goes away
    // before it can report: its client left, so the future is dropped.
    tokio::time::advance(Duration::from_secs(6)).await;
    let (probe, probe_pools) = (router.clone(), pools.clone());
    let abandoned = tokio::spawn(async move {
        let candidates = probe_pools.iter().collect::<Vec<_>>();
        probe
            .select(
                &RoutingRequest::new("qwen").with_session(Some("s")),
                &candidates,
            )
            .await
            .map(|target| target.pool.id.clone())
    });
    tokio::task::yield_now().await;
    abandoned.abort();
    assert_eq!(store.reads.load(Ordering::Relaxed), 2);

    // The breaker is still open, and no request is admitted until the next
    // cooldown: a lost probe costs one more cooldown, not the whole breaker.
    pick(&router, pools.as_ref(), "qwen", Some("s")).await;
    assert_eq!(store.reads.load(Ordering::Relaxed), 2);
    tokio::time::advance(Duration::from_secs(6)).await;
    store.recovered.store(true, Ordering::Relaxed);
    pick(&router, pools.as_ref(), "qwen", Some("s")).await;
    assert_eq!(store.reads.load(Ordering::Relaxed), 3);
}
