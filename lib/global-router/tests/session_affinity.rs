// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use dynamo_global_common::{PoolId, PoolKey};
use dynamo_global_router::affinity::{
    AffinityBinding, AffinityConfig, AffinityError, AffinityKey, AffinitySnapshot, AffinityStore,
    InMemoryAffinityStore, SessionAffinity, StoreBackedSessionAffinity, WriteOutcome,
};
use dynamo_global_router::{
    AffinitySelector, InFlightTracker, Pool, PoolSelector, PoolSignals, RoutingRequest,
};

fn key() -> AffinityKey {
    AffinityKey {
        scope: "test".into(),
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
    AffinitySelector::new(
        Arc::new(PoolSignals::new(Arc::new(InFlightTracker::new()))),
        region.into(),
        2.0,
        tracker(store),
        AffinityConfig {
            scope: scope.into(),
            ..Default::default()
        },
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
    let mut other_policy = key();
    other_policy.policy = "new-policy".into();
    assert!(store.read(&other_policy).await.unwrap().is_none());
    assert_eq!(store.len(), 3);
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
        .compare_exchange(&key(), None, &expired)
        .await
        .unwrap();
    let affinity = tracker(store.clone());
    let observed = affinity.lookup(&key()).await.unwrap();
    assert_eq!(observed.pool, None);
    assert_eq!(observed.revision.as_deref(), Some("expired"));
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
            .compare_exchange(&key(), Some("expired"), &expired)
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
    store.compare_exchange(&key(), None, &old).await.unwrap();
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
        expected: Option<&str>,
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
        .compare_exchange(&key(), None, &original)
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
            .compare_exchange(&key(), None, &original)
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

/// Demonstrates replacing the whole affinity mechanism without any store.
struct FixedAffinity(PoolId);

#[async_trait]
impl SessionAffinity for FixedAffinity {
    async fn lookup(&self, _: &AffinityKey) -> Result<AffinitySnapshot, AffinityError> {
        Ok(AffinitySnapshot {
            pool: Some(self.0.clone()),
            revision: None,
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
