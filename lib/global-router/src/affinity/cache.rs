// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
// The runtime's clock, so a paused test clock expires entries as a real one does.
use tokio::time::Instant;

use super::stats::CacheStats;
use super::{AffinityBinding, AffinityError, AffinityKey, AffinityStore, Expected, WriteOutcome};

/// A read-through cache in front of a remote store, so a chatty session costs
/// one lookup per `ttl` rather than one per request.
///
/// A cached entry is what the backing store last told us, including that no
/// record existed. It is not authority. A session another replica has moved
/// keeps its old pin on this replica for at most `ttl`, since keeping a pin
/// writes nothing; soft affinity allows that, and the entry then ages out.
/// A write still carries the expectation the entry came with, so a stale entry
/// never overwrites a newer binding: it loses its conditional write, which
/// invalidates the entry and sends the retry to the store.
///
/// `ttl` is independent of a binding's own idle expiry: the entry holds
/// `expires_at`, and the tracker applies it, so a cached binding that has
/// since expired is still read as expired.
///
/// Entries are evicted by [`sweep`](Self::sweep), which the host runs off the
/// request path, and the cache refuses new keys at `capacity` rather than
/// growing without bound.
pub struct CachedAffinityStore {
    inner: Arc<dyn AffinityStore>,
    entries: DashMap<AffinityKey, Entry>,
    ttl: Duration,
    capacity: usize,
    stats: Arc<CacheStats>,
}

#[derive(Clone, Debug)]
struct Entry {
    /// What the store held: `None` is a record that was absent, cached so a
    /// new session does not re-read on every request before its first write.
    binding: Option<AffinityBinding>,
    read_at: Instant,
}

impl CachedAffinityStore {
    /// Wraps `inner`. `ttl` bounds how stale a read may be and `capacity`
    /// bounds how many sessions are held.
    pub fn new(
        inner: Arc<dyn AffinityStore>,
        ttl: Duration,
        capacity: usize,
    ) -> Result<Self, AffinityError> {
        Self::with_stats(inner, ttl, capacity, Arc::default())
    }

    /// As [`new`](Self::new), reporting into counters the host also exports.
    pub fn with_stats(
        inner: Arc<dyn AffinityStore>,
        ttl: Duration,
        capacity: usize,
        stats: Arc<CacheStats>,
    ) -> Result<Self, AffinityError> {
        if ttl.is_zero() {
            return Err(AffinityError::Configuration(
                "affinity cache TTL must be positive; omit the cache instead".into(),
            ));
        }
        if capacity == 0 {
            return Err(AffinityError::Configuration(
                "affinity cache capacity must be positive".into(),
            ));
        }
        Ok(Self {
            inner,
            entries: DashMap::new(),
            ttl,
            capacity,
            stats,
        })
    }

    /// Includes entries past their TTL until the next sweep.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Off-request-path eviction of entries past their TTL.
    pub fn sweep(&self) {
        let now = Instant::now();
        self.entries
            .retain(|_, entry| now.duration_since(entry.read_at) < self.ttl);
    }

    fn fresh(&self, key: &AffinityKey) -> Option<Option<AffinityBinding>> {
        let entry = self.entries.get(key)?;
        (entry.read_at.elapsed() < self.ttl).then(|| entry.binding.clone())
    }

    /// Records what the store last held. A key the cache does not already
    /// hold is dropped at capacity rather than evicting another session.
    fn remember(&self, key: &AffinityKey, binding: Option<AffinityBinding>) {
        let entry = Entry {
            binding,
            read_at: Instant::now(),
        };
        match self.entries.get_mut(key) {
            Some(mut existing) => *existing = entry,
            None if self.entries.len() < self.capacity => {
                self.entries.insert(key.clone(), entry);
            }
            None => {
                self.stats.invalidations.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn forget(&self, key: &AffinityKey) {
        if self.entries.remove(key).is_some() {
            self.stats.invalidations.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[async_trait]
impl AffinityStore for CachedAffinityStore {
    async fn read(&self, key: &AffinityKey) -> Result<Option<AffinityBinding>, AffinityError> {
        if let Some(binding) = self.fresh(key) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(binding);
        }
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        // An error, including an undecodable record, is never cached: the
        // caller repairs or retries against the store itself.
        let binding = self
            .inner
            .read(key)
            .await
            .inspect_err(|_| self.forget(key))?;
        self.remember(key, binding.clone());
        Ok(binding)
    }

    async fn compare_exchange(
        &self,
        key: &AffinityKey,
        expected: &Expected,
        binding: &AffinityBinding,
    ) -> Result<WriteOutcome, AffinityError> {
        match self.inner.compare_exchange(key, expected, binding).await {
            // This replica now knows the record exactly, revision included.
            Ok(WriteOutcome::Applied) => {
                self.remember(key, Some(binding.clone()));
                Ok(WriteOutcome::Applied)
            }
            // Someone else holds the record; whatever we cached is wrong, so
            // the caller's retry reads through.
            Ok(WriteOutcome::Conflict) => {
                self.forget(key);
                Ok(WriteOutcome::Conflict)
            }
            // A failed write may still have committed, so the cached entry
            // cannot be trusted either way.
            Err(error) => {
                self.forget(key);
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use dynamo_global_common::{PoolId, PoolIdDeriver, PoolKey, V1PoolIdDeriver};

    use super::super::InMemoryAffinityStore;
    use super::*;

    /// Counts what actually reached the backing store.
    struct Counting {
        inner: InMemoryAffinityStore,
        reads: AtomicUsize,
        writes: AtomicUsize,
    }

    impl Counting {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                inner: InMemoryAffinityStore::default(),
                reads: AtomicUsize::new(0),
                writes: AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl AffinityStore for Counting {
        async fn read(&self, key: &AffinityKey) -> Result<Option<AffinityBinding>, AffinityError> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.inner.read(key).await
        }

        async fn compare_exchange(
            &self,
            key: &AffinityKey,
            expected: &Expected,
            binding: &AffinityBinding,
        ) -> Result<WriteOutcome, AffinityError> {
            self.writes.fetch_add(1, Ordering::Relaxed);
            self.inner.compare_exchange(key, expected, binding).await
        }
    }

    fn key() -> AffinityKey {
        AffinityKey {
            scope: "test".into(),
            tenant: "anonymous".into(),
            policy: "headroom-v1".into(),
            model: "qwen".into(),
            session: "s".into(),
        }
    }

    fn pool() -> PoolId {
        V1PoolIdDeriver.derive(&PoolKey::new("east", "dynamo", "blue").unwrap())
    }

    fn binding(revision: &str) -> AffinityBinding {
        AffinityBinding {
            pool: pool(),
            revision: revision.into(),
            updated_at: 1,
            expires_at: u64::MAX,
        }
    }

    fn cache(inner: Arc<dyn AffinityStore>, ttl: Duration) -> CachedAffinityStore {
        CachedAffinityStore::new(inner, ttl, 8).unwrap()
    }

    #[tokio::test]
    async fn repeat_reads_within_the_ttl_do_not_reach_the_store() {
        let store = Counting::new();
        let cache = cache(store.clone(), Duration::from_secs(60));
        for _ in 0..5 {
            assert!(cache.read(&key()).await.unwrap().is_none());
        }
        // Including the absence of a record, so a new session reads once.
        assert_eq!(store.reads.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn an_applied_write_is_readable_without_another_lookup() {
        let store = Counting::new();
        let cache = cache(store.clone(), Duration::from_secs(60));
        assert_eq!(
            cache
                .compare_exchange(&key(), &Expected::Absent, &binding("one"))
                .await
                .unwrap(),
            WriteOutcome::Applied
        );
        assert_eq!(cache.read(&key()).await.unwrap().unwrap().revision, "one");
        assert_eq!(store.reads.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn a_conflict_invalidates_so_the_retry_reads_through() {
        let store = Counting::new();
        let cache = cache(store.clone(), Duration::from_secs(60));
        cache.read(&key()).await.unwrap();
        // Another replica creates the record behind the cache's back.
        store
            .inner
            .compare_exchange(&key(), &Expected::Absent, &binding("theirs"))
            .await
            .unwrap();
        // The stale "absent" entry loses its conditional write...
        assert_eq!(
            cache
                .compare_exchange(&key(), &Expected::Absent, &binding("ours"))
                .await
                .unwrap(),
            WriteOutcome::Conflict
        );
        // ...and the next read sees the truth rather than the stale entry.
        assert_eq!(
            cache.read(&key()).await.unwrap().unwrap().revision,
            "theirs"
        );
        assert_eq!(store.reads.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn a_failed_write_is_never_left_cached() {
        struct Failing;

        #[async_trait]
        impl AffinityStore for Failing {
            async fn read(
                &self,
                _: &AffinityKey,
            ) -> Result<Option<AffinityBinding>, AffinityError> {
                Ok(None)
            }

            async fn compare_exchange(
                &self,
                _: &AffinityKey,
                _: &Expected,
                _: &AffinityBinding,
            ) -> Result<WriteOutcome, AffinityError> {
                Err(AffinityError::Backend("down".into()))
            }
        }

        let cache = cache(Arc::new(Failing), Duration::from_secs(60));
        cache.read(&key()).await.unwrap();
        assert_eq!(cache.len(), 1);
        assert!(
            cache
                .compare_exchange(&key(), &Expected::Absent, &binding("ours"))
                .await
                .is_err()
        );
        assert!(cache.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn an_entry_past_its_ttl_is_re_read_and_swept() {
        let store = Counting::new();
        let cache = cache(store.clone(), Duration::from_millis(50));
        cache.read(&key()).await.unwrap();
        tokio::time::advance(Duration::from_millis(80)).await;
        cache.read(&key()).await.unwrap();
        assert_eq!(store.reads.load(Ordering::Relaxed), 2);
        tokio::time::advance(Duration::from_millis(80)).await;
        cache.sweep();
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn a_full_cache_keeps_serving_without_growing() {
        let store = Counting::new();
        let cache = CachedAffinityStore::new(store.clone(), Duration::from_secs(60), 2).unwrap();
        for session in 0..5 {
            let mut key = key();
            key.session = session.to_string();
            assert!(cache.read(&key).await.unwrap().is_none());
        }
        assert_eq!(cache.len(), 2);
        // Every read still answered, the three uncached ones from the store.
        assert_eq!(store.reads.load(Ordering::Relaxed), 5);
    }

    #[tokio::test]
    async fn a_cached_read_never_outlives_the_binding_it_holds() {
        let store = Counting::new();
        let cache = cache(store.clone(), Duration::from_secs(60));
        let mut expiring = binding("one");
        expiring.expires_at = 1;
        store
            .inner
            .compare_exchange(&key(), &Expected::Absent, &expiring)
            .await
            .unwrap();
        // The cache hands back `expires_at` unchanged; judging it is the
        // tracker's job, so a long TTL cannot resurrect a dead binding.
        assert_eq!(cache.read(&key()).await.unwrap().unwrap().expires_at, 1);
    }

    #[tokio::test]
    async fn an_unreadable_record_is_not_cached_as_absent() {
        struct Corrupt;

        #[async_trait]
        impl AffinityStore for Corrupt {
            async fn read(
                &self,
                _: &AffinityKey,
            ) -> Result<Option<AffinityBinding>, AffinityError> {
                Err(AffinityError::InvalidRecord("schemaVersion".into()))
            }

            async fn compare_exchange(
                &self,
                _: &AffinityKey,
                _: &Expected,
                _: &AffinityBinding,
            ) -> Result<WriteOutcome, AffinityError> {
                Ok(WriteOutcome::Applied)
            }
        }

        let cache = cache(Arc::new(Corrupt), Duration::from_secs(60));
        assert!(matches!(
            cache.read(&key()).await,
            Err(AffinityError::InvalidRecord(_))
        ));
        assert!(cache.is_empty());
    }
}
