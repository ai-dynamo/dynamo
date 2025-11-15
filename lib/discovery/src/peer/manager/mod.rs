// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod local;

use anyhow::Result;
use dashmap::DashMap;
use futures::future::{self, BoxFuture, Either, FutureExt, Ready, Shared};
use std::sync::Arc;

use local::LocalPeerDiscovery;

use crate::peer::{DiscoveryQueryError, InstanceId, PeerDiscovery, PeerInfo, WorkerId};

type QueryResult = Result<PeerInfo, DiscoveryQueryError>;
type MaybeAsyncQueryResult = Either<Ready<QueryResult>, Shared<BoxFuture<'static, QueryResult>>>;

/// Cache of shared query futures to deduplicate concurrent remote lookups.
///
/// Shared futures are kept permanently to eliminate race conditions.
/// Memory is bounded by unique peer count, and PeerInfo is cheap to clone.
#[derive(Debug, Default)]
struct PendingQueries {
    by_worker_id: DashMap<WorkerId, Shared<BoxFuture<'static, QueryResult>>>,
    by_instance_id: DashMap<InstanceId, Shared<BoxFuture<'static, QueryResult>>>,
}

#[derive(Debug)]
pub struct PeerDiscoveryManager {
    local: LocalPeerDiscovery,
    remotes: Vec<Arc<dyn PeerDiscovery>>,
    pending: Arc<PendingQueries>,
}

impl PeerDiscoveryManager {
    pub async fn new(
        local_peer: Option<PeerInfo>,
        sources: Vec<Arc<dyn PeerDiscovery>>,
    ) -> Result<Self> {
        let local = LocalPeerDiscovery::default();

        if let Some(local_peer) = &local_peer {
            let instance_id = local_peer.instance_id;
            let worker_address = local_peer.worker_address.clone();

            // register local peer with local discovery
            local
                .register_instance(instance_id, worker_address.clone())
                .map_err(|e| anyhow::anyhow!("Failed to register local peer: {}", e))?;

            // register local peer with remote discoveries
            for remote in &sources {
                remote
                    .register_instance(instance_id, worker_address.clone())
                    .await?;
            }
        }

        // TODO: Unregister local peer and remotes when the manager is dropped
        // Since drop is not async, we'll need to create a task to unregister the remote instances and
        // trigger that task during the drop implementation.

        Ok(Self {
            local,
            remotes: sources,
            pending: Arc::new(PendingQueries::default()),
        })
    }

    pub async fn discover_by_worker_id(&self, worker_id: WorkerId) -> MaybeAsyncQueryResult {
        // Fast path: check local cache
        if let Ok(peer) = self.local.discover_by_worker_id(worker_id) {
            return Either::Left(future::ready(Ok(peer)));
        }

        if self.remotes.is_empty() {
            return Either::Left(future::ready(Err(DiscoveryQueryError::NotFound)));
        }

        // Check if there's already a pending query for this worker_id
        if let Some(shared_future) = self.pending.by_worker_id.get(&worker_id) {
            return Either::Right(shared_future.clone());
        }

        // Create a new shared future for this query
        let local = self.local.clone();
        let remotes = self.remotes.clone();
        let pending = self.pending.clone();

        use dashmap::mapref::entry::Entry;
        let shared_future = match self.pending.by_worker_id.entry(worker_id) {
            Entry::Occupied(entry) => {
                // Another thread beat us to it, use their future
                entry.get().clone()
            }
            Entry::Vacant(entry) => {
                // We're the first, create the shared future
                let shared = async move {
                    // Query remotes sequentially
                    for remote in &remotes {
                        match remote.discover_by_worker_id(worker_id).await {
                            Ok(peer_info) => {
                                // Cache the result in local store (ignore errors)
                                if let Err(e) = local.register_instance(
                                    peer_info.instance_id,
                                    peer_info.worker_address.clone(),
                                ) {
                                    tracing::debug!(
                                        "Failed to register peer info in local store: {}",
                                        e
                                    );
                                }
                                return Ok(peer_info);
                            }
                            Err(DiscoveryQueryError::NotFound) => continue,
                            Err(e) => {
                                // Clean up failed future from cache to allow retry
                                pending.by_worker_id.remove(&worker_id);
                                return Err(e);
                            }
                        }
                    }
                    // Clean up NotFound result from cache to allow retry
                    pending.by_worker_id.remove(&worker_id);
                    Err(DiscoveryQueryError::NotFound)
                }
                .boxed()
                .shared();

                entry.insert(shared.clone());
                shared
            }
        };

        Either::Right(shared_future)
    }

    pub async fn discover_by_instance_id(&self, instance_id: InstanceId) -> MaybeAsyncQueryResult {
        // Fast path: check local cache
        if let Ok(peer) = self.local.discover_by_instance_id(instance_id) {
            return Either::Left(future::ready(Ok(peer)));
        }

        // Check if there's already a pending query for this instance_id
        if let Some(shared_future) = self.pending.by_instance_id.get(&instance_id) {
            return Either::Right(shared_future.clone());
        }

        // Create a new shared future for this query
        let local = self.local.clone();
        let remotes = self.remotes.clone();
        let pending = self.pending.clone();

        use dashmap::mapref::entry::Entry;
        let shared_future = match self.pending.by_instance_id.entry(instance_id) {
            Entry::Occupied(entry) => {
                // Another thread beat us to it, use their future
                entry.get().clone()
            }
            Entry::Vacant(entry) => {
                // We're the first, create the shared future
                let shared = async move {
                    // Query remotes sequentially
                    for remote in &remotes {
                        match remote.discover_by_instance_id(instance_id).await {
                            Ok(peer_info) => {
                                // Cache the result in local store (ignore errors)
                                if let Err(e) = local.register_instance(
                                    peer_info.instance_id,
                                    peer_info.worker_address.clone(),
                                ) {
                                    tracing::debug!(
                                        "Failed to register peer info in local store: {}",
                                        e
                                    );
                                }
                                return Ok(peer_info);
                            }
                            Err(DiscoveryQueryError::NotFound) => continue,
                            Err(e) => {
                                // Clean up failed future from cache to allow retry
                                pending.by_instance_id.remove(&instance_id);
                                return Err(e);
                            }
                        }
                    }
                    // Clean up NotFound result from cache to allow retry
                    pending.by_instance_id.remove(&instance_id);
                    Err(DiscoveryQueryError::NotFound)
                }
                .boxed()
                .shared();

                entry.insert(shared.clone());
                shared
            }
        };

        Either::Right(shared_future)
    }
}

#[cfg(test)]
mod tests {
    use crate::peer::{DiscoveryError, WorkerAddress};

    use super::*;
    use bytes::Bytes;
    use parking_lot::Mutex as StdMutex;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::{Barrier, Notify};

    // Test timeout duration
    const TEST_TIMEOUT: Duration = Duration::from_secs(5);

    fn make_test_address() -> WorkerAddress {
        WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:5555"))
    }

    // ============================================================================
    // Mock Discovery Infrastructure
    // ============================================================================

    /// Improved mock discovery with pre-configured responses and proper synchronization.
    #[derive(Debug, Clone)]
    struct MockDiscovery {
        inner: Arc<MockDiscoveryInner>,
    }

    #[derive(Debug)]
    struct MockDiscoveryInner {
        // Track call counts
        worker_id_calls: AtomicUsize,
        instance_id_calls: AtomicUsize,
        register_calls: AtomicUsize,
        unregister_calls: AtomicUsize,

        // Pre-configured responses
        worker_responses: StdMutex<HashMap<WorkerId, QueryResult>>,
        instance_responses: StdMutex<HashMap<InstanceId, QueryResult>>,

        // Notification for test synchronization
        worker_call_notify: Arc<Notify>,
        instance_call_notify: Arc<Notify>,

        // Control whether to return immediately or simulate delay
        simulate_delay: AtomicBool,
        delay_duration: StdMutex<Duration>,
    }

    impl MockDiscovery {
        fn new() -> Self {
            Self {
                inner: Arc::new(MockDiscoveryInner {
                    worker_id_calls: AtomicUsize::new(0),
                    instance_id_calls: AtomicUsize::new(0),
                    register_calls: AtomicUsize::new(0),
                    unregister_calls: AtomicUsize::new(0),
                    worker_responses: StdMutex::new(HashMap::new()),
                    instance_responses: StdMutex::new(HashMap::new()),
                    worker_call_notify: Arc::new(Notify::new()),
                    instance_call_notify: Arc::new(Notify::new()),
                    simulate_delay: AtomicBool::new(false),
                    delay_duration: StdMutex::new(Duration::from_millis(100)),
                }),
            }
        }

        /// Set response for a specific worker_id (must be called before query)
        fn set_worker_response(&self, worker_id: WorkerId, result: QueryResult) {
            self.inner.worker_responses.lock().insert(worker_id, result);
        }

        /// Set response for a specific instance_id (must be called before query)
        fn set_instance_response(&self, instance_id: InstanceId, result: QueryResult) {
            self.inner
                .instance_responses
                .lock()
                .insert(instance_id, result);
        }

        /// Enable simulated delay for responses
        fn enable_delay(&self, duration: Duration) {
            *self.inner.delay_duration.lock() = duration;
            self.inner.simulate_delay.store(true, Ordering::SeqCst);
        }

        /// Get call counts
        fn worker_id_call_count(&self) -> usize {
            self.inner.worker_id_calls.load(Ordering::SeqCst)
        }

        fn instance_id_call_count(&self) -> usize {
            self.inner.instance_id_calls.load(Ordering::SeqCst)
        }

        fn register_call_count(&self) -> usize {
            self.inner.register_calls.load(Ordering::SeqCst)
        }

        #[allow(dead_code)]
        fn unregister_call_count(&self) -> usize {
            self.inner.unregister_calls.load(Ordering::SeqCst)
        }

        /// Wait for at least N worker_id queries to be made
        #[allow(dead_code)]
        async fn wait_for_worker_calls(&self, min_calls: usize) {
            loop {
                if self.worker_id_call_count() >= min_calls {
                    return;
                }
                // Subscribe BEFORE checking again to avoid race
                let notified = self.inner.worker_call_notify.notified();
                if self.worker_id_call_count() >= min_calls {
                    return;
                }
                notified.await;
            }
        }

        /// Wait for at least N instance_id queries to be made
        #[allow(dead_code)]
        async fn wait_for_instance_calls(&self, min_calls: usize) {
            loop {
                if self.instance_id_call_count() >= min_calls {
                    return;
                }
                // Subscribe BEFORE checking again to avoid race
                let notified = self.inner.instance_call_notify.notified();
                if self.instance_id_call_count() >= min_calls {
                    return;
                }
                notified.await;
            }
        }
    }

    impl PeerDiscovery for MockDiscovery {
        fn discover_by_worker_id(
            &self,
            worker_id: WorkerId,
        ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>> {
            self.inner.worker_id_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.worker_call_notify.notify_waiters();

            let result = self
                .inner
                .worker_responses
                .lock()
                .get(&worker_id)
                .cloned()
                .unwrap_or(Err(DiscoveryQueryError::NotFound));

            let should_delay = self.inner.simulate_delay.load(Ordering::SeqCst);
            let delay = *self.inner.delay_duration.lock();

            Box::pin(async move {
                if should_delay {
                    tokio::time::sleep(delay).await;
                }
                result
            })
        }

        fn discover_by_instance_id(
            &self,
            instance_id: InstanceId,
        ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>> {
            self.inner.instance_id_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.instance_call_notify.notify_waiters();

            let result = self
                .inner
                .instance_responses
                .lock()
                .get(&instance_id)
                .cloned()
                .unwrap_or(Err(DiscoveryQueryError::NotFound));

            let should_delay = self.inner.simulate_delay.load(Ordering::SeqCst);
            let delay = *self.inner.delay_duration.lock();

            Box::pin(async move {
                if should_delay {
                    tokio::time::sleep(delay).await;
                }
                result
            })
        }

        fn register_instance(
            &self,
            _instance_id: InstanceId,
            _worker_address: WorkerAddress,
        ) -> BoxFuture<'static, Result<(), DiscoveryError>> {
            self.inner.register_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { Ok(()) })
        }

        fn unregister_instance(
            &self,
            _instance_id: InstanceId,
        ) -> BoxFuture<'static, Result<(), DiscoveryError>> {
            self.inner.unregister_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { Ok(()) })
        }
    }

    // ============================================================================
    // PeerDiscoveryManager Tests
    // ============================================================================

    #[tokio::test]
    async fn test_manager_local_cache_hit() {
        let result = tokio::time::timeout(TEST_TIMEOUT, async {
            let local_instance = InstanceId::new_v4();
            let local_address = make_test_address();
            let local_peer = PeerInfo::new(local_instance, local_address.clone());

            let manager = PeerDiscoveryManager::new(Some(local_peer.clone()), vec![])
                .await
                .unwrap();

            // Query should hit local cache immediately
            let result = manager
                .discover_by_worker_id(local_instance.worker_id())
                .await;

            match result {
                Either::Left(ready) => {
                    let peer = ready.into_inner().unwrap();
                    assert_eq!(peer.instance_id(), local_instance);
                    assert_eq!(peer.worker_address(), &local_address);
                }
                Either::Right(_) => panic!("Expected immediate ready future, got async"),
            }
        })
        .await;

        assert!(result.is_ok(), "Test timed out after {:?}", TEST_TIMEOUT);
    }

    #[tokio::test]
    async fn test_manager_no_remotes_returns_not_found() {
        let result = tokio::time::timeout(TEST_TIMEOUT, async {
            let local_instance = InstanceId::new_v4();
            let local_address = make_test_address();
            let local_peer = PeerInfo::new(local_instance, local_address);

            let manager = PeerDiscoveryManager::new(Some(local_peer), vec![])
                .await
                .unwrap();

            // Query for unknown worker_id with no remotes
            let unknown_worker_id = WorkerId::from_u64(999);
            let result = manager.discover_by_worker_id(unknown_worker_id).await;

            match result {
                Either::Left(ready) => {
                    let err = ready.into_inner().unwrap_err();
                    assert!(matches!(err, DiscoveryQueryError::NotFound));
                }
                Either::Right(_) => panic!("Expected immediate not found, got async"),
            }
        })
        .await;

        assert!(result.is_ok(), "Test timed out after {:?}", TEST_TIMEOUT);
    }

    #[tokio::test]
    async fn test_manager_remote_query_on_miss() {
        let result = tokio::time::timeout(TEST_TIMEOUT, async {
            let local_instance = InstanceId::new_v4();
            let local_address = make_test_address();
            let local_peer = PeerInfo::new(local_instance, local_address);

            let mock = Arc::new(MockDiscovery::new());
            let query_worker_id = WorkerId::from_u64(42);
            let remote_instance = InstanceId::new_v4();
            let remote_address = make_test_address();
            let remote_peer = PeerInfo::new(remote_instance, remote_address.clone());

            // Pre-configure mock response
            mock.set_worker_response(query_worker_id, Ok(remote_peer.clone()));

            let manager = PeerDiscoveryManager::new(
                Some(local_peer),
                vec![mock.clone() as Arc<dyn PeerDiscovery>],
            )
            .await
            .unwrap();

            // Query should go to remote
            let result = manager.discover_by_worker_id(query_worker_id).await;

            match result {
                Either::Right(fut) => {
                    let peer = fut.await.unwrap();
                    assert_eq!(peer.instance_id(), remote_instance);
                    assert_eq!(peer.worker_address(), &remote_address);
                }
                Either::Left(_) => panic!("Expected async future for remote query"),
            }

            // Verify mock was called
            assert_eq!(mock.worker_id_call_count(), 1);
        })
        .await;

        assert!(result.is_ok(), "Test timed out after {:?}", TEST_TIMEOUT);
    }

    #[tokio::test]
    async fn test_manager_concurrent_deduplication_worker_id() {
        let result = tokio::time::timeout(TEST_TIMEOUT, async {
            let local_instance = InstanceId::new_v4();
            let local_address = make_test_address();
            let local_peer = PeerInfo::new(local_instance, local_address);

            let mock = Arc::new(MockDiscovery::new());
            let query_worker_id = WorkerId::from_u64(42);
            let peer_instance = InstanceId::new_v4();
            let peer_address = make_test_address();
            let peer_info = PeerInfo::new(peer_instance, peer_address);

            // Pre-configure mock response with delay
            mock.set_worker_response(query_worker_id, Ok(peer_info.clone()));
            mock.enable_delay(Duration::from_millis(100));

            let manager = Arc::new(
                PeerDiscoveryManager::new(
                    Some(local_peer),
                    vec![mock.clone() as Arc<dyn PeerDiscovery>],
                )
                .await
                .unwrap(),
            );

            // Use barrier to synchronize query starts
            let barrier = Arc::new(Barrier::new(11)); // 10 queries + main thread
            let mut handles = vec![];

            for _ in 0..10 {
                let mgr = manager.clone();
                let bar = barrier.clone();
                handles.push(tokio::spawn(async move {
                    bar.wait().await;
                    let maybe_async = mgr.discover_by_worker_id(query_worker_id).await;
                    // Actually await the future to trigger the remote call
                    match maybe_async {
                        Either::Right(fut) => Either::Right(fut.await),
                        Either::Left(ready) => Either::Left(ready.into_inner()),
                    }
                }));
            }

            // Start all queries simultaneously
            barrier.wait().await;

            // Give tasks time to start polling the shared future
            tokio::time::sleep(Duration::from_millis(50)).await;

            // Verify deduplication: only ONE remote call despite 10 concurrent queries
            assert_eq!(
                mock.worker_id_call_count(),
                1,
                "Deduplication failed: mock was called more than once"
            );

            // All 10 queries should eventually succeed with same result
            for handle in handles {
                let query_result = handle.await.unwrap();
                match query_result {
                    Either::Right(result) => {
                        let peer = result.unwrap();
                        assert_eq!(peer.instance_id(), peer_instance);
                    }
                    Either::Left(result) => {
                        let peer = result.unwrap();
                        assert_eq!(peer.instance_id(), peer_instance);
                    }
                }
            }
        })
        .await;

        assert!(result.is_ok(), "Test timed out after {:?}", TEST_TIMEOUT);
    }

    #[tokio::test]
    async fn test_manager_concurrent_deduplication_instance_id() {
        let result = tokio::time::timeout(TEST_TIMEOUT, async {
            let local_instance = InstanceId::new_v4();
            let local_address = make_test_address();
            let local_peer = PeerInfo::new(local_instance, local_address);

            let mock = Arc::new(MockDiscovery::new());
            let query_instance_id = InstanceId::new_v4();
            let peer_address = make_test_address();
            let peer_info = PeerInfo::new(query_instance_id, peer_address);

            // Pre-configure mock response with delay
            mock.set_instance_response(query_instance_id, Ok(peer_info.clone()));
            mock.enable_delay(Duration::from_millis(100));

            let manager = Arc::new(
                PeerDiscoveryManager::new(
                    Some(local_peer),
                    vec![mock.clone() as Arc<dyn PeerDiscovery>],
                )
                .await
                .unwrap(),
            );

            // Use barrier to synchronize
            let barrier = Arc::new(Barrier::new(11));
            let mut handles = vec![];

            for _ in 0..10 {
                let mgr = manager.clone();
                let bar = barrier.clone();
                handles.push(tokio::spawn(async move {
                    bar.wait().await;
                    let maybe_async = mgr.discover_by_instance_id(query_instance_id).await;
                    // Actually await the future to trigger the remote call
                    match maybe_async {
                        Either::Right(fut) => Either::Right(fut.await),
                        Either::Left(ready) => Either::Left(ready.into_inner()),
                    }
                }));
            }

            barrier.wait().await;

            // Give tasks time to start polling the shared future
            tokio::time::sleep(Duration::from_millis(50)).await;

            // Verify deduplication
            assert_eq!(
                mock.instance_id_call_count(),
                1,
                "Deduplication failed for instance_id queries"
            );

            for handle in handles {
                let query_result = handle.await.unwrap();
                match query_result {
                    Either::Right(result) => {
                        let peer = result.unwrap();
                        assert_eq!(peer.instance_id(), query_instance_id);
                    }
                    Either::Left(result) => {
                        let peer = result.unwrap();
                        assert_eq!(peer.instance_id(), query_instance_id);
                    }
                }
            }
        })
        .await;

        assert!(result.is_ok(), "Test timed out after {:?}", TEST_TIMEOUT);
    }

    #[tokio::test]
    async fn test_manager_different_ids_independent() {
        let result = tokio::time::timeout(TEST_TIMEOUT, async {
            let local_instance = InstanceId::new_v4();
            let local_address = make_test_address();
            let local_peer = PeerInfo::new(local_instance, local_address);

            let mock = Arc::new(MockDiscovery::new());

            let worker_id_1 = WorkerId::from_u64(1);
            let worker_id_2 = WorkerId::from_u64(2);
            let peer1 = PeerInfo::new(InstanceId::new_v4(), make_test_address());
            let peer2 = PeerInfo::new(InstanceId::new_v4(), make_test_address());

            // Pre-configure responses
            mock.set_worker_response(worker_id_1, Ok(peer1.clone()));
            mock.set_worker_response(worker_id_2, Ok(peer2.clone()));

            let manager = Arc::new(
                PeerDiscoveryManager::new(
                    Some(local_peer),
                    vec![mock.clone() as Arc<dyn PeerDiscovery>],
                )
                .await
                .unwrap(),
            );

            // Query both IDs concurrently
            let mgr1 = manager.clone();
            let mgr2 = manager.clone();

            let handle1 =
                tokio::spawn(async move { mgr1.discover_by_worker_id(worker_id_1).await });
            let handle2 =
                tokio::spawn(async move { mgr2.discover_by_worker_id(worker_id_2).await });

            let (result1, result2) = tokio::join!(handle1, handle2);
            let query1 = result1.unwrap();
            let query2 = result2.unwrap();

            // Both should be async futures (remote queries)
            match (query1, query2) {
                (Either::Right(fut1), Either::Right(fut2)) => {
                    let p1 = fut1.await.unwrap();
                    let p2 = fut2.await.unwrap();
                    assert_eq!(p1.instance_id(), peer1.instance_id());
                    assert_eq!(p2.instance_id(), peer2.instance_id());
                    assert_ne!(p1.instance_id(), p2.instance_id());
                }
                _ => panic!("Expected async futures for both queries"),
            }

            // Each ID should have triggered one call (no cross-deduplication)
            assert_eq!(mock.worker_id_call_count(), 2);
        })
        .await;

        assert!(result.is_ok(), "Test timed out after {:?}", TEST_TIMEOUT);
    }

    #[tokio::test]
    async fn test_manager_sequential_remote_fallback() {
        let result = tokio::time::timeout(TEST_TIMEOUT, async {
            let local_instance = InstanceId::new_v4();
            let local_address = make_test_address();
            let local_peer = PeerInfo::new(local_instance, local_address);

            let mock1 = Arc::new(MockDiscovery::new());
            let mock2 = Arc::new(MockDiscovery::new());

            let query_worker_id = WorkerId::from_u64(42);
            let peer_info = PeerInfo::new(InstanceId::new_v4(), make_test_address());

            // First mock returns NotFound, second succeeds
            mock1.set_worker_response(query_worker_id, Err(DiscoveryQueryError::NotFound));
            mock2.set_worker_response(query_worker_id, Ok(peer_info.clone()));

            let manager = PeerDiscoveryManager::new(
                Some(local_peer),
                vec![
                    mock1.clone() as Arc<dyn PeerDiscovery>,
                    mock2.clone() as Arc<dyn PeerDiscovery>,
                ],
            )
            .await
            .unwrap();

            let result = manager.discover_by_worker_id(query_worker_id).await;

            match result {
                Either::Right(fut) => {
                    let peer = fut.await.unwrap();
                    assert_eq!(peer.instance_id(), peer_info.instance_id());
                }
                Either::Left(_) => panic!("Expected async future"),
            }

            // Both mocks should have been called (fallback)
            assert_eq!(mock1.worker_id_call_count(), 1);
            assert_eq!(mock2.worker_id_call_count(), 1);
        })
        .await;

        assert!(result.is_ok(), "Test timed out after {:?}", TEST_TIMEOUT);
    }

    #[tokio::test]
    async fn test_manager_all_remotes_fail() {
        let result = tokio::time::timeout(TEST_TIMEOUT, async {
            let local_instance = InstanceId::new_v4();
            let local_address = make_test_address();
            let local_peer = PeerInfo::new(local_instance, local_address);

            let mock1 = Arc::new(MockDiscovery::new());
            let mock2 = Arc::new(MockDiscovery::new());

            let query_worker_id = WorkerId::from_u64(42);

            // Both mocks return NotFound
            mock1.set_worker_response(query_worker_id, Err(DiscoveryQueryError::NotFound));
            mock2.set_worker_response(query_worker_id, Err(DiscoveryQueryError::NotFound));

            let manager = PeerDiscoveryManager::new(
                Some(local_peer),
                vec![
                    mock1.clone() as Arc<dyn PeerDiscovery>,
                    mock2.clone() as Arc<dyn PeerDiscovery>,
                ],
            )
            .await
            .unwrap();

            let result = manager.discover_by_worker_id(query_worker_id).await;

            match result {
                Either::Right(fut) => {
                    let err = fut.await.unwrap_err();
                    assert!(matches!(err, DiscoveryQueryError::NotFound));
                }
                Either::Left(_) => panic!("Expected async future"),
            }

            // Both should have been tried
            assert_eq!(mock1.worker_id_call_count(), 1);
            assert_eq!(mock2.worker_id_call_count(), 1);
        })
        .await;

        assert!(result.is_ok(), "Test timed out after {:?}", TEST_TIMEOUT);
    }

    #[tokio::test]
    async fn test_manager_cache_population_after_remote_success() {
        let result = tokio::time::timeout(TEST_TIMEOUT, async {
            let local_instance = InstanceId::new_v4();
            let local_address = make_test_address();
            let local_peer = PeerInfo::new(local_instance, local_address);

            let mock = Arc::new(MockDiscovery::new());
            let query_worker_id = WorkerId::from_u64(42);
            let remote_instance = InstanceId::new_v4();
            let remote_peer = PeerInfo::new(remote_instance, make_test_address());

            mock.set_worker_response(query_worker_id, Ok(remote_peer.clone()));

            let manager = Arc::new(
                PeerDiscoveryManager::new(
                    Some(local_peer),
                    vec![mock.clone() as Arc<dyn PeerDiscovery>],
                )
                .await
                .unwrap(),
            );

            // First query - goes to remote
            let result1 = manager.discover_by_worker_id(query_worker_id).await;
            match result1 {
                Either::Right(fut) => {
                    let peer = fut.await.unwrap();
                    assert_eq!(peer.instance_id(), remote_instance);
                }
                Either::Left(_) => panic!("Expected async future"),
            }

            // Give time for caching
            tokio::time::sleep(Duration::from_millis(100)).await;

            // Second query - should hit local cache OR shared future cache
            let result2 = manager.discover_by_worker_id(query_worker_id).await;
            match result2 {
                Either::Left(ready) => {
                    // Cache hit!
                    let peer = ready.into_inner().unwrap();
                    assert_eq!(peer.instance_id(), remote_instance);
                }
                Either::Right(fut) => {
                    // Shared future cache (also valid)
                    let peer = fut.await.unwrap();
                    assert_eq!(peer.instance_id(), remote_instance);
                }
            }

            // Mock should have been called only once (not twice)
            assert_eq!(mock.worker_id_call_count(), 1);
        })
        .await;

        assert!(result.is_ok(), "Test timed out after {:?}", TEST_TIMEOUT);
    }

    #[tokio::test]
    async fn test_manager_register_propagates_to_remotes() {
        let result = tokio::time::timeout(TEST_TIMEOUT, async {
            let local_instance = InstanceId::new_v4();
            let local_address = make_test_address();
            let local_peer = PeerInfo::new(local_instance, local_address);

            let mock1 = Arc::new(MockDiscovery::new());
            let mock2 = Arc::new(MockDiscovery::new());

            // Creating the manager already calls register on remotes (for local_peer)
            let _manager = PeerDiscoveryManager::new(
                Some(local_peer),
                vec![
                    mock1.clone() as Arc<dyn PeerDiscovery>,
                    mock2.clone() as Arc<dyn PeerDiscovery>,
                ],
            )
            .await
            .unwrap();

            // Both remotes should have received register call
            assert_eq!(mock1.register_call_count(), 1);
            assert_eq!(mock2.register_call_count(), 1);
        })
        .await;

        assert!(result.is_ok(), "Test timed out after {:?}", TEST_TIMEOUT);
    }
}
