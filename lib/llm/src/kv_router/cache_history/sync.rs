// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    future::Future,
    sync::{Arc, Weak},
    time::Duration,
};

use dynamo_runtime::{
    component::Endpoint,
    traits::DistributedRuntimeProvider,
    transports::event_plane::{EventPublisher, EventSubscriber},
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::CacheHistory;
use crate::kv_router::{KvRouter, metrics::RouterRequestMetrics};

const SUBJECT: &str = "cache-history-v1";
const MAX_BATCH_HASHES: usize = 4096;
const QUEUE_BATCHES: usize = 256;
const INITIAL_RETRY_DELAY: Duration = Duration::from_millis(100);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(5);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct Domain {
    model: String,
    block_size: u32,
    is_eagle: bool,
}

#[derive(Serialize, Deserialize)]
struct CompletedHashes {
    source: String,
    domain: Domain,
    hashes: Vec<u64>,
}

fn apply_peer(
    history: &CacheHistory,
    source: &str,
    domain: &Domain,
    event: CompletedHashes,
) -> Option<super::CacheHistoryStats> {
    if event.source == source || event.domain != *domain || event.hashes.len() > MAX_BATCH_HASHES {
        return None;
    }
    history.insert(event.hashes.into_iter())
}

pub(super) struct HistorySync {
    sender: mpsc::Sender<Vec<u64>>,
    cancel: CancellationToken,
}

impl Drop for HistorySync {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl HistorySync {
    pub(super) fn publish(&self, mut hashes: impl Iterator<Item = u64>) {
        loop {
            let batch: Vec<_> = hashes.by_ref().take(MAX_BATCH_HASHES).collect();
            if batch.is_empty() {
                break;
            }
            if self.sender.try_send(batch).is_err() {
                tracing::warn!("Cache history replica queue unavailable; dropping peer update");
                break;
            }
        }
    }
}

impl CacheHistory {
    pub(crate) fn start_replica_sync(
        self: &Arc<Self>,
        router: &KvRouter,
        metrics: Arc<RouterRequestMetrics>,
    ) {
        if !router.kv_router_config().router_replica_sync {
            return;
        }
        let endpoint = router.client().endpoint.clone();
        let domain = Domain {
            model: router.tracking_model_name.clone(),
            block_size: router.block_size(),
            is_eagle: router.is_eagle(),
        };
        self.start_sync(endpoint, domain, metrics);
    }

    fn start_sync(
        self: &Arc<Self>,
        endpoint: Endpoint,
        domain: Domain,
        metrics: Arc<RouterRequestMetrics>,
    ) {
        let cancel = CancellationToken::new();
        let (sender, receiver) = mpsc::channel(QUEUE_BATCHES);
        if self
            .sync
            .set(HistorySync {
                sender,
                cancel: cancel.clone(),
            })
            .is_err()
        {
            return;
        }
        let history = Arc::downgrade(self);
        tokio::spawn(async move {
            let result = tokio::select! {
                _ = cancel.cancelled() => return,
                result = run(endpoint, domain, history, receiver, metrics, cancel.clone()) => result,
            };
            if let Err(error) = result {
                tracing::warn!(%error, "Cache history replica sync stopped");
            }
        });
    }
}

async fn run(
    endpoint: Endpoint,
    domain: Domain,
    history: Weak<CacheHistory>,
    mut receiver: mpsc::Receiver<Vec<u64>>,
    metrics: Arc<RouterRequestMetrics>,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    let transport = endpoint.drt().default_event_transport_kind();
    let source = uuid::Uuid::new_v4().to_string();
    loop {
        let connected = connect_with_retry(
            || async {
                let subscriber =
                    EventSubscriber::for_endpoint_with_transport(&endpoint, SUBJECT, transport)
                        .await?
                        .typed::<CompletedHashes>();
                let publisher =
                    EventPublisher::for_endpoint_with_transport(&endpoint, SUBJECT, transport)
                        .await?;
                Ok((publisher, subscriber))
            },
            &cancel,
        )
        .await;
        let Some((publisher, mut subscriber)) = connected else {
            return Ok(());
        };
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                hashes = receiver.recv() => {
                    let Some(hashes) = hashes else { return Ok(()) };
                    let event = CompletedHashes { source: source.clone(), domain: domain.clone(), hashes };
                    if let Err(error) = publisher.publish(&event).await {
                        tracing::warn!(%error, "Failed to publish completed cache history hashes; reconnecting");
                        break;
                    }
                }
                event = subscriber.next() => {
                    match event {
                        Some(Ok((_, event))) => {
                            let Some(history) = history.upgrade() else { return Ok(()) };
                            if let Some(stats) = apply_peer(&history, &source, &domain, event) {
                                metrics.set_cache_history_retained(stats);
                            }
                        }
                        Some(Err(error)) => tracing::warn!(%error, "Invalid cache history replica event"),
                        None => {
                            tracing::warn!("Cache history replica stream ended; reconnecting");
                            break;
                        }
                    }
                }
            }
        }
        tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            _ = tokio::time::sleep(INITIAL_RETRY_DELAY) => {}
        }
    }
}

async fn connect_with_retry<F, Fut, T>(mut connect: F, cancel: &CancellationToken) -> Option<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = anyhow::Result<T>>,
{
    let mut delay = INITIAL_RETRY_DELAY;
    loop {
        let result = tokio::select! {
            _ = cancel.cancelled() => return None,
            result = connect() => result,
        };
        match result {
            Ok(connection) => return Some(connection),
            Err(error) => {
                tracing::warn!(%error, "Cache history replica transport unavailable; retrying")
            }
        }
        tokio::select! {
            _ = cancel.cancelled() => return None,
            _ = tokio::time::sleep(delay) => {}
        }
        delay = (delay * 2).min(MAX_RETRY_DELAY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn transient_connect_failure_retries_with_backoff() {
        let cancel = CancellationToken::new();
        let mut attempts = 0;
        let start = tokio::time::Instant::now();
        let result = connect_with_retry(
            || {
                attempts += 1;
                std::future::ready(if attempts < 3 {
                    Err(anyhow::anyhow!("offline"))
                } else {
                    Ok(42)
                })
            },
            &cancel,
        )
        .await;
        assert_eq!(result, Some(42));
        assert_eq!(attempts, 3);
        assert_eq!(start.elapsed(), Duration::from_millis(300));
    }

    #[tokio::test(start_paused = true)]
    async fn cancellation_stops_retrying_an_unavailable_transport() {
        let cancel = CancellationToken::new();
        let stop = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            stop.cancel();
        });
        let mut attempts = 0;
        let result = connect_with_retry(
            || {
                attempts += 1;
                std::future::ready(Err::<(), _>(anyhow::anyhow!("offline")))
            },
            &cancel,
        )
        .await;
        assert!(result.is_none());
        assert_eq!(attempts, 1);
    }

    fn domain() -> Domain {
        Domain {
            model: "model-a".into(),
            block_size: 16,
            is_eagle: false,
        }
    }

    fn event(hashes: &[u64]) -> CompletedHashes {
        CompletedHashes {
            source: "peer".into(),
            domain: domain(),
            hashes: hashes.to_vec(),
        }
    }

    #[test]
    fn peer_and_local_insertions_have_identical_fifo_behavior_without_rebroadcast() {
        let local = CacheHistory::with_capacity(3, 16);
        let peer = CacheHistory::with_capacity(3, 16);
        let (sender, mut receiver) = mpsc::channel(8);
        assert!(
            peer.sync
                .set(HistorySync {
                    sender,
                    cancel: CancellationToken::new()
                })
                .is_ok()
        );
        for batch in [&[10, 20, 30][..], &[10, 40], &[20, 50], &[20, 50]] {
            local.record_completed(batch.iter().copied());
            apply_peer(&peer, "local", &domain(), event(batch));
            assert_eq!(*local.fifo.lock(), *peer.fifo.lock());
            assert_eq!(local.stats(), peer.stats());
        }
        assert!(receiver.try_recv().is_err());
        peer.record_completed([20, 50].into_iter());
        assert_eq!(receiver.try_recv().unwrap(), vec![20, 50]);
        assert_eq!(*local.fifo.lock(), *peer.fifo.lock());
    }

    #[test]
    fn rejects_self_and_incompatible_hash_domains() {
        let history = CacheHistory::with_capacity(3, 16);
        apply_peer(&history, "peer", &domain(), event(&[10]));
        for incompatible in [
            Domain {
                model: "model-b".into(),
                ..domain()
            },
            Domain {
                block_size: 8,
                ..domain()
            },
            Domain {
                is_eagle: true,
                ..domain()
            },
        ] {
            apply_peer(&history, "local", &incompatible, event(&[10]));
        }
        assert_eq!(history.stats().retained_entries, 0);
        apply_peer(&history, "local", &domain(), event(&[10]));
        assert_eq!(history.previously_computed_tokens(&[10]), 16);
    }

    #[test]
    fn publication_is_chunked_and_does_not_block_when_full() {
        let (sender, mut receiver) = mpsc::channel(2);
        let sync = HistorySync {
            sender,
            cancel: CancellationToken::new(),
        };
        sync.publish(0..MAX_BATCH_HASHES as u64 + 3);
        assert_eq!(receiver.try_recv().unwrap().len(), MAX_BATCH_HASHES);
        assert_eq!(receiver.try_recv().unwrap().len(), 3);
        sync.publish(0..3 * MAX_BATCH_HASHES as u64);
        assert_eq!(receiver.len(), 2);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn two_frontends_learn_completed_hashes_over_the_event_plane() -> anyhow::Result<()> {
        let runtime = dynamo_runtime::Runtime::from_current()?;
        let distributed = dynamo_runtime::DistributedRuntime::new(
            runtime,
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await?;
        let component = distributed
            .namespace(format!("history-sync-{}", uuid::Uuid::new_v4()))?
            .component("workers")?;
        let endpoint = component.endpoint("generate");
        let metrics = RouterRequestMetrics::from_component(&component);
        let first = Arc::new(CacheHistory::with_capacity(3, 16));
        let second = Arc::new(CacheHistory::with_capacity(3, 16));
        first.start_sync(endpoint.clone(), domain(), metrics.clone());
        second.start_sync(endpoint, domain(), metrics);
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                first.record_completed([10, 20].into_iter());
                if second.previously_computed_tokens(&[10, 20]) == 32 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await?;
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                second.record_completed([10, 20, 30].into_iter());
                if first.previously_computed_tokens(&[10, 20, 30]) == 48 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await?;
        drop((first, second));
        distributed.shutdown();
        Ok(())
    }
}
