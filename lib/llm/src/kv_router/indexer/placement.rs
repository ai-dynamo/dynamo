// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use dynamo_kv_router::protocols::RouterEvent;
use futures::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard, broadcast};
use tokio_util::sync::CancellationToken;

use super::Indexer;

const PLACEMENT_JOURNAL_CAPACITY: usize = 1024;

pub(crate) type PlacementStream =
    Pin<Box<dyn Stream<Item = Result<PlacementUpdate>> + Send + 'static>>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum PlacementUpdate {
    Snapshot {
        cursor: u64,
        events: Vec<RouterEvent>,
    },
    Events {
        cursor: u64,
        events: Vec<RouterEvent>,
    },
}

#[derive(Clone)]
pub struct PlacementJournal {
    inner: Arc<PlacementJournalInner>,
}

struct PlacementJournalInner {
    mutation: RwLock<()>,
    cursor: AtomicU64,
    sender: broadcast::Sender<PlacementDelta>,
    cancel: CancellationToken,
}

#[derive(Clone)]
struct PlacementDelta {
    cursor: u64,
    events: Vec<RouterEvent>,
}

impl PlacementJournal {
    pub(super) fn new(cancel: CancellationToken) -> Self {
        let (sender, _) = broadcast::channel(PLACEMENT_JOURNAL_CAPACITY);
        Self {
            inner: Arc::new(PlacementJournalInner {
                mutation: RwLock::new(()),
                cursor: AtomicU64::new(0),
                sender,
                cancel,
            }),
        }
    }

    pub(super) async fn lock_shared(&self) -> RwLockReadGuard<'_, ()> {
        self.inner.mutation.read().await
    }

    pub(super) async fn lock_exclusive(&self) -> RwLockWriteGuard<'_, ()> {
        self.inner.mutation.write().await
    }

    fn cursor(&self) -> u64 {
        self.inner.cursor.load(Ordering::Relaxed)
    }

    fn subscribe(&self) -> broadcast::Receiver<PlacementDelta> {
        self.inner.sender.subscribe()
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.inner.cancel.clone()
    }

    pub(super) fn has_subscribers(&self) -> bool {
        self.inner.sender.receiver_count() != 0
    }

    fn advance_cursor(&self) -> u64 {
        self.inner
            .cursor
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("KV placement cursor overflowed")
            + 1
    }

    pub(super) fn advance(&self) {
        self.advance_cursor();
    }

    pub(super) fn publish(&self, events: Vec<RouterEvent>) {
        if events.is_empty() {
            return;
        }
        let cursor = self.advance_cursor();
        let _ = self.inner.sender.send(PlacementDelta { cursor, events });
    }
}

#[derive(Clone)]
pub(crate) struct PlacementFeed {
    indexer: Indexer,
}

impl PlacementFeed {
    pub(super) fn new(indexer: Indexer) -> Self {
        Self { indexer }
    }

    pub(crate) async fn stream(&self) -> Result<PlacementStream> {
        match &self.indexer {
            Indexer::KvIndexer {
                placement: Some(journal),
                ..
            }
            | Indexer::Concurrent {
                placement: Some(journal),
                ..
            } => local_stream(self.indexer.clone(), journal.clone()).await,
            Indexer::Remote { primary, .. } if primary.use_kv_events() => {
                primary.placement_stream().await
            }
            Indexer::KvIndexer {
                placement: None, ..
            }
            | Indexer::Concurrent {
                placement: None, ..
            }
            | Indexer::Remote { .. }
            | Indexer::None => anyhow::bail!("KV placement feed is not available"),
        }
    }
}

async fn local_stream(indexer: Indexer, journal: PlacementJournal) -> Result<PlacementStream> {
    // The exclusive lock waits for accepted mutations and blocks later ones. The dump is
    // therefore a linearizable snapshot, and every later journal cursor is a delta.
    let guard = journal.lock_exclusive().await;
    let mut receiver = journal.subscribe();
    let cancel = journal.cancellation_token();
    let cursor = journal.cursor();
    let events = indexer
        .dump_events()
        .await
        .context("failed to dump KV placement index")?;
    drop(guard);

    Ok(Box::pin(async_stream::try_stream! {
        yield PlacementUpdate::Snapshot { cursor, events };
        loop {
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = receiver.recv() => result,
            };
            match result {
                Ok(delta) => yield PlacementUpdate::Events {
                    cursor: delta.cursor,
                    events: delta.events,
                },
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    Err(anyhow::anyhow!(
                        "KV placement consumer lagged by {skipped} updates"
                    ))?;
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_router::indexer::{LowerTierIndexers, test_util::store_event};
    use dynamo_kv_router::indexer::{KvIndexer, KvIndexerMetrics};
    use dynamo_kv_router::protocols::StorageTier;
    use futures::StreamExt;
    use tokio_util::sync::CancellationToken;

    fn indexer() -> Indexer {
        indexer_with_cancel(CancellationToken::new())
    }

    fn indexer_with_cancel(cancel: CancellationToken) -> Indexer {
        Indexer::KvIndexer {
            primary: KvIndexer::new(
                CancellationToken::new(),
                4,
                Arc::new(KvIndexerMetrics::new_unregistered()),
            ),
            lower_tier: LowerTierIndexers::new(1, 4),
            approx: None,
            primary_records_routing_decisions: false,
            placement: Some(PlacementJournal::new(cancel)),
        }
    }

    #[tokio::test]
    async fn feed_starts_with_index_snapshot_then_accepted_deltas() {
        let indexer = indexer();
        indexer
            .try_apply_event(store_event(7, 0, 1, &[], &[11], StorageTier::Device))
            .await
            .unwrap();

        let mut stream = indexer.placement_feed().unwrap().stream().await.unwrap();
        let PlacementUpdate::Snapshot { cursor, events } = stream.next().await.unwrap().unwrap()
        else {
            panic!("placement feed must start with a snapshot");
        };
        assert_eq!(cursor, 1);
        assert_eq!(events.len(), 1);

        indexer
            .try_apply_event(store_event(7, 0, 2, &[11], &[12], StorageTier::Device))
            .await
            .unwrap();
        let PlacementUpdate::Events { cursor, events } = stream.next().await.unwrap().unwrap()
        else {
            panic!("accepted mutation must be journaled as a delta");
        };
        assert_eq!(cursor, 2);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.event_id, 2);
    }

    #[tokio::test]
    async fn rank_replacement_is_one_cursor_update() {
        let indexer = indexer();
        let mut stream = indexer.placement_feed().unwrap().stream().await.unwrap();
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            PlacementUpdate::Snapshot { cursor: 0, .. }
        ));

        indexer
            .replace_worker_dp_rank_and_wait(
                7,
                0,
                vec![store_event(7, 0, 9, &[], &[11], StorageTier::Device)],
            )
            .await
            .unwrap();
        let PlacementUpdate::Events { cursor, events } = stream.next().await.unwrap().unwrap()
        else {
            panic!("rank replacement must be journaled");
        };
        assert_eq!(cursor, 1);
        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0].event.data,
            dynamo_kv_router::protocols::KvCacheEventData::Cleared
        ));
    }

    #[tokio::test]
    async fn feed_closes_when_its_router_lifecycle_ends() {
        let cancel = CancellationToken::new();
        let indexer = indexer_with_cancel(cancel.clone());
        let mut stream = indexer.placement_feed().unwrap().stream().await.unwrap();
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            PlacementUpdate::Snapshot { .. }
        ));

        cancel.cancel();
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn lagged_feed_fails_instead_of_skipping_a_cursor() {
        let indexer = indexer();
        let journal = match &indexer {
            Indexer::KvIndexer {
                placement: Some(journal),
                ..
            } => journal.clone(),
            _ => unreachable!(),
        };
        let mut stream = indexer.placement_feed().unwrap().stream().await.unwrap();

        for event_id in 1..=PLACEMENT_JOURNAL_CAPACITY as u64 + 1 {
            journal.publish(vec![store_event(
                7,
                0,
                event_id,
                &[],
                &[event_id],
                StorageTier::Device,
            )]);
        }

        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            PlacementUpdate::Snapshot { .. }
        ));
        assert!(stream.next().await.unwrap().is_err());
    }
}
