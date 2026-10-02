// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine replay framing and bounded, ordered replay/live merge.

use anyhow::{Result, bail};
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq)]
enum ReplayFrame {
    Batch(u64, Vec<u8>),
    End,
}

fn decode_replay(frames: Vec<Vec<u8>>) -> Result<ReplayFrame> {
    // DEALER removes identity; SGLang omits the topic included by vLLM.
    let (sequence, payload) = match frames.as_slice() {
        [delimiter, sequence, payload] if delimiter.is_empty() => (sequence, payload),
        [delimiter, _, sequence, payload] if delimiter.is_empty() => (sequence, payload),
        _ => bail!("invalid KV replay envelope"),
    };
    let sequence: [u8; 8] = sequence.as_slice().try_into()?;
    if sequence == (-1_i64).to_be_bytes() && payload.is_empty() {
        return Ok(ReplayFrame::End);
    }
    if payload.is_empty() || sequence == (-1_i64).to_be_bytes() {
        bail!("invalid KV replay end marker");
    }
    Ok(ReplayFrame::Batch(
        u64::from_be_bytes(sequence),
        payload.clone(),
    ))
}

/// One cursor per engine incarnation and rank. Never infer history loss from
/// a timeout: an ordered replay suffix can confirm an observed hole.
#[derive(Debug)]
struct ReplayCursor {
    next: u64,
    pending: BTreeMap<u64, Vec<u8>>,
    bytes: usize,
    limit: usize,
}

impl ReplayCursor {
    fn new(max_bytes: usize) -> Self {
        Self {
            next: 0,
            pending: BTreeMap::new(),
            bytes: 0,
            limit: max_bytes,
        }
    }
    fn next(&self) -> u64 {
        self.next
    }
    fn insert(&mut self, sequence: u64, payload: Vec<u8>) -> Result<()> {
        if sequence < self.next || self.pending.contains_key(&sequence) {
            return Ok(());
        }
        if self.pending.len() >= 4096 || payload.len() > self.limit.saturating_sub(self.bytes) {
            bail!("KV replay/live merge buffer exhausted");
        }
        self.bytes += payload.len();
        self.pending.insert(sequence, payload);
        Ok(())
    }
    fn pop_contiguous(&mut self) -> Result<Option<(u64, Vec<u8>)>> {
        let Some(payload) = self.pending.remove(&self.next) else {
            return Ok(None);
        };
        let sequence = self.next;
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("KV sequence exhausted"))?;
        self.bytes -= payload.len();
        Ok(Some((sequence, payload)))
    }
    fn gap(&self) -> Option<(u64, u64)> {
        self.pending
            .first_key_value()
            .and_then(|(&got, _)| (got > self.next).then_some((self.next, got)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn end_marker_is_explicit() {
        assert_eq!(
            decode_replay(vec![vec![], (-1_i64).to_be_bytes().to_vec(), vec![]]).unwrap(),
            ReplayFrame::End
        );
        assert!(decode_replay(vec![vec![], 0_u64.to_be_bytes().to_vec(), vec![]]).is_err());
        assert!(decode_replay(vec![]).is_err());
    }
    #[test]
    fn overlap_is_applied_once_in_order() {
        let mut cursor = ReplayCursor::new(8);
        cursor.insert(2, vec![2]).unwrap();
        assert_eq!(cursor.gap(), Some((0, 2)));
        cursor.insert(0, vec![0]).unwrap();
        cursor.insert(1, vec![1]).unwrap();
        cursor.insert(2, vec![2]).unwrap();
        for seq in 0..3 {
            assert_eq!(
                cursor.pop_contiguous().unwrap(),
                Some((seq, vec![seq as u8]))
            );
        }
        cursor.insert(1, vec![1]).unwrap();
        assert_eq!(cursor.pop_contiguous().unwrap(), None);
        assert_eq!(cursor.next(), 3);
        cursor.insert(4, vec![4]).unwrap();
        assert_eq!(cursor.gap(), Some((3, 4)));
    }
    #[test]
    fn expired_history_and_buffer_bounds() {
        let mut cursor = ReplayCursor::new(2);
        cursor.insert(500, vec![1, 1]).unwrap();
        assert_eq!(cursor.gap(), Some((0, 500)));
        cursor.insert(500, vec![1, 1]).unwrap();
        assert!(cursor.insert(501, vec![2]).is_err());
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplaySource {
    pub endpoint: String,
    pub replay_endpoint: String,
    pub topic: String,
    pub dp_rank: u32,
}

#[derive(Clone, Debug)]
enum RecoveryStatus {
    Recovering,
    Ready,
    MissingHistory { expected: u64, got: u64 },
}

/// Result of one bounded attempt against an initially empty local index.
/// Retry and engine shutdown policy are deliberately left to the caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BootstrapOutcome {
    Success,
    MissingHistory {
        dp_rank: u32,
        expected: u64,
        got: u64,
    },
    Uncertain {
        reason: String,
    },
}

/// Bootstrap all sources, leaving only live listeners running on success.
/// The caller must provide an empty index and cancel successful listeners before
/// any later attempt. Failed attempts may have partially applied events; they
/// never permit serving. The deadline covers replay and index application.
pub async fn bootstrap_sources<T: RecoveryTarget + 'static>(
    sources: Vec<ReplaySource>,
    publishers: BTreeMap<u32, std::sync::Arc<T>>,
    worker_id: u64,
    cancel: tokio_util::sync::CancellationToken,
    timeout: std::time::Duration,
) -> BootstrapOutcome {
    let session = cancel.child_token();
    let mut tasks = tokio::task::JoinSet::new();
    let mut statuses = Vec::new();
    for source in sources {
        let rank = source.dp_rank;
        let Some(publisher) = publishers.get(&rank).cloned() else {
            session.cancel();
            tasks.abort_all();
            return BootstrapOutcome::Uncertain {
                reason: format!("missing publisher for rank {rank}"),
            };
        };
        let (tx, rx) = tokio::sync::watch::channel(RecoveryStatus::Recovering);
        statuses.push((rank, rx));
        let cancel = session.clone();
        tasks.spawn(async move {
            let result = tokio::select! {
                _ = cancel.cancelled() => Ok(()),
                result = run_source(source, publisher, worker_id, cancel.clone(), tx) => result,
            };
            if let Err(error) = &result {
                tracing::warn!(rank, %error, "KV listener stopped");
            }
            result
        });
    }
    let result = tokio::time::timeout(timeout, async {
        loop {
            for (rank, status) in &statuses {
                if let RecoveryStatus::MissingHistory { expected, got } = *status.borrow() {
                    return BootstrapOutcome::MissingHistory { dp_rank: *rank, expected, got };
                }
            }
            if statuses.iter().all(|(_, status)| matches!(*status.borrow(), RecoveryStatus::Ready)) {
                return BootstrapOutcome::Success;
            }
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return BootstrapOutcome::Uncertain { reason: "startup cancelled".into() },
                result = tasks.join_next() => return BootstrapOutcome::Uncertain { reason: format!("KV listener ended during bootstrap: {result:?}") },
                _ = futures::future::select_all(statuses.iter_mut().map(|(_, rx)| Box::pin(rx.changed()))) => {},
            }
        }
    }).await.unwrap_or_else(|_| BootstrapOutcome::Uncertain { reason: "KV bootstrap deadline exceeded".into() });
    if result == BootstrapOutcome::Success {
        // No controller remains to monitor engines, reconnect, or change routes.
        tasks.detach_all();
    } else {
        session.cancel();
        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
    }
    result
}

/// Local application target. Keeping transport separate allows socket-level
/// replay tests without starting a Dynamo runtime or a GPU engine.
pub trait RecoveryTarget: Send + Sync {
    fn block_size(&self) -> u32;
    fn next_event_id(&self) -> u64;
    fn apply(
        &self,
        events: Vec<dynamo_kv_router::protocols::PlacementEvent>,
    ) -> impl std::future::Future<Output = Result<()>> + Send;
    fn publish_live(
        &self,
        events: Vec<dynamo_kv_router::protocols::PlacementEvent>,
    ) -> impl std::future::Future<Output = Result<()>> + Send {
        self.apply(events)
    }
}

impl RecoveryTarget for dynamo_llm::kv_router::publisher::KvEventPublisher {
    fn block_size(&self) -> u32 {
        self.kv_block_size()
    }
    fn next_event_id(&self) -> u64 {
        self.next_event_id()
    }
    async fn apply(&self, events: Vec<dynamo_kv_router::protocols::PlacementEvent>) -> Result<()> {
        self.publish_recovery_batch(events).await
    }
    async fn publish_live(
        &self,
        events: Vec<dynamo_kv_router::protocols::PlacementEvent>,
    ) -> Result<()> {
        self.publish_placement_batch(events)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
    }
}

/// Receive replay and live events during startup, then consume live events.
/// No engine lifecycle or runtime sequence-gap policy is implemented here.
async fn run_source<T: RecoveryTarget + 'static>(
    source: ReplaySource,
    publisher: std::sync::Arc<T>,
    worker_id: u64,
    cancel: tokio_util::sync::CancellationToken,
    status: tokio::sync::watch::Sender<RecoveryStatus>,
) -> Result<()> {
    use dynamo_kv_router::{protocols::*, zmq_wire::*};
    use futures::{FutureExt, SinkExt, StreamExt};
    use std::time::Duration;
    use tmq::{Context, Multipart};

    const LIMIT: usize = 32 * 1024 * 1024;
    const TIMEOUT: Duration = Duration::from_secs(5);
    struct UnreadyOnExit(tokio::sync::watch::Sender<RecoveryStatus>);
    impl Drop for UnreadyOnExit {
        fn drop(&mut self) {
            self.0.send_replace(RecoveryStatus::Recovering);
        }
    }
    let _guard = UnreadyOnExit(status.clone());
    let context = Context::new();
    let monitor_events =
        zmq::SocketEvent::HANDSHAKE_SUCCEEDED as i32 | zmq::SocketEvent::DISCONNECTED as i32;
    let mut live = tmq::subscribe(&context)
        .set_linger(0)
        .set_ipv6(true)
        .monitor("inproc://live-monitor", monitor_events)
        .connect(&source.endpoint)?
        .subscribe(source.topic.as_bytes())?;
    let mut live_monitor = tmq::pair(&context).connect("inproc://live-monitor")?;
    let mut replay = tmq::dealer(&context)
        .set_linger(0)
        .set_ipv6(true)
        .set_rcvhwm(8)
        .set_maxmsgsize(LIMIT as i64)
        .monitor("inproc://replay-monitor", monitor_events)
        .connect(&source.replay_endpoint)?;
    let mut replay_monitor = tmq::pair(&context).connect("inproc://replay-monitor")?;
    async fn connected(monitor: &mut tmq::pair::Pair) -> Result<()> {
        let event = monitor
            .next()
            .await
            .ok_or_else(|| anyhow::anyhow!("KV socket monitor ended"))??;
        let data = event
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("empty socket monitor event"))?;
        let kind = u16::from_ne_bytes(
            data.get(..2)
                .ok_or_else(|| anyhow::anyhow!("short monitor event"))?
                .try_into()?,
        );
        anyhow::ensure!(
            kind == zmq::SocketEvent::HANDSHAKE_SUCCEEDED as u16,
            "KV socket disconnected during connection"
        );
        Ok(())
    }
    tokio::select! {
        _ = cancel.cancelled() => return Ok(()),
        result = tokio::time::timeout(TIMEOUT, async { tokio::try_join!(connected(&mut live_monitor), connected(&mut replay_monitor)) }) => { result??; }
    }
    let mut cursor = ReplayCursor::new(LIMIT);
    let mut normalizer = ZmqEventNormalizer::new(publisher.block_size());
    loop {
        let request = Multipart::from(vec![Vec::new(), cursor.next().to_be_bytes().to_vec()]);
        tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            result = tokio::time::timeout(TIMEOUT, replay.send(request)) => { result??; }
        }
        // Validate the local application path even for an empty replay.
        publisher.apply(Vec::new()).await?;
        // An inactivity timeout is uncertainty, never evidence of expired history.
        let mut deadline = tokio::time::Instant::now() + TIMEOUT;
        loop {
            let (sequence, payload) = tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                _ = live_monitor.next() => bail!("KV live connection lost"),
                _ = replay_monitor.next() => bail!("KV replay connection lost"),
                _ = tokio::time::sleep_until(deadline) => bail!("KV replay timed out; history availability is unknown"),
                message = replay.next() => {
                    let frames = message.ok_or_else(|| anyhow::anyhow!("KV replay socket ended"))??.into_iter().map(|f| f.to_vec()).collect();
                    deadline = tokio::time::Instant::now() + TIMEOUT;
                    match decode_replay(frames)? {
                        ReplayFrame::End => break,
                        ReplayFrame::Batch(seq, payload) => {
                            // Replay is ordered. A suffix beyond the next required
                            // batch proves loss without buffering that suffix.
                            if seq > cursor.next() {
                                status.send_replace(RecoveryStatus::MissingHistory { expected: cursor.next(), got: seq });
                                cancel.cancelled().await;
                                return Ok(());
                            }
                            (seq, payload)
                        }
                    }
                }
                message = live.next() => decode_live(message.ok_or_else(|| anyhow::anyhow!("KV live socket ended"))??)?,
            };
            apply_batch(
                &mut cursor,
                sequence,
                payload,
                &mut normalizer,
                publisher.as_ref(),
                WorkerWithDpRank::new(worker_id, source.dp_rank),
                &status,
            )
            .await?;
        }
        // Replay End can arrive while live messages are already queued. Merge
        // those before deciding whether the startup handoff still has a gap.
        while let Some(message) = live.next().now_or_never() {
            let (sequence, payload) =
                decode_live(message.ok_or_else(|| anyhow::anyhow!("KV live socket ended"))??)?;
            apply_batch(
                &mut cursor,
                sequence,
                payload,
                &mut normalizer,
                publisher.as_ref(),
                WorkerWithDpRank::new(worker_id, source.dp_rank),
                &status,
            )
            .await?;
        }
        if cursor.gap().is_none() {
            break;
        }
        // Resolve overlap gaps before completing bootstrap. No replay is
        // requested after this source enters live consumption.
    }
    drop(replay);
    drop(replay_monitor);
    drop(live_monitor);
    let bootstrap_next = cursor.next();
    drop(cursor);
    status.send_replace(RecoveryStatus::Ready);
    loop {
        let message = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Ok(()),
            message = live.next() => message.ok_or_else(|| anyhow::anyhow!("KV live socket ended"))??,
        };
        let (sequence, payload) = match decode_live(message) {
            Ok(batch) => batch,
            Err(error) => {
                tracing::warn!(%error, "Invalid live KV envelope");
                continue;
            }
        };
        // Discard queued overlap with bootstrap; otherwise use ordinary live
        // admission without completion acknowledgements or runtime gap repair.
        if sequence < bootstrap_next {
            continue;
        }
        match normalize_payload(
            &payload,
            &mut normalizer,
            publisher.as_ref(),
            WorkerWithDpRank::new(worker_id, source.dp_rank),
        ) {
            Ok(events) => publisher.publish_live(events).await?,
            Err(error) => tracing::warn!(%error, "Invalid live KV batch"),
        }
    }
}

fn decode_live(frames: tmq::Multipart) -> Result<(u64, Vec<u8>)> {
    let frames: Vec<_> = frames.into_iter().collect();
    let [_, sequence, payload] = frames.as_slice() else {
        bail!("invalid live KV envelope");
    };
    Ok((
        u64::from_be_bytes(sequence[..].try_into()?),
        payload.to_vec(),
    ))
}

async fn apply_batch<T: RecoveryTarget>(
    cursor: &mut ReplayCursor,
    sequence: u64,
    payload: Vec<u8>,
    normalizer: &mut dynamo_kv_router::zmq_wire::ZmqEventNormalizer,
    publisher: &T,
    worker: dynamo_kv_router::protocols::WorkerWithDpRank,
    status: &tokio::sync::watch::Sender<RecoveryStatus>,
) -> Result<()> {
    cursor.insert(sequence, payload)?;
    if cursor.gap().is_some() {
        status.send_replace(RecoveryStatus::Recovering);
    }
    while let Some((_, payload)) = cursor.pop_contiguous()? {
        publisher
            .apply(normalize_payload(&payload, normalizer, publisher, worker)?)
            .await?;
    }
    Ok(())
}

fn normalize_payload<T: RecoveryTarget>(
    payload: &[u8],
    normalizer: &mut dynamo_kv_router::zmq_wire::ZmqEventNormalizer,
    publisher: &T,
    worker: dynamo_kv_router::protocols::WorkerWithDpRank,
) -> Result<Vec<dynamo_kv_router::protocols::PlacementEvent>> {
    let batch = dynamo_kv_router::zmq_wire::decode_event_batch(payload)?;
    anyhow::ensure!(
        !batch
            .data_parallel_rank
            .is_some_and(|rank| rank < 0 || rank as u32 != worker.dp_rank),
        "KV batch belongs to the wrong DP rank"
    );
    let mut events = Vec::with_capacity(batch.events.len());
    for raw in batch.events {
        if let Some(raw) = normalizer.preprocess(raw, worker)
            && let Some(event) =
                normalizer.normalize_preprocessed(raw, publisher.next_event_id(), worker)
        {
            events.push(event);
        }
    }
    Ok(events)
}

#[cfg(test)]
mod socket_tests {
    use super::*;
    use dynamo_kv_router::protocols::{KvCacheEvent, PlacementEvent};
    use futures::{SinkExt, StreamExt};
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    };
    use std::time::Duration;
    use tmq::{AsZmqSocket, Multipart};
    use tokio::sync::watch;
    use tokio_util::sync::CancellationToken;

    #[derive(Default)]
    struct Target {
        events: Mutex<Vec<KvCacheEvent>>,
        next: AtomicU64,
        fail: bool,
        block_first: Option<(Arc<tokio::sync::Notify>, Arc<tokio::sync::Notify>)>,
    }
    impl RecoveryTarget for Target {
        fn block_size(&self) -> u32 {
            4
        }
        fn next_event_id(&self) -> u64 {
            self.next.fetch_add(1, Ordering::Relaxed)
        }
        async fn apply(&self, events: Vec<PlacementEvent>) -> Result<()> {
            anyhow::ensure!(!self.fail, "injected application failure");
            if events
                .first()
                .is_some_and(|event| event.event.event_id == 0)
                && let Some((entered, release)) = &self.block_first
            {
                entered.notify_one();
                release.notified().await;
            }
            self.events
                .lock()
                .unwrap()
                .extend(events.into_iter().map(|e| e.event));
            Ok(())
        }
    }
    struct Fixture {
        live: tmq::publish::Publish,
        replay: tmq::router::Router,
        source: ReplaySource,
    }
    impl Fixture {
        fn new() -> Self {
            let context = tmq::Context::new();
            let live = tmq::publish(&context)
                .set_linger(0)
                .bind("tcp://127.0.0.1:*")
                .unwrap();
            let replay = tmq::router(&context)
                .set_linger(0)
                .bind("tcp://127.0.0.1:*")
                .unwrap();
            let source = ReplaySource {
                endpoint: live.get_socket().get_last_endpoint().unwrap().unwrap(),
                replay_endpoint: replay.get_socket().get_last_endpoint().unwrap().unwrap(),
                topic: String::new(),
                dp_rank: 0,
            };
            Self {
                live,
                replay,
                source,
            }
        }
        async fn request(&mut self, expected: u64) -> Vec<u8> {
            let request = tokio::time::timeout(Duration::from_secs(10), self.replay.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            let frames: Vec<_> = request.into_iter().collect();
            assert_eq!(&frames[2][..], &expected.to_be_bytes());
            frames[0].to_vec()
        }
        async fn send(&mut self, id: &[u8], seq: i64, payload: Vec<u8>) {
            self.replay
                .send(Multipart::from(vec![
                    id.to_vec(),
                    vec![],
                    seq.to_be_bytes().to_vec(),
                    payload,
                ]))
                .await
                .unwrap();
        }
    }
    fn payload(rank: i32) -> Vec<u8> {
        rmp_serde::to_vec(&(0.0_f64, vec![vec!["AllBlocksCleared"]], Some(rank))).unwrap()
    }
    async fn until(
        rx: &mut watch::Receiver<RecoveryStatus>,
        pred: impl Fn(&RecoveryStatus) -> bool,
    ) {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if pred(&rx.borrow_and_update()) {
                    return;
                }
                rx.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn bootstrap_waits_for_every_rank_then_leaves_live_listeners_running() {
        let mut first = Fixture::new();
        let mut second = Fixture::new();
        second.source.dp_rank = 1;
        let a = Arc::new(Target::default());
        let b = Arc::new(Target::default());
        let cancel = CancellationToken::new();
        let task = tokio::spawn(bootstrap_sources(
            vec![first.source.clone(), second.source.clone()],
            BTreeMap::from([(0, a.clone()), (1, b.clone())]),
            1,
            cancel.clone(),
            Duration::from_secs(10),
        ));
        let id = first.request(0).await;
        first.send(&id, 0, payload(0)).await;
        first.send(&id, -1, vec![]).await;
        let id = second.request(0).await;
        assert!(!task.is_finished());
        second.send(&id, 0, payload(1)).await;
        second.send(&id, -1, vec![]).await;
        assert_eq!(task.await.unwrap(), BootstrapOutcome::Success);
        assert_eq!(a.events.lock().unwrap().len(), 1);
        assert_eq!(b.events.lock().unwrap().len(), 1);
        second
            .live
            .send(Multipart::from(vec![
                vec![],
                2_u64.to_be_bytes().to_vec(),
                payload(1),
            ]))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while b.events.lock().unwrap().len() < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        cancel.cancel();
    }

    #[tokio::test]
    async fn bootstrap_distinguishes_missing_history_from_deadline() {
        for missing in [true, false] {
            let mut fixture = Fixture::new();
            let cancel = CancellationToken::new();
            let task = tokio::spawn(bootstrap_sources(
                vec![fixture.source.clone()],
                BTreeMap::from([(0, Arc::new(Target::default()))]),
                1,
                cancel,
                Duration::from_secs(1),
            ));
            let id = fixture.request(0).await;
            if missing {
                fixture.send(&id, 500, payload(0)).await;
            }
            let outcome = task.await.unwrap();
            if missing {
                assert_eq!(
                    outcome,
                    BootstrapOutcome::MissingHistory {
                        dp_rank: 0,
                        expected: 0,
                        got: 500
                    }
                );
            } else {
                assert!(matches!(outcome, BootstrapOutcome::Uncertain { .. }));
            }
        }
    }

    #[tokio::test]
    async fn bootstrap_deadline_includes_local_application() {
        struct BlockedTarget(std::sync::atomic::AtomicBool);
        impl RecoveryTarget for BlockedTarget {
            fn block_size(&self) -> u32 {
                16
            }
            fn next_event_id(&self) -> u64 {
                0
            }
            async fn apply(&self, _: Vec<PlacementEvent>) -> Result<()> {
                self.0.store(true, Ordering::Relaxed);
                std::future::pending().await
            }
        }
        let fixture = Fixture::new();
        let target = Arc::new(BlockedTarget(std::sync::atomic::AtomicBool::new(false)));
        let outcome = bootstrap_sources(
            vec![fixture.source],
            BTreeMap::from([(0, target.clone())]),
            1,
            CancellationToken::new(),
            Duration::from_secs(1),
        )
        .await;
        assert!(
            target.0.load(Ordering::Relaxed),
            "must reach local application before timing out"
        );
        assert!(matches!(outcome, BootstrapOutcome::Uncertain { .. }));
    }

    #[tokio::test]
    async fn empty_completed_replay_can_bootstrap_with_validated_retention() {
        let mut fixture = Fixture::new();
        let target = Arc::new(Target::default());
        let cancel = CancellationToken::new();
        let (tx, mut rx) = watch::channel(RecoveryStatus::Recovering);
        let task = tokio::spawn(run_source(
            fixture.source.clone(),
            target.clone(),
            1,
            cancel.clone(),
            tx,
        ));
        let id = fixture.request(0).await;
        assert!(matches!(*rx.borrow(), RecoveryStatus::Recovering));
        fixture.send(&id, -1, vec![]).await;
        until(&mut rx, |s| matches!(s, RecoveryStatus::Ready)).await;
        assert!(target.events.lock().unwrap().is_empty());
        cancel.cancel();
        task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn bootstrap_merges_overlap_then_consumes_live_without_replay() {
        let mut fixture = Fixture::new();
        let target = Arc::new(Target::default());
        let cancel = CancellationToken::new();
        let (tx, mut rx) = watch::channel(RecoveryStatus::Recovering);
        let task = tokio::spawn(run_source(
            fixture.source.clone(),
            target.clone(),
            1,
            cancel.clone(),
            tx,
        ));
        let id = fixture.request(0).await;
        fixture
            .live
            .send(Multipart::from(vec![
                vec![],
                1_u64.to_be_bytes().to_vec(),
                payload(0),
            ]))
            .await
            .unwrap();
        fixture.send(&id, 0, payload(0)).await;
        fixture.send(&id, 1, payload(0)).await;
        fixture.send(&id, -1, vec![]).await;
        until(&mut rx, |s| matches!(s, RecoveryStatus::Ready)).await;
        assert_eq!(target.events.lock().unwrap().len(), 2);
        // A live gap after bootstrap does not request replay or withdraw
        // readiness. Continuous repair is outside this scope.
        fixture
            .live
            .send(Multipart::from(vec![
                vec![],
                3_u64.to_be_bytes().to_vec(),
                payload(0),
            ]))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while target.events.lock().unwrap().len() < 3 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(target.events.lock().unwrap().len(), 3);
        assert!(matches!(*rx.borrow(), RecoveryStatus::Ready));
        // Longer than the former periodic replay interval.
        assert!(
            tokio::time::timeout(Duration::from_millis(1200), fixture.replay.next())
                .await
                .is_err()
        );
        assert!(matches!(*rx.borrow(), RecoveryStatus::Ready));
        cancel.cancel();
        task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn bootstrap_resolves_live_overlap_gap_before_success() {
        let mut fixture = Fixture::new();
        let target = Arc::new(Target::default());
        let cancel = CancellationToken::new();
        let (tx, mut rx) = watch::channel(RecoveryStatus::Recovering);
        let task = tokio::spawn(run_source(
            fixture.source.clone(),
            target.clone(),
            1,
            cancel.clone(),
            tx,
        ));
        let id = fixture.request(0).await;
        fixture
            .live
            .send(Multipart::from(vec![
                vec![],
                3_u64.to_be_bytes().to_vec(),
                payload(0),
            ]))
            .await
            .unwrap();
        // Wait until the live batch has been buffered ahead of sequence zero.
        tokio::time::timeout(Duration::from_secs(5), rx.changed())
            .await
            .unwrap()
            .unwrap();
        fixture.send(&id, 0, payload(0)).await;
        fixture.send(&id, -1, vec![]).await;
        let id = fixture.request(1).await;
        assert!(matches!(*rx.borrow(), RecoveryStatus::Recovering));
        fixture.send(&id, 1, payload(0)).await;
        fixture.send(&id, 2, payload(0)).await;
        fixture.send(&id, 3, payload(0)).await;
        fixture.send(&id, -1, vec![]).await;
        until(&mut rx, |s| matches!(s, RecoveryStatus::Ready)).await;
        assert_eq!(target.events.lock().unwrap().len(), 4);
        cancel.cancel();
        task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn queued_live_gap_at_replay_end_is_resolved_before_success() {
        let mut fixture = Fixture::new();
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let target = Arc::new(Target {
            block_first: Some((entered.clone(), release.clone())),
            ..Default::default()
        });
        let cancel = CancellationToken::new();
        let (tx, mut rx) = watch::channel(RecoveryStatus::Recovering);
        let task = tokio::spawn(run_source(
            fixture.source.clone(),
            target.clone(),
            1,
            cancel.clone(),
            tx,
        ));
        let id = fixture.request(0).await;
        fixture.send(&id, 0, payload(0)).await;
        tokio::time::timeout(Duration::from_secs(5), entered.notified())
            .await
            .unwrap();
        fixture
            .live
            .send(Multipart::from(vec![
                vec![],
                3_u64.to_be_bytes().to_vec(),
                payload(0),
            ]))
            .await
            .unwrap();
        fixture.send(&id, -1, vec![]).await;
        // Let both socket transports deliver while application is held at the
        // explicit barrier. The end marker and live batch must be queued.
        tokio::time::sleep(Duration::from_millis(50)).await;
        release.notify_one();
        let id = fixture.request(1).await;
        assert!(matches!(*rx.borrow(), RecoveryStatus::Recovering));
        fixture.send(&id, 1, payload(0)).await;
        fixture.send(&id, 2, payload(0)).await;
        fixture.send(&id, -1, vec![]).await;
        until(&mut rx, |s| matches!(s, RecoveryStatus::Ready)).await;
        assert_eq!(target.events.lock().unwrap().len(), 4);
        cancel.cancel();
        task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn expired_prefix_is_reported_before_end_or_large_suffix() {
        let mut fixture = Fixture::new();
        let cancel = CancellationToken::new();
        let (tx, mut rx) = watch::channel(RecoveryStatus::Recovering);
        let task = tokio::spawn(run_source(
            fixture.source.clone(),
            Arc::new(Target::default()),
            1,
            cancel.clone(),
            tx,
        ));
        let id = fixture.request(0).await;
        fixture.send(&id, 500, payload(0)).await;
        until(&mut rx, |s| {
            matches!(
                s,
                RecoveryStatus::MissingHistory {
                    expected: 0,
                    got: 500
                }
            )
        })
        .await;
        cancel.cancel();
        task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn failed_application_and_wrong_rank_never_become_ready() {
        for (fail, rank) in [(true, 0), (false, 1)] {
            let mut fixture = Fixture::new();
            let (tx, rx) = watch::channel(RecoveryStatus::Recovering);
            let task = tokio::spawn(run_source(
                fixture.source.clone(),
                Arc::new(Target {
                    fail,
                    ..Default::default()
                }),
                1,
                CancellationToken::new(),
                tx,
            ));
            let id = fixture.request(0).await;
            fixture.send(&id, 0, payload(rank)).await;
            fixture.send(&id, -1, vec![]).await;
            assert!(
                tokio::time::timeout(Duration::from_secs(5), task)
                    .await
                    .unwrap()
                    .unwrap()
                    .is_err()
            );
            assert!(matches!(*rx.borrow(), RecoveryStatus::Recovering));
        }
    }

    #[tokio::test]
    async fn disconnect_during_bootstrap_is_uncertainty() {
        let mut fixture = Fixture::new();
        let (tx, rx) = watch::channel(RecoveryStatus::Recovering);
        let task = tokio::spawn(run_source(
            fixture.source.clone(),
            Arc::new(Target::default()),
            1,
            CancellationToken::new(),
            tx,
        ));
        let _id = fixture.request(0).await;
        drop(fixture.live);
        assert!(
            tokio::time::timeout(Duration::from_secs(5), task)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
        assert!(matches!(*rx.borrow(), RecoveryStatus::Recovering));
    }
    #[tokio::test]
    async fn replay_timeout_is_uncertainty_not_missing_history() {
        let mut fixture = Fixture::new();
        let (tx, rx) = watch::channel(RecoveryStatus::Recovering);
        let task = tokio::spawn(run_source(
            fixture.source.clone(),
            Arc::new(Target::default()),
            1,
            CancellationToken::new(),
            tx,
        ));
        let _id = fixture.request(0).await;
        // Keep the connection open but withhold every replay response.
        let result = tokio::time::timeout(Duration::from_secs(8), task)
            .await
            .unwrap()
            .unwrap();
        assert!(result.unwrap_err().to_string().contains("timed out"));
        assert!(matches!(*rx.borrow(), RecoveryStatus::Recovering));
    }
}
