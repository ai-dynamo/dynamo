// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine replay framing and bounded, ordered replay/live merge.

use anyhow::{Result, bail};
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq)]
pub enum ReplayFrame {
    Batch(u64, Vec<u8>),
    End,
}

pub fn decode_replay(frames: Vec<Vec<u8>>) -> Result<ReplayFrame> {
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
pub struct ReplayCursor {
    next: u64,
    pending: BTreeMap<u64, Vec<u8>>,
    bytes: usize,
    limit: usize,
}

impl ReplayCursor {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            next: 0,
            pending: BTreeMap::new(),
            bytes: 0,
            limit: max_bytes,
        }
    }
    pub fn next(&self) -> u64 {
        self.next
    }
    pub fn insert(&mut self, sequence: u64, payload: Vec<u8>) -> Result<()> {
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
    pub fn pop_contiguous(&mut self) -> Result<Option<(u64, Vec<u8>)>> {
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
    pub fn gap(&self) -> Option<(u64, u64)> {
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

#[derive(Clone, Debug)]
pub struct ReplaySource {
    pub endpoint: String,
    pub replay_endpoint: String,
    pub topic: String,
    pub dp_rank: u32,
}

#[derive(Clone, Debug)]
pub enum RecoveryStatus {
    Recovering,
    Ready,
    MissingHistory { expected: u64, got: u64 },
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
}

/// Socket lifetimes are one recovery session: automatic reconnect is disabled
/// so replacement-engine sequences cannot enter an existing cursor. The owner
/// verifies the engine instance again before publishing readiness.
pub async fn run_source<T: RecoveryTarget + 'static>(
    source: ReplaySource,
    publisher: std::sync::Arc<T>,
    worker_id: u64,
    cancel: tokio_util::sync::CancellationToken,
    status: tokio::sync::watch::Sender<RecoveryStatus>,
) -> Result<()> {
    use dynamo_kv_router::{protocols::*, zmq_wire::*};
    use futures::{SinkExt, StreamExt};
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
        .set_reconnect_ivl(-1)
        .set_rcvhwm(8)
        .set_maxmsgsize(LIMIT as i64)
        .monitor("inproc://live-monitor", monitor_events)
        .connect(&source.endpoint)?
        .subscribe(source.topic.as_bytes())?;
    let mut live_monitor = tmq::pair(&context).connect("inproc://live-monitor")?;
    let mut replay = tmq::dealer(&context)
        .set_linger(0)
        .set_ipv6(true)
        .set_reconnect_ivl(-1)
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
        // An inactivity timeout is uncertainty, never evidence of expired history.
        let mut deadline = tokio::time::Instant::now() + TIMEOUT;
        loop {
            let (sequence, payload) = tokio::select! {
                biased;
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
        if cursor.gap().is_none() {
            status.send_replace(RecoveryStatus::Ready);
        }
        // Consume live immediately. Periodic replay also repairs a lost final
        // PUB batch when there is no subsequent live sequence to expose a gap.
        let poll_at = tokio::time::Instant::now() + Duration::from_secs(1);
        loop {
            let message = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Ok(()),
                _ = live_monitor.next() => bail!("KV live connection lost"),
                _ = replay_monitor.next() => bail!("KV replay connection lost"),
                _ = tokio::time::sleep_until(poll_at) => break,
                message = live.next() => message.ok_or_else(|| anyhow::anyhow!("KV live socket ended"))??,
            };
            let (sequence, payload) = decode_live(message)?;
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
            if cursor.gap().is_some() {
                break;
            }
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
        let batch = dynamo_kv_router::zmq_wire::decode_event_batch(&payload)?;
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
        publisher.apply(events).await?;
    }
    Ok(())
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
    async fn replay_overlap_and_later_live_gap_are_recovered_in_order() {
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
        // A later live batch exposes a hole; replay supplies the missing batch
        // and overlaps the live batch. Every batch is applied exactly once.
        fixture
            .live
            .send(Multipart::from(vec![
                vec![],
                3_u64.to_be_bytes().to_vec(),
                payload(0),
            ]))
            .await
            .unwrap();
        let id = fixture.request(2).await;
        fixture.send(&id, 2, payload(0)).await;
        fixture.send(&id, 3, payload(0)).await;
        fixture.send(&id, -1, vec![]).await;
        until(&mut rx, |s| matches!(s, RecoveryStatus::Ready)).await;
        tokio::time::timeout(Duration::from_secs(5), async {
            while target.events.lock().unwrap().len() < 4 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
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
    async fn disconnected_publisher_ends_session_without_reconnecting() {
        let mut fixture = Fixture::new();
        let (tx, mut rx) = watch::channel(RecoveryStatus::Recovering);
        let task = tokio::spawn(run_source(
            fixture.source.clone(),
            Arc::new(Target::default()),
            1,
            CancellationToken::new(),
            tx,
        ));
        let id = fixture.request(0).await;
        fixture.send(&id, -1, vec![]).await;
        until(&mut rx, |s| matches!(s, RecoveryStatus::Ready)).await;
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
