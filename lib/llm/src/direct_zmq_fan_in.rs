// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, time::Duration};

use anyhow::Result;
use dynamo_runtime::{
    component::Endpoint,
    discovery::{
        DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
        EventChannelInstanceId, EventChannelQuery, EventScope, EventTransport,
    },
    traits::DistributedRuntimeProvider,
    transports::event_plane::{ValidatedEnvelope, ValidatedZmqSource, ValidatedZmqSourceError},
};
use futures::StreamExt;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const MAX_BACKOFF: Duration = Duration::from_secs(5);
const SOURCE_JOIN_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContinuityMode {
    Disabled,
    TrackFromZero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FanInEvent {
    SourceStarted,
    SourceStopped,
    Reconnect,
    Replacement,
    EnvelopeDecodeError,
    IdentityMismatch,
    SequenceGap { missing: u64 },
    OutOfOrder,
    ForcedAbort,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FanInObservation {
    pub publisher_id: u64,
    pub generation: u64,
    pub event: FanInEvent,
}

struct SourceTask {
    publisher_id: u64,
    endpoint: String,
    generation: u64,
    cancel: CancellationToken,
    handle: JoinHandle<Option<u64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Continuity {
    InOrder,
    Gap(u64),
    OutOfOrder,
}

#[derive(Debug, Clone, Copy)]
struct SequenceCursor {
    mode: ContinuityMode,
    high_watermark: Option<u64>,
}

impl SequenceCursor {
    fn new(mode: ContinuityMode, high_watermark: Option<u64>) -> Self {
        Self {
            mode,
            high_watermark,
        }
    }

    fn observe(&mut self, sequence: u64) -> Option<Continuity> {
        if self.mode == ContinuityMode::Disabled {
            return None;
        }

        let Some(high_watermark) = self.high_watermark else {
            self.high_watermark = Some(sequence);
            return Some(if sequence == 0 {
                Continuity::InOrder
            } else {
                Continuity::Gap(sequence)
            });
        };

        if sequence <= high_watermark {
            return Some(Continuity::OutOfOrder);
        }

        self.high_watermark = Some(sequence);
        let missing = sequence - high_watermark - 1;
        Some(if missing == 0 {
            Continuity::InOrder
        } else {
            Continuity::Gap(missing)
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn start_direct_zmq_fan_in<H, O>(
    endpoint: Endpoint,
    topic: &'static str,
    rcvhwm: i32,
    excluded_publisher_id: Option<u64>,
    continuity_mode: ContinuityMode,
    cancellation_token: CancellationToken,
    handler: H,
    observer: O,
) -> Result<JoinHandle<()>>
where
    H: Fn(ValidatedEnvelope) -> Result<()> + Clone + Send + Sync + 'static,
    O: Fn(FanInObservation) + Clone + Send + Sync + 'static,
{
    let query =
        DiscoveryQuery::EventChannels(EventChannelQuery::endpoint_topic(endpoint.id(), topic));
    let watch_cancel = cancellation_token.child_token();
    let initial_watch = endpoint
        .drt()
        .discovery()
        .list_and_watch(query.clone(), Some(watch_cancel.clone()))
        .await?;

    Ok(tokio::spawn(run_supervisor(
        endpoint,
        topic,
        query,
        rcvhwm,
        excluded_publisher_id,
        continuity_mode,
        cancellation_token,
        handler,
        observer,
        Some((initial_watch, watch_cancel)),
    )))
}

#[allow(clippy::too_many_arguments)]
async fn run_supervisor<H, O>(
    endpoint: Endpoint,
    topic: &'static str,
    query: DiscoveryQuery,
    rcvhwm: i32,
    excluded_publisher_id: Option<u64>,
    continuity_mode: ContinuityMode,
    cancellation_token: CancellationToken,
    handler: H,
    observer: O,
    mut next_watch: Option<(
        dynamo_runtime::discovery::DiscoveryStream,
        CancellationToken,
    )>,
) where
    H: Fn(ValidatedEnvelope) -> Result<()> + Clone + Send + Sync + 'static,
    O: Fn(FanInObservation) + Clone + Send + Sync + 'static,
{
    let expected_scope = EventScope::Endpoint {
        endpoint: endpoint.id(),
    };
    let discovery = endpoint.drt().discovery();
    let mut resume_cursors = HashMap::<u64, u64>::new();
    let mut next_generation = 1_u64;
    let mut retry_delay = INITIAL_BACKOFF;

    loop {
        let (mut watch, watch_cancel) = if let Some(watch) = next_watch.take() {
            watch
        } else {
            let watch_cancel = cancellation_token.child_token();
            let watch = tokio::select! {
                _ = cancellation_token.cancelled() => break,
                watch = discovery.list_and_watch(query.clone(), Some(watch_cancel.clone())) => watch,
            };
            let watch = match watch {
                Ok(watch) => watch,
                Err(error) => {
                    tracing::warn!(%error, topic, "failed to watch direct-ZMQ publishers");
                    if !sleep_or_cancel(retry_delay, &cancellation_token).await {
                        break;
                    }
                    retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
                    continue;
                }
            };
            (watch, watch_cancel)
        };
        retry_delay = INITIAL_BACKOFF;
        let mut sources = HashMap::<u64, SourceTask>::new();
        let mut restart_watch = true;

        loop {
            let event = tokio::select! {
                biased;
                _ = cancellation_token.cancelled() => {
                    restart_watch = false;
                    break;
                }
                event = watch.next() => event,
            };
            let Some(event) = event else {
                tracing::warn!(topic, "direct-ZMQ discovery watch ended");
                break;
            };

            match event {
                Ok(DiscoveryEvent::Added(DiscoveryInstance::EventChannel {
                    scope,
                    topic: discovered_topic,
                    instance_id,
                    transport,
                })) if scope == expected_scope && discovered_topic == topic => {
                    if excluded_publisher_id == Some(instance_id) {
                        continue;
                    }
                    let EventTransport::Zmq { endpoint } = transport else {
                        tracing::warn!(
                            publisher_id = instance_id,
                            topic,
                            "ignoring non-ZMQ event channel in direct fan-in"
                        );
                        continue;
                    };
                    if let Some(source) = sources.get(&instance_id)
                        && source.endpoint == endpoint
                    {
                        continue;
                    }

                    let high_watermark = if let Some(existing) = sources.remove(&instance_id) {
                        observe(
                            &observer,
                            instance_id,
                            existing.generation,
                            FanInEvent::Replacement,
                        );
                        stop_source(existing, &observer).await
                    } else {
                        resume_cursors.remove(&instance_id)
                    };
                    let generation = next_generation;
                    next_generation = next_generation.wrapping_add(1);
                    sources.insert(
                        instance_id,
                        spawn_source(
                            instance_id,
                            endpoint,
                            generation,
                            high_watermark,
                            topic,
                            rcvhwm,
                            continuity_mode,
                            handler.clone(),
                            observer.clone(),
                            cancellation_token.child_token(),
                        ),
                    );
                }
                Ok(DiscoveryEvent::Removed(DiscoveryInstanceId::EventChannel(
                    EventChannelInstanceId {
                        scope,
                        topic: discovered_topic,
                        instance_id,
                    },
                ))) if scope == expected_scope && discovered_topic == topic => {
                    resume_cursors.remove(&instance_id);
                    if let Some(source) = sources.remove(&instance_id) {
                        stop_source(source, &observer).await;
                    }
                }
                Ok(DiscoveryEvent::Added(_)) | Ok(DiscoveryEvent::Removed(_)) => {}
                Err(error) => {
                    tracing::warn!(%error, topic, "direct-ZMQ discovery watch failed");
                    break;
                }
            }
        }

        watch_cancel.cancel();
        for (publisher_id, high_watermark) in stop_sources(sources, &observer).await {
            resume_cursors.insert(publisher_id, high_watermark);
        }
        if !restart_watch {
            break;
        }
        if !sleep_or_cancel(retry_delay, &cancellation_token).await {
            break;
        }
        retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_source<H, O>(
    publisher_id: u64,
    endpoint: String,
    generation: u64,
    high_watermark: Option<u64>,
    topic: &'static str,
    rcvhwm: i32,
    continuity_mode: ContinuityMode,
    handler: H,
    observer: O,
    cancel: CancellationToken,
) -> SourceTask
where
    H: Fn(ValidatedEnvelope) -> Result<()> + Clone + Send + Sync + 'static,
    O: Fn(FanInObservation) + Clone + Send + Sync + 'static,
{
    let task_endpoint = endpoint.clone();
    let task_cancel = cancel.clone();
    observe(
        &observer,
        publisher_id,
        generation,
        FanInEvent::SourceStarted,
    );
    let handle = tokio::spawn(async move {
        run_source(
            publisher_id,
            task_endpoint,
            generation,
            high_watermark,
            topic,
            rcvhwm,
            continuity_mode,
            handler,
            observer,
            task_cancel,
        )
        .await
    });
    SourceTask {
        publisher_id,
        endpoint,
        generation,
        cancel,
        handle,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_source<H, O>(
    publisher_id: u64,
    endpoint: String,
    generation: u64,
    high_watermark: Option<u64>,
    topic: &'static str,
    rcvhwm: i32,
    continuity_mode: ContinuityMode,
    handler: H,
    observer: O,
    cancel: CancellationToken,
) -> Option<u64>
where
    H: Fn(ValidatedEnvelope) -> Result<()> + Clone + Send + Sync + 'static,
    O: Fn(FanInObservation) + Clone + Send + Sync + 'static,
{
    let mut cursor = SequenceCursor::new(continuity_mode, high_watermark);
    let mut retry_delay = INITIAL_BACKOFF;
    let mut connected_once = false;

    loop {
        let source = tokio::select! {
            _ = cancel.cancelled() => break,
            source = ValidatedZmqSource::connect(&endpoint, topic, publisher_id, rcvhwm) => source,
        };
        let mut source = match source {
            Ok(source) => source,
            Err(error) => {
                tracing::warn!(%error, publisher_id, generation, %endpoint, topic, "failed to connect direct-ZMQ source");
                observe(&observer, publisher_id, generation, FanInEvent::Reconnect);
                if !sleep_or_cancel(retry_delay, &cancel).await {
                    break;
                }
                retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
                continue;
            }
        };
        if connected_once {
            observe(&observer, publisher_id, generation, FanInEvent::Reconnect);
        }
        connected_once = true;
        retry_delay = INITIAL_BACKOFF;

        if !consume_connection(
            publisher_id,
            generation,
            &mut source,
            &mut cursor,
            &handler,
            &observer,
            &cancel,
        )
        .await
        {
            break;
        }
        if !sleep_or_cancel(retry_delay, &cancel).await {
            break;
        }
        retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
    }

    cursor.high_watermark
}

async fn consume_connection<H, O>(
    publisher_id: u64,
    generation: u64,
    source: &mut ValidatedZmqSource,
    cursor: &mut SequenceCursor,
    handler: &H,
    observer: &O,
    cancel: &CancellationToken,
) -> bool
where
    H: Fn(ValidatedEnvelope) -> Result<()> + Send + Sync,
    O: Fn(FanInObservation) + Send + Sync,
{
    loop {
        let envelope = tokio::select! {
            biased;
            _ = cancel.cancelled() => return false,
            envelope = source.next() => envelope,
        };
        let Some(envelope) = envelope else {
            return true;
        };
        let envelope = match envelope {
            Ok(envelope) => envelope,
            Err(ValidatedZmqSourceError::Receive(error)) => {
                tracing::warn!(%error, publisher_id, generation, "direct-ZMQ source receive failed");
                return true;
            }
            Err(ValidatedZmqSourceError::EnvelopeDecode(error)) => {
                tracing::warn!(%error, publisher_id, generation, "dropping malformed direct-ZMQ envelope");
                observe(
                    observer,
                    publisher_id,
                    generation,
                    FanInEvent::EnvelopeDecodeError,
                );
                continue;
            }
            Err(error @ ValidatedZmqSourceError::IdentityMismatch { .. }) => {
                tracing::warn!(%error, publisher_id, generation, "dropping misattributed direct-ZMQ envelope");
                observe(
                    observer,
                    publisher_id,
                    generation,
                    FanInEvent::IdentityMismatch,
                );
                continue;
            }
        };

        match cursor.observe(envelope.sequence) {
            None | Some(Continuity::InOrder) => {}
            Some(Continuity::Gap(missing)) => observe(
                observer,
                publisher_id,
                generation,
                FanInEvent::SequenceGap { missing },
            ),
            Some(Continuity::OutOfOrder) => {
                observe(observer, publisher_id, generation, FanInEvent::OutOfOrder)
            }
        }
        if let Err(error) = handler(envelope) {
            tracing::warn!(%error, publisher_id, generation, "direct-ZMQ source handler rejected an envelope");
        }
        tokio::task::consume_budget().await;
    }
}

async fn stop_source<O>(source: SourceTask, observer: &O) -> Option<u64>
where
    O: Fn(FanInObservation) + Send + Sync,
{
    stop_source_with_timeout(source, observer, SOURCE_JOIN_TIMEOUT).await
}

async fn stop_source_with_timeout<O>(
    source: SourceTask,
    observer: &O,
    join_timeout: Duration,
) -> Option<u64>
where
    O: Fn(FanInObservation) + Send + Sync,
{
    source.cancel.cancel();
    let publisher_id = source.publisher_id;
    let generation = source.generation;
    let mut handle = source.handle;
    let high_watermark = match tokio::time::timeout(join_timeout, &mut handle).await {
        Ok(Ok(high_watermark)) => high_watermark,
        Ok(Err(error)) if error.is_cancelled() => None,
        Ok(Err(error)) => {
            tracing::warn!(%error, publisher_id, generation, "direct-ZMQ source task failed during shutdown");
            None
        }
        Err(_) => {
            handle.abort();
            let _ = handle.await;
            observe(observer, publisher_id, generation, FanInEvent::ForcedAbort);
            None
        }
    };
    observe(
        observer,
        publisher_id,
        generation,
        FanInEvent::SourceStopped,
    );
    high_watermark
}

async fn stop_sources<O>(sources: HashMap<u64, SourceTask>, observer: &O) -> HashMap<u64, u64>
where
    O: Fn(FanInObservation) + Clone + Send + Sync,
{
    for source in sources.values() {
        source.cancel.cancel();
    }
    futures::future::join_all(sources.into_iter().map(|(publisher_id, source)| {
        let observer = observer.clone();
        async move { (publisher_id, stop_source(source, &observer).await) }
    }))
    .await
    .into_iter()
    .filter_map(|(publisher_id, high_watermark)| {
        high_watermark.map(|high_watermark| (publisher_id, high_watermark))
    })
    .collect()
}

fn observe<O>(observer: &O, publisher_id: u64, generation: u64, event: FanInEvent)
where
    O: Fn(FanInObservation),
{
    observer(FanInObservation {
        publisher_id,
        generation,
        event,
    });
}

async fn sleep_or_cancel(delay: Duration, cancellation_token: &CancellationToken) -> bool {
    tokio::select! {
        _ = cancellation_token.cancelled() => false,
        _ = tokio::time::sleep(delay) => true,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        sync::{
            Arc, Condvar, Mutex,
            atomic::{AtomicBool, Ordering},
        },
    };

    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        transports::event_plane::{EventPublisher, EventTransportKind},
    };

    use super::*;

    struct BlockingGate(Arc<(Mutex<bool>, Condvar)>);

    impl BlockingGate {
        fn new() -> Self {
            Self(Arc::new((Mutex::new(false), Condvar::new())))
        }

        fn open(&self) {
            let (lock, ready) = &*self.0;
            *lock.lock().unwrap() = true;
            ready.notify_all();
        }
    }

    impl Drop for BlockingGate {
        fn drop(&mut self) {
            self.open();
        }
    }

    #[test]
    fn continuity_tracks_initial_gaps_and_retains_high_watermark() {
        let mut cursor = SequenceCursor::new(ContinuityMode::TrackFromZero, None);
        assert_eq!(cursor.observe(3), Some(Continuity::Gap(3)));
        assert_eq!(cursor.observe(4), Some(Continuity::InOrder));
        assert_eq!(cursor.observe(7), Some(Continuity::Gap(2)));
        assert_eq!(cursor.observe(6), Some(Continuity::OutOfOrder));

        let mut resumed = SequenceCursor::new(ContinuityMode::TrackFromZero, cursor.high_watermark);
        assert_eq!(resumed.observe(8), Some(Continuity::InOrder));
        assert_eq!(
            SequenceCursor::new(ContinuityMode::Disabled, None).observe(9),
            None
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[serial_test::serial]
    async fn fan_in_tracks_source_recreation_and_excludes_local_publisher() -> Result<()> {
        const TOPIC: &str = "direct-zmq-fan-in-lifecycle";

        let runtime = Runtime::from_current()?;
        let distributed =
            DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
        let endpoint = distributed
            .namespace(format!("direct-zmq-fan-in-{}", uuid::Uuid::new_v4()))?
            .component("frontend")?
            .endpoint("generate");
        let local =
            EventPublisher::for_endpoint_with_transport(&endpoint, TOPIC, EventTransportKind::Zmq)
                .await?;
        let (observation_tx, mut observation_rx) = tokio::sync::mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        let supervisor = start_direct_zmq_fan_in(
            endpoint.clone(),
            TOPIC,
            32,
            Some(local.publisher_id()),
            ContinuityMode::Disabled,
            cancel.clone(),
            |_| Ok(()),
            move |observation| {
                let _ = observation_tx.send(observation);
            },
        )
        .await?;
        let mut active = HashSet::new();

        for _ in 0..2 {
            let mut publishers = Vec::with_capacity(32);
            for _ in 0..32 {
                publishers.push(
                    EventPublisher::for_endpoint_with_transport(
                        &endpoint,
                        TOPIC,
                        EventTransportKind::Zmq,
                    )
                    .await?,
                );
            }
            wait_for_source_count(&mut observation_rx, &mut active, 32).await;
            assert!(
                !active
                    .iter()
                    .any(|(publisher_id, _)| { *publisher_id == local.publisher_id() })
            );

            drop(publishers);
            wait_for_source_count(&mut observation_rx, &mut active, 0).await;
        }

        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), supervisor)
            .await
            .expect("fan-in supervisor should stop after cancellation")
            .expect("fan-in supervisor should not panic");
        assert!(active.is_empty());
        drop(local);
        distributed.shutdown();
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[serial_test::serial]
    async fn one_blocked_source_does_not_stall_another_source() -> Result<()> {
        const TOPIC: &str = "direct-zmq-fan-in-concurrency";

        let runtime = Runtime::from_current()?;
        let distributed =
            DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
        let endpoint = distributed
            .namespace(format!("direct-zmq-concurrency-{}", uuid::Uuid::new_v4()))?
            .component("frontend")?
            .endpoint("generate");
        let publisher_a =
            EventPublisher::for_endpoint_with_transport(&endpoint, TOPIC, EventTransportKind::Zmq)
                .await?;
        let publisher_b =
            EventPublisher::for_endpoint_with_transport(&endpoint, TOPIC, EventTransportKind::Zmq)
                .await?;
        let publisher_a_id = publisher_a.publisher_id();
        let publisher_b_id = publisher_b.publisher_id();
        let release = BlockingGate::new();
        let block_enabled = Arc::new(AtomicBool::new(false));
        let blocked_once = Arc::new(AtomicBool::new(false));
        let order_violation = Arc::new(AtomicBool::new(false));
        let last_sequences = Arc::new(Mutex::new(HashMap::<u64, u64>::new()));
        let (a_seen_tx, mut a_seen_rx) = tokio::sync::mpsc::unbounded_channel();
        let (a_started_tx, mut a_started_rx) = tokio::sync::mpsc::unbounded_channel();
        let (a_released_tx, mut a_released_rx) = tokio::sync::mpsc::unbounded_channel();
        let (b_seen_tx, mut b_seen_rx) = tokio::sync::mpsc::unbounded_channel();

        let handler_release = release.0.clone();
        let handler_block_enabled = block_enabled.clone();
        let handler_blocked_once = blocked_once.clone();
        let handler_order_violation = order_violation.clone();
        let handler_last_sequences = last_sequences.clone();
        let handler = move |envelope: ValidatedEnvelope| {
            let mut last_sequences = handler_last_sequences.lock().unwrap();
            if last_sequences
                .insert(envelope.publisher_id, envelope.sequence)
                .is_some_and(|previous| previous >= envelope.sequence)
            {
                handler_order_violation.store(true, Ordering::Relaxed);
            }
            drop(last_sequences);

            if envelope.publisher_id == publisher_a_id {
                let _ = a_seen_tx.send(());
                if handler_block_enabled.load(Ordering::Relaxed)
                    && !handler_blocked_once.swap(true, Ordering::Relaxed)
                {
                    let _ = a_started_tx.send(());
                    tokio::task::block_in_place(|| {
                        let (lock, ready) = &*handler_release;
                        let released = lock.lock().unwrap();
                        let _ = ready
                            .wait_timeout_while(released, Duration::from_secs(5), |released| {
                                !*released
                            })
                            .unwrap();
                    });
                    let _ = a_released_tx.send(());
                }
            }
            if envelope.publisher_id == publisher_b_id {
                let _ = b_seen_tx.send(());
            }
            Ok(())
        };
        let cancel = CancellationToken::new();
        let supervisor = start_direct_zmq_fan_in(
            endpoint,
            TOPIC,
            32,
            None,
            ContinuityMode::Disabled,
            cancel.clone(),
            handler,
            |_| {},
        )
        .await?;

        publish_until_observed(&publisher_a, &mut a_seen_rx, Duration::from_secs(5)).await;
        publish_until_observed(&publisher_b, &mut b_seen_rx, Duration::from_secs(5)).await;
        block_enabled.store(true, Ordering::Relaxed);
        publish_until_observed(&publisher_a, &mut a_started_rx, Duration::from_secs(5)).await;
        publish_until_observed(&publisher_b, &mut b_seen_rx, Duration::from_secs(1)).await;
        assert!(
            a_released_rx.try_recv().is_err(),
            "source B must progress while source A remains blocked"
        );
        release.open();
        tokio::time::timeout(Duration::from_secs(1), a_released_rx.recv())
            .await
            .expect("source A handler should resume after release")
            .expect("source A release observer should stay open");

        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let both_seen = {
                    let seen = last_sequences.lock().unwrap();
                    seen.contains_key(&publisher_a_id) && seen.contains_key(&publisher_b_id)
                };
                if both_seen {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("both sources should make progress");
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), supervisor)
            .await
            .expect("fan-in supervisor should stop after cancellation")
            .expect("fan-in supervisor should not panic");
        assert!(!order_violation.load(Ordering::Relaxed));
        drop(publisher_a);
        drop(publisher_b);
        distributed.shutdown();
        Ok(())
    }

    #[tokio::test]
    async fn source_shutdown_joins_and_forces_abort_after_timeout() {
        let observations = Arc::new(Mutex::new(Vec::new()));
        let observer = {
            let observations = observations.clone();
            move |observation| observations.lock().unwrap().push(observation)
        };

        let cooperative_cancel = CancellationToken::new();
        let task_cancel = cooperative_cancel.clone();
        let cooperative = SourceTask {
            publisher_id: 1,
            endpoint: "cooperative".to_string(),
            generation: 1,
            cancel: cooperative_cancel,
            handle: tokio::spawn(async move {
                task_cancel.cancelled().await;
                Some(17)
            }),
        };
        assert_eq!(
            stop_source_with_timeout(cooperative, &observer, Duration::from_secs(1)).await,
            Some(17)
        );

        let stuck_handle = tokio::spawn(std::future::pending::<Option<u64>>());
        let abort_handle = stuck_handle.abort_handle();
        let stuck = SourceTask {
            publisher_id: 2,
            endpoint: "stuck".to_string(),
            generation: 2,
            cancel: CancellationToken::new(),
            handle: stuck_handle,
        };
        assert_eq!(
            stop_source_with_timeout(stuck, &observer, Duration::from_millis(10)).await,
            None
        );
        assert!(abort_handle.is_finished());
        assert!(observations.lock().unwrap().iter().any(|observation| {
            observation.publisher_id == 2 && observation.event == FanInEvent::ForcedAbort
        }));
    }

    async fn wait_for_source_count(
        observations: &mut tokio::sync::mpsc::UnboundedReceiver<FanInObservation>,
        active: &mut HashSet<(u64, u64)>,
        expected: usize,
    ) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while active.len() != expected {
                let observation = observations
                    .recv()
                    .await
                    .expect("fan-in observation channel closed");
                let key = (observation.publisher_id, observation.generation);
                match observation.event {
                    FanInEvent::SourceStarted => assert!(active.insert(key)),
                    FanInEvent::SourceStopped => assert!(active.remove(&key)),
                    _ => {}
                }
            }
        })
        .await
        .unwrap_or_else(|_| panic!("source count did not reach {expected}"));
    }

    async fn publish_until_observed(
        publisher: &EventPublisher,
        observed: &mut tokio::sync::mpsc::UnboundedReceiver<()>,
        timeout: Duration,
    ) {
        tokio::time::timeout(timeout, async {
            loop {
                publisher.publish(&()).await.unwrap();
                if tokio::time::timeout(Duration::from_millis(20), observed.recv())
                    .await
                    .is_ok()
                {
                    break;
                }
            }
        })
        .await
        .expect("source should receive a warmed-up message");
    }
}
