// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, ffi::OsString, sync::Arc, time::Duration};

use dynamo_kv_router::{
    ActiveSequencesMultiWorker, SequencePublisher,
    protocols::{ActiveSequenceEventBatch, MAX_REPLICA_BATCH_EVENTS},
};
use dynamo_runtime::{
    component::Endpoint,
    discovery::{
        DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
        EventChannelInstanceId, EventChannelQuery, EventScope, EventTransport, EventTransportKind,
    },
    traits::DistributedRuntimeProvider,
    transports::event_plane::{Codec, ZmqSubTransport, uses_direct_zmq},
};
use futures::StreamExt;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::ActiveSequencesMulti;
use crate::kv_router::{ACTIVE_SEQUENCES_SUBJECT, metrics::ActiveSequenceZmqIngressMetrics};

const DIRECT_ZMQ_ENV: &str = "DYN_ROUTER_ACTIVE_SEQUENCE_DIRECT_ZMQ";
const RCVHWM_ENV: &str = "DYN_ROUTER_ACTIVE_SEQUENCE_ZMQ_RCVHWM";
const DEFAULT_RCVHWM: i32 = 1024;
const INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const MAX_BACKOFF: Duration = Duration::from_secs(5);
const SOURCE_JOIN_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DirectZmqSequenceConfig {
    enabled: bool,
    pub(super) rcvhwm: i32,
}

impl DirectZmqSequenceConfig {
    pub(super) fn from_env() -> Self {
        Self::from_lookup(|key| std::env::var_os(key))
    }

    fn from_lookup(mut get_env: impl FnMut(&str) -> Option<OsString>) -> Self {
        let enabled = match get_env(DIRECT_ZMQ_ENV) {
            None => true,
            Some(raw) => match raw.to_string_lossy().trim().to_ascii_lowercase().as_str() {
                "0" | "false" => false,
                "1" | "true" => true,
                value => {
                    tracing::warn!(
                        env = DIRECT_ZMQ_ENV,
                        %value,
                        "invalid direct-ZMQ active-sequence setting; using enabled default"
                    );
                    true
                }
            },
        };

        let rcvhwm = match get_env(RCVHWM_ENV) {
            None => DEFAULT_RCVHWM,
            Some(raw) => match raw.to_string_lossy().trim().parse::<i32>() {
                Ok(value) if value > 0 => value,
                _ => {
                    tracing::warn!(
                        env = RCVHWM_ENV,
                        value = %raw.to_string_lossy(),
                        default = DEFAULT_RCVHWM,
                        "invalid direct-ZMQ active-sequence receive HWM; using default"
                    );
                    DEFAULT_RCVHWM
                }
            },
        };

        Self { enabled, rcvhwm }
    }

    pub(super) fn should_use_direct(self, transport_kind: EventTransportKind) -> bool {
        self.should_use_direct_for_topology(transport_kind, uses_direct_zmq(transport_kind))
    }

    fn should_use_direct_for_topology(
        self,
        transport_kind: EventTransportKind,
        direct_zmq_topology: bool,
    ) -> bool {
        self.enabled && transport_kind == EventTransportKind::Zmq && direct_zmq_topology
    }
}

struct SourceTask {
    endpoint: String,
    generation: u64,
    cancel: CancellationToken,
    handle: JoinHandle<Option<u64>>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct SequenceCursor {
    high_watermark: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Continuity {
    InOrder,
    Gap(u64),
    OutOfOrder,
}

impl SequenceCursor {
    fn from_high_watermark(high_watermark: Option<u64>) -> Self {
        Self { high_watermark }
    }

    fn observe(&mut self, sequence: u64) -> Continuity {
        let Some(high_watermark) = self.high_watermark else {
            self.high_watermark = Some(sequence);
            return if sequence == 0 {
                Continuity::InOrder
            } else {
                Continuity::Gap(sequence)
            };
        };

        if sequence <= high_watermark {
            return Continuity::OutOfOrder;
        }

        self.high_watermark = Some(sequence);
        let missing = sequence - high_watermark - 1;
        if missing == 0 {
            Continuity::InOrder
        } else {
            Continuity::Gap(missing)
        }
    }
}

pub(super) fn start(
    endpoint: Endpoint,
    tracker: Arc<ActiveSequencesMulti>,
    rcvhwm: i32,
    cancellation_token: CancellationToken,
) {
    tokio::spawn(run_supervisor(
        endpoint,
        tracker,
        rcvhwm,
        cancellation_token,
    ));
}

async fn run_supervisor<P>(
    endpoint: Endpoint,
    tracker: Arc<ActiveSequencesMultiWorker<P>>,
    rcvhwm: i32,
    cancellation_token: CancellationToken,
) where
    P: SequencePublisher + 'static,
{
    run_supervisor_inner(endpoint, tracker, rcvhwm, cancellation_token, |_| {}).await;
}

async fn run_supervisor_inner<P, F>(
    endpoint: Endpoint,
    tracker: Arc<ActiveSequencesMultiWorker<P>>,
    rcvhwm: i32,
    cancellation_token: CancellationToken,
    mut observe_source_count: F,
) where
    P: SequencePublisher + 'static,
    F: FnMut(usize),
{
    let expected_scope = EventScope::Endpoint {
        endpoint: endpoint.id(),
    };
    let query = DiscoveryQuery::EventChannels(EventChannelQuery::endpoint_topic(
        endpoint.id(),
        ACTIVE_SEQUENCES_SUBJECT,
    ));
    let discovery = endpoint.drt().discovery();
    let metrics = ActiveSequenceZmqIngressMetrics::from_component(endpoint.component());
    let mut resume_cursors = HashMap::<u64, u64>::new();
    let mut next_generation = 1_u64;
    let mut retry_delay = INITIAL_BACKOFF;

    loop {
        let watch_cancel = cancellation_token.child_token();
        let watch = tokio::select! {
            _ = cancellation_token.cancelled() => break,
            watch = discovery.list_and_watch(query.clone(), Some(watch_cancel.clone())) => watch,
        };
        let mut watch = match watch {
            Ok(watch) => watch,
            Err(error) => {
                tracing::warn!(%error, "failed to watch active-sequence ZMQ publishers");
                if !sleep_or_cancel(retry_delay, &cancellation_token).await {
                    break;
                }
                retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
                continue;
            }
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
                tracing::warn!("active-sequence ZMQ discovery watch ended");
                break;
            };

            match event {
                Ok(DiscoveryEvent::Added(DiscoveryInstance::EventChannel {
                    scope,
                    topic,
                    instance_id,
                    transport,
                })) if scope == expected_scope && topic == ACTIVE_SEQUENCES_SUBJECT => {
                    let EventTransport::Zmq { endpoint } = transport else {
                        tracing::warn!(
                            publisher_id = instance_id,
                            "ignoring non-ZMQ active-sequence event channel"
                        );
                        continue;
                    };

                    if let Some(existing) = sources.get(&instance_id) {
                        if existing.endpoint == endpoint {
                            continue;
                        }
                    }

                    let high_watermark = if let Some(existing) = sources.remove(&instance_id) {
                        tracing::warn!(
                            publisher_id = instance_id,
                            old_endpoint = %existing.endpoint,
                            new_endpoint = %endpoint,
                            "replacing active-sequence ZMQ source endpoint"
                        );
                        metrics.record_replacement();
                        stop_source(existing, &metrics).await
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
                            rcvhwm,
                            tracker.clone(),
                            metrics.clone(),
                            cancellation_token.child_token(),
                        ),
                    );
                    observe_source_count(sources.len());
                }
                Ok(DiscoveryEvent::Removed(DiscoveryInstanceId::EventChannel(
                    EventChannelInstanceId {
                        scope,
                        topic,
                        instance_id,
                    },
                ))) if scope == expected_scope && topic == ACTIVE_SEQUENCES_SUBJECT => {
                    resume_cursors.remove(&instance_id);
                    if let Some(source) = sources.remove(&instance_id) {
                        stop_source(source, &metrics).await;
                        observe_source_count(sources.len());
                    }
                }
                Ok(DiscoveryEvent::Added(_)) | Ok(DiscoveryEvent::Removed(_)) => {}
                Err(error) => {
                    tracing::warn!(%error, "active-sequence ZMQ discovery watch failed");
                    break;
                }
            }
        }

        watch_cancel.cancel();
        for (publisher_id, high_watermark) in stop_sources(sources, &metrics).await {
            resume_cursors.insert(publisher_id, high_watermark);
        }
        observe_source_count(0);
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
fn spawn_source<P>(
    publisher_id: u64,
    endpoint: String,
    generation: u64,
    high_watermark: Option<u64>,
    rcvhwm: i32,
    tracker: Arc<ActiveSequencesMultiWorker<P>>,
    metrics: Arc<ActiveSequenceZmqIngressMetrics>,
    cancel: CancellationToken,
) -> SourceTask
where
    P: SequencePublisher + 'static,
{
    let task_endpoint = endpoint.clone();
    let task_cancel = cancel.clone();
    metrics.source_started();
    let handle = tokio::spawn(async move {
        run_source(
            publisher_id,
            task_endpoint,
            generation,
            high_watermark,
            rcvhwm,
            tracker,
            metrics,
            task_cancel,
        )
        .await
    });
    SourceTask {
        endpoint,
        generation,
        cancel,
        handle,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_source<P>(
    publisher_id: u64,
    endpoint: String,
    generation: u64,
    high_watermark: Option<u64>,
    rcvhwm: i32,
    tracker: Arc<ActiveSequencesMultiWorker<P>>,
    metrics: Arc<ActiveSequenceZmqIngressMetrics>,
    cancel: CancellationToken,
) -> Option<u64>
where
    P: SequencePublisher + 'static,
{
    let mut cursor = SequenceCursor::from_high_watermark(high_watermark);
    let mut retry_delay = INITIAL_BACKOFF;
    let mut connected_once = false;

    loop {
        let stream = tokio::select! {
            _ = cancel.cancelled() => break,
            stream = ZmqSubTransport::connect_single_consumer_with_rcvhwm(
                &endpoint,
                ACTIVE_SEQUENCES_SUBJECT,
                rcvhwm,
            ) => stream,
        };
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                tracing::warn!(
                    %error,
                    publisher_id,
                    generation,
                    %endpoint,
                    "failed to connect direct-ZMQ active-sequence source"
                );
                metrics.record_reconnect();
                if !sleep_or_cancel(retry_delay, &cancel).await {
                    break;
                }
                retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
                continue;
            }
        };
        if connected_once {
            metrics.record_reconnect();
        }
        connected_once = true;
        retry_delay = INITIAL_BACKOFF;

        if !consume_connection(
            publisher_id,
            generation,
            &mut stream,
            &tracker,
            &metrics,
            &cancel,
            &mut cursor,
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

async fn consume_connection<P>(
    publisher_id: u64,
    generation: u64,
    stream: &mut dynamo_runtime::transports::event_plane::zmq_transport::ZmqWireStream,
    tracker: &ActiveSequencesMultiWorker<P>,
    metrics: &ActiveSequenceZmqIngressMetrics,
    cancel: &CancellationToken,
    cursor: &mut SequenceCursor,
) -> bool
where
    P: SequencePublisher + 'static,
{
    let codec = Codec::default();
    loop {
        let message = tokio::select! {
            biased;
            _ = cancel.cancelled() => return false,
            message = stream.next() => message,
        };
        let Some(message) = message else {
            return true;
        };
        let message = match message {
            Ok(message) => message,
            Err(error) => {
                tracing::warn!(
                    %error,
                    publisher_id,
                    generation,
                    "direct-ZMQ active-sequence source stream failed"
                );
                return true;
            }
        };

        let envelope = match codec.decode_envelope(&message.payload) {
            Ok(envelope) => envelope,
            Err(error) => {
                tracing::warn!(
                    %error,
                    publisher_id,
                    generation,
                    "failed to decode direct-ZMQ active-sequence envelope"
                );
                metrics.record_envelope_decode_error();
                continue;
            }
        };
        if envelope.publisher_id != publisher_id
            || envelope.publisher_id != message.publisher_id
            || envelope.sequence != message.sequence
            || envelope.topic != ACTIVE_SEQUENCES_SUBJECT
        {
            tracing::warn!(
                publisher_id,
                generation,
                frame_publisher_id = message.publisher_id,
                frame_sequence = message.sequence,
                envelope_publisher_id = envelope.publisher_id,
                envelope_sequence = envelope.sequence,
                envelope_topic = %envelope.topic,
                "dropping direct-ZMQ active-sequence envelope with inconsistent attribution"
            );
            metrics.record_identity_error();
            continue;
        }

        match cursor.observe(envelope.sequence) {
            Continuity::InOrder => {}
            Continuity::Gap(missing) => metrics.record_gap(missing),
            Continuity::OutOfOrder => metrics.record_out_of_order(),
        }

        let batch = match codec.decode_payload::<ActiveSequenceEventBatch>(&envelope.payload) {
            Ok(batch) => batch,
            Err(error) => {
                tracing::warn!(
                    %error,
                    publisher_id,
                    generation,
                    "failed to decode direct-ZMQ active-sequence batch"
                );
                metrics.record_payload_decode_error();
                continue;
            }
        };
        if batch.events.len() > MAX_REPLICA_BATCH_EVENTS {
            tracing::warn!(
                publisher_id,
                generation,
                event_count = batch.events.len(),
                max_event_count = MAX_REPLICA_BATCH_EVENTS,
                "dropping oversized direct-ZMQ active-sequence batch"
            );
            metrics.record_payload_decode_error();
            continue;
        }
        let event_count = batch.events.len();
        metrics.record_received(event_count);
        tracker.apply_replica_batch(batch.events);
        metrics.record_handled(event_count);
        tokio::task::consume_budget().await;
    }
}

async fn stop_source(source: SourceTask, metrics: &ActiveSequenceZmqIngressMetrics) -> Option<u64> {
    stop_source_with_timeout(source, metrics, SOURCE_JOIN_TIMEOUT).await
}

async fn stop_sources(
    sources: HashMap<u64, SourceTask>,
    metrics: &ActiveSequenceZmqIngressMetrics,
) -> HashMap<u64, u64> {
    for source in sources.values() {
        source.cancel.cancel();
    }
    futures::future::join_all(
        sources
            .into_iter()
            .map(|(publisher_id, source)| async move {
                (publisher_id, stop_source(source, metrics).await)
            }),
    )
    .await
    .into_iter()
    .filter_map(|(publisher_id, high_watermark)| {
        high_watermark.map(|high_watermark| (publisher_id, high_watermark))
    })
    .collect()
}

async fn stop_source_with_timeout(
    source: SourceTask,
    metrics: &ActiveSequenceZmqIngressMetrics,
    join_timeout: Duration,
) -> Option<u64> {
    tracing::debug!(
        generation = source.generation,
        endpoint = %source.endpoint,
        "stopping direct-ZMQ active-sequence source"
    );
    source.cancel.cancel();
    let mut handle = source.handle;
    let high_watermark = match tokio::time::timeout(join_timeout, &mut handle).await {
        Ok(Ok(high_watermark)) => high_watermark,
        Ok(Err(error)) if error.is_cancelled() => None,
        Ok(Err(error)) => {
            tracing::warn!(%error, "direct-ZMQ active-sequence source task failed during shutdown");
            None
        }
        Err(_) => {
            handle.abort();
            let _ = handle.await;
            metrics.record_forced_abort();
            None
        }
    };
    metrics.source_stopped();
    high_watermark
}

async fn sleep_or_cancel(delay: Duration, cancellation_token: &CancellationToken) -> bool {
    tokio::select! {
        _ = cancellation_token.cancelled() => false,
        _ = tokio::time::sleep(delay) => true,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::{
        NoopSequencePublisher,
        protocols::{ActiveSequenceEvent, ActiveSequenceEventData, WorkerWithDpRank},
    };
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        transports::event_plane::{EventPublisher, EventTransportTx, ZmqPubTransport},
    };

    use super::*;

    fn lookup(direct: Option<&str>, rcvhwm: Option<&str>) -> impl FnMut(&str) -> Option<OsString> {
        let direct = direct.map(OsString::from);
        let rcvhwm = rcvhwm.map(OsString::from);
        move |key| match key {
            DIRECT_ZMQ_ENV => direct.clone(),
            RCVHWM_ENV => rcvhwm.clone(),
            _ => None,
        }
    }

    #[test]
    fn parses_direct_zmq_configuration() {
        assert_eq!(
            DirectZmqSequenceConfig::from_lookup(lookup(None, None)),
            DirectZmqSequenceConfig {
                enabled: true,
                rcvhwm: DEFAULT_RCVHWM,
            }
        );
        assert_eq!(
            DirectZmqSequenceConfig::from_lookup(lookup(Some("false"), Some("37"))),
            DirectZmqSequenceConfig {
                enabled: false,
                rcvhwm: 37,
            }
        );
        for invalid in ["0", "-1", "not-a-number", "2147483648"] {
            assert_eq!(
                DirectZmqSequenceConfig::from_lookup(lookup(None, Some(invalid))).rcvhwm,
                DEFAULT_RCVHWM
            );
        }
        assert!(DirectZmqSequenceConfig::from_lookup(lookup(Some("invalid"), None)).enabled);
    }

    #[test]
    fn selects_only_direct_zmq_and_honors_rollback() {
        let enabled = DirectZmqSequenceConfig::from_lookup(lookup(None, None));
        let rollback = DirectZmqSequenceConfig::from_lookup(lookup(Some("0"), None));

        assert!(enabled.should_use_direct_for_topology(EventTransportKind::Zmq, true));
        assert!(!enabled.should_use_direct_for_topology(EventTransportKind::Zmq, false));
        assert!(!enabled.should_use_direct_for_topology(EventTransportKind::Nats, false));
        assert!(!rollback.should_use_direct_for_topology(EventTransportKind::Zmq, true));
        assert!(!rollback.should_use_direct_for_topology(EventTransportKind::Zmq, false));
        assert!(!rollback.should_use_direct_for_topology(EventTransportKind::Nats, false));
    }

    #[test]
    fn observes_initial_forward_and_out_of_order_sequences() {
        let mut cursor = SequenceCursor::default();
        assert_eq!(cursor.observe(3), Continuity::Gap(3));
        assert_eq!(cursor.observe(4), Continuity::InOrder);
        assert_eq!(cursor.observe(7), Continuity::Gap(2));
        assert_eq!(cursor.observe(6), Continuity::OutOfOrder);
        assert_eq!(cursor.high_watermark, Some(7));

        let mut resumed = SequenceCursor::from_high_watermark(cursor.high_watermark);
        assert_eq!(resumed.observe(8), Continuity::InOrder);
    }

    fn add_event(request_id: impl Into<String>) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id: request_id.into(),
            worker: WorkerWithDpRank::new(0, 0),
            data: ActiveSequenceEventData::AddRequest {
                token_sequence: Some(vec![1]),
                track_prefill_tokens: false,
                expected_output_tokens: None,
                prefill_load_hint: None,
            },
            router_id: 99,
            lora_name: None,
        }
    }

    fn free_event(request_id: impl Into<String>) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id: request_id.into(),
            worker: WorkerWithDpRank::new(0, 0),
            data: ActiveSequenceEventData::Free,
            router_id: 99,
            lora_name: None,
        }
    }

    async fn publish_batch(
        publisher: &ZmqPubTransport,
        publisher_id: u64,
        sequence: u64,
        events: Vec<ActiveSequenceEvent>,
    ) -> anyhow::Result<()> {
        let codec = Codec::default();
        let payload = codec.encode_payload(&ActiveSequenceEventBatch { events })?;
        let envelope = codec.encode_envelope_parts(
            publisher_id,
            sequence,
            0,
            ACTIVE_SEQUENCES_SUBJECT,
            &payload,
        )?;
        publisher.publish(ACTIVE_SEQUENCES_SUBJECT, envelope).await
    }

    #[tokio::test]
    async fn real_zmq_batch_ingress_validates_identity_and_survives_rebind() -> anyhow::Result<()> {
        let runtime = Runtime::from_current()?;
        let distributed =
            DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
        let component = distributed
            .namespace(format!(
                "active-sequence-direct-zmq-{}",
                uuid::Uuid::new_v4()
            ))?
            .component("frontend")?;
        let metrics = ActiveSequenceZmqIngressMetrics::from_component(&component);
        let tracker = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            4,
            HashMap::from([(0, (0, 1))]),
            true,
            1,
            "test",
        ));
        let worker = WorkerWithDpRank::new(0, 0);
        let publisher_id = 42;
        let (publisher, bind_endpoint) =
            ZmqPubTransport::bind("tcp://127.0.0.1:0", ACTIVE_SEQUENCES_SUBJECT).await?;
        let connect_endpoint = bind_endpoint.replace("0.0.0.0", "127.0.0.1");
        let cancel = CancellationToken::new();
        let source = tokio::spawn(run_source(
            publisher_id,
            connect_endpoint,
            1,
            None,
            DEFAULT_RCVHWM,
            tracker.clone(),
            metrics,
            cancel.clone(),
        ));

        let mut sequence = 0;
        let mut sent_requests = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let request_id = format!("warmup-{sequence}");
                publish_batch(
                    &publisher,
                    publisher_id,
                    sequence,
                    vec![add_event(request_id.clone())],
                )
                .await
                .unwrap();
                sequence += 1;
                sent_requests.push(request_id);
                if tracker.active_blocks()[&worker] > 0 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("warmed direct-ZMQ source should apply a batch");

        let requests_before_out_of_order = tracker.active_request_counts()[&worker];
        let out_of_order_request = "out-of-order".to_string();
        publish_batch(
            &publisher,
            publisher_id,
            0,
            vec![add_event(out_of_order_request.clone())],
        )
        .await?;
        tokio::time::timeout(Duration::from_secs(1), async {
            while tracker.active_request_counts()[&worker] == requests_before_out_of_order {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("out-of-order envelopes should remain observational and be applied");
        sent_requests.push(out_of_order_request);

        let requests_before_mismatch = tracker.active_request_counts()[&worker];
        publish_batch(
            &publisher,
            publisher_id + 1,
            sequence,
            vec![add_event("identity-mismatch")],
        )
        .await?;
        sequence += 1;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            tracker.active_request_counts()[&worker],
            requests_before_mismatch
        );

        let free_events = sent_requests
            .iter()
            .cloned()
            .map(free_event)
            .collect::<Vec<_>>();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                publish_batch(&publisher, publisher_id, sequence, free_events.clone())
                    .await
                    .unwrap();
                sequence += 1;
                if tracker.active_blocks()[&worker] == 0 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("warmed source should drain before rebind");

        drop(publisher);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let publisher = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match ZmqPubTransport::bind(&bind_endpoint, ACTIVE_SEQUENCES_SUBJECT).await {
                    Ok((publisher, _)) => break publisher,
                    Err(_) => tokio::time::sleep(Duration::from_millis(20)).await,
                }
            }
        })
        .await
        .expect("publisher should rebind to its original endpoint");

        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                publish_batch(
                    &publisher,
                    publisher_id,
                    sequence,
                    vec![add_event("after-rebind")],
                )
                .await
                .unwrap();
                sequence += 1;
                if tracker.active_blocks()[&worker] > 0 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("direct-ZMQ source should receive after publisher rebind");

        cancel.cancel();
        let final_cursor = tokio::time::timeout(Duration::from_secs(1), source)
            .await
            .expect("source should stop after cancellation")
            .expect("source task should not panic");
        assert!(final_cursor.is_some());
        distributed.shutdown();
        Ok(())
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn supervisor_tracks_recreation_and_drains_every_source() -> anyhow::Result<()> {
        let runtime = Runtime::from_current()?;
        let distributed =
            DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
        let endpoint = distributed
            .namespace(format!(
                "active-sequence-supervisor-lifecycle-{}",
                uuid::Uuid::new_v4()
            ))?
            .component("frontend")?
            .endpoint("generate");
        let tracker = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            4,
            HashMap::from([(0, (0, 1))]),
            true,
            1,
            "test",
        ));
        let cancel = CancellationToken::new();
        let (source_count_tx, mut source_count_rx) = tokio::sync::mpsc::unbounded_channel();
        let supervisor = tokio::spawn(run_supervisor_inner(
            endpoint.clone(),
            tracker,
            DEFAULT_RCVHWM,
            cancel.clone(),
            move |count| {
                let _ = source_count_tx.send(count);
            },
        ));

        for _ in 0..2 {
            let mut publishers = Vec::with_capacity(32);
            for _ in 0..32 {
                publishers.push(
                    EventPublisher::for_endpoint_with_transport(
                        &endpoint,
                        ACTIVE_SEQUENCES_SUBJECT,
                        EventTransportKind::Zmq,
                    )
                    .await?,
                );
            }
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    match source_count_rx.recv().await {
                        Some(32) => break,
                        Some(_) => {}
                        None => panic!("source-count observer closed before reaching 32"),
                    }
                }
            })
            .await
            .expect("supervisor should track all 32 publishers");

            drop(publishers);
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    match source_count_rx.recv().await {
                        Some(0) => break,
                        Some(_) => {}
                        None => panic!("source-count observer closed before reaching zero"),
                    }
                }
            })
            .await
            .expect("supervisor should drain all removed publishers");
        }

        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), supervisor)
            .await
            .expect("supervisor should join after cancellation")
            .expect("supervisor task should not panic");
        distributed.shutdown();
        Ok(())
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn source_shutdown_joins_cooperative_tasks_and_aborts_stuck_tasks() -> anyhow::Result<()>
    {
        let runtime = Runtime::from_current()?;
        let distributed =
            DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
        let component = distributed
            .namespace(format!(
                "active-sequence-source-shutdown-{}",
                uuid::Uuid::new_v4()
            ))?
            .component("frontend")?;
        let metrics = ActiveSequenceZmqIngressMetrics::from_component(&component);

        let cooperative_cancel = CancellationToken::new();
        let task_cancel = cooperative_cancel.clone();
        let cooperative = SourceTask {
            endpoint: "cooperative".to_string(),
            generation: 1,
            cancel: cooperative_cancel,
            handle: tokio::spawn(async move {
                task_cancel.cancelled().await;
                Some(17)
            }),
        };
        metrics.source_started();
        assert_eq!(
            stop_source_with_timeout(cooperative, &metrics, Duration::from_secs(1)).await,
            Some(17)
        );

        let stuck_cancel = CancellationToken::new();
        let stuck_handle = tokio::spawn(std::future::pending::<Option<u64>>());
        let stuck_abort = stuck_handle.abort_handle();
        let stuck = SourceTask {
            endpoint: "stuck".to_string(),
            generation: 2,
            cancel: stuck_cancel,
            handle: stuck_handle,
        };
        metrics.source_started();
        assert_eq!(
            stop_source_with_timeout(stuck, &metrics, Duration::from_millis(10)).await,
            None
        );
        assert!(stuck_abort.is_finished());

        for generation_base in [100_u64, 200] {
            let mut sources = HashMap::new();
            for publisher_id in 0..32 {
                let cancel = CancellationToken::new();
                let task_cancel = cancel.clone();
                let generation = generation_base + publisher_id;
                metrics.source_started();
                sources.insert(
                    publisher_id,
                    SourceTask {
                        endpoint: format!("source-{publisher_id}"),
                        generation,
                        cancel,
                        handle: tokio::spawn(async move {
                            task_cancel.cancelled().await;
                            Some(generation)
                        }),
                    },
                );
            }
            assert_eq!(sources.len(), 32);
            let cursors = stop_sources(sources, &metrics).await;
            assert_eq!(cursors.len(), 32);
            assert_eq!(cursors[&0], generation_base);
            assert_eq!(cursors[&31], generation_base + 31);
        }

        distributed.shutdown();
        Ok(())
    }
}
