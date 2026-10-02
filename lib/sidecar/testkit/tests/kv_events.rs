// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::DisaggregationMode;
use dynamo_kv_router::indexer::{
    KvIndexerInterface, KvIndexerMetrics, LocalKvIndexer, WorkerKvQueryKind, WorkerKvQueryRequest,
    WorkerKvQueryResponse,
};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KV_EVENT_SUBJECT, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
    KvCacheStoreData, KvCacheStoredBlockData, RouterEvent, StorageTier, compute_block_hash_for_seq,
};
use dynamo_llm::discovery::KvEventSource;
use dynamo_mocker::common::protocols::{RawKvEvent, RawKvEventSink};
use dynamo_mocker::services::zmq_events::ZmqKvEventSink;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery, EventSourceQuery};
use dynamo_runtime::pipeline::{
    AddressedPushRouter, AddressedRequest, AsyncEngine, ManyOut, SingleIn,
};
use dynamo_runtime::transports::event_plane::EventSubscriber;
use dynamo_sidecar_testkit::{bounded, control::Controller};
use futures::{StreamExt, stream::SelectAll};
use serde_json::json;

#[path = "support/process.rs"]
#[allow(dead_code)]
mod process;
#[allow(dead_code)]
mod support;

use process::Environment;
use support::{FixtureConfig, ProcessFixture, sglang, vllm};

trait KvFixture: ProcessFixture {
    fn advertise(&self, sinks: &[ZmqKvEventSink]);
}

impl KvFixture for vllm::Fixture {
    fn advertise(&self, sinks: &[ZmqKvEventSink]) {
        self.override_kv_sources(
            sinks
                .iter()
                .enumerate()
                .map(|(rank, sink)| dynamo_vllm_sidecar::proto::KvEventSource {
                    transport: "zmq".into(),
                    endpoint: sink.endpoint().into(),
                    topic: String::new(),
                    data_parallel_rank: Some(rank as u32),
                    replay_endpoint: String::new(),
                    encoding: "msgpack".into(),
                    schema_version: 1,
                    ..Default::default()
                })
                .collect(),
        );
    }
}

impl KvFixture for sglang::Fixture {
    fn advertise(&self, sinks: &[ZmqKvEventSink]) {
        let port: u16 = sinks[0]
            .endpoint()
            .rsplit(':')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        self.override_discovery(
            serde_json::Value::Null,
            vec![json!({
                "dp_size": 2,
                "enable_dp_attention": true,
                "kv_events": {
                    "publisher": "zmq", "endpoint_host": "*", "endpoint_port_base": port,
                    "topic": "", "block_size": 4, "dp_size": 2
                }
            })],
        );
    }
}

fn ranked_sinks() -> Vec<ZmqKvEventSink> {
    // SGLang advertises a base port plus rank; bind the pair before launching it.
    for _ in 0..32 {
        let first = ZmqKvEventSink::bind(None, None, 0, 4).unwrap();
        let port: u16 = first
            .endpoint()
            .rsplit(':')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        if let Some(port) = port.checked_add(1)
            && let Ok(second) = ZmqKvEventSink::bind(Some(port), None, 1, 4)
        {
            return vec![first, second];
        }
    }
    panic!("could not bind adjacent native KV source ports");
}

fn stored(rank: u32, tag: u32) -> RawKvEvent {
    let tokens = vec![tag; 4];
    RawKvEvent {
        event: KvCacheEvent {
            event_id: 0,
            dp_rank: rank,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(u64::from(tag)),
                    tokens_hash: compute_block_hash_for_seq(&tokens, 4, Default::default())[0],
                    mm_extra_info: None,
                }],
            }),
        },
        block_token_ids: Some(vec![tokens]),
        storage_tier: StorageTier::Device,
    }
}

fn removed(rank: u32, tag: u32) -> RawKvEvent {
    RawKvEvent {
        event: KvCacheEvent {
            event_id: 1,
            dp_rank: rank,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(u64::from(tag))],
            }),
        },
        block_token_ids: None,
        storage_tier: StorageTier::Device,
    }
}

async fn sources(env: &Environment) -> Vec<KvEventSource> {
    env.runtime
        .discovery()
        .list(DiscoveryQuery::EventSources(EventSourceQuery::all()))
        .await
        .unwrap()
        .into_iter()
        .filter_map(|instance| match instance {
            DiscoveryInstance::EventSource { metadata, .. } => {
                Some(serde_json::from_value(metadata).unwrap())
            }
            _ => None,
        })
        .collect()
}

async fn query(
    env: &Environment,
    source: &KvEventSource,
    start_event_id: Option<u64>,
) -> WorkerKvQueryResponse {
    let component = env
        .runtime
        .namespace(&env.namespace)
        .unwrap()
        .component("backend")
        .unwrap();
    let router = AddressedPushRouter::from_runtime_provider(&component)
        .await
        .unwrap();
    let target = source
        .recovery_target
        .clone()
        .expect("Worker must expose its local indexer");
    let request = SingleIn::new(WorkerKvQueryRequest {
        worker_id: source.worker.worker_id,
        dp_rank: source.worker.dp_rank,
        start_event_id,
        end_event_id: None,
        supports_tree_dump_failed: true,
        kind: WorkerKvQueryKind::Recovery,
    })
    .map(|request| AddressedRequest::for_instance(request, target));
    let mut output: ManyOut<WorkerKvQueryResponse> = router.generate(request).await.unwrap();
    bounded("Worker local index recovery", output.next())
        .await
        .unwrap()
}

fn expected_tag(source: &KvEventSource) -> u32 {
    match source.kv_state_endpoint.component.as_str() {
        "prefill" => 100 + source.worker.dp_rank,
        "backend" => 110 + source.worker.dp_rank,
        component => panic!("unexpected KV source component: {component}"),
    }
}

#[tokio::test]
async fn vllm_disaggregated_ranked_events_reach_event_plane_and_worker_indexes() {
    ranked_events_reach_event_plane_and_worker_indexes::<vllm::Fixture>().await;
}

#[tokio::test]
async fn sglang_disaggregated_ranked_events_reach_event_plane_and_worker_indexes() {
    ranked_events_reach_event_plane_and_worker_indexes::<sglang::Fixture>().await;
}

async fn ranked_events_reach_event_plane_and_worker_indexes<F: KvFixture>() {
    let env = Environment::new().await;
    let mut peers = Vec::new();
    let mut processes = Vec::new();
    let mut publishers = Vec::new();
    let mut subscriptions = SelectAll::new();
    for (role, mode, component) in [
        (0, DisaggregationMode::Prefill, "prefill"),
        (1, DisaggregationMode::Decode, "backend"),
    ] {
        let sinks = ranked_sinks();
        let peer = F::start(
            Controller::default(),
            FixtureConfig {
                model: env.model.clone(),
                disaggregation_mode: mode,
                ..Default::default()
            },
        )
        .await;
        peer.advertise(&sinks);
        let subscription =
            EventSubscriber::for_endpoint(&env.endpoint(component), KV_EVENT_SUBJECT)
                .await
                .unwrap()
                .typed::<Vec<RouterEvent>>();
        subscriptions.push(
            futures::stream::unfold(subscription, |mut subscription| async move {
                subscription.next().await.map(|event| (event, subscription))
            })
            .boxed(),
        );
        processes.push(env.spawn::<F>(&peer.endpoint(), mode, 5));
        env.ready(component).await;
        peers.push(peer);
        publishers.extend(
            sinks
                .into_iter()
                .enumerate()
                .map(|(rank, sink)| (sink, rank as u32, 100 + role * 10 + rank as u32)),
        );
    }
    let sources = bounded("all prefill/decode rank advertisements", async {
        loop {
            let sources = sources(&env).await;
            if sources.len() == 4 {
                break sources;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert_eq!(
        sources
            .iter()
            .map(|s| s.publisher_id)
            .collect::<HashSet<_>>()
            .len(),
        4
    );
    assert_eq!(
        sources
            .iter()
            .map(|s| s.worker.worker_id)
            .collect::<HashSet<_>>()
            .len(),
        2
    );
    for rank in 0..2 {
        assert_eq!(
            sources.iter().filter(|s| s.worker.dp_rank == rank).count(),
            2
        );
    }

    let cancel = tokio_util::sync::CancellationToken::new();
    let indexer = LocalKvIndexer::new(
        cancel.clone(),
        4,
        Arc::new(KvIndexerMetrics::new_unregistered()),
        100,
    );
    bounded("native and event-plane subscription handshakes", async {
        let mut ready = HashSet::new();
        while ready.len() < 4 {
            for (sink, rank, tag) in &publishers {
                sink.publish_batch(vec![stored(*rank, tag + 1000), removed(*rank, tag + 1000)])
                    .unwrap();
            }
            if let Ok(Some(batch)) =
                tokio::time::timeout(Duration::from_millis(50), subscriptions.next()).await
            {
                let (publisher, events) = batch.unwrap();
                if events
                    .iter()
                    .any(|event| matches!(event.event.data, KvCacheEventData::Removed(_)))
                {
                    ready.insert(publisher.publisher_id);
                }
            }
        }
    })
    .await;
    for (sink, rank, tag) in &publishers {
        sink.publish(stored(*rank, *tag)).unwrap();
    }
    let mut observed = HashSet::new();
    bounded("events from every native source", async {
        while observed.len() < 4 {
            if let Ok(Some(batch)) =
                tokio::time::timeout(Duration::from_millis(50), subscriptions.next()).await
            {
                let (publisher, events) = batch.unwrap();
                let source = sources
                    .iter()
                    .find(|source| source.publisher_id == publisher.publisher_id)
                    .unwrap();
                for event in events {
                    assert_eq!(event.worker_id, source.worker.worker_id);
                    assert_eq!(event.event.dp_rank, source.worker.dp_rank);
                    if let KvCacheEventData::Stored(data) = &event.event.data
                        && data.blocks.iter().all(|block| block.block_hash.0 < 1000)
                    {
                        observed.extend(data.blocks.iter().map(|block| block.block_hash.0));
                        indexer.apply_event_with_buffer(event).await.unwrap();
                    }
                }
            }
        }
    })
    .await;
    assert_eq!(observed, HashSet::from([100, 101, 110, 111]));

    let mut cursors = HashMap::new();
    for source in &sources {
        let (events, cursor) = bounded("native events in Worker local index", async {
            loop {
                match query(&env, source, None).await {
                    WorkerKvQueryResponse::TreeDump {
                        events,
                        last_event_id,
                        ..
                    } if !events.is_empty() => {
                        break (events, last_event_id);
                    }
                    WorkerKvQueryResponse::TreeDump { .. } => {}
                    response => panic!("expected local index snapshot: {response:?}"),
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        cursors.insert(source.publisher_id, cursor);
        let blocks = events
            .iter()
            .flat_map(|event| {
                assert_eq!(event.worker_id, source.worker.worker_id);
                assert_eq!(event.event.dp_rank, source.worker.dp_rank);
                match &event.event.data {
                    KvCacheEventData::Stored(data) => data.blocks.clone(),
                    _ => Vec::new(),
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(blocks.len(), 1);
        let tag = expected_tag(source);
        assert_eq!(blocks[0].block_hash.0, u64::from(tag));
        assert_eq!(
            blocks[0].tokens_hash,
            compute_block_hash_for_seq(&[tag; 4], 4, Default::default())[0]
        );
        let scores = indexer
            .find_matches_for_request(&[tag; 4], None, None, None)
            .await
            .unwrap();
        assert_eq!(scores.scores.get(&source.worker), Some(&1));
    }

    let mut removed = HashSet::new();
    for batch in [&publishers[..1], &publishers[1..]] {
        let expected_count = removed.len() + batch.len();
        for (sink, rank, tag) in batch {
            sink.publish(self::removed(*rank, *tag)).unwrap();
        }
        bounded("removals from the selected native sources", async {
            while removed.len() < expected_count {
                let (publisher, events) = subscriptions.next().await.unwrap().unwrap();
                let source = sources
                    .iter()
                    .find(|source| source.publisher_id == publisher.publisher_id)
                    .unwrap();
                for event in events {
                    assert_eq!(event.worker_id, source.worker.worker_id);
                    assert_eq!(event.event.dp_rank, source.worker.dp_rank);
                    if let KvCacheEventData::Removed(data) = &event.event.data {
                        assert_eq!(
                            data.block_hashes,
                            vec![ExternalSequenceBlockHash(u64::from(expected_tag(source)))]
                        );
                        removed.extend(data.block_hashes.iter().map(|hash| hash.0));
                    }
                    indexer.apply_event_with_buffer(event).await.unwrap();
                }
            }
        })
        .await;
        for source in &sources {
            let tag = expected_tag(source);
            let is_removed = removed.contains(&u64::from(tag));
            let scores = indexer
                .find_matches_for_request(&[tag; 4], None, None, None)
                .await
                .unwrap();
            assert_eq!(
                scores
                    .scores
                    .get(&source.worker)
                    .copied()
                    .unwrap_or_default(),
                u32::from(!is_removed)
            );
            let previous_cursor = cursors[&source.publisher_id];
            let is_removed_in_batch = batch.iter().any(|(_, _, removed_tag)| *removed_tag == tag);
            let cursor = bounded(
                "Worker local index eviction replay and survivor isolation",
                async {
                    loop {
                        match query(&env, source, Some(previous_cursor + 1)).await {
                            WorkerKvQueryResponse::Events {
                                events,
                                last_event_id,
                            } => {
                                assert!(is_removed_in_batch);
                                assert_eq!(events.len(), 1);
                                assert_eq!(events[0].worker_id, source.worker.worker_id);
                                assert_eq!(events[0].event.dp_rank, source.worker.dp_rank);
                                let KvCacheEventData::Removed(data) = &events[0].event.data else {
                                    panic!("expected removal: {:?}", events[0]);
                                };
                                assert_eq!(
                                    data.block_hashes,
                                    vec![ExternalSequenceBlockHash(u64::from(tag))]
                                );
                                assert!(last_event_id > previous_cursor);
                                break last_event_id;
                            }
                            WorkerKvQueryResponse::TooNew {
                                newest_available, ..
                            } => {
                                assert_eq!(newest_available, previous_cursor);
                                if !is_removed_in_batch {
                                    break previous_cursor;
                                }
                            }
                            response => panic!("expected buffered local recovery: {response:?}"),
                        }
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                },
            )
            .await;
            cursors.insert(source.publisher_id, cursor);
        }
    }
    for child in &mut processes {
        child.shutdown().await;
    }
    for peer in &mut peers {
        peer.shutdown().await;
    }
    cancel.cancel();
}
