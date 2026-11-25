// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end assertions for native KV events through sidecar.

use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::{
    GenerateContext, KvEventSource, LLMEngine, OnPublisherReady, PreprocessedRequest,
    SamplingOptions, StopConditions,
};
use dynamo_kv_events::event::{KV_EVENT_SUBJECT, KvCacheEvent, KvCacheEventData, RouterEvent};
use dynamo_kv_hashing::routing::compute_block_hash_for_seq;
use dynamo_kv_indexer::host::{KvIndexerInterface, KvIndexerMetrics, LocalKvIndexer};
use dynamo_llm::kv_router::publisher::{KvEventPublisher, KvEventSourceConfig};
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::transports::event_plane::{EventSubscriber, TypedEventSubscriber};
use dynamo_runtime::{DistributedRuntime, Runtime};
use futures::StreamExt;

async fn generate(engine: &impl LLMEngine, tokens: Vec<u32>) {
    let request = PreprocessedRequest::builder()
        .model("mocker-model".to_string())
        .token_ids(tokens)
        .sampling_options(SamplingOptions {
            temperature: Some(0.0),
            ..Default::default()
        })
        .output_options(Default::default())
        .stop_conditions(StopConditions {
            max_tokens: Some(1),
            ignore_eos: Some(true),
            ..Default::default()
        })
        .build()
        .unwrap();
    let context = dynamo_backend_common::testing::mock_context();
    let mut output = engine
        .generate(request, GenerateContext::new(context, None))
        .await
        .unwrap();
    while let Some(output) = output.next().await {
        output.unwrap();
    }
}

/// The sidecar's push source for the engine's one rank, a publisher for it,
/// and a subscriber to what that publisher sends routers.
async fn push_source(
    engine: &impl LLMEngine,
    block_size: u32,
    idempotent_stores: bool,
) -> (
    OnPublisherReady,
    Arc<KvEventPublisher>,
    TypedEventSubscriber<Vec<RouterEvent>>,
    DistributedRuntime,
) {
    let mut sources = engine.kv_event_sources().await.unwrap();
    assert_eq!(sources.len(), 1);
    let KvEventSource::Push {
        on_ready,
        dp_rank,
        idempotent_stores: source_idempotent_stores,
    } = sources.remove(0)
    else {
        panic!("expected the sidecar's push source");
    };
    assert_eq!(dp_rank, 0);
    assert_eq!(source_idempotent_stores, idempotent_stores);

    let drt = DistributedRuntime::new(
        Runtime::from_current().unwrap(),
        DistributedConfig::process_local(),
    )
    .await
    .unwrap();
    let endpoint = drt
        .namespace("mocker-kv-test")
        .unwrap()
        .component("worker")
        .unwrap()
        .endpoint("generate");
    let subscriber = EventSubscriber::for_endpoint(&endpoint, KV_EVENT_SUBJECT)
        .await
        .unwrap()
        .typed::<Vec<RouterEvent>>();
    let relay = Arc::new(
        KvEventPublisher::new_with_local_indexer(
            endpoint,
            block_size,
            Some(KvEventSourceConfig::Push { idempotent_stores }),
            false,
            dp_rank,
            None,
        )
        .unwrap(),
    );
    (on_ready, relay, subscriber, drt)
}

/// The engine's KV events reach a router index through the sidecar's push
/// source: stores, evictions, then the cache clear `clear_cache` triggers, if any.
pub async fn check_kv_events(
    engine: &impl LLMEngine,
    block_size: u32,
    idempotent_stores: bool,
    clear_cache: Option<&dyn Fn()>,
) {
    let (on_ready, relay, mut subscriber, _drt) =
        push_source(engine, block_size, idempotent_stores).await;
    on_ready(relay.clone()).unwrap();

    // Distinct warmup prompts produce fresh stores even if an earlier event
    // was published before the subscriptions connected.
    tokio::time::timeout(Duration::from_secs(10), async {
        for n in 100.. {
            generate(engine, vec![n; block_size as usize]).await;
            if let Ok(Some(batch)) =
                tokio::time::timeout(Duration::from_millis(50), subscriber.next()).await
            {
                batch.unwrap();
                break;
            }
        }
    })
    .await
    .expect("KV relay did not become ready");

    let cancel = dynamo_runtime::CancellationToken::new();
    let indexer = LocalKvIndexer::new(
        cancel.clone(),
        block_size,
        Arc::new(KvIndexerMetrics::new_unregistered()),
        100,
    );
    let prompt = vec![11, 22, 33, 44, 55, 66, 77, 88];
    let prompt_hashes = compute_block_hash_for_seq(&prompt, block_size, Default::default());
    let expected_blocks = u32::try_from(prompt_hashes.len()).unwrap();
    assert!(expected_blocks > 0);
    generate(engine, prompt.clone()).await;
    let mut last_event_id = None;
    let mut stored_hashes = Vec::new();

    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let (_, events) = subscriber.next().await.unwrap().unwrap();
            for event in events {
                assert_eq!(event.event.dp_rank, 0);
                if let Some(last) = last_event_id {
                    assert!(event.event.event_id > last);
                }
                last_event_id = Some(event.event.event_id);
                if let KvCacheEventData::Stored(data) = &event.event.data {
                    stored_hashes.extend(
                        data.blocks
                            .iter()
                            .filter(|block| prompt_hashes.contains(&block.tokens_hash))
                            .map(|block| block.block_hash),
                    );
                }
                indexer.apply_event_with_buffer(event).await.unwrap();
            }
            let scores = indexer
                .find_matches_for_request(&prompt, None, None, None)
                .await
                .unwrap();
            if scores
                .scores
                .values()
                .any(|&blocks| blocks == expected_blocks)
            {
                assert_eq!(stored_hashes.len(), prompt_hashes.len());
                break;
            }
        }
    })
    .await
    .expect("stored prompt did not reach the router index");

    // The test engine has eight blocks. Force eviction with unrelated prompts.
    for n in 200..212 {
        generate(engine, vec![n; 2 * block_size as usize]).await;
    }
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let (_, events) = subscriber.next().await.unwrap().unwrap();
            for event in events {
                assert_eq!(event.event.dp_rank, 0);
                assert!(event.event.event_id > last_event_id.unwrap());
                last_event_id = Some(event.event.event_id);
                if let KvCacheEventData::Removed(data) = &event.event.data {
                    stored_hashes.retain(|hash| !data.block_hashes.contains(hash));
                }
                indexer.apply_event_with_buffer(event).await.unwrap();
            }
            let scores = indexer
                .find_matches_for_request(&prompt, None, None, None)
                .await
                .unwrap();
            if stored_hashes.is_empty() && scores.scores.values().all(|&blocks| blocks == 0) {
                break;
            }
        }
    })
    .await
    .expect("eviction did not remove the prompt from the router index");

    // Only the gRPC stream carries a clear, so this also shows which feed ran.
    if let Some(clear_cache) = clear_cache {
        clear_cache();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let (_, events) = subscriber.next().await.unwrap().unwrap();
                let mut events = events.iter();
                if events.any(|event| matches!(event.event.data, KvCacheEventData::Cleared)) {
                    break;
                }
            }
        })
        .await
        .expect("the cache clear did not reach the router");
    }
    engine.cleanup().await.unwrap();
    relay.stop_and_wait().await.unwrap();
    cancel.cancel();
}

/// A sidecar that starts next to an engine holding a prompt's blocks gives
/// them to a router index without a new request, then follows the engine's
/// stream. When that stream is lost, it clears the index and gives the
/// engine's blocks again.
pub async fn check_kv_snapshot(
    engine: &impl LLMEngine,
    block_size: u32,
    idempotent_stores: bool,
    drop_stream: &dyn Fn(),
) {
    let prompt = vec![11, 22, 33, 44, 55, 66, 77, 88];
    generate(engine, prompt.clone()).await;

    let (on_ready, relay, mut subscriber, _drt) =
        push_source(engine, block_size, idempotent_stores).await;
    // The snapshot is published once, so the subscription must be connected first.
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            relay
                .publish(KvCacheEvent {
                    event_id: relay.next_event_id(),
                    data: KvCacheEventData::Cleared,
                    dp_rank: 0,
                })
                .unwrap();
            if let Ok(Some(batch)) =
                tokio::time::timeout(Duration::from_millis(50), subscriber.next()).await
            {
                batch.unwrap();
                break;
            }
        }
    })
    .await
    .expect("KV relay did not become ready");
    on_ready(relay.clone()).unwrap();

    let cancel = dynamo_runtime::CancellationToken::new();
    let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
    let indexer = LocalKvIndexer::new(cancel.clone(), block_size, metrics, 100);
    // Applies the sidecar's events until the index holds `tokens`, after a
    // clear when `needs_clear`. Returns whether a clear arrived.
    let mut index_holds = async |tokens: &[u32], needs_clear: bool| {
        let expected_blocks = u32::try_from(tokens.len()).unwrap() / block_size;
        let mut is_cleared = false;
        loop {
            let (_, events) = subscriber.next().await.unwrap().unwrap();
            for event in events {
                is_cleared |= matches!(event.event.data, KvCacheEventData::Cleared);
                indexer.apply_event_with_buffer(event).await.unwrap();
            }
            let scores = indexer
                .find_matches_for_request(tokens, None, None, None)
                .await
                .unwrap();
            let is_held = scores
                .scores
                .values()
                .any(|&blocks| blocks == expected_blocks);
            if is_held && (is_cleared || !needs_clear) {
                return is_cleared;
            }
        }
    };
    let deadline = Duration::from_secs(10);

    tokio::time::timeout(deadline, index_holds(&prompt, false))
        .await
        .expect("the engine's blocks did not reach the router index");

    // Stores on the stream that resumed from the snapshot's cursor reload nothing.
    // A stream resumed from another position reloads after the first of them.
    for live_prompt in [[21, 22, 23, 24], [31, 32, 33, 34]] {
        generate(engine, live_prompt.to_vec()).await;
        let is_reloaded = tokio::time::timeout(deadline, index_holds(&live_prompt, false))
            .await
            .expect("a live store did not reach the router index");
        assert!(!is_reloaded);
    }

    drop_stream();
    tokio::time::timeout(deadline, index_holds(&prompt, true))
        .await
        .expect("the lost stream was not cleared and reloaded");
    engine.cleanup().await.unwrap();
    relay.stop_and_wait().await.unwrap();
    cancel.cancel();
}
