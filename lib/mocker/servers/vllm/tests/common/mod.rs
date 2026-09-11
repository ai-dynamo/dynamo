// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end assertions for native vLLM KV events through sidecar.

use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::{
    GenerateContext, KvEventSource, LLMEngine, PreprocessedRequest, StopConditions,
};
use dynamo_kv_router::indexer::{KvIndexerInterface, KvIndexerMetrics, LocalKvIndexer};
use dynamo_kv_router::protocols::{
    KV_EVENT_SUBJECT, KvCacheEventData, RouterEvent, compute_block_hash_for_seq,
};
use dynamo_llm::kv_router::publisher::{KvEventPublisher, KvEventSourceConfig};
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use dynamo_runtime::{DistributedRuntime, Runtime};
use futures::StreamExt;

async fn generate(engine: &impl LLMEngine, tokens: Vec<u32>) {
    let request = PreprocessedRequest::builder()
        .model("mocker-model".to_string())
        .token_ids(tokens)
        .sampling_options(Default::default())
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

pub async fn check_kv_events(engine: &impl LLMEngine, block_size: u32) {
    let sources = engine.kv_event_sources().await.unwrap();
    assert_eq!(sources.len(), 1);
    let KvEventSource::Zmq {
        endpoint: source_endpoint,
        topic,
        dp_rank,
    } = &sources[0]
    else {
        panic!("expected a native ZMQ source");
    };
    assert_eq!(*dp_rank, 0);
    assert!(topic.is_empty());
    assert!(source_endpoint.starts_with("tcp://127.0.0.1:"));
    assert_ne!(source_endpoint.rsplit(':').next(), Some("0"));

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
    let mut subscriber = EventSubscriber::for_endpoint(&endpoint, KV_EVENT_SUBJECT)
        .await
        .unwrap()
        .typed::<Vec<RouterEvent>>();
    let mut relay = KvEventPublisher::new_with_local_indexer(
        endpoint,
        block_size,
        Some(KvEventSourceConfig::Zmq {
            endpoint: source_endpoint.clone(),
            topic: topic.clone(),
            image_token_id: None,
            video_token_id: None,
        }),
        false,
        *dp_rank,
        None,
    )
    .unwrap();

    // Exercise both real ZMQ subscription handshakes. Distinct warmup prompts
    // produce fresh stores even if an earlier event was dropped before connect.
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
    relay.shutdown();
    cancel.cancel();
    engine.cleanup().await.unwrap();
}
