// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end assertions for native KV events through sidecar.

use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::{
    GenerateContext, KvEventSource, LLMEngine, PreprocessedRequest, SamplingOptions, StopConditions,
};
use dynamo_kv_events::event::{KV_EVENT_SUBJECT, KvCacheEventData, RouterEvent};
use dynamo_kv_hashing::routing::compute_block_hash_for_seq;
use dynamo_kv_hashing::{ExternalSequenceBlockHash, LocalBlockHash};
use dynamo_kv_indexer::host::{KvIndexerInterface, KvIndexerMetrics, LocalKvIndexer};
use dynamo_llm::kv_router::publisher::{KvEventPublisher, KvEventSourceConfig};
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::transports::event_plane::EventSubscriber;
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

pub async fn check_kv_events(engine: &impl LLMEngine, block_size: u32) {
    let mut relay = KvRelay::start(engine, block_size).await;
    let prompt = relay.store_prompt(engine).await;

    // The test engine has eight blocks. Force eviction with unrelated prompts.
    for n in 200..212 {
        generate(engine, vec![n; 2 * block_size as usize]).await;
    }
    relay
        .wait_until(&prompt, "eviction removed the prompt", |scores, stored| {
            stored == 0 && scores.iter().all(|&blocks| blocks == 0)
        })
        .await;
    relay.shutdown();
    engine.cleanup().await.unwrap();
}

/// Relays an engine's native KV events into a router index, as a worker does.
pub struct KvRelay {
    block_size: u32,
    subscriber: dynamo_runtime::transports::event_plane::TypedEventSubscriber<Vec<RouterEvent>>,
    publisher: KvEventPublisher,
    indexer: LocalKvIndexer,
    cancel: dynamo_runtime::CancellationToken,
    last_event_id: Option<u64>,
    prompt_hashes: Vec<LocalBlockHash>,
    stored_hashes: Vec<ExternalSequenceBlockHash>,
    _drt: DistributedRuntime,
}

impl KvRelay {
    pub async fn start(engine: &impl LLMEngine, block_size: u32) -> Self {
        let sources = engine.kv_event_sources().await.unwrap();
        assert_eq!(sources.len(), 1);
        let KvEventSource::Zmq {
            endpoint: source_endpoint,
            topic,
            dp_rank,
            ..
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
        let publisher = KvEventPublisher::new_with_local_indexer(
            endpoint,
            block_size,
            Some(KvEventSourceConfig::Zmq {
                endpoint: source_endpoint.clone(),
                topic: topic.clone(),
                image_token_id: None,
                video_token_id: None,
                liveness: None,
                expected_rank: None,
                engine_owned_mm_identity: false,
                idempotent_stores: false,
                recovery: None,
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
        Self {
            block_size,
            subscriber,
            publisher,
            indexer,
            cancel,
            last_event_id: None,
            prompt_hashes: Vec::new(),
            stored_hashes: Vec::new(),
            _drt: drt,
        }
    }

    /// Generates from a fresh two-block prompt and waits until the router
    /// index matches every full block of it. Returns the prompt.
    pub async fn store_prompt(&mut self, engine: &impl LLMEngine) -> Vec<u32> {
        let prompt: Vec<u32> = (1_000..1_000 + 2 * self.block_size).collect();
        self.prompt_hashes =
            compute_block_hash_for_seq(&prompt, self.block_size, Default::default());
        let expected_blocks = u32::try_from(self.prompt_hashes.len()).unwrap();
        generate(engine, prompt.clone()).await;
        let expected_hashes = self.prompt_hashes.len();
        self.wait_until(
            &prompt,
            "the stored prompt reached the router index",
            |scores, stored| scores.contains(&expected_blocks) && stored == expected_hashes,
        )
        .await;
        prompt
    }

    /// Applies relayed events until `done(index scores, stored prompt blocks)` holds.
    async fn wait_until(
        &mut self,
        prompt: &[u32],
        what: &str,
        done: impl Fn(&[u32], usize) -> bool,
    ) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                self.apply_next().await;
                let scores = self
                    .indexer
                    .find_matches_for_request(prompt, None, None, None)
                    .await
                    .unwrap();
                let scores: Vec<u32> = scores.scores.values().copied().collect();
                if done(&scores, self.stored_hashes.len()) {
                    break;
                }
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out before {what}"));
    }

    async fn apply_next(&mut self) {
        let (_, events) = self.subscriber.next().await.unwrap().unwrap();
        for event in events {
            assert_eq!(event.event.dp_rank, 0);
            if let Some(last) = self.last_event_id {
                assert!(event.event.event_id > last);
            }
            self.last_event_id = Some(event.event.event_id);
            match &event.event.data {
                KvCacheEventData::Stored(data) => self.stored_hashes.extend(
                    data.blocks
                        .iter()
                        .filter(|block| self.prompt_hashes.contains(&block.tokens_hash))
                        .map(|block| block.block_hash),
                ),
                KvCacheEventData::Removed(data) => self
                    .stored_hashes
                    .retain(|hash| !data.block_hashes.contains(hash)),
                _ => {}
            }
            self.indexer.apply_event_with_buffer(event).await.unwrap();
        }
    }

    pub fn shutdown(mut self) {
        self.publisher.shutdown();
        self.cancel.cancel();
    }
}
