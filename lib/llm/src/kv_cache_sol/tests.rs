// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use super::*;
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::transports::event_plane::{EventPublisher, EventSubscriber};
use dynamo_runtime::{DistributedRuntime, Runtime};
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

fn domain(eagle: bool) -> KvCacheSolDomain {
    KvCacheSolDomain {
        model: "model".to_string(),
        model_config: "config".to_string(),
        block_size: 4,
        is_eagle: eagle,
    }
}

fn start_event(id: &str, time: u64, hashes: Vec<u64>, tokens: u64) -> KvCacheSolEvent {
    KvCacheSolEvent {
        producer_sequence: 0,
        occurred_at_ms: time,
        request_id: id.to_string(),
        domain: domain(false),
        kind: KvCacheSolEventKind::RequestStart {
            prompt_sequence_hashes: hashes,
            prompt_tokens: tokens,
        },
    }
}

fn start_event_in_domain(
    id: &str,
    time: u64,
    hashes: Vec<u64>,
    tokens: u64,
    model_config: &str,
) -> KvCacheSolEvent {
    let mut event = start_event(id, time, hashes, tokens);
    event.domain.model_config = model_config.to_string();
    event
}

fn end_event(id: &str, time: u64, sequence: u64) -> KvCacheSolEvent {
    KvCacheSolEvent {
        producer_sequence: sequence,
        occurred_at_ms: time,
        request_id: id.to_string(),
        domain: domain(false),
        kind: KvCacheSolEventKind::RequestEnd {
            continuation_sequence_hashes: BTreeMap::new(),
            observed_cached_tokens: Some(0),
        },
    }
}

#[test]
fn longest_prefix_hits_and_horizon_expiry() {
    let mut core = KvCacheSolCore::new(Duration::from_millis(100), 100, 100);
    let first = core.apply(1, start_event("a", 10, vec![11, 12], 8));
    assert!(matches!(first, SolObservation::Start { hit_tokens: 0, .. }));
    let second = core.apply(1, start_event("b", 20, vec![11, 12, 13], 12));
    assert!(matches!(
        second,
        SolObservation::Start { hit_tokens: 8, .. }
    ));
    let expired = core.apply(1, start_event("c", 121, vec![11, 12], 8));
    assert!(matches!(
        expired,
        SolObservation::Start { hit_tokens: 0, .. }
    ));
}

#[test]
fn hot_prefix_refresh_keeps_expiry_index_bounded() {
    let mut cache = DomainCache::default();
    assert!(cache.insert(11, 100));
    for expires_at in 101..=100_000 {
        assert!(!cache.insert(11, expires_at));
    }

    assert_eq!(cache.expirations.len(), 1);
    assert_eq!(cache.expiry_queue.len(), 1);
    cache.prune(100_000);
    assert!(cache.expirations.is_empty());
    assert!(cache.expiry_queue.is_empty());
}

#[test]
fn refresh_with_an_older_timestamp_never_shortens_expiry() {
    let mut cache = DomainCache::default();
    assert!(cache.insert(11, 200));
    assert!(!cache.insert(11, 100));
    cache.prune(150);

    assert_eq!(cache.expirations.get(&11), Some(&200));
    assert_eq!(cache.expiry_queue.len(), 1);
}

#[test]
fn expired_inactive_domain_releases_global_capacity() {
    let mut core = KvCacheSolCore::new(Duration::from_millis(100), 1, 100);
    core.apply(1, start_event_in_domain("a", 10, vec![11], 4, "config-a"));
    assert_eq!(core.cache_blocks(), 1);

    let observation = core.apply(1, start_event_in_domain("b", 111, vec![22], 4, "config-b"));
    assert!(matches!(
        observation,
        SolObservation::Start { hit_tokens: 0, .. }
    ));
    assert_eq!(core.cache_blocks(), 1);
    assert!(!core.degraded());
    assert!(
        core.caches
            .keys()
            .all(|domain| domain.model_config != "config-a")
    );
}

#[test]
fn cache_block_capacity_is_a_hard_bound_under_domain_churn() {
    let mut core = KvCacheSolCore::new(Duration::from_millis(10), 8, 10_000);
    for index in 0..10_000_u64 {
        core.apply(
            1,
            start_event_in_domain(
                &format!("request-{index}"),
                index,
                vec![index],
                4,
                &format!("config-{}", index % 16),
            ),
        );
        assert!(core.cache_blocks() <= 8);
    }
    assert!(core.caches.len() <= 16);
}

#[test]
fn identical_request_ids_from_different_publishers_do_not_collide() {
    let mut core = KvCacheSolCore::new(Duration::from_secs(1), 100, 100);
    core.apply(1, start_event("shared-id", 10, vec![11], 4));
    core.apply(2, start_event("shared-id", 11, vec![22], 4));
    assert_eq!(core.pending_requests(), 2);

    core.apply(1, end_event("shared-id", 20, 1));
    core.apply(2, end_event("shared-id", 21, 1));
    assert_eq!(core.pending_requests(), 0);
    assert!(!core.degraded());
}

#[test]
fn pending_request_capacity_is_a_hard_bound() {
    let mut core = KvCacheSolCore::new(Duration::from_secs(1), 100, 2);
    core.apply(1, start_event("a", 10, vec![11], 4));
    core.apply(1, start_event("b", 11, vec![12], 4));
    core.apply(1, start_event("c", 12, vec![13], 4));

    assert_eq!(core.pending_requests(), 2);
    assert!(core.degraded());
}

#[test]
fn degraded_and_pending_state_expire_after_the_horizon() {
    let mut core = KvCacheSolCore::new(Duration::from_millis(100), 0, 100);
    core.apply(1, start_event("a", 10, vec![11], 4));

    assert!(core.degraded());
    assert_eq!(core.pending_requests(), 1);

    core.advance_to(110);
    assert!(!core.degraded());
    assert_eq!(core.pending_requests(), 0);
}

#[tokio::test]
async fn queue_overflow_and_worker_close_are_sticky_producer_failures() -> Result<()> {
    let runtime = Runtime::from_current()?;
    let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
    let namespace = drt.namespace(format!("kv-sol-queue-{}", uuid::Uuid::new_v4()))?;
    let component = namespace.component("producer")?;
    let metrics = Arc::new(ProducerMetrics::new(&component)?);
    let (tx, rx) = mpsc::channel(1);
    let producer = KvCacheSolProducer {
        tx,
        missing_sequences: Arc::new(AtomicU64::new(0)),
        metrics: metrics.clone(),
        block_size: 4,
        is_eagle: false,
        router_computes_hashes: false,
    };

    producer.try_send(RawEvent::End(RawEnd {
        occurred_at_ms: 1,
        request_id: "first".to_string(),
        output_tokens: BTreeMap::new(),
        observed_cached_tokens: None,
    }));
    producer.try_send(RawEvent::End(RawEnd {
        occurred_at_ms: 2,
        request_id: "overflow".to_string(),
        output_tokens: BTreeMap::new(),
        observed_cached_tokens: None,
    }));

    assert_eq!(metrics.queue_depth.get(), 1);
    assert_eq!(metrics.degraded.get(), 1);
    assert_eq!(
        metrics
            .events_dropped_total
            .with_label_values(&["queue_full"])
            .get(),
        1
    );
    assert_eq!(producer.missing_sequences.load(Ordering::Relaxed), 1);

    drop(rx);
    producer.try_send(RawEvent::End(RawEnd {
        occurred_at_ms: 3,
        request_id: "closed".to_string(),
        output_tokens: BTreeMap::new(),
        observed_cached_tokens: None,
    }));
    assert_eq!(metrics.queue_depth.get(), 0);
    assert_eq!(
        metrics
            .events_dropped_total
            .with_label_values(&["worker_closed"])
            .get(),
        1
    );
    assert_eq!(producer.missing_sequences.load(Ordering::Relaxed), 2);

    drt.shutdown();
    Ok(())
}

#[tokio::test]
async fn dropping_an_unpolled_stream_publishes_request_end() -> Result<()> {
    let runtime = Runtime::from_current()?;
    let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
    let namespace = drt.namespace(format!("kv-sol-drop-{}", uuid::Uuid::new_v4()))?;
    let component = namespace.component("producer")?;
    let metrics = Arc::new(ProducerMetrics::new(&component)?);
    let (tx, mut rx) = mpsc::channel(1);
    let producer = KvCacheSolProducer {
        tx,
        missing_sequences: Arc::new(AtomicU64::new(0)),
        metrics,
        block_size: 4,
        is_eagle: false,
        router_computes_hashes: false,
    };
    let guard = ResponseEndGuard {
        producer,
        request_id: Some("unpolled".to_string()),
        output_tokens: BTreeMap::new(),
        observed_cached_tokens: None,
    };
    let stream = async_stream::stream! {
        let _guard = guard;
        if false {
            yield ();
        }
    };

    drop(stream);
    assert!(matches!(rx.try_recv(), Ok(RawEvent::End(_))));
    drt.shutdown();
    Ok(())
}

#[tokio::test]
async fn process_local_event_plane_round_trip_drains_cleanly() -> Result<()> {
    let runtime = Runtime::from_current()?;
    let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
    let namespace = drt.namespace(format!("kv-sol-wire-{}", uuid::Uuid::new_v4()))?;
    let component = namespace.component("estimator")?;
    let subscriber = EventSubscriber::for_namespace(&namespace, KV_CACHE_SOL_TOPIC)
        .await?
        .typed::<KvCacheSolEvent>();
    let publisher = EventPublisher::for_namespace(&namespace, KV_CACHE_SOL_TOPIC).await?;
    let metrics = Arc::new(EstimatorMetrics::new(&component, Duration::from_secs(60))?);
    let cancel = CancellationToken::new();
    let task = tokio::spawn(run_estimator(
        subscriber,
        KvCacheSolCore::new(Duration::from_secs(60), 100, 100),
        metrics.clone(),
        cancel.clone(),
    ));

    // Avoid the PUB/SUB slow-joiner window before sending the first event.
    tokio::time::sleep(Duration::from_millis(150)).await;
    publisher
        .publish(&start_event("wire", now_ms(), vec![11], 4))
        .await?;
    publisher.publish(&end_event("wire", now_ms(), 1)).await?;

    timeout(Duration::from_secs(5), async {
        loop {
            if metrics
                .events_total
                .with_label_values(&["request_end"])
                .get()
                == 1
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await?;

    let labels = &["model", "config"];
    assert_eq!(
        metrics
            .events_total
            .with_label_values(&["request_start"])
            .get(),
        1
    );
    assert_eq!(metrics.requests_total.with_label_values(labels).get(), 1);
    assert_eq!(
        metrics.prompt_tokens_total.with_label_values(labels).get(),
        4
    );
    assert_eq!(metrics.pending_requests.get(), 0);
    assert_eq!(metrics.degraded.get(), 0);

    cancel.cancel();
    task.await?;
    drt.shutdown();
    Ok(())
}

#[test]
fn continuation_completes_partial_standard_block() {
    let tokens = vec![1, 2, 3, 4, 5, 6];
    let start = RawStart {
        occurred_at_ms: 0,
        request_id: "r".to_string(),
        domain: domain(false),
        prompt: PromptHashInput::Tokens {
            token_ids: tokens.clone(),
            block_mm_infos: None,
        },
        next_block_mm_info: None,
        lora_name: None,
        cache_namespace: None,
    };
    let local = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default());
    let seq = compute_seq_hash_for_block(&local);
    let mut continuation = HashContinuation::new(&start, &seq, tokens[4..].to_vec());
    let next = continuation.append(&[7, 8, 9, 10, 11, 12]);
    let all = compute_block_hash_for_seq(
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        4,
        BlockHashOptions::default(),
    );
    assert_eq!(next, compute_seq_hash_for_block(&all)[1..]);
}

#[test]
fn continuation_matches_eagle_hashing() {
    let tokens = vec![1, 2, 3, 4, 5, 6];
    let mut start = RawStart {
        occurred_at_ms: 0,
        request_id: "r".to_string(),
        domain: domain(true),
        prompt: PromptHashInput::Tokens {
            token_ids: tokens.clone(),
            block_mm_infos: None,
        },
        next_block_mm_info: None,
        lora_name: None,
        cache_namespace: None,
    };
    start.domain.is_eagle = true;
    let options = BlockHashOptions {
        is_eagle: Some(true),
        ..Default::default()
    };
    let local = compute_block_hash_for_seq(&tokens, 4, options);
    let seq = compute_seq_hash_for_block(&local);
    let mut continuation = HashContinuation::new(&start, &seq, tokens[4..].to_vec());
    let next = continuation.append(&[7, 8, 9, 10, 11, 12]);
    let all = compute_block_hash_for_seq(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 4, options);
    assert_eq!(next, compute_seq_hash_for_block(&all)[1..]);
}

#[test]
fn long_continuation_matches_single_pass_hashing() {
    let prompt = vec![1, 2, 3, 4, 5, 6];
    let generated: Vec<_> = (0..16_384).map(|token| token as TokenIdType).collect();
    let start = RawStart {
        occurred_at_ms: 0,
        request_id: "long".to_string(),
        domain: domain(false),
        prompt: PromptHashInput::Tokens {
            token_ids: prompt.clone(),
            block_mm_infos: None,
        },
        next_block_mm_info: None,
        lora_name: None,
        cache_namespace: None,
    };
    let prompt_hashes = compute_seq_hash_for_block(&compute_block_hash_for_seq(
        &prompt,
        4,
        BlockHashOptions::default(),
    ));
    let mut continuation = HashContinuation::new(&start, &prompt_hashes, prompt[4..].to_vec());
    let actual = continuation.append(&generated);

    let mut all = prompt;
    all.extend_from_slice(&generated);
    let expected = compute_seq_hash_for_block(&compute_block_hash_for_seq(
        &all,
        4,
        BlockHashOptions::default(),
    ));
    assert_eq!(actual, expected[1..]);
}

#[test]
fn continuation_preserves_lora_namespace_and_multimodal_hashing() {
    let mm = BlockExtraInfo {
        mm_objects: vec![dynamo_kv_router::protocols::BlockMmObjectInfo {
            mm_hash: 42,
            offsets: vec![(0, 2)],
        }],
    };
    let tokens = vec![1, 2, 3, 4, 5, 6];
    let prompt_mm_infos = vec![None, Some(mm.clone())];
    let start = RawStart {
        occurred_at_ms: 0,
        request_id: "r".to_string(),
        domain: domain(false),
        prompt: PromptHashInput::Tokens {
            token_ids: tokens.clone(),
            block_mm_infos: Some(prompt_mm_infos.clone()),
        },
        next_block_mm_info: Some(mm.clone()),
        lora_name: Some("adapter".to_string()),
        cache_namespace: Some("tenant".to_string()),
    };
    let options = BlockHashOptions {
        block_mm_infos: Some(&prompt_mm_infos),
        lora_name: start.lora_name.as_deref(),
        cache_namespace: start.cache_namespace.as_deref(),
        is_eagle: Some(false),
    };
    let initial = compute_block_hash_for_seq(&tokens, 4, options);
    let initial_seq = compute_seq_hash_for_block(&initial);
    let mut continuation = HashContinuation::new(&start, &initial_seq, tokens[4..].to_vec());
    let next = continuation.append(&[7, 8]);
    let full_tokens = [1, 2, 3, 4, 5, 6, 7, 8];
    let full_mm = [None, Some(mm)];
    let expected = compute_seq_hash_for_block(&compute_block_hash_for_seq(
        &full_tokens,
        4,
        BlockHashOptions {
            block_mm_infos: Some(&full_mm),
            lora_name: Some("adapter"),
            cache_namespace: Some("tenant"),
            is_eagle: Some(false),
        },
    ));
    assert_eq!(next, expected[1..]);
}

#[test]
fn response_continuation_becomes_a_future_prompt_hit() {
    let tokens = vec![1, 2, 3, 4, 5, 6];
    let local = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default());
    let prompt_hashes = compute_seq_hash_for_block(&local);
    let start = RawStart {
        occurred_at_ms: 10,
        request_id: "a".to_string(),
        domain: domain(false),
        prompt: PromptHashInput::Tokens {
            token_ids: tokens.clone(),
            block_mm_infos: None,
        },
        next_block_mm_info: None,
        lora_name: None,
        cache_namespace: None,
    };
    let mut continuation = HashContinuation::new(&start, &prompt_hashes, tokens[4..].to_vec());
    let response_hashes = continuation.append(&[7, 8]);

    let mut core = KvCacheSolCore::new(Duration::from_secs(1), 100, 100);
    core.apply(1, start_event("a", 10, prompt_hashes.clone(), 6));
    core.apply(
        1,
        KvCacheSolEvent {
            producer_sequence: 1,
            occurred_at_ms: 20,
            request_id: "a".to_string(),
            domain: domain(false),
            kind: KvCacheSolEventKind::RequestEnd {
                continuation_sequence_hashes: BTreeMap::from([(0, response_hashes.clone())]),
                observed_cached_tokens: Some(0),
            },
        },
    );
    let next = core.apply(
        1,
        start_event("b", 30, vec![prompt_hashes[0], response_hashes[0]], 8),
    );
    assert!(matches!(next, SolObservation::Start { hit_tokens: 8, .. }));
}

#[test]
fn publisher_sequence_follows_worker_order_and_preserves_real_gaps() {
    let missing = AtomicU64::new(0);
    let mut next = 0;
    assert_eq!(reserve_publisher_sequence(&mut next, &missing), 0);
    assert_eq!(reserve_publisher_sequence(&mut next, &missing), 1);

    missing.fetch_add(3, Ordering::Relaxed);
    assert_eq!(reserve_publisher_sequence(&mut next, &missing), 5);
    assert_eq!(reserve_publisher_sequence(&mut next, &missing), 6);
}

#[test]
fn event_wire_payload_contains_no_raw_tokens() {
    let start_json = serde_json::to_string(&start_event("a", 10, vec![11, 12], 8)).unwrap();
    assert!(!start_json.contains("token_ids"));
    assert!(start_json.contains("prompt_sequence_hashes"));

    let mut end = end_event("a", 20, 1);
    end.kind = KvCacheSolEventKind::RequestEnd {
        continuation_sequence_hashes: BTreeMap::from([(0, vec![101, 102])]),
        observed_cached_tokens: Some(4),
    };
    let end_json = serde_json::to_string(&end).unwrap();
    assert!(!end_json.contains("token_ids"));
    assert!(end_json.contains("continuation_sequence_hashes"));
}
