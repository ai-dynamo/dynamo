// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Out-of-band estimator for the workload-specific KV-cache speed-of-light hit rate.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use dynamo_kv_router::protocols::{
    BlockExtraInfo, BlockHashOptions, compute_block_hash_for_seq, compute_next_seq_hash,
    compute_seq_hash_for_block,
};
use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, ManyOut, Operator, PipelineOperator, ResponseStream,
    ServerStreamingEngine, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;
use futures::StreamExt;
use prometheus::{IntCounterVec, IntGauge};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::protocols::TokenIdType;
use crate::protocols::common::llm_backend::{BackendOutput, LLMEngineOutput, PreprocessedRequest};
use crate::protocols::common::timing::RequestTracker;
use dynamo_runtime::engine::Data;

const KV_CACHE_SOL_TOPIC: &str = "kv-cache-sol-v1";
const DEFAULT_QUEUE_CAPACITY: usize = 256;
const MAX_PRODUCER_INFLIGHT_REQUESTS: usize = 100_000;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn env_enabled(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
struct KvCacheSolDomain {
    pub model: String,
    pub model_config: String,
    pub block_size: u32,
    pub is_eagle: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum KvCacheSolEventKind {
    RequestStart {
        prompt_sequence_hashes: Vec<u64>,
        prompt_tokens: u64,
    },
    RequestEnd {
        continuation_sequence_hashes: BTreeMap<u32, Vec<u64>>,
        observed_cached_tokens: Option<u64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct KvCacheSolEvent {
    pub producer_sequence: u64,
    pub occurred_at_ms: u64,
    pub request_id: String,
    pub domain: KvCacheSolDomain,
    pub kind: KvCacheSolEventKind,
}

struct ProducerMetrics {
    events_total: IntCounterVec,
    events_dropped_total: IntCounterVec,
    skipped_requests_total: IntCounterVec,
    queue_depth: IntGauge,
    degraded: IntGauge,
}

impl ProducerMetrics {
    fn new(component: &Component) -> Result<Self> {
        let metrics = component.metrics();
        Ok(Self {
            events_total: metrics.create_intcountervec(
                "kv_cache_sol_events_total",
                "KV-cache speed-of-light events published by type",
                &["event_type"],
                &[],
            )?,
            events_dropped_total: metrics.create_intcountervec(
                "kv_cache_sol_events_dropped_total",
                "KV-cache speed-of-light events dropped before publication",
                &["reason"],
                &[],
            )?,
            skipped_requests_total: metrics.create_intcountervec(
                "kv_cache_sol_skipped_requests_total",
                "Requests excluded from KV-cache speed-of-light estimation",
                &["reason"],
                &[],
            )?,
            queue_depth: metrics.create_intgauge(
                "kv_cache_sol_queue_depth",
                "Events waiting in the local KV-cache speed-of-light publisher queue",
                &[],
            )?,
            degraded: metrics.create_intgauge(
                "kv_cache_sol_producer_degraded",
                "Whether this producer has lost or rejected a KV-cache speed-of-light event",
                &[],
            )?,
        })
    }

    fn record_drop(&self, reason: &str) {
        self.events_dropped_total.with_label_values(&[reason]).inc();
        self.degraded.set(1);
    }
}

#[derive(Clone)]
struct KvCacheSolProducer {
    tx: mpsc::Sender<RawEvent>,
    missing_sequences: Arc<AtomicU64>,
    metrics: Arc<ProducerMetrics>,
    block_size: u32,
    is_eagle: bool,
    router_computes_hashes: bool,
}

enum PromptHashInput {
    Router {
        tracker: Arc<RequestTracker>,
        prompt_tokens: usize,
        tail_token_ids: Vec<TokenIdType>,
    },
    Tokens {
        token_ids: Vec<TokenIdType>,
        block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    },
}

struct RawStart {
    occurred_at_ms: u64,
    request_id: String,
    domain: KvCacheSolDomain,
    prompt: PromptHashInput,
    next_block_mm_info: Option<BlockExtraInfo>,
    lora_name: Option<String>,
    cache_namespace: Option<String>,
}

struct RawEnd {
    occurred_at_ms: u64,
    request_id: String,
    output_tokens: BTreeMap<u32, Vec<TokenIdType>>,
    observed_cached_tokens: Option<usize>,
}

enum RawEvent {
    Start(RawStart),
    End(RawEnd),
}

impl KvCacheSolProducer {
    pub fn from_env(
        component: Component,
        block_size: u32,
        is_eagle: bool,
        router_computes_hashes: bool,
    ) -> Result<Option<Self>> {
        if !env_enabled("DYN_KV_CACHE_SOL_ENABLED") {
            return Ok(None);
        }
        if block_size == 0 {
            anyhow::bail!("DYN_KV_CACHE_SOL_ENABLED requires a non-zero KV cache block size");
        }
        let queue_capacity = std::env::var("DYN_KV_CACHE_SOL_QUEUE_CAPACITY")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()
            .context("invalid DYN_KV_CACHE_SOL_QUEUE_CAPACITY")?
            .unwrap_or(DEFAULT_QUEUE_CAPACITY);
        if queue_capacity == 0 {
            anyhow::bail!("DYN_KV_CACHE_SOL_QUEUE_CAPACITY must be greater than zero");
        }

        let metrics = Arc::new(ProducerMetrics::new(&component)?);
        let (tx, rx) = mpsc::channel(queue_capacity);
        let missing_sequences = Arc::new(AtomicU64::new(0));
        let producer = Self {
            tx,
            missing_sequences: missing_sequences.clone(),
            metrics: metrics.clone(),
            block_size,
            is_eagle,
            router_computes_hashes,
        };
        let runtime = component.drt().runtime().secondary();
        runtime.spawn(publisher_worker(component, rx, metrics, missing_sequences));
        Ok(Some(producer))
    }

    fn try_send(&self, event: RawEvent) {
        match self.tx.try_send(event) {
            Ok(()) => self
                .metrics
                .queue_depth
                .set(self.tx.max_capacity().saturating_sub(self.tx.capacity()) as i64),
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.missing_sequences.fetch_add(1, Ordering::Relaxed);
                self.metrics.record_drop("queue_full");
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.missing_sequences.fetch_add(1, Ordering::Relaxed);
                self.metrics.queue_depth.set(0);
                self.metrics.record_drop("worker_closed");
            }
        }
    }

    fn prepare_start(
        &self,
        request_id: String,
        request: &mut PreprocessedRequest,
    ) -> Option<(RawStart, Arc<RequestTracker>)> {
        if request.is_probe {
            self.metrics
                .skipped_requests_total
                .with_label_values(&["health_probe"])
                .inc();
            return None;
        }
        if request.prompt_embeds.is_some() {
            self.metrics
                .skipped_requests_total
                .with_label_values(&["prompt_embeds"])
                .inc();
            return None;
        }
        let Some(model_config) = request.mdc_sum.clone() else {
            self.metrics
                .skipped_requests_total
                .with_label_values(&["missing_model_config"])
                .inc();
            return None;
        };

        let tracker = request
            .tracker
            .get_or_insert_with(|| Arc::new(RequestTracker::new()))
            .clone();
        tracker.enable_kv_cache_sol_tracking();
        let (tokens, block_mm_infos) = request.block_mm_routing_info();
        let completed_blocks =
            tokens.len().saturating_sub(usize::from(self.is_eagle)) / self.block_size as usize;
        let next_block_mm_info = block_mm_infos
            .and_then(|infos| infos.get(completed_blocks))
            .cloned()
            .flatten();
        let prompt = if self.router_computes_hashes {
            let retained = self.block_size as usize + usize::from(self.is_eagle);
            PromptHashInput::Router {
                tracker: tracker.clone(),
                prompt_tokens: tokens.len(),
                tail_token_ids: tokens[tokens.len().saturating_sub(retained)..].to_vec(),
            }
        } else {
            PromptHashInput::Tokens {
                token_ids: tokens.to_vec(),
                block_mm_infos: block_mm_infos.map(ToOwned::to_owned),
            }
        };
        let routing = request.routing.as_ref();
        let domain = KvCacheSolDomain {
            model: request.model.clone(),
            model_config,
            block_size: self.block_size,
            is_eagle: self.is_eagle,
        };
        Some((
            RawStart {
                occurred_at_ms: now_ms(),
                request_id,
                domain,
                prompt,
                next_block_mm_info,
                lora_name: routing.and_then(|hints| hints.lora_name.clone()),
                cache_namespace: routing.and_then(|hints| hints.cache_namespace.clone()),
            },
            tracker,
        ))
    }

    fn start(&self, start: RawStart) {
        self.try_send(RawEvent::Start(start));
    }

    fn end(
        &self,
        request_id: String,
        output_tokens: BTreeMap<u32, Vec<TokenIdType>>,
        observed_cached_tokens: Option<usize>,
    ) {
        self.try_send(RawEvent::End(RawEnd {
            occurred_at_ms: now_ms(),
            request_id,
            output_tokens,
            observed_cached_tokens,
        }));
    }
}

#[derive(Clone)]
struct HashContinuation {
    block_size: u32,
    is_eagle: bool,
    tail: Vec<TokenIdType>,
    parent_sequence_hash: Option<u64>,
    next_block_mm_info: Option<BlockExtraInfo>,
    lora_name: Option<String>,
    cache_namespace: Option<String>,
}

impl HashContinuation {
    fn new(start: &RawStart, sequence_hashes: &[u64], tail: Vec<TokenIdType>) -> Self {
        Self {
            block_size: start.domain.block_size,
            is_eagle: start.domain.is_eagle,
            tail,
            parent_sequence_hash: sequence_hashes.last().copied(),
            next_block_mm_info: start.next_block_mm_info.clone(),
            lora_name: start.lora_name.clone(),
            cache_namespace: start.cache_namespace.clone(),
        }
    }

    fn append(&mut self, tokens: &[TokenIdType]) -> Vec<u64> {
        self.tail.extend_from_slice(tokens);
        let stride = self.block_size as usize;
        let window = stride + usize::from(self.is_eagle);
        let mut hashes = Vec::new();
        let mut consumed = 0;
        while self.tail.len().saturating_sub(consumed) >= window {
            let mm_infos = self.next_block_mm_info.take().map(|info| vec![Some(info)]);
            let local = compute_block_hash_for_seq(
                &self.tail[consumed..consumed + window],
                self.block_size,
                BlockHashOptions {
                    block_mm_infos: mm_infos.as_deref(),
                    lora_name: self.lora_name.as_deref(),
                    cache_namespace: self.cache_namespace.as_deref(),
                    is_eagle: Some(self.is_eagle),
                },
            )[0];
            let sequence = self
                .parent_sequence_hash
                .map_or(local.0, |parent| compute_next_seq_hash(parent, local));
            hashes.push(sequence);
            self.parent_sequence_hash = Some(sequence);
            consumed += stride;
        }
        if consumed > 0 {
            self.tail.drain(..consumed);
        }
        hashes
    }
}

struct RequestHashState {
    domain: KvCacheSolDomain,
    continuation: HashContinuation,
}

async fn publisher_worker(
    component: Component,
    mut rx: mpsc::Receiver<RawEvent>,
    metrics: Arc<ProducerMetrics>,
    missing_sequences: Arc<AtomicU64>,
) {
    let publisher =
        match EventPublisher::for_namespace(component.namespace(), KV_CACHE_SOL_TOPIC).await {
            Ok(publisher) => publisher,
            Err(error) => {
                metrics.queue_depth.set(0);
                metrics.record_drop("publisher_init");
                tracing::error!(%error, "failed to initialize KV-cache speed-of-light publisher");
                return;
            }
        };
    let mut requests: HashMap<String, RequestHashState> = HashMap::new();
    let mut next_sequence = 0_u64;
    while let Some(raw) = rx.recv().await {
        metrics.queue_depth.set(rx.len() as i64);
        let sequence = reserve_publisher_sequence(&mut next_sequence, &missing_sequences);
        let event = match raw {
            RawEvent::Start(start) => {
                if requests.len() >= MAX_PRODUCER_INFLIGHT_REQUESTS {
                    metrics.record_drop("inflight_capacity");
                    continue;
                }
                if requests.contains_key(&start.request_id) {
                    metrics.record_drop("duplicate_request_id");
                    continue;
                }
                let (sequence_hashes, prompt_tokens, tail) = match &start.prompt {
                    PromptHashInput::Router {
                        tracker,
                        prompt_tokens,
                        tail_token_ids,
                    } => {
                        let Some(hashes) = tracker.kv_cache_sol_prompt_hashes() else {
                            metrics.record_drop("missing_router_hashes");
                            continue;
                        };
                        let tail_start =
                            hashes.sequence_hashes.len() * start.domain.block_size as usize;
                        if tail_start > *prompt_tokens {
                            metrics.record_drop("invalid_router_hashes");
                            continue;
                        }
                        let tail_len = prompt_tokens - tail_start;
                        if tail_len > tail_token_ids.len() {
                            metrics.record_drop("invalid_router_hashes");
                            continue;
                        }
                        (
                            hashes.sequence_hashes.clone(),
                            *prompt_tokens,
                            tail_token_ids[tail_token_ids.len() - tail_len..].to_vec(),
                        )
                    }
                    PromptHashInput::Tokens {
                        token_ids,
                        block_mm_infos,
                    } => {
                        let local = compute_block_hash_for_seq(
                            token_ids,
                            start.domain.block_size,
                            BlockHashOptions {
                                block_mm_infos: block_mm_infos.as_deref(),
                                lora_name: start.lora_name.as_deref(),
                                cache_namespace: start.cache_namespace.as_deref(),
                                is_eagle: Some(start.domain.is_eagle),
                            },
                        );
                        let sequence_hashes = compute_seq_hash_for_block(&local);
                        let tail_start = sequence_hashes.len() * start.domain.block_size as usize;
                        (
                            sequence_hashes,
                            token_ids.len(),
                            token_ids[tail_start.min(token_ids.len())..].to_vec(),
                        )
                    }
                };
                let continuation = HashContinuation::new(&start, &sequence_hashes, tail);
                requests.insert(
                    start.request_id.clone(),
                    RequestHashState {
                        domain: start.domain.clone(),
                        continuation,
                    },
                );
                KvCacheSolEvent {
                    producer_sequence: sequence,
                    occurred_at_ms: start.occurred_at_ms,
                    request_id: start.request_id,
                    domain: start.domain,
                    kind: KvCacheSolEventKind::RequestStart {
                        prompt_sequence_hashes: sequence_hashes,
                        prompt_tokens: prompt_tokens as u64,
                    },
                }
            }
            RawEvent::End(end) => {
                let Some(state) = requests.remove(&end.request_id) else {
                    metrics.record_drop("orphan_end");
                    continue;
                };
                let mut continuation_sequence_hashes = BTreeMap::new();
                for (choice, tokens) in end.output_tokens {
                    let mut continuation = state.continuation.clone();
                    continuation_sequence_hashes.insert(choice, continuation.append(&tokens));
                }
                KvCacheSolEvent {
                    producer_sequence: sequence,
                    occurred_at_ms: end.occurred_at_ms,
                    request_id: end.request_id,
                    domain: state.domain,
                    kind: KvCacheSolEventKind::RequestEnd {
                        continuation_sequence_hashes,
                        observed_cached_tokens: end
                            .observed_cached_tokens
                            .map(|value| value as u64),
                    },
                }
            }
        };
        let event_type = match event.kind {
            KvCacheSolEventKind::RequestStart { .. } => "request_start",
            KvCacheSolEventKind::RequestEnd { .. } => "request_end",
        };
        match publisher.publish(&event).await {
            Ok(()) => metrics.events_total.with_label_values(&[event_type]).inc(),
            Err(error) => {
                metrics.record_drop("publish_error");
                tracing::warn!(%error, "failed to publish KV-cache speed-of-light event");
            }
        }
    }
}

fn reserve_publisher_sequence(next_sequence: &mut u64, missing_sequences: &AtomicU64) -> u64 {
    *next_sequence = next_sequence.saturating_add(missing_sequences.swap(0, Ordering::AcqRel));
    let sequence = *next_sequence;
    *next_sequence = next_sequence.saturating_add(1);
    sequence
}

pub(crate) struct KvCacheSolTap {
    producer: KvCacheSolProducer,
}

impl KvCacheSolTap {
    pub(crate) fn from_env(
        component: Component,
        block_size: u32,
        is_eagle: bool,
        router_computes_hashes: bool,
    ) -> Result<Option<Arc<Self>>> {
        Ok(
            KvCacheSolProducer::from_env(component, block_size, is_eagle, router_computes_hashes)?
                .map(|producer| Arc::new(Self { producer })),
        )
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn into_operator_for<Resp>(
        self: &Arc<Self>,
    ) -> Arc<
        PipelineOperator<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<Resp>>,
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<Resp>>,
        >,
    >
    where
        Resp: SolResponse,
    {
        Operator::into_operator(self)
    }
}

pub(crate) trait SolResponse: Data {
    fn sol_token_ids(&self) -> &[TokenIdType];
    fn sol_choice_index(&self) -> u32;
    fn sol_observed_cached_tokens(&self) -> Option<usize>;
}

macro_rules! impl_sol_response {
    ($response:ty) => {
        impl SolResponse for $response {
            fn sol_token_ids(&self) -> &[TokenIdType] {
                &self.token_ids
            }

            fn sol_choice_index(&self) -> u32 {
                self.index.unwrap_or(0)
            }

            fn sol_observed_cached_tokens(&self) -> Option<usize> {
                self.completion_usage
                    .as_ref()
                    .and_then(|usage| usage.prompt_tokens_details.as_ref())
                    .and_then(|details| details.cached_tokens)
                    .map(|value| value as usize)
            }
        }
    };
}

impl_sol_response!(LLMEngineOutput);
impl_sol_response!(BackendOutput);

struct ResponseEndGuard {
    producer: KvCacheSolProducer,
    request_id: Option<String>,
    output_tokens: BTreeMap<u32, Vec<TokenIdType>>,
    observed_cached_tokens: Option<usize>,
}

impl ResponseEndGuard {
    fn finish(&mut self) {
        let Some(request_id) = self.request_id.take() else {
            return;
        };
        self.producer.end(
            request_id,
            std::mem::take(&mut self.output_tokens),
            self.observed_cached_tokens,
        );
    }
}

impl Drop for ResponseEndGuard {
    fn drop(&mut self) {
        self.finish();
    }
}

#[async_trait]
impl<Resp>
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<Resp>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<Resp>>,
    > for KvCacheSolTap
where
    Resp: SolResponse,
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
    ) -> Result<ManyOut<Annotated<Resp>>> {
        let producer = &self.producer;
        let (mut request, context) = request.into_parts();
        let request_id = context.id().to_string();
        let pending_start = producer.prepare_start(request_id.clone(), &mut request);
        let input = context.map(|_| request);
        let mut downstream = match next.generate(input).await {
            Ok(stream) => stream,
            Err(error) => {
                if let Some((start, tracker)) = pending_start {
                    producer.start(start);
                    producer.end(
                        request_id,
                        BTreeMap::new(),
                        tracker.worker_observed_cached_tokens(),
                    );
                }
                return Err(error);
            }
        };
        let Some((start, tracker)) = pending_start else {
            return Ok(downstream);
        };
        producer.start(start);

        let stream_context = downstream.context();
        let producer = producer.clone();
        let end_guard = ResponseEndGuard {
            producer,
            request_id: Some(request_id),
            output_tokens: BTreeMap::new(),
            observed_cached_tokens: tracker.worker_observed_cached_tokens(),
        };
        let wrapped = async_stream::stream! {
            let mut end_guard = end_guard;
            while let Some(item) = downstream.next().await {
                if let Some(output) = item.data.as_ref() {
                    end_guard
                        .output_tokens
                        .entry(output.sol_choice_index())
                        .or_default()
                        .extend_from_slice(output.sol_token_ids());
                    if end_guard.observed_cached_tokens.is_none() {
                        end_guard.observed_cached_tokens = output.sol_observed_cached_tokens();
                        if let Some(value) = end_guard.observed_cached_tokens {
                            tracker.record_worker_observed_cached_tokens(value);
                        }
                    }
                }
                yield item;
            }
            end_guard.finish();
        };
        Ok(ResponseStream::new(Box::pin(wrapped), stream_context))
    }
}

mod estimator;
pub use estimator::KvCacheSolEstimator;

#[cfg(test)]
use estimator::{DomainCache, EstimatorMetrics, KvCacheSolCore, SolObservation, run_estimator};
#[cfg(test)]
mod tests;
