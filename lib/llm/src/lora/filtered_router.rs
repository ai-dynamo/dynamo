// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA-filtered router wrapper for non-KV routing modes.
//!
//! Implements 2-stage routing:
//!   Stage 1 — LoRA filter narrows the candidate worker set.
//!   Stage 2 — The inner PushRouter selects from candidates.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures::Stream;
use rand::Rng;

use dynamo_runtime::{
    engine::{
        AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineStream, Data,
    },
    pipeline::{Error, ManyOut, RouterMode, SingleIn, network::egress::push_router::PushRouter},
    protocols::annotated::Annotated,
};

use crate::lora::filter::LoraFilter;
use crate::lora::load_estimator::LoadEstimator;
use crate::preprocessor::PreprocessedRequest;
use crate::protocols::common::llm_backend::LLMEngineOutput;

/// Decrements the [`LoadEstimator`] counter for a LoRA when dropped.
struct LoadGuard {
    estimator: Arc<LoadEstimator>,
    lora_name: String,
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        self.estimator.decrement_load(&self.lora_name);
    }
}

/// Thin wrapper around the inner response stream that holds a [`LoadGuard`].
struct LoadTrackingStream<S> {
    inner: S,
    _guard: LoadGuard,
}

impl<S> std::fmt::Debug for LoadTrackingStream<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadTrackingStream")
            .field("lora", &self._guard.lora_name)
            .finish()
    }
}

impl<S> Stream for LoadTrackingStream<S>
where
    S: Stream + Unpin,
{
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<S> AsyncEngineContextProvider for LoadTrackingStream<S>
where
    S: AsyncEngineContextProvider + Unpin,
{
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.inner.context()
    }
}

impl<T: Data, S> AsyncEngineStream<T> for LoadTrackingStream<S> where
    S: Stream<Item = T> + AsyncEngineContextProvider + Send + Unpin
{
}

/// Wraps a `PushRouter` with a LoRA pre-filter stage and load tracking.
pub struct LoraFilteredRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    filter: Arc<LoraFilter>,
    load_estimator: Arc<LoadEstimator>,
    router_mode: RouterMode,
    round_robin_counter: AtomicU64,
}

impl LoraFilteredRouter {
    pub fn new(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        filter: Arc<LoraFilter>,
        load_estimator: Arc<LoadEstimator>,
        router_mode: RouterMode,
    ) -> Self {
        tracing::info!(
            ?router_mode,
            "LoRA-filtered router created (2-stage: LoRA filter → {:?})",
            router_mode
        );
        Self {
            inner,
            filter,
            load_estimator,
            router_mode,
            round_robin_counter: AtomicU64::new(0),
        }
    }

    fn select_from(&self, candidates: &[u64]) -> Option<u64> {
        if candidates.is_empty() {
            return None;
        }
        match self.router_mode {
            RouterMode::RoundRobin => {
                let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
                Some(candidates[counter % candidates.len()])
            }
            RouterMode::Random => {
                let idx = rand::rng().random::<u64>() as usize;
                Some(candidates[idx % candidates.len()])
            }
            // Direct, KV, and advanced modes are never routed through LoraFilteredRouter
            // (Direct uses DirectRoutingRouter; KV uses KvPushRouter; advanced modes
            // bypass LoRA filtering in common.rs). These arms exist only for match
            // exhaustiveness.
            RouterMode::Direct
            | RouterMode::PowerOfTwoChoices
            | RouterMode::LeastLoaded
            | RouterMode::DeviceAwareWeighted => {
                tracing::warn!(
                    ?self.router_mode,
                    "LoraFilteredRouter::select_from called with unexpected router mode, using first candidate"
                );
                Some(candidates[0])
            }
            RouterMode::KV => {
                tracing::error!("LoraFilteredRouter should not be used with KV routing mode");
                Some(candidates[0])
            }
        }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for LoraFilteredRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let lora_name = request
            .routing
            .as_ref()
            .and_then(|r| r.lora_name.as_deref())
            .map(|s| s.to_string());

        // Base-model (non-LoRA) request: nothing to filter or load-track. Delegate straight to
        // the inner load-aware push router so base traffic on a LoRA-enabled deployment keeps the
        // unmodified hot path (no avail/free scans, no set allocation, no LoadGuard).
        let Some(lora_name) = lora_name else {
            return self.inner.generate(request).await;
        };

        self.load_estimator.increment_load(&lora_name);
        let guard = LoadGuard {
            estimator: self.load_estimator.clone(),
            lora_name: lora_name.clone(),
        };

        // Stage 1: narrow to the LoRA's replica set against the FULL routable set, so the
        // intended replicas are always represented even when they are currently busy.
        //
        // Filtering the "free" subset first would be wrong: if a LoRA's replicas are all busy
        // but some unrelated worker is free, the filter would see no replica intersection in the
        // free set and fall back to returning those unrelated free workers — scattering adapter
        // traffic off the replica set exactly when it is saturated (and forcing a cold adapter
        // load on a non-replica worker). Filtering the routable set keeps the replica constraint.
        let routable = self.inner.client.instance_ids_avail();
        let replica_candidates = self
            .filter
            .filter_worker_ids_for_lora(Some(lora_name.as_str()), &routable);

        // Stage 2: among the replica candidates, prefer free (non-overloaded) workers to match
        // PushRouter's load-aware selection. Only when every replica candidate is busy do we fall
        // back to the (busy) replica set itself, rather than degrading to non-replica workers —
        // so the replica-set constraint is preserved even under saturation.
        let free: std::collections::HashSet<u64> =
            self.inner.client.instance_ids_free().into_iter().collect();
        let free_replica_candidates: Vec<u64> = replica_candidates
            .iter()
            .copied()
            .filter(|id| free.contains(id))
            .collect();
        let candidates = if free_replica_candidates.is_empty() {
            replica_candidates
        } else {
            free_replica_candidates
        };

        if candidates.is_empty() {
            return Err(anyhow::anyhow!(
                "No workers available after LoRA filtering (lora={})",
                lora_name
            ));
        }

        let target = self
            .select_from(&candidates)
            .expect("candidates is non-empty");
        tracing::debug!(
            lora = %lora_name,
            worker_id = target,
            candidates = candidates.len(),
            routable = routable.len(),
            free = free.len(),
            "LoRA-filtered router selected worker"
        );

        // Constrain the inner router's vanished-target fallback to the LoRA candidate set, so a
        // race where `target` disappears mid-dispatch reselects another replica-set worker rather
        // than escaping to an arbitrary worker outside the placement table.
        let candidate_set: std::collections::HashSet<u64> = candidates.iter().copied().collect();
        let response_stream = self
            .inner
            .direct_within(request, target, Some(&candidate_set))
            .await?;
        let tracking = LoadTrackingStream {
            inner: response_stream,
            _guard: guard,
        };
        Ok(Box::pin(tracking))
    }
}
