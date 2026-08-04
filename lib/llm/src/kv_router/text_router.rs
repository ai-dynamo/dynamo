// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reported-load KV routing for backends that receive OpenAI text requests.
//!
//! Text-input backends cannot use the token path's prompt-block index because
//! tokenization happens behind the frontend. They instead report per-rank KV
//! occupancy and queue depth. This module chooses an exact `(worker, DP rank)`
//! from those reports and uses the shared session-affinity coordinator to keep
//! later requests on the same target.

use std::{
    cmp::Ordering,
    collections::HashMap,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

use dynamo_runtime::{
    component::Client,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Data, Error, ManyOut, PushRouter, ResponseStream,
        SingleIn, async_trait,
    },
    protocols::maybe_error::MaybeError,
};
use futures::Stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    discovery::{KvWorkerMonitor, RuntimeConfigWatch, WorkerLoadState},
    protocols::{
        common::extensions::NvExt,
        openai::{
            chat_completions::NvCreateChatCompletionRequest, completions::NvCreateCompletionRequest,
        },
    },
    session_affinity::{
        AffinityAcquire, AffinityCoordinator, AffinityTarget, affinity_id, invalid_argument,
    },
};

#[derive(Clone, Copy, Debug)]
struct CandidateLoad {
    target: AffinityTarget,
    kv_used_blocks: Option<u64>,
    kv_total_blocks: Option<u64>,
    num_waiting_reqs: Option<u64>,
}

impl CandidateLoad {
    fn usable_kv_load(&self) -> Option<(u64, u64)> {
        let used = self.kv_used_blocks?;
        let total = self.kv_total_blocks.filter(|total| *total > 0)?;
        Some((used, total))
    }

    fn has_usable_kv_load(&self) -> bool {
        self.usable_kv_load().is_some()
    }
}

#[derive(Default)]
struct TextKvRouterState {
    frontend_inflight: HashMap<AffinityTarget, u64>,
    fallback_counter: u64,
}

#[derive(Clone)]
pub(crate) struct TextKvRouter {
    client: Client,
    monitor: KvWorkerMonitor,
    runtime_configs: RuntimeConfigWatch,
    state: Arc<Mutex<TextKvRouterState>>,
}

impl TextKvRouter {
    pub(crate) fn new(
        client: Client,
        monitor: KvWorkerMonitor,
        runtime_configs: RuntimeConfigWatch,
    ) -> Self {
        Self {
            client,
            monitor,
            runtime_configs,
            state: Arc::new(Mutex::new(TextKvRouterState::default())),
        }
    }

    fn target_is_available(&self, target: AffinityTarget) -> bool {
        if !self.client.instance_ids_avail().contains(&target.worker_id) {
            return false;
        }
        if !self
            .runtime_configs
            .borrow()
            .contains_key(&target.worker_id)
        {
            return false;
        }
        candidate_targets(
            &[target.worker_id],
            &self.monitor.load_states_for(&[target.worker_id]),
        )
        .contains(&target)
    }

    fn select(
        &self,
        explicit: Option<AffinityTarget>,
    ) -> Result<(AffinityTarget, TextKvPermit), Error> {
        let mut worker_ids = self.client.instance_ids_free();
        let model_workers = self.runtime_configs.borrow();
        worker_ids.retain(|worker_id| model_workers.contains_key(worker_id));
        drop(model_workers);
        worker_ids.sort_unstable();
        if worker_ids.is_empty() {
            return Err(anyhow::anyhow!(
                "no free workers available for text KV routing on endpoint {}",
                self.client.endpoint.id()
            ));
        }

        if let Some(explicit) = explicit {
            if !worker_ids.contains(&explicit.worker_id) {
                return Err(invalid_argument(format!(
                    "worker {} is not available for text KV routing",
                    explicit.worker_id
                )));
            }
            worker_ids.retain(|worker_id| *worker_id == explicit.worker_id);
        }

        let load_states = self.monitor.load_states_for(&worker_ids);
        let mut candidates = candidate_loads(&worker_ids, &load_states);
        if let Some(explicit) = explicit
            && let Some(dp_rank) = explicit.dp_rank
        {
            candidates.retain(|candidate| candidate.target.dp_rank == Some(dp_rank));
        }
        if candidates.is_empty() {
            return Err(invalid_argument(
                "requested text KV routing target is unavailable",
            ));
        }

        let mut state = self.state.lock().unwrap();
        let target = choose_candidate(&candidates, &mut state);
        *state.frontend_inflight.entry(target).or_default() += 1;
        drop(state);
        Ok((target, TextKvPermit::new(self.state.clone(), target)))
    }

    fn reserve_existing(&self, target: AffinityTarget) -> TextKvPermit {
        let mut state = self.state.lock().unwrap();
        *state.frontend_inflight.entry(target).or_default() += 1;
        drop(state);
        TextKvPermit::new(self.state.clone(), target)
    }
}

pub(crate) trait TextKvRequest: Data + Serialize {
    fn nvext(&self) -> Option<&NvExt>;
    fn nvext_mut(&mut self) -> &mut Option<NvExt>;
}

impl TextKvRequest for NvCreateChatCompletionRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn nvext_mut(&mut self) -> &mut Option<NvExt> {
        &mut self.nvext
    }
}

impl TextKvRequest for NvCreateCompletionRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn nvext_mut(&mut self) -> &mut Option<NvExt> {
        &mut self.nvext
    }
}

fn explicit_target(request: &impl TextKvRequest) -> Result<Option<AffinityTarget>, Error> {
    let Some(nvext) = request.nvext() else {
        return Ok(None);
    };
    let worker_id = nvext.decode_worker_id.or(nvext.backend_instance_id);
    if worker_id.is_none() && nvext.dp_rank.is_some() {
        return Err(invalid_argument(
            "DP rank requires an explicit worker for text KV routing",
        ));
    }
    Ok(worker_id.map(|worker_id| AffinityTarget {
        worker_id,
        dp_rank: nvext.dp_rank,
    }))
}

fn apply_target(request: &mut impl TextKvRequest, target: AffinityTarget) {
    match target.dp_rank {
        Some(dp_rank) => {
            request
                .nvext_mut()
                .get_or_insert_with(NvExt::default)
                .dp_rank = Some(dp_rank);
        }
        None => {
            if let Some(nvext) = request.nvext_mut().as_mut() {
                nvext.dp_rank = None;
            }
        }
    }
}

pub(crate) struct TextKvPushRouter<T, U>
where
    T: TextKvRequest,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    inner: PushRouter<T, U>,
    selector: Arc<TextKvRouter>,
    affinity: Option<AffinityCoordinator>,
}

impl<T, U> TextKvPushRouter<T, U>
where
    T: TextKvRequest,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    pub(crate) fn new(
        inner: PushRouter<T, U>,
        selector: Arc<TextKvRouter>,
        affinity: Option<AffinityCoordinator>,
    ) -> Self {
        Self {
            inner,
            selector,
            affinity,
        }
    }

    async fn acquire_routable(
        &self,
        session_id: &crate::protocols::common::extensions::SessionAffinityId,
        explicit: Option<AffinityTarget>,
        request_context: &dyn dynamo_runtime::pipeline::AsyncEngineContext,
    ) -> Result<AffinityAcquire, Error> {
        let affinity = self
            .affinity
            .as_ref()
            .expect("affinity acquisition requires an enabled coordinator");
        let operation = affinity
            .acquire_with_context(session_id, explicit, request_context)
            .await?;
        let Some(target) = operation.target() else {
            return Ok(operation);
        };
        if self.selector.target_is_available(target) {
            return Ok(operation);
        }

        operation.invalidate();
        affinity
            .acquire_with_context(session_id, explicit, request_context)
            .await
    }

    async fn route(&self, mut request: SingleIn<T>) -> Result<ManyOut<U>, Error> {
        let explicit = explicit_target(request.content())?;
        let session_id = if self.affinity.is_some() {
            affinity_id(&request)?
        } else {
            None
        };

        let mut operation = match session_id.as_ref() {
            Some(session_id) => Some(
                self.acquire_routable(session_id, explicit, request.context().as_ref())
                    .await?,
            ),
            None => None,
        };

        let (target, mut permit) = match operation.as_ref().and_then(AffinityAcquire::target) {
            Some(target) => (target, self.selector.reserve_existing(target)),
            None => self.selector.select(explicit)?,
        };
        apply_target(&mut *request, target);

        tracing::info!(
            router_mode = "kv",
            kv_routing_mechanism = "reported-load",
            worker_id = target.worker_id,
            dp_rank = ?target.dp_rank,
            affinity_reuse = operation.as_ref().is_some_and(|operation| operation.target().is_some()),
            "Selected text-input KV target"
        );

        let stream = match self.inner.dispatch_exact(request, target.worker_id).await {
            Ok(stream) => stream,
            Err(error) => {
                permit.release();
                if let Some(operation) = operation.take() {
                    operation.invalidate();
                }
                return Err(error);
            }
        };
        let stream = permit.into_tracked_stream(stream);
        match operation {
            Some(operation) => operation.into_stream(target, stream),
            None => Ok(stream),
        }
    }
}

#[async_trait]
impl<T, U> AsyncEngine<SingleIn<T>, ManyOut<U>, Error> for TextKvPushRouter<T, U>
where
    T: TextKvRequest,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(&self, request: SingleIn<T>) -> Result<ManyOut<U>, Error> {
        self.route(request).await
    }
}

fn candidate_targets(
    worker_ids: &[u64],
    load_states: &HashMap<u64, WorkerLoadState>,
) -> Vec<AffinityTarget> {
    let mut targets = Vec::new();
    for worker_id in worker_ids {
        let Some(load_state) = load_states.get(worker_id) else {
            targets.push(AffinityTarget {
                worker_id: *worker_id,
                dp_rank: None,
            });
            continue;
        };

        let dp_size = load_state.data_parallel_size.max(1);
        let uses_dp_routing = load_state.data_parallel_start_rank != 0 || dp_size > 1;
        for offset in 0..dp_size {
            targets.push(AffinityTarget {
                worker_id: *worker_id,
                dp_rank: uses_dp_routing
                    .then_some(load_state.data_parallel_start_rank.saturating_add(offset)),
            });
        }
    }
    targets.sort_unstable();
    targets
}

fn candidate_loads(
    worker_ids: &[u64],
    load_states: &HashMap<u64, WorkerLoadState>,
) -> Vec<CandidateLoad> {
    candidate_targets(worker_ids, load_states)
        .into_iter()
        .map(|target| {
            let rank = target.dp_rank.unwrap_or(0);
            let state = load_states.get(&target.worker_id);
            CandidateLoad {
                target,
                kv_used_blocks: state.and_then(|state| state.kv_used_blocks.get(&rank).copied()),
                kv_total_blocks: state.and_then(|state| state.kv_total_blocks.get(&rank).copied()),
                num_waiting_reqs: state
                    .and_then(|state| state.num_waiting_reqs.get(&rank).copied()),
            }
        })
        .collect()
}

fn choose_candidate(candidates: &[CandidateLoad], state: &mut TextKvRouterState) -> AffinityTarget {
    let has_usable_load = candidates.iter().any(CandidateLoad::has_usable_kv_load);
    if !has_usable_load {
        let target = candidates[state.fallback_counter as usize % candidates.len()].target;
        state.fallback_counter = state.fallback_counter.wrapping_add(1);
        return target;
    }

    let mut best = Vec::new();
    for candidate in candidates
        .iter()
        .filter(|candidate| candidate.has_usable_kv_load())
    {
        let Some(current_best) = best.first().copied() else {
            best.push(candidate);
            continue;
        };
        match compare_candidate(candidate, current_best, state) {
            Ordering::Less => {
                best.clear();
                best.push(candidate);
            }
            Ordering::Equal => best.push(candidate),
            Ordering::Greater => {}
        }
    }

    let index = if best.len() == 1 {
        0
    } else {
        rand::rng().random_range(0..best.len())
    };
    best[index].target
}

fn compare_candidate(
    left: &CandidateLoad,
    right: &CandidateLoad,
    state: &TextKvRouterState,
) -> Ordering {
    let kv_occupancy = match (left.usable_kv_load(), right.usable_kv_load()) {
        (Some((left_used, left_total)), Some((right_used, right_total))) => (u128::from(left_used)
            * u128::from(right_total))
        .cmp(&(u128::from(right_used) * u128::from(left_total))),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    };

    kv_occupancy
        .then_with(|| compare_optional_load(left.num_waiting_reqs, right.num_waiting_reqs))
        .then_with(|| {
            state
                .frontend_inflight
                .get(&left.target)
                .copied()
                .unwrap_or_default()
                .cmp(
                    &state
                        .frontend_inflight
                        .get(&right.target)
                        .copied()
                        .unwrap_or_default(),
                )
        })
}

fn compare_optional_load(left: Option<u64>, right: Option<u64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

struct TextKvPermit {
    state: Arc<Mutex<TextKvRouterState>>,
    target: AffinityTarget,
    released: bool,
}

impl TextKvPermit {
    fn new(state: Arc<Mutex<TextKvRouterState>>, target: AffinityTarget) -> Self {
        Self {
            state,
            target,
            released: false,
        }
    }

    fn release(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        let mut state = self.state.lock().unwrap();
        let Some(inflight) = state.frontend_inflight.get_mut(&self.target) else {
            return;
        };
        *inflight = inflight.saturating_sub(1);
        if *inflight == 0 {
            state.frontend_inflight.remove(&self.target);
        }
    }

    fn into_tracked_stream<U: Data>(self, stream: ManyOut<U>) -> ManyOut<U> {
        let context = stream.context();
        ResponseStream::new(
            Box::pin(TextKvTrackedStream {
                stream,
                permit: Some(self),
            }),
            context,
        )
    }
}

impl Drop for TextKvPermit {
    fn drop(&mut self) {
        self.release();
    }
}

struct TextKvTrackedStream<U: Data> {
    stream: ManyOut<U>,
    permit: Option<TextKvPermit>,
}

impl<U: Data> Stream for TextKvTrackedStream<U> {
    type Item = U;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll = self.stream.as_mut().poll_next(cx);
        if matches!(poll, Poll::Ready(None)) {
            drop(self.permit.take());
        }
        poll
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        worker_id: u64,
        dp_rank: Option<u32>,
        used: Option<u64>,
        total: Option<u64>,
        waiting: Option<u64>,
    ) -> CandidateLoad {
        CandidateLoad {
            target: AffinityTarget { worker_id, dp_rank },
            kv_used_blocks: used,
            kv_total_blocks: total,
            num_waiting_reqs: waiting,
        }
    }

    #[test]
    fn chooses_lowest_normalized_kv_occupancy() {
        let candidates = [
            candidate(1, Some(0), Some(40), Some(100), Some(0)),
            candidate(2, Some(1), Some(300), Some(1000), Some(9)),
        ];
        let mut state = TextKvRouterState::default();
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[1].target
        );
    }

    #[test]
    fn queue_then_frontend_inflight_break_equal_occupancy() {
        let candidates = [
            candidate(1, Some(0), Some(30), Some(100), Some(2)),
            candidate(2, Some(1), Some(300), Some(1000), Some(1)),
        ];
        let mut state = TextKvRouterState::default();
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[1].target
        );

        let candidates = [
            candidate(1, Some(0), Some(30), Some(100), Some(1)),
            candidate(2, Some(1), Some(300), Some(1000), Some(1)),
        ];
        state.frontend_inflight.insert(candidates[1].target, 1);
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[0].target
        );
    }

    #[test]
    fn usable_load_sorts_before_missing_load() {
        let usable = candidate(1, Some(0), Some(30), Some(100), Some(0));
        let missing = candidate(2, Some(1), None, Some(100), Some(0));
        let state = TextKvRouterState::default();

        assert_eq!(compare_candidate(&usable, &missing, &state), Ordering::Less);
        assert_eq!(
            compare_candidate(&missing, &usable, &state),
            Ordering::Greater
        );
    }

    #[test]
    fn unknown_load_uses_stable_round_robin() {
        let candidates = [
            candidate(1, None, None, None, None),
            candidate(2, None, None, None, Some(0)),
        ];
        let mut state = TextKvRouterState::default();
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[0].target
        );
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[1].target
        );
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[0].target
        );
    }

    #[test]
    fn expands_dp_workers_but_not_plain_workers() {
        let mut states = HashMap::new();
        let mut dp = WorkerLoadState::default();
        dp.data_parallel_start_rank = 4;
        dp.data_parallel_size = 2;
        states.insert(20, dp);
        states.insert(10, WorkerLoadState::default());

        assert_eq!(
            candidate_targets(&[10, 20], &states),
            vec![
                AffinityTarget {
                    worker_id: 10,
                    dp_rank: None,
                },
                AffinityTarget {
                    worker_id: 20,
                    dp_rank: Some(4),
                },
                AffinityTarget {
                    worker_id: 20,
                    dp_rank: Some(5),
                },
            ]
        );
    }

    #[test]
    fn frontend_inflight_is_released_with_the_permit() {
        let state = Arc::new(Mutex::new(TextKvRouterState::default()));
        let target = AffinityTarget {
            worker_id: 10,
            dp_rank: Some(2),
        };
        state.lock().unwrap().frontend_inflight.insert(target, 1);
        let permit = TextKvPermit::new(state.clone(), target);
        drop(permit);
        assert!(
            !state
                .lock()
                .unwrap()
                .frontend_inflight
                .contains_key(&target)
        );
    }
}
