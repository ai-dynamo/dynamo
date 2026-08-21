// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result};
use dynamo_kv_router::{
    config::KvRouterConfig,
    protocols::{PrefillLoadHint, WorkerConfigLike, WorkerWithDpRank},
    sequences::topology::WorkerDpRange,
};
use dynamo_runtime::{
    CancellationToken, component::Endpoint, pipeline::PushRouter,
    traits::DistributedRuntimeProvider,
};
use parking_lot::{Mutex, RwLock};
use tokio::time::Instant;

use crate::{
    discovery::RuntimeConfigWatch,
    kv_router::sequence::{ActiveSequencesMulti, SequenceRequest, create_multi_worker_sequences},
    local_model::runtime_config::ModelRuntimeConfig,
    preprocessor::PreprocessedRequest,
    protocols::common::llm_backend::LLMEngineOutput,
};

use dynamo_runtime::protocols::annotated::Annotated;

/// Lazily constructed state behind [`WorkerInputs::LOAD`](dynamo_kv_router::selector::WorkerInputs::LOAD).
///
/// This deliberately owns the same slot tracker used by KV scheduling. Builtin policies may read
/// only active requests today, but the capability retains active prefill tokens and decode blocks
/// as well so `LOAD` has one meaning across all routing policies.
// TODO: P2C and least-loaded can be further optimized with an active-request-only tracker if it
// preserves atomic selection and reservation.
pub(crate) struct RoutingLoadState {
    slots: Arc<ActiveSequencesMulti>,
    // This snapshot is published only after `slots` has reconciled the same topology, so
    // selection never chooses a worker whose DP ranks have not been registered yet.
    workers: RwLock<HashMap<u64, ModelRuntimeConfig>>,
    block_size: usize,
    config: KvRouterConfig,
    selection_gate: Mutex<()>,
    cancellation_token: CancellationToken,
}

impl RoutingLoadState {
    pub(crate) async fn start(
        endpoint: Endpoint,
        block_size: u32,
        workers: RuntimeConfigWatch,
        config: KvRouterConfig,
        worker_type: &'static str,
    ) -> Result<Arc<Self>> {
        let initial_workers: HashMap<u64, ModelRuntimeConfig> = workers.borrow().clone();
        let router_id = endpoint.drt().discovery().instance_id();
        let cancellation_token = endpoint.drt().primary_token().child_token();
        let slots = create_multi_worker_sequences(
            endpoint,
            block_size as usize,
            initial_workers.clone(),
            config.router_replica_sync,
            router_id,
            worker_type,
            cancellation_token.child_token(),
        )
        .await
        .context("create routing load slot tracker")?;

        let state = Arc::new(Self {
            slots,
            workers: RwLock::new(initial_workers),
            block_size: block_size as usize,
            config,
            selection_gate: Mutex::new(()),
            cancellation_token,
        });
        Self::watch_worker_topology(&state, workers);
        Ok(state)
    }

    fn watch_worker_topology(state: &Arc<Self>, mut workers: RuntimeConfigWatch) {
        let weak = Arc::downgrade(state);
        let cancellation_token = state.cancellation_token.child_token();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => break,
                    changed = workers.changed() => {
                        if changed.is_err() {
                            break;
                        }
                        let Some(state) = weak.upgrade() else {
                            break;
                        };
                        let configured_workers = workers.borrow_and_update().clone();
                        let ranges = configured_workers
                            .iter()
                            .map(|(&worker_id, config)| {
                                WorkerDpRange::new(
                                    worker_id,
                                    config.data_parallel_start_rank(),
                                    config.data_parallel_size(),
                                )
                            })
                            .collect::<Vec<_>>();
                        let _selection = state.selection_gate.lock();
                        if let Err(error) = state.slots.reconcile_workers(ranges) {
                            tracing::error!(%error, "Invalid routing load worker topology update");
                        } else {
                            *state.workers.write() = configured_workers;
                        }
                    }
                }
            }
        });
    }

    pub(crate) fn select_and_reserve(
        self: &Arc<Self>,
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        request_id: &str,
        request: &PreprocessedRequest,
        pinned_worker: Option<(u64, Option<u32>)>,
    ) -> Result<RoutingLoadReservation> {
        // Cache-key extraction and cache-index reads are request work. Do them before the gate
        // so DeviceAware requests do not serialize that work with other admissions.
        let device_aware = pinned_worker
            .is_none()
            .then(|| router.prepare_device_aware_selection(Some(request)))
            .flatten();
        let tracked_request = self.tracked_request(request_id, request);

        // Serialize the policy decision with its slot booking. Without this boundary, concurrent
        // P2C decisions could all observe the same pre-admission load.
        let _selection = self.selection_gate.lock();
        let workers = self.workers.read();
        if let Some((worker_id, _)) = pinned_worker {
            anyhow::ensure!(
                workers.contains_key(&worker_id),
                "worker {worker_id} has no runtime configuration"
            );
        }
        let worker_id = router.select_target_with_prepared_load(
            device_aware.as_ref(),
            pinned_worker.map(|target| target.0),
            |worker_id| workers.contains_key(&worker_id),
            |worker_id| {
                workers.get(&worker_id).map_or(0, |config| {
                    self.slots.active_request_count_for_worker(
                        worker_id,
                        config.data_parallel_start_rank(),
                        config.data_parallel_size(),
                    ) as u64
                })
            },
        )?;
        let worker = match pinned_worker {
            Some((pinned_worker_id, Some(dp_rank))) => {
                debug_assert_eq!(worker_id, pinned_worker_id);
                WorkerWithDpRank::new(worker_id, dp_rank)
            }
            _ => self.least_loaded_rank(&workers, worker_id)?,
        };
        self.slots
            .add_request_if_registered(tracked_request.sequence_request(worker), Instant::now())
            .with_context(|| format!("reserve routing load on worker {worker:?}"))?;
        drop(workers);

        Ok(RoutingLoadReservation {
            state: Arc::clone(self),
            tracked_request,
            worker,
            armed: true,
        })
    }

    /// Select using the current caller-owned load without reserving it.
    pub(crate) fn peek(
        &self,
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<WorkerWithDpRank> {
        let device_aware = router.prepare_device_aware_selection(None);
        let _selection = self.selection_gate.lock();
        let workers = self.workers.read();
        let worker_id = router.select_target_with_prepared_load(
            device_aware.as_ref(),
            None,
            |worker_id| workers.contains_key(&worker_id),
            |worker_id| {
                workers.get(&worker_id).map_or(0, |config| {
                    self.slots.active_request_count_for_worker(
                        worker_id,
                        config.data_parallel_start_rank(),
                        config.data_parallel_size(),
                    ) as u64
                })
            },
        )?;
        self.least_loaded_rank(&workers, worker_id)
    }

    pub(crate) fn is_configured_worker(&self, worker_id: u64) -> bool {
        self.workers.read().contains_key(&worker_id)
    }

    fn least_loaded_rank(
        &self,
        workers: &HashMap<u64, ModelRuntimeConfig>,
        worker_id: u64,
    ) -> Result<WorkerWithDpRank> {
        let config = workers
            .get(&worker_id)
            .with_context(|| format!("worker {worker_id} has no runtime configuration"))?;
        let start_rank = config.data_parallel_start_rank();
        let end_rank = start_rank.saturating_add(config.data_parallel_size());
        (start_rank..end_rank)
            .map(|dp_rank| WorkerWithDpRank::new(worker_id, dp_rank))
            .min_by_key(|worker| self.slots.active_request_count(*worker))
            .with_context(|| format!("worker {worker_id} has no data-parallel ranks"))
    }

    fn tracked_request(&self, request_id: &str, request: &PreprocessedRequest) -> TrackedRequest {
        let routing = request.routing.as_ref();
        let expected_output_tokens = routing.and_then(|routing| routing.expected_output_tokens);
        let lora_name = routing.and_then(|routing| routing.lora_name.clone());
        let track_prefill_tokens = self
            .config
            .track_prefill_tokens(request.router_config_override.as_ref());
        let prefill_load_hint = track_prefill_tokens.then_some(PrefillLoadHint {
            initial_effective_prefill_tokens: request.token_ids.len(),
            expected_prefill_duration: None,
        });
        let full_blocks = request.token_ids.len() / self.block_size;
        let token_sequence = self.config.random_seq_hashes_for_tracking(full_blocks);

        TrackedRequest {
            request_id: request_id.to_string(),
            token_sequence,
            track_prefill_tokens,
            expected_output_tokens,
            prefill_load_hint,
            lora_name,
        }
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(crate) fn track_output_blocks(&self) -> bool {
        self.config.router_track_output_blocks
    }

    #[cfg(test)]
    pub(crate) fn active_request_count_for_test(&self, worker: WorkerWithDpRank) -> usize {
        self.slots.active_request_count(worker)
    }

    #[cfg(test)]
    pub(crate) fn active_blocks_for_test(&self, worker: WorkerWithDpRank) -> usize {
        self.slots
            .active_blocks()
            .get(&worker)
            .copied()
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn active_tokens_for_test(&self, worker: WorkerWithDpRank) -> usize {
        self.slots
            .active_tokens(Instant::now())
            .get(&worker)
            .copied()
            .unwrap_or(0)
    }
}

impl Drop for RoutingLoadState {
    fn drop(&mut self) {
        self.cancellation_token.cancel();
    }
}

#[derive(Clone)]
struct TrackedRequest {
    request_id: String,
    token_sequence: Option<Vec<u64>>,
    track_prefill_tokens: bool,
    expected_output_tokens: Option<u32>,
    prefill_load_hint: Option<PrefillLoadHint>,
    lora_name: Option<String>,
}

impl TrackedRequest {
    fn sequence_request(&self, worker: WorkerWithDpRank) -> SequenceRequest {
        SequenceRequest {
            request_id: self.request_id.clone(),
            token_sequence: self.token_sequence.clone(),
            track_prefill_tokens: self.track_prefill_tokens,
            expected_output_tokens: self.expected_output_tokens,
            prefill_load_hint: self.prefill_load_hint,
            worker,
            lora_name: self.lora_name.clone(),
        }
    }
}

/// One request booking in the shared slot tracker.
pub(crate) struct RoutingLoadReservation {
    state: Arc<RoutingLoadState>,
    tracked_request: TrackedRequest,
    worker: WorkerWithDpRank,
    armed: bool,
}

impl RoutingLoadReservation {
    pub(crate) fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    pub(crate) fn block_size(&self) -> usize {
        self.state.block_size()
    }

    pub(crate) fn track_output_blocks(&self) -> bool {
        self.state.track_output_blocks()
    }

    pub(crate) fn retarget(&mut self, worker_id: u64) -> Result<WorkerWithDpRank> {
        if self.worker.worker_id == worker_id {
            return Ok(self.worker);
        }

        let _selection = self.state.selection_gate.lock();
        let workers = self.state.workers.read();
        let next_worker = self.state.least_loaded_rank(&workers, worker_id)?;
        self.state
            .slots
            .free_if_worker(
                &self.tracked_request.request_id,
                self.worker,
                Instant::now(),
            )
            .context("release routing load before transport fallback")?;
        self.armed = false;
        self.state
            .slots
            .add_request_if_registered(
                self.tracked_request.sequence_request(next_worker),
                Instant::now(),
            )
            .with_context(|| format!("reserve routing load fallback on worker {next_worker:?}"))?;
        self.worker = next_worker;
        self.armed = true;
        Ok(self.worker)
    }

    pub(crate) fn mark_prefill_completed(&self) -> Result<()> {
        self.state
            .slots
            .mark_prefill_completed(&self.tracked_request.request_id, Instant::now())
            .context("mark routing load prefill completed")
    }

    pub(crate) fn add_output_block(&self, decay_fraction: Option<f64>) -> Result<()> {
        self.state
            .slots
            .add_output_block(&self.tracked_request.request_id, decay_fraction)
            .context("add routing load output block")
    }

    fn release(&mut self) -> Result<()> {
        if !self.armed {
            return Ok(());
        }
        self.state
            .slots
            .free_if_worker(
                &self.tracked_request.request_id,
                self.worker,
                Instant::now(),
            )
            .context("release routing load reservation")?;
        self.armed = false;
        Ok(())
    }
}

impl Drop for RoutingLoadReservation {
    fn drop(&mut self) {
        if let Err(error) = self.release() {
            tracing::warn!(
                request_id = %self.tracked_request.request_id,
                worker = ?self.worker,
                %error,
                "Failed to release routing load reservation"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, mpsc},
        time::Duration,
    };

    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        pipeline::{MultimodalCacheIndex, RouterMode},
    };
    use tokio::sync::watch;

    use super::*;
    use crate::protocols::common::timing::WORKER_TYPE_DECODE;

    struct NotifyingCacheIndex {
        worker_id: u64,
        entered: mpsc::Sender<()>,
    }

    impl MultimodalCacheIndex for NotifyingCacheIndex {
        fn workers_with_cache_key_hits(&self, cache_keys: &[String]) -> Vec<(u64, usize)> {
            let _ = self.entered.send(());
            vec![(self.worker_id, cache_keys.len())]
        }

        fn remove_worker(&self, _worker_id: u64) {}
    }

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn device_aware_cache_preparation_does_not_wait_for_selection_gate() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("device-aware-cache-before-selection-gate".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate".to_string());
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();
        let (_workers_tx, workers) =
            watch::channel(HashMap::from([(worker_id, ModelRuntimeConfig::default())]));
        let state = RoutingLoadState::start(
            endpoint,
            16,
            workers,
            KvRouterConfig::default(),
            WORKER_TYPE_DECODE,
        )
        .await
        .unwrap();

        let (entered_tx, entered_rx) = mpsc::channel();
        let router = PushRouter::from_client_with_state(
            client,
            RouterMode::DeviceAwareWeighted,
            None,
            Some(Arc::new(NotifyingCacheIndex {
                worker_id,
                entered: entered_tx,
            })),
            Some(Arc::new(|_| vec!["image-key".to_string()])),
        )
        .await
        .unwrap();

        let selection_gate = state.selection_gate.lock();
        let (result_tx, result_rx) = mpsc::channel();
        let state_for_selection = Arc::clone(&state);
        let router_for_selection = router.clone();
        let runtime_handle = tokio::runtime::Handle::current();
        let selection = std::thread::spawn(move || {
            let _runtime = runtime_handle.enter();
            let reservation = state_for_selection.select_and_reserve(
                &router_for_selection,
                "request-1",
                &request(),
                None,
            );
            result_tx.send(reservation).unwrap();
        });

        let prepared_before_gate = entered_rx.recv_timeout(Duration::from_millis(250)).is_ok();
        drop(selection_gate);
        let reservation = result_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("selection must complete after the gate is released")
            .expect("selection must reserve the configured worker");
        selection.join().unwrap();
        drop(reservation);

        assert!(
            prepared_before_gate,
            "DeviceAware cache preparation must not wait for the selection gate"
        );

        runtime.shutdown();
    }
}
