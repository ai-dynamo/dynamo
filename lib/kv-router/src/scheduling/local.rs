// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use rustc_hash::{FxHashMap, FxHashSet};
use tokio::sync::watch;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::policy::{RouterSchedulingPolicy, SchedulingPolicy};
use super::prefill_load::PrefillLoadEstimator;
use super::queue::SchedulerQueue;
use super::selector::{DefaultWorkerSelector, WorkerSelector};
use super::types::{
    KvSchedulerError, PotentialLoad, SchedulingRequest, SchedulingResponse, TierOverlapBlocks,
};
use crate::protocols::{OverlapScores, WorkerConfigLike, WorkerId, WorkerWithDpRank};
use crate::sequences::{
    ActiveSequencesMultiWorker, SequenceError, SequencePublisher, SequenceRequest,
};
use dynamo_tokens::SequenceHash;

pub struct LocalScheduler<P, C, S = RouterSchedulingPolicy, Sel = DefaultWorkerSelector>
where
    P: SequencePublisher,
    C: WorkerConfigLike,
    S: SchedulingPolicy,
    Sel: WorkerSelector<C>,
{
    slots: Arc<ActiveSequencesMultiWorker<P>>,
    queue: Arc<SchedulerQueue<P, C, S, Sel>>,
    queue_updates: watch::Sender<()>,
    track_prefill_tokens_default: bool,
    worker_type: &'static str,
}

impl<P, C, S, Sel> LocalScheduler<P, C, S, Sel>
where
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Clone + PartialEq + Send + Sync + 'static,
    S: SchedulingPolicy + 'static,
    Sel: WorkerSelector<C> + Send + Sync + 'static,
{
    fn worker_dp_range(workers: &HashMap<WorkerId, C>) -> HashMap<WorkerId, (u32, u32)> {
        workers
            .iter()
            .map(|(&id, cfg)| {
                (
                    id,
                    (cfg.data_parallel_start_rank(), cfg.data_parallel_size()),
                )
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        threshold_frac: Option<f64>,
        block_size: u32,
        selector: Sel,
        policy: S,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        recheck_interval: Duration,
        track_prefill_tokens_default: bool,
        cancellation_token: CancellationToken,
        worker_type: &'static str,
        monitor_worker_configs: bool,
    ) -> Self {
        if monitor_worker_configs {
            let slots_monitor = Arc::clone(&slots);
            let mut monitor_rx = workers_with_configs.clone();
            let mut last_workers = monitor_rx.borrow().clone();
            let monitor_cancel_token = cancellation_token.clone();
            tokio::spawn(async move {
                tracing::trace!("LocalScheduler workers monitoring task started");

                loop {
                    tokio::select! {
                        _ = monitor_cancel_token.cancelled() => {
                            tracing::trace!("LocalScheduler workers monitoring task shutting down");
                            break;
                        }
                        result = monitor_rx.changed() => {
                            if result.is_err() {
                                tracing::warn!("LocalScheduler worker config watch dropped, shutting down");
                                break;
                            }
                        }
                    }

                    let current_workers = monitor_rx.borrow_and_update().clone();
                    if current_workers == last_workers {
                        continue;
                    }

                    let dp_range = Self::worker_dp_range(&current_workers);
                    slots_monitor.update_workers(&dp_range);
                    last_workers = current_workers;
                }
            });
        }

        let queue = Arc::new(SchedulerQueue::new(
            Arc::clone(&slots),
            workers_with_configs,
            threshold_frac,
            block_size,
            selector,
            policy,
            prefill_load_estimator,
        ));
        let (queue_updates, _) = watch::channel(());
        let queue_remote_updates = Arc::clone(&queue);
        let queue_periodic_updates = Arc::clone(&queue);
        let mut remote_state_updates = slots.subscribe_remote_state_changes();
        let remote_update_cancel_token = cancellation_token.clone();
        let queue_updates_remote = queue_updates.clone();

        tokio::spawn(async move {
            tracing::trace!("LocalScheduler remote state listener started");

            loop {
                tokio::select! {
                    _ = remote_update_cancel_token.cancelled() => {
                        tracing::trace!("LocalScheduler remote state listener shutting down");
                        break;
                    }
                    result = remote_state_updates.changed() => {
                        if result.is_err() {
                            tracing::trace!("LocalScheduler remote state listener shutting down");
                            break;
                        }
                        queue_remote_updates.update().await;
                        let _ = queue_updates_remote.send(());
                    }
                }
            }
        });

        tokio::spawn(async move {
            let mut recheck_interval = tokio::time::interval(recheck_interval);
            tracing::trace!("LocalScheduler periodic queue update task started");

            loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        tracing::trace!("LocalScheduler periodic queue update task shutting down");
                        break;
                    }
                    _ = recheck_interval.tick() => {
                        queue_periodic_updates.update().await;
                    }
                }
            }
        });

        Self {
            slots,
            queue,
            queue_updates,
            track_prefill_tokens_default,
            worker_type,
        }
    }

    #[expect(clippy::too_many_arguments)]
    pub async fn schedule(
        &self,
        maybe_request_id: Option<String>,
        isl_tokens: usize,
        token_seq: Option<Vec<SequenceHash>>,
        overlaps: OverlapScores,
        tier_overlap_blocks: TierOverlapBlocks,
        effective_overlap_blocks: HashMap<WorkerWithDpRank, f64>,
        effective_cached_tokens: HashMap<WorkerWithDpRank, usize>,
        router_config_override: Option<&super::config::RouterConfigOverride>,
        update_states: bool,
        lora_name: Option<String>,
        priority_jump: f64,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
    ) -> Result<SchedulingResponse, KvSchedulerError> {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let track_prefill_tokens = router_config_override
            .and_then(|cfg| cfg.track_prefill_tokens)
            .unwrap_or(self.track_prefill_tokens_default);
        let request = SchedulingRequest {
            maybe_request_id,
            token_seq,
            isl_tokens,
            overlaps,
            tier_overlap_blocks,
            effective_overlap_blocks,
            effective_cached_tokens,
            decode_blocks: FxHashMap::default(),
            prefill_tokens: FxHashMap::default(),
            track_prefill_tokens,
            router_config_override: router_config_override.cloned(),
            update_states,
            lora_name,
            priority_jump,
            expected_output_tokens,
            pinned_worker,
            allowed_worker_ids,
            resp_tx: Some(resp_tx),
        };

        self.queue.enqueue(request).await;

        resp_rx
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?
    }

    pub fn register_workers(&self, worker_ids: &HashSet<WorkerId>) {
        self.queue.register_workers(worker_ids);
    }

    pub async fn add_request(&self, req: SequenceRequest) -> Result<(), SequenceError> {
        self.slots.add_request(req, Instant::now())
    }

    pub async fn mark_prefill_completed(&self, request_id: &str) -> Result<(), SequenceError> {
        self.slots
            .mark_prefill_completed(&request_id.to_string(), Instant::now())?;
        self.queue.update().await;
        Ok(())
    }

    pub async fn free(&self, request_id: &str) -> Result<(), SequenceError> {
        self.slots.free(&request_id.to_string(), Instant::now())?;
        self.queue.update().await;
        Ok(())
    }

    pub fn pending_count(&self) -> usize {
        self.queue.pending_count()
    }

    pub fn pending_isl_tokens(&self) -> usize {
        self.queue.pending_isl_tokens()
    }

    pub fn worker_type(&self) -> &'static str {
        self.worker_type
    }

    pub fn subscribe_queue_updates(&self) -> watch::Receiver<()> {
        self.queue_updates.subscribe()
    }

    pub fn add_output_block(
        &self,
        request_id: &str,
        decay_fraction: Option<f64>,
    ) -> Result<(), SequenceError> {
        self.slots
            .add_output_block(&request_id.to_string(), decay_fraction)
    }

    pub fn get_potential_loads(
        &self,
        token_seq: Option<Vec<SequenceHash>>,
        isl_tokens: usize,
        _overlaps: OverlapScores,
        effective_cached_tokens: HashMap<WorkerWithDpRank, usize>,
        track_prefill_tokens: bool,
    ) -> Vec<PotentialLoad> {
        let decay_now = Instant::now();
        let (decode_blocks, prefill_tokens) = self
            .slots
            .potential_blocks_and_tokens_with_prefill_tracking(
                token_seq.as_deref(),
                isl_tokens,
                effective_cached_tokens,
                track_prefill_tokens,
                decay_now,
            );

        let mut workers: FxHashSet<WorkerWithDpRank> = FxHashSet::default();
        workers.extend(decode_blocks.keys().copied());
        workers.extend(prefill_tokens.keys().copied());

        let mut loads = Vec::with_capacity(workers.len());
        for worker in workers {
            loads.push(PotentialLoad {
                worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                potential_prefill_tokens: prefill_tokens
                    .get(&worker)
                    .copied()
                    .unwrap_or(isl_tokens),
                potential_decode_blocks: decode_blocks.get(&worker).copied().unwrap_or(0),
            });
        }

        loads
    }

    pub fn get_active_lora_counts(&self) -> HashMap<String, usize> {
        self.slots.get_active_lora_counts()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::sync::{mpsc, watch};

    use super::*;
    use crate::protocols::{ActiveSequenceEvent, ActiveSequenceEventData, OverlapScores};
    use crate::scheduling::PrefillLoadEstimator;
    use crate::scheduling::policy::FcfsPolicy;
    use crate::scheduling::selector::DefaultWorkerSelector;
    use crate::sequences::SequenceSubscriber;
    use crate::test_utils::{NoopSequencePublisher, SimpleWorkerConfig};

    struct TestSequenceSubscriber {
        rx: mpsc::UnboundedReceiver<ActiveSequenceEvent>,
    }

    impl SequenceSubscriber for TestSequenceSubscriber {
        async fn next_event(&mut self) -> Option<anyhow::Result<ActiveSequenceEvent>> {
            self.rx.recv().await.map(Ok)
        }
    }

    struct FixedPrefillLoadEstimator {
        duration: Duration,
    }

    impl PrefillLoadEstimator for FixedPrefillLoadEstimator {
        fn predict_prefill_duration(
            &self,
            _batch_size: usize,
            _effective_isl: usize,
            _prefix: usize,
        ) -> anyhow::Result<Duration> {
            Ok(self.duration)
        }
    }

    #[allow(clippy::type_complexity)]
    fn make_scheduler(
        workers: HashMap<WorkerId, SimpleWorkerConfig>,
        threshold_frac: Option<f64>,
        monitor_worker_configs: bool,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    ) -> (
        Arc<LocalScheduler<NoopSequencePublisher, SimpleWorkerConfig, FcfsPolicy>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
        watch::Sender<HashMap<WorkerId, SimpleWorkerConfig>>,
        CancellationToken,
    ) {
        let dp_range = workers
            .iter()
            .map(|(&id, cfg)| (id, (cfg.data_parallel_start_rank, cfg.data_parallel_size)))
            .collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            64,
            dp_range,
            false,
            0,
            "test",
        ));
        let (cfg_tx, cfg_rx) = watch::channel(workers);
        let cancel_token = CancellationToken::new();
        let scheduler = Arc::new(LocalScheduler::new(
            Arc::clone(&slots),
            cfg_rx,
            threshold_frac,
            64,
            DefaultWorkerSelector::new(None, "test"),
            FcfsPolicy,
            prefill_load_estimator,
            Duration::from_secs(60),
            true,
            cancel_token.clone(),
            "test",
            monitor_worker_configs,
        ));
        (scheduler, slots, cfg_tx, cancel_token)
    }

    fn start_replica_sync(
        slots: &Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
        cancel_token: &CancellationToken,
    ) -> mpsc::UnboundedSender<ActiveSequenceEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        slots.start_replica_sync(TestSequenceSubscriber { rx }, cancel_token.clone());
        tx
    }

    async fn wait_for_pending_count(
        scheduler: &Arc<LocalScheduler<NoopSequencePublisher, SimpleWorkerConfig, FcfsPolicy>>,
        expected: usize,
    ) {
        tokio::time::timeout(Duration::from_millis(250), async {
            loop {
                if scheduler.pending_count() == expected {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_schedule_books_request_into_active_sequences() {
        let mut workers = HashMap::new();
        workers.insert(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(64),
                ..Default::default()
            },
        );
        let (scheduler, _slots, _cfg_tx, cancel_token) = make_scheduler(workers, None, true, None);

        let response = scheduler
            .schedule(
                Some("req-1".to_string()),
                64,
                Some(vec![1, 2, 3, 4]),
                OverlapScores::default(),
                TierOverlapBlocks::default(),
                HashMap::new(),
                HashMap::new(),
                None,
                true,
                Some("adapter-a".to_string()),
                0.0,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        assert_eq!(response.best_worker.worker_id, 0);
        assert_eq!(
            scheduler.get_active_lora_counts(),
            HashMap::from([(String::from("adapter-a"), 1)])
        );

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_schedule_override_can_disable_prefill_tracking() {
        let mut workers = HashMap::new();
        workers.insert(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(64),
                ..Default::default()
            },
        );
        let (scheduler, slots, _cfg_tx, cancel_token) = make_scheduler(workers, None, true, None);

        scheduler
            .schedule(
                Some("req-1".to_string()),
                64,
                Some(vec![1, 2, 3, 4]),
                OverlapScores::default(),
                TierOverlapBlocks::default(),
                HashMap::new(),
                HashMap::new(),
                Some(&crate::config::RouterConfigOverride {
                    track_prefill_tokens: Some(false),
                    ..Default::default()
                }),
                true,
                None,
                0.0,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            slots
                .active_tokens(Instant::now())
                .get(&WorkerWithDpRank::new(0, 0))
                .copied(),
            Some(0)
        );

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_schedule_uses_weighted_cached_tokens_for_active_tracking() {
        let mut workers = HashMap::new();
        workers.insert(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(64),
                ..Default::default()
            },
        );
        let (scheduler, slots, _cfg_tx, cancel_token) = make_scheduler(workers, None, true, None);
        let worker = WorkerWithDpRank::new(0, 0);

        let response = scheduler
            .schedule(
                Some("req-1".to_string()),
                64,
                Some(vec![1, 2, 3, 4]),
                OverlapScores::default(),
                TierOverlapBlocks::default(),
                HashMap::from([(worker, 0.75)]),
                HashMap::from([(worker, 48)]),
                None,
                true,
                None,
                0.0,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        assert_eq!(response.best_worker, worker);
        assert_eq!(response.cached_tokens, 48);
        assert_eq!(response.overlap_blocks, 1);
        assert_eq!(
            slots.active_tokens(Instant::now()).get(&worker).copied(),
            Some(16),
            "weighted cached tokens should reduce tracked prefill load",
        );

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_mark_prefill_completed_drains_pending_queue() {
        let mut workers = HashMap::new();
        workers.insert(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(64),
                ..Default::default()
            },
        );
        let (scheduler, _slots, _cfg_tx, cancel_token) =
            make_scheduler(workers, Some(0.5), true, None);

        scheduler
            .schedule(
                Some("req-1".to_string()),
                64,
                Some(vec![1, 2, 3, 4]),
                OverlapScores::default(),
                TierOverlapBlocks::default(),
                HashMap::new(),
                HashMap::new(),
                None,
                true,
                None,
                0.0,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        let queued = {
            let scheduler = Arc::clone(&scheduler);
            tokio::spawn(async move {
                scheduler
                    .schedule(
                        Some("req-2".to_string()),
                        64,
                        Some(vec![5, 6, 7, 8]),
                        OverlapScores::default(),
                        TierOverlapBlocks::default(),
                        HashMap::new(),
                        HashMap::new(),
                        None,
                        true,
                        None,
                        0.0,
                        None,
                        None,
                        None,
                    )
                    .await
            })
        };

        wait_for_pending_count(&scheduler, 1).await;

        scheduler.mark_prefill_completed("req-1").await.unwrap();
        queued.await.unwrap().unwrap();
        assert_eq!(scheduler.pending_count(), 0);

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_remote_mark_prefill_completed_drains_pending_queue() {
        let mut workers = HashMap::new();
        workers.insert(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(64),
                ..Default::default()
            },
        );
        let (scheduler, slots, _cfg_tx, cancel_token) =
            make_scheduler(workers, Some(0.5), true, None);
        let event_tx = start_replica_sync(&slots, &cancel_token);

        scheduler
            .schedule(
                Some("req-1".to_string()),
                64,
                Some(vec![1, 2, 3, 4]),
                OverlapScores::default(),
                TierOverlapBlocks::default(),
                HashMap::new(),
                HashMap::new(),
                None,
                true,
                None,
                0.0,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        let queued = {
            let scheduler = Arc::clone(&scheduler);
            tokio::spawn(async move {
                scheduler
                    .schedule(
                        Some("req-2".to_string()),
                        64,
                        Some(vec![5, 6, 7, 8]),
                        OverlapScores::default(),
                        TierOverlapBlocks::default(),
                        HashMap::new(),
                        HashMap::new(),
                        None,
                        true,
                        None,
                        0.0,
                        None,
                        None,
                        None,
                    )
                    .await
            })
        };

        wait_for_pending_count(&scheduler, 1).await;

        event_tx
            .send(ActiveSequenceEvent {
                request_id: "req-1".to_string(),
                worker: WorkerWithDpRank::new(0, 0),
                data: ActiveSequenceEventData::MarkPrefillCompleted,
                router_id: 1,
                lora_name: None,
            })
            .unwrap();

        tokio::time::timeout(Duration::from_millis(250), async {
            queued.await.unwrap().unwrap();
        })
        .await
        .unwrap();
        assert_eq!(scheduler.pending_count(), 0);

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_remote_queue_update_notification_fires_after_drain() {
        let mut workers = HashMap::new();
        workers.insert(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(64),
                ..Default::default()
            },
        );
        let (scheduler, slots, _cfg_tx, cancel_token) =
            make_scheduler(workers, Some(0.5), true, None);
        let event_tx = start_replica_sync(&slots, &cancel_token);
        let mut queue_updates = scheduler.subscribe_queue_updates();

        scheduler
            .schedule(
                Some("req-1".to_string()),
                64,
                Some(vec![1, 2, 3, 4]),
                OverlapScores::default(),
                TierOverlapBlocks::default(),
                HashMap::new(),
                HashMap::new(),
                None,
                true,
                None,
                0.0,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        let queued = {
            let scheduler = Arc::clone(&scheduler);
            tokio::spawn(async move {
                scheduler
                    .schedule(
                        Some("req-2".to_string()),
                        64,
                        Some(vec![5, 6, 7, 8]),
                        OverlapScores::default(),
                        TierOverlapBlocks::default(),
                        HashMap::new(),
                        HashMap::new(),
                        None,
                        true,
                        None,
                        0.0,
                        None,
                        None,
                        None,
                    )
                    .await
            })
        };

        wait_for_pending_count(&scheduler, 1).await;

        event_tx
            .send(ActiveSequenceEvent {
                request_id: "req-1".to_string(),
                worker: WorkerWithDpRank::new(0, 0),
                data: ActiveSequenceEventData::Free,
                router_id: 1,
                lora_name: None,
            })
            .unwrap();

        tokio::time::timeout(Duration::from_millis(250), queue_updates.changed())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(scheduler.pending_count(), 0);
        queued.await.unwrap().unwrap();

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_remote_free_drains_pending_queue() {
        let mut workers = HashMap::new();
        workers.insert(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(64),
                ..Default::default()
            },
        );
        let (scheduler, slots, _cfg_tx, cancel_token) =
            make_scheduler(workers, Some(0.5), true, None);
        let event_tx = start_replica_sync(&slots, &cancel_token);

        scheduler
            .schedule(
                Some("req-1".to_string()),
                64,
                Some(vec![1, 2, 3, 4]),
                OverlapScores::default(),
                TierOverlapBlocks::default(),
                HashMap::new(),
                HashMap::new(),
                None,
                true,
                None,
                0.0,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        let queued = {
            let scheduler = Arc::clone(&scheduler);
            tokio::spawn(async move {
                scheduler
                    .schedule(
                        Some("req-2".to_string()),
                        64,
                        Some(vec![5, 6, 7, 8]),
                        OverlapScores::default(),
                        TierOverlapBlocks::default(),
                        HashMap::new(),
                        HashMap::new(),
                        None,
                        true,
                        None,
                        0.0,
                        None,
                        None,
                        None,
                    )
                    .await
            })
        };

        wait_for_pending_count(&scheduler, 1).await;

        event_tx
            .send(ActiveSequenceEvent {
                request_id: "req-1".to_string(),
                worker: WorkerWithDpRank::new(0, 0),
                data: ActiveSequenceEventData::Free,
                router_id: 1,
                lora_name: None,
            })
            .unwrap();

        tokio::time::timeout(Duration::from_millis(250), async {
            queued.await.unwrap().unwrap();
        })
        .await
        .unwrap();
        assert_eq!(scheduler.pending_count(), 0);

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_free_updates_active_state() {
        let mut workers = HashMap::new();
        workers.insert(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(64),
                ..Default::default()
            },
        );
        let (scheduler, _slots, _cfg_tx, cancel_token) = make_scheduler(workers, None, true, None);

        scheduler
            .schedule(
                Some("req-1".to_string()),
                64,
                Some(vec![1, 2, 3, 4]),
                OverlapScores::default(),
                TierOverlapBlocks::default(),
                HashMap::new(),
                HashMap::new(),
                None,
                true,
                Some("adapter-a".to_string()),
                0.0,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            scheduler.get_active_lora_counts(),
            HashMap::from([(String::from("adapter-a"), 1)])
        );

        scheduler.free("req-1").await.unwrap();
        assert!(scheduler.get_active_lora_counts().is_empty());

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_get_potential_loads_matches_slots() {
        let mut workers = HashMap::new();
        workers.insert(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(256),
                ..Default::default()
            },
        );
        workers.insert(
            1,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(256),
                ..Default::default()
            },
        );
        let (scheduler, slots, _cfg_tx, cancel_token) = make_scheduler(workers, None, true, None);
        let token_seq = vec![11, 22, 33, 44];
        let overlaps = OverlapScores::default();
        let cached_tokens = HashMap::new();

        let (decode_blocks, prefill_tokens) =
            slots.potential_blocks_and_tokens(Some(&token_seq), 128, cached_tokens.clone());
        let mut expected: Vec<_> = decode_blocks
            .keys()
            .map(|worker| PotentialLoad {
                worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                potential_prefill_tokens: prefill_tokens.get(worker).copied().unwrap_or(128),
                potential_decode_blocks: decode_blocks.get(worker).copied().unwrap_or(0),
            })
            .collect();
        expected.sort_by_key(|load| (load.worker_id, load.dp_rank));

        let mut actual =
            scheduler.get_potential_loads(Some(token_seq), 128, overlaps, cached_tokens, true);
        actual.sort_by_key(|load| (load.worker_id, load.dp_rank));

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual.worker_id, expected.worker_id);
            assert_eq!(actual.dp_rank, expected.dp_rank);
            assert_eq!(
                actual.potential_prefill_tokens,
                expected.potential_prefill_tokens
            );
            assert_eq!(
                actual.potential_decode_blocks,
                expected.potential_decode_blocks
            );
        }

        cancel_token.cancel();
    }

    #[tokio::test(start_paused = true)]
    async fn test_get_potential_loads_uses_decayed_prefill_tokens() {
        let mut workers = HashMap::new();
        workers.insert(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(256),
                ..Default::default()
            },
        );
        let estimator: Arc<dyn PrefillLoadEstimator> = Arc::new(FixedPrefillLoadEstimator {
            duration: Duration::from_secs(10),
        });
        let (scheduler, _slots, _cfg_tx, cancel_token) =
            make_scheduler(workers, None, true, Some(estimator));

        scheduler
            .schedule(
                Some("req-1".to_string()),
                100,
                Some(vec![1, 2, 3, 4]),
                OverlapScores::default(),
                TierOverlapBlocks::default(),
                HashMap::new(),
                HashMap::new(),
                None,
                true,
                None,
                0.0,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        tokio::time::advance(Duration::from_secs(6)).await;

        let loads =
            scheduler.get_potential_loads(None, 0, OverlapScores::default(), HashMap::new(), true);
        assert_eq!(loads.len(), 1);
        assert_eq!(loads[0].potential_prefill_tokens, 40);

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_register_workers_uses_default_dp_fallback() {
        let (scheduler, _slots, _cfg_tx, cancel_token) =
            make_scheduler(HashMap::new(), None, false, None);

        scheduler.register_workers(&HashSet::from([42]));
        let loads =
            scheduler.get_potential_loads(None, 64, OverlapScores::default(), HashMap::new(), true);

        assert_eq!(loads.len(), 1);
        assert_eq!(loads[0].worker_id, 42);
        assert_eq!(loads[0].dp_rank, 0);

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_worker_watch_updates_slot_ranges() {
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());
        let (scheduler, _slots, cfg_tx, cancel_token) = make_scheduler(workers, None, true, None);

        assert_eq!(
            scheduler
                .get_potential_loads(None, 64, OverlapScores::default(), HashMap::new(), true,)
                .len(),
            1
        );

        let mut updated_workers = HashMap::new();
        updated_workers.insert(
            0,
            SimpleWorkerConfig {
                data_parallel_size: 2,
                ..Default::default()
            },
        );
        updated_workers.insert(1, SimpleWorkerConfig::default());
        cfg_tx.send(updated_workers).unwrap();

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if scheduler
                    .get_potential_loads(None, 64, OverlapScores::default(), HashMap::new(), true)
                    .len()
                    == 3
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_get_potential_loads_can_ignore_prefill_tokens() {
        let mut workers = HashMap::new();
        workers.insert(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(256),
                ..Default::default()
            },
        );
        let (scheduler, _slots, _cfg_tx, cancel_token) = make_scheduler(workers, None, true, None);

        scheduler
            .schedule(
                Some("req-1".to_string()),
                64,
                Some(vec![11, 22]),
                OverlapScores::default(),
                TierOverlapBlocks::default(),
                HashMap::new(),
                HashMap::new(),
                None,
                true,
                None,
                0.0,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        let loads = scheduler.get_potential_loads(
            None,
            64,
            OverlapScores::default(),
            HashMap::new(),
            false,
        );
        assert_eq!(loads.len(), 1);
        assert_eq!(loads[0].potential_prefill_tokens, 64);

        cancel_token.cancel();
    }
}
