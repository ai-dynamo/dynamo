// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::protocols::{DpRank, WorkerId};
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use tokio::time::Instant;

use super::worker_query_directory::WorkerQueryEndpointDirectory;
use super::worker_query_state::RecoveryKey;
use crate::discovery::RuntimeConfigWatch;
use crate::kv_router::metrics::RouterWorkerStatusMetrics;
use crate::local_model::runtime_config::ModelRuntimeConfig;

const REGISTRATION_GRACE_PERIOD: Duration = Duration::from_secs(10);
const ERROR_REPEAT_INTERVAL: Duration = Duration::from_secs(60);
const HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, PartialEq, Eq)]
struct MismatchReport {
    worker_id: WorkerId,
    missing_dp_ranks: Vec<DpRank>,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct HealthEvaluation {
    reports: Vec<MismatchReport>,
    recovered_workers: Vec<WorkerId>,
    mismatch_worker_count: usize,
}

#[derive(Debug)]
struct ReportedMismatch {
    missing_dp_ranks: Vec<DpRank>,
    reported_at: Instant,
}

#[derive(Debug, Default)]
struct KvEventSourceHealthMonitor {
    missing_since: HashMap<RecoveryKey, Instant>,
    reported: HashMap<WorkerId, ReportedMismatch>,
}

impl KvEventSourceHealthMonitor {
    fn evaluate(
        &mut self,
        now: Instant,
        workers_with_configs: &HashMap<WorkerId, ModelRuntimeConfig>,
        discovered_endpoints: &HashSet<RecoveryKey>,
    ) -> HealthEvaluation {
        let expected_endpoints = expected_query_endpoints(workers_with_configs);
        let missing_endpoints: HashSet<_> = expected_endpoints
            .difference(discovered_endpoints)
            .copied()
            .collect();

        self.missing_since
            .retain(|key, _| missing_endpoints.contains(key));
        for key in &missing_endpoints {
            self.missing_since.entry(*key).or_insert(now);
        }

        let mut matured_by_worker: BTreeMap<WorkerId, Vec<DpRank>> = BTreeMap::new();
        for (key, missing_since) in &self.missing_since {
            if now.duration_since(*missing_since) >= REGISTRATION_GRACE_PERIOD {
                matured_by_worker.entry(key.0).or_default().push(key.1);
            }
        }
        for ranks in matured_by_worker.values_mut() {
            ranks.sort_unstable();
        }

        let recovered_workers: Vec<_> = self
            .reported
            .keys()
            .filter(|worker_id| !matured_by_worker.contains_key(worker_id))
            .copied()
            .collect();
        for worker_id in &recovered_workers {
            self.reported.remove(worker_id);
        }

        let mut reports = Vec::new();
        for (worker_id, missing_dp_ranks) in &matured_by_worker {
            let should_report = self.reported.get(worker_id).is_none_or(|previous| {
                previous.missing_dp_ranks != *missing_dp_ranks
                    || now.duration_since(previous.reported_at) >= ERROR_REPEAT_INTERVAL
            });
            if !should_report {
                continue;
            }

            self.reported.insert(
                *worker_id,
                ReportedMismatch {
                    missing_dp_ranks: missing_dp_ranks.clone(),
                    reported_at: now,
                },
            );
            reports.push(MismatchReport {
                worker_id: *worker_id,
                missing_dp_ranks: missing_dp_ranks.clone(),
            });
        }

        HealthEvaluation {
            reports,
            recovered_workers,
            mismatch_worker_count: matured_by_worker.len(),
        }
    }
}

fn expected_query_endpoints(
    workers_with_configs: &HashMap<WorkerId, ModelRuntimeConfig>,
) -> HashSet<RecoveryKey> {
    workers_with_configs
        .iter()
        .filter(|(_, config)| config.enable_local_indexer)
        .flat_map(|(worker_id, config)| {
            (0..config.data_parallel_size).filter_map(move |offset| {
                config
                    .data_parallel_start_rank
                    .checked_add(offset)
                    .map(|dp_rank| (*worker_id, dp_rank))
            })
        })
        .collect()
}

pub(super) fn spawn_kv_event_source_health_monitor(
    component: Component,
    mut workers_with_configs: RuntimeConfigWatch,
    query_endpoints: Arc<WorkerQueryEndpointDirectory>,
) {
    let cancellation_token = component.drt().primary_token();
    let metrics = RouterWorkerStatusMetrics::from_component(&component);

    tokio::spawn(async move {
        let mut monitor = KvEventSourceHealthMonitor::default();
        let mut interval = tokio::time::interval(HEALTH_CHECK_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                _ = interval.tick() => {}
                result = workers_with_configs.changed() => {
                    if result.is_err() {
                        break;
                    }
                }
            }

            let evaluation = {
                let configs = workers_with_configs.borrow_and_update();
                let discovered_endpoints = query_endpoints.keys();
                monitor.evaluate(Instant::now(), &configs, &discovered_endpoints)
            };

            metrics.set_kv_event_source_mismatch_workers(evaluation.mismatch_worker_count);

            for report in evaluation.reports {
                tracing::error!(
                    worker_id = report.worker_id,
                    missing_dp_ranks = ?report.missing_dp_ranks,
                    "KV EVENT ROUTING MISCONFIGURATION: worker {} is missing worker-local KV indexer query endpoints for DP ranks {:?}. The router expects worker KV events, so cache overlap will remain empty for these ranks and cache-aware routing will be ineffective. For Dynamo vLLM this usually means --kv-events-config was omitted. Add --kv-events-config '{{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:PORT\",\"enable_kv_cache_events\":true}}' to the worker, or intentionally use --no-router-kv-events for approximate routing. Continuing to serve.",
                    report.worker_id,
                    report.missing_dp_ranks,
                );
            }

            for worker_id in evaluation.recovered_workers {
                tracing::info!(
                    worker_id,
                    "KV event source mismatch cleared: all expected worker-local KV indexer query endpoints are now registered"
                );
            }
        }

        metrics.set_kv_event_source_mismatch_workers(0);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker_config(data_parallel_start_rank: u32, data_parallel_size: u32) -> ModelRuntimeConfig {
        ModelRuntimeConfig {
            data_parallel_start_rank,
            data_parallel_size,
            enable_local_indexer: true,
            ..Default::default()
        }
    }

    #[test]
    fn missing_endpoint_matures_after_grace_and_repeats_once_per_minute() {
        let mut monitor = KvEventSourceHealthMonitor::default();
        let configs = HashMap::from([(7, worker_config(0, 1))]);
        let discovered = HashSet::new();
        let now = Instant::now();

        assert_eq!(
            monitor.evaluate(now, &configs, &discovered),
            HealthEvaluation::default()
        );

        let matured = monitor.evaluate(now + REGISTRATION_GRACE_PERIOD, &configs, &discovered);
        assert_eq!(matured.mismatch_worker_count, 1);
        assert_eq!(
            matured.reports,
            vec![MismatchReport {
                worker_id: 7,
                missing_dp_ranks: vec![0],
            }]
        );

        let quiet = monitor.evaluate(
            now + REGISTRATION_GRACE_PERIOD + Duration::from_secs(1),
            &configs,
            &discovered,
        );
        assert_eq!(quiet.mismatch_worker_count, 1);
        assert!(quiet.reports.is_empty());

        let repeated = monitor.evaluate(
            now + REGISTRATION_GRACE_PERIOD + ERROR_REPEAT_INTERVAL,
            &configs,
            &discovered,
        );
        assert_eq!(repeated.reports.len(), 1);
    }

    #[test]
    fn endpoint_registered_during_grace_never_reports() {
        let mut monitor = KvEventSourceHealthMonitor::default();
        let configs = HashMap::from([(7, worker_config(0, 1))]);
        let mut discovered = HashSet::new();
        let now = Instant::now();

        monitor.evaluate(now, &configs, &discovered);
        discovered.insert((7, 0));

        let evaluation = monitor.evaluate(
            now + REGISTRATION_GRACE_PERIOD + Duration::from_secs(1),
            &configs,
            &discovered,
        );
        assert_eq!(evaluation, HealthEvaluation::default());
    }

    #[test]
    fn partial_dp_registration_reports_missing_rank_then_recovers() {
        let mut monitor = KvEventSourceHealthMonitor::default();
        let configs = HashMap::from([(7, worker_config(3, 2))]);
        let mut discovered = HashSet::from([(7, 3)]);
        let now = Instant::now();

        monitor.evaluate(now, &configs, &discovered);
        let partial = monitor.evaluate(now + REGISTRATION_GRACE_PERIOD, &configs, &discovered);
        assert_eq!(
            partial.reports,
            vec![MismatchReport {
                worker_id: 7,
                missing_dp_ranks: vec![4],
            }]
        );

        discovered.insert((7, 4));
        let recovered = monitor.evaluate(
            now + REGISTRATION_GRACE_PERIOD + Duration::from_secs(1),
            &configs,
            &discovered,
        );
        assert_eq!(recovered.mismatch_worker_count, 0);
        assert_eq!(recovered.recovered_workers, vec![7]);
    }

    #[test]
    fn worker_removal_clears_active_mismatch() {
        let mut monitor = KvEventSourceHealthMonitor::default();
        let mut configs = HashMap::from([(7, worker_config(0, 1))]);
        let discovered = HashSet::new();
        let now = Instant::now();

        monitor.evaluate(now, &configs, &discovered);
        let matured = monitor.evaluate(now + REGISTRATION_GRACE_PERIOD, &configs, &discovered);
        assert_eq!(matured.mismatch_worker_count, 1);

        configs.clear();
        let recovered = monitor.evaluate(
            now + REGISTRATION_GRACE_PERIOD + Duration::from_secs(1),
            &configs,
            &discovered,
        );
        assert_eq!(recovered.mismatch_worker_count, 0);
        assert_eq!(recovered.recovered_workers, vec![7]);
    }

    #[test]
    fn workers_without_local_indexers_are_not_expected_to_register() {
        let mut monitor = KvEventSourceHealthMonitor::default();
        let mut config = worker_config(0, 1);
        config.enable_local_indexer = false;
        let configs = HashMap::from([(7, config)]);
        let discovered = HashSet::new();
        let now = Instant::now();

        monitor.evaluate(now, &configs, &discovered);
        let evaluation = monitor.evaluate(
            now + REGISTRATION_GRACE_PERIOD + Duration::from_secs(1),
            &configs,
            &discovered,
        );
        assert_eq!(evaluation, HealthEvaluation::default());
    }
}
