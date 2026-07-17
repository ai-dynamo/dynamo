// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::protocols::{DpRank, WorkerId};
use dynamo_runtime::component::Component;
use dynamo_runtime::protocols::EndpointId;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::worker_query_directory::WorkerQueryEndpointDirectory;
use super::worker_query_state::RecoveryKey;
use crate::discovery::RuntimeConfigWatch;
use crate::kv_router::metrics::RouterWorkerStatusMetrics;
use crate::local_model::runtime_config::ModelRuntimeConfig;

const REGISTRATION_GRACE_PERIOD: Duration = Duration::from_secs(10);
const EXPECTED_WORKER_ABSENCE_HOLD_DOWN: Duration = Duration::from_secs(10);
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
    cleared_workers: Vec<WorkerId>,
    mismatch_worker_count: usize,
}

#[derive(Debug)]
struct MissingEndpoint {
    missing_since: Instant,
    unexpected_since: Option<Instant>,
}

#[derive(Debug)]
struct ReportedMismatch {
    missing_dp_ranks: Vec<DpRank>,
    reported_at: Instant,
}

#[derive(Debug, Default)]
struct KvEventSourceHealthMonitor {
    missing_endpoints: HashMap<RecoveryKey, MissingEndpoint>,
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
        for key in expected_endpoints.difference(discovered_endpoints) {
            self.missing_endpoints
                .entry(*key)
                .and_modify(|missing| missing.unexpected_since = None)
                .or_insert(MissingEndpoint {
                    missing_since: now,
                    unexpected_since: None,
                });
        }

        self.missing_endpoints.retain(|key, missing| {
            if discovered_endpoints.contains(key) {
                return false;
            }
            if expected_endpoints.contains(key) {
                missing.unexpected_since = None;
                return true;
            }

            let unexpected_since = missing.unexpected_since.get_or_insert(now);
            now.duration_since(*unexpected_since) < EXPECTED_WORKER_ABSENCE_HOLD_DOWN
        });

        let mut matured_by_worker: BTreeMap<WorkerId, Vec<DpRank>> = BTreeMap::new();
        for (key, missing) in &self.missing_endpoints {
            let worker_is_expected = expected_endpoints.contains(key);
            let rank_was_reported = self
                .reported
                .get(&key.0)
                .is_some_and(|reported| reported.missing_dp_ranks.contains(&key.1));
            if now.duration_since(missing.missing_since) >= REGISTRATION_GRACE_PERIOD
                && (worker_is_expected || rank_was_reported)
            {
                matured_by_worker.entry(key.0).or_default().push(key.1);
            }
        }
        for ranks in matured_by_worker.values_mut() {
            ranks.sort_unstable();
        }

        let cleared_workers: Vec<_> = self
            .reported
            .keys()
            .filter(|worker_id| !matured_by_worker.contains_key(worker_id))
            .copied()
            .collect();
        for worker_id in &cleared_workers {
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
            cleared_workers,
            mismatch_worker_count: matured_by_worker.len(),
        }
    }
}

// TODO: Centralize DP-rank expansion with the scheduler/topology code once a shared
// cross-crate helper exists.
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
    source_endpoint: EndpointId,
    mut workers_with_configs: RuntimeConfigWatch,
    query_endpoints: Arc<WorkerQueryEndpointDirectory>,
    model: String,
    worker_type: &'static str,
    cancellation_token: CancellationToken,
) {
    let metrics = RouterWorkerStatusMetrics::from_component(&component);
    let target_namespace = component.namespace().name().to_string();
    let target_component = component.name().to_string();
    let target_endpoint = source_endpoint.name;

    tokio::spawn(async move {
        let mut monitor = KvEventSourceHealthMonitor::default();
        let mut interval = tokio::time::interval(HEALTH_CHECK_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
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

            metrics.set_kv_event_source_mismatch_workers(
                &model,
                worker_type,
                &target_namespace,
                &target_component,
                &target_endpoint,
                evaluation.mismatch_worker_count,
            );

            for report in evaluation.reports {
                tracing::error!(
                    worker_id = report.worker_id,
                    missing_dp_ranks = ?report.missing_dp_ranks,
                    model,
                    worker_type,
                    target_namespace,
                    target_component,
                    target_endpoint,
                    "KV event routing degraded: expected worker-local KV indexer query endpoints were not discovered. Cache overlap will remain empty for the affected ranks and cache-aware routing will be ineffective. For Dynamo vLLM, verify that prefix caching is enabled (--no-enable-prefix-caching is not set) and that the worker has --kv-events-config '{{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:PORT\",\"enable_kv_cache_events\":true}}'; if both are configured, check router endpoint discovery health. To intentionally use approximate routing, set --no-router-kv-events. Continuing to serve.",
                );
            }

            for worker_id in evaluation.cleared_workers {
                tracing::info!(
                    worker_id,
                    model,
                    worker_type,
                    target_namespace,
                    target_component,
                    target_endpoint,
                    "KV event source mismatch no longer active; endpoints are registered or the worker is no longer expected"
                );
            }
        }
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
        assert_eq!(recovered.cleared_workers, vec![7]);
    }

    #[test]
    fn transient_worker_absence_during_grace_preserves_missing_timer() {
        let mut monitor = KvEventSourceHealthMonitor::default();
        let mut configs = HashMap::from([(7, worker_config(0, 1))]);
        let discovered = HashSet::new();
        let now = Instant::now();

        monitor.evaluate(now, &configs, &discovered);
        configs.clear();
        assert_eq!(
            monitor.evaluate(now + Duration::from_secs(5), &configs, &discovered),
            HealthEvaluation::default()
        );
        assert_eq!(
            monitor.evaluate(now + REGISTRATION_GRACE_PERIOD, &configs, &discovered),
            HealthEvaluation::default()
        );

        configs.insert(7, worker_config(0, 1));
        let matured = monitor.evaluate(
            now + REGISTRATION_GRACE_PERIOD + Duration::from_secs(1),
            &configs,
            &discovered,
        );
        assert_eq!(matured.mismatch_worker_count, 1);
        assert_eq!(matured.reports.len(), 1);
    }

    #[test]
    fn transient_worker_absence_keeps_reported_mismatch_active() {
        let mut monitor = KvEventSourceHealthMonitor::default();
        let mut configs = HashMap::from([(7, worker_config(0, 1))]);
        let discovered = HashSet::new();
        let now = Instant::now();

        monitor.evaluate(now, &configs, &discovered);
        monitor.evaluate(now + REGISTRATION_GRACE_PERIOD, &configs, &discovered);

        let absence_started = now + REGISTRATION_GRACE_PERIOD + Duration::from_secs(1);
        configs.clear();
        let absent = monitor.evaluate(absence_started, &configs, &discovered);
        assert_eq!(absent.mismatch_worker_count, 1);
        assert!(absent.cleared_workers.is_empty());

        configs.insert(7, worker_config(0, 1));
        let restored = monitor.evaluate(
            absence_started + EXPECTED_WORKER_ABSENCE_HOLD_DOWN - Duration::from_secs(1),
            &configs,
            &discovered,
        );
        assert_eq!(restored.mismatch_worker_count, 1);
        assert!(restored.reports.is_empty());
        assert!(restored.cleared_workers.is_empty());
    }

    #[test]
    fn sustained_worker_removal_clears_active_mismatch_after_hold_down() {
        let mut monitor = KvEventSourceHealthMonitor::default();
        let mut configs = HashMap::from([(7, worker_config(0, 1))]);
        let discovered = HashSet::new();
        let now = Instant::now();

        monitor.evaluate(now, &configs, &discovered);
        let matured = monitor.evaluate(now + REGISTRATION_GRACE_PERIOD, &configs, &discovered);
        assert_eq!(matured.mismatch_worker_count, 1);

        configs.clear();
        let absence_started = now + REGISTRATION_GRACE_PERIOD + Duration::from_secs(1);
        let held = monitor.evaluate(absence_started, &configs, &discovered);
        assert_eq!(held.mismatch_worker_count, 1);
        assert!(held.cleared_workers.is_empty());

        let cleared = monitor.evaluate(
            absence_started + EXPECTED_WORKER_ABSENCE_HOLD_DOWN,
            &configs,
            &discovered,
        );
        assert_eq!(cleared.mismatch_worker_count, 0);
        assert_eq!(cleared.cleared_workers, vec![7]);
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
