// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use dynamo_runtime::metrics::prometheus_names::{frontend_service, name_prefix};
use prometheus::{
    IntGaugeVec, Opts, Registry,
    core::{Collector, Desc},
    proto::{Metric, MetricFamily},
};

use super::metrics::{
    WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE, WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE,
    WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE,
};
use crate::discovery::ModelManager;
use crate::discovery::worker_inventory::{WorkerGroupObservation, WorkerGroupState};
use crate::kv_router::metrics::WORKER_LOAD_METRICS;

const GROUP_LABELS: &[&str] = &[
    "model",
    "target_namespace",
    "target_component",
    "target_endpoint",
    "worker_type",
];
const COUNT_STATES: &[&str] = &["discovered", "available", "pending", "excluded"];

type InventorySnapshot = Vec<(WorkerGroupObservation, HashSet<u64>)>;
type InventoryProvider = Arc<dyn Fn() -> InventorySnapshot + Send + Sync>;
type AllowedRanks = HashMap<(u64, String), HashSet<u32>>;

fn worker_last_metric_names() -> [String; 3] {
    [
        frontend_service::WORKER_LAST_TIME_TO_FIRST_TOKEN_SECONDS,
        frontend_service::WORKER_LAST_INPUT_SEQUENCE_TOKENS,
        frontend_service::WORKER_LAST_INTER_TOKEN_LATENCY_SECONDS,
    ]
    .map(|suffix| format!("{}_{suffix}", name_prefix::FRONTEND))
}

#[derive(Clone, Copy)]
struct WorkerState {
    state: &'static str,
    reason: &'static str,
}

impl WorkerState {
    fn priority(self) -> u8 {
        match self.state {
            "available" => 2,
            "pending" => 1,
            _ => 0,
        }
    }
}

#[derive(Default)]
struct PoolSnapshot {
    workers: HashMap<u64, WorkerState>,
}

/// Observability outlives a withdrawn router. Request writers keep only observed values;
/// discovery and committed serving state decide which worker series may be exported.
struct WorkerMetricsCollector {
    inventory: InventoryProvider,
    values: Vec<Box<dyn Collector>>,
    counts: IntGaugeVec,
    states: IntGaugeVec,
}

impl WorkerMetricsCollector {
    fn new(
        inventory: InventoryProvider,
        values: Vec<Box<dyn Collector>>,
    ) -> Result<Self, prometheus::Error> {
        let metric_opts =
            |suffix, help| Opts::new(format!("{}_{suffix}", name_prefix::FRONTEND), help);
        let mut count_labels = GROUP_LABELS.to_vec();
        count_labels.push("state");
        let mut state_labels = GROUP_LABELS.to_vec();
        state_labels.extend(["router_worker_id", "state", "reason"]);
        Ok(Self {
            inventory,
            values,
            counts: IntGaugeVec::new(
                metric_opts(
                    frontend_service::ROUTER_WORKERS,
                    "Number of discovered workers by current router state (not DP ranks)",
                ),
                &count_labels,
            )?,
            states: IntGaugeVec::new(
                metric_opts(
                    frontend_service::ROUTER_WORKER_STATE,
                    "Current router state of a discovered worker (1 = current state)",
                ),
                &state_labels,
            )?,
        })
    }

    fn worker_state(
        group: &WorkerGroupObservation,
        available: &HashSet<u64>,
        id: u64,
    ) -> WorkerState {
        if group.checksum_mismatches.contains(&id) {
            return WorkerState {
                state: "excluded",
                reason: "checksum_mismatch",
            };
        }
        if group.workers[&id].data_parallel_rank_range().is_err() {
            return WorkerState {
                state: "excluded",
                reason: "invalid_config",
            };
        }
        if group.committed.contains(&id) {
            return if available.contains(&id) {
                WorkerState {
                    state: "available",
                    reason: "none",
                }
            } else {
                WorkerState {
                    state: "excluded",
                    reason: "unavailable",
                }
            };
        }
        match group.state {
            WorkerGroupState::Pending => WorkerState {
                state: "pending",
                reason: "initializing",
            },
            WorkerGroupState::MaterializationFailed => WorkerState {
                state: "excluded",
                reason: "materialization_failed",
            },
            WorkerGroupState::CommitBlocked => WorkerState {
                state: "excluded",
                reason: "commit_blocked",
            },
            _ => WorkerState {
                state: "excluded",
                reason: "unavailable",
            },
        }
    }

    fn allowed_sample(metric: &Metric, allowed: &AllowedRanks, allow_unset_rank: bool) -> bool {
        let label = |name: &str| {
            metric
                .get_label()
                .iter()
                .find(|label| label.name() == name)
                .map(|label| label.value())
        };
        let Some(worker) = label("worker_id").and_then(|id| id.parse::<u64>().ok()) else {
            return false;
        };
        let (Some(worker_type), Some(rank)) = (label("worker_type"), label("dp_rank")) else {
            return false;
        };
        let Some(ranks) = allowed.get(&(worker, worker_type.to_string())) else {
            return false;
        };
        (allow_unset_rank && rank == "none")
            || rank.parse::<u32>().is_ok_and(|rank| ranks.contains(&rank))
    }
}

impl Collector for WorkerMetricsCollector {
    fn desc(&self) -> Vec<&Desc> {
        let mut desc = self.counts.desc();
        desc.extend(self.states.desc());
        for collector in &self.values {
            desc.extend(collector.desc());
        }
        desc
    }

    fn collect(&self) -> Vec<MetricFamily> {
        let mut pools: BTreeMap<[String; 5], PoolSnapshot> = BTreeMap::new();
        let mut allowed = AllowedRanks::new();
        for (group, available) in (self.inventory)() {
            let labels = [
                group.model.clone(),
                group.endpoint.namespace.clone(),
                group.endpoint.component.clone(),
                group.endpoint.name.clone(),
                group.worker_type.to_string(),
            ];
            let pool = pools.entry(labels).or_default();
            for &id in group.workers.keys() {
                let state = Self::worker_state(&group, &available, id);
                pool.workers
                    .entry(id)
                    .and_modify(|current| {
                        if state.priority() > current.priority() {
                            *current = state;
                        }
                    })
                    .or_insert(state);
                if state.state == "available" {
                    allowed
                        .entry((id, group.worker_type.to_string()))
                        .or_default()
                        .extend(
                            group.workers[&id]
                                .data_parallel_rank_range()
                                .expect("validated worker rank range"),
                        );
                }
            }
        }

        // Fresh vectors prevent shared reset/collect races between concurrent scrapes.
        let metrics = Self::new(Arc::clone(&self.inventory), Vec::new())
            .expect("validated worker metric descriptors");
        for (labels, pool) in pools {
            let base: Vec<_> = labels.iter().map(String::as_str).collect();
            for state in COUNT_STATES {
                let count = if *state == "discovered" {
                    pool.workers.len()
                } else {
                    pool.workers
                        .values()
                        .filter(|worker| worker.state == *state)
                        .count()
                };
                let mut values = base.clone();
                values.push(state);
                metrics.counts.with_label_values(&values).set(count as i64);
            }
            for (id, worker) in pool.workers {
                let id = id.to_string();
                let mut values = base.clone();
                values.extend([&id, worker.state, worker.reason]);
                metrics.states.with_label_values(&values).set(1);
            }
        }
        let mut result = metrics.counts.collect();
        result.extend(metrics.states.collect());
        let worker_last_metric_names = worker_last_metric_names();
        for collector in &self.values {
            for mut family in collector.collect() {
                let allow_unset_rank = worker_last_metric_names
                    .iter()
                    .any(|name| name == family.name());
                let retained: Vec<_> = family
                    .take_metric()
                    .into_iter()
                    .filter(|metric| Self::allowed_sample(metric, &allowed, allow_unset_rank))
                    .collect();
                if !retained.is_empty() {
                    family.set_metric(retained);
                    result.push(family);
                }
            }
        }
        result
    }
}

pub(super) fn register_worker_metrics(
    registry: &Registry,
    manager: Arc<ModelManager>,
) -> Result<(), prometheus::Error> {
    let inventory = Arc::new(move || {
        manager
            .worker_inventory()
            .into_iter()
            .map(|(key, group)| {
                let available = manager.worker_group_available_ids(&key, &group.endpoint);
                (group, available)
            })
            .collect()
    });
    let values: Vec<Box<dyn Collector>> = vec![
        Box::new(WORKER_LOAD_METRICS.active_decode_blocks.clone()),
        Box::new(WORKER_LOAD_METRICS.active_prefill_tokens.clone()),
        Box::new(WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE.clone()),
        Box::new(WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE.clone()),
        Box::new(WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE.clone()),
    ];
    registry.register(Box::new(WorkerMetricsCollector::new(inventory, values)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::worker_inventory::WorkerInventory;
    use crate::local_model::runtime_config::ModelRuntimeConfig;
    use dynamo_runtime::protocols::EndpointId;
    use parking_lot::Mutex;

    struct Fixture {
        registry: Registry,
        inventory: Arc<WorkerInventory>,
        available: Arc<Mutex<HashSet<u64>>>,
        values: Vec<IntGaugeVec>,
    }

    impl Fixture {
        fn new() -> Self {
            let inventory = Arc::new(WorkerInventory::default());
            let available = Arc::new(Mutex::new(HashSet::new()));
            let source = Arc::clone(&inventory);
            let availability = Arc::clone(&available);
            let provider = Arc::new(move || {
                source
                    .snapshot()
                    .into_iter()
                    .map(|(_, group)| (group, availability.lock().clone()))
                    .collect()
            });
            let values: Vec<_> = [
                frontend_service::WORKER_ACTIVE_DECODE_BLOCKS,
                frontend_service::WORKER_ACTIVE_PREFILL_TOKENS,
                frontend_service::WORKER_LAST_TIME_TO_FIRST_TOKEN_SECONDS,
                frontend_service::WORKER_LAST_INPUT_SEQUENCE_TOKENS,
                frontend_service::WORKER_LAST_INTER_TOKEN_LATENCY_SECONDS,
            ]
            .into_iter()
            .map(|suffix| {
                IntGaugeVec::new(
                    Opts::new(
                        format!("{}_{suffix}", name_prefix::FRONTEND),
                        "Observed value",
                    ),
                    &["worker_id", "dp_rank", "worker_type"],
                )
                .unwrap()
            })
            .collect();
            let registry = Registry::new();
            registry
                .register(Box::new(
                    WorkerMetricsCollector::new(
                        provider,
                        values
                            .iter()
                            .map(|value| Box::new(value.clone()) as Box<dyn Collector>)
                            .collect(),
                    )
                    .unwrap(),
                ))
                .unwrap();
            Self {
                registry,
                inventory,
                available,
                values,
            }
        }

        fn observe(&self, ids: &[u64], committed: &[u64], state: WorkerGroupState) {
            self.observe_with_rejections(ids, committed, &[], state);
        }

        fn observe_with_rejections(
            &self,
            ids: &[u64],
            committed: &[u64],
            checksum_mismatches: &[u64],
            state: WorkerGroupState,
        ) {
            self.inventory.publish(
                "group".into(),
                Some(WorkerGroupObservation {
                    model: "model".into(),
                    endpoint: EndpointId {
                        namespace: "ns".into(),
                        component: "decode".into(),
                        name: "generate".into(),
                    },
                    worker_type: "decode",
                    workers: ids
                        .iter()
                        .map(|&id| {
                            (
                                id,
                                ModelRuntimeConfig {
                                    data_parallel_size: 2,
                                    ..Default::default()
                                },
                            )
                        })
                        .collect(),
                    committed: committed.iter().copied().collect(),
                    checksum_mismatches: checksum_mismatches.iter().copied().collect(),
                    state,
                }),
            );
        }

        fn sample(&self, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
            self.registry
                .gather()
                .iter()
                .find(|family| family.name() == name)
                .and_then(|family| {
                    family
                        .get_metric()
                        .iter()
                        .find(|sample| {
                            labels.iter().all(|(name, value)| {
                                sample
                                    .get_label()
                                    .iter()
                                    .any(|label| label.name() == *name && label.value() == *value)
                            })
                        })
                        .map(|sample| sample.get_gauge().value())
                })
        }

        fn count(&self, state: &str) -> Option<f64> {
            self.sample("dynamo_frontend_router_workers", &[("state", state)])
        }
    }

    #[test]
    fn late_values_cannot_revive_removed_workers_or_invalid_ranks() {
        let f = Fixture::new();
        f.observe(&[1, 2], &[1, 2], WorkerGroupState::Ready);
        *f.available.lock() = HashSet::from([1, 2]);
        for gauge in &f.values {
            for labels in [
                ["1", "0", "decode"],
                ["1", "1", "decode"],
                ["1", "99", "decode"],
                ["1", "none", "decode"],
                ["2", "0", "decode"],
            ] {
                gauge.with_label_values(&labels).set(7);
            }
        }
        assert_eq!(f.count("available"), Some(2.0));
        for (index, gauge) in f.values.iter().enumerate() {
            let name = &gauge.desc()[0].fq_name;
            assert_eq!(
                f.sample(name, &[("worker_id", "1"), ("dp_rank", "0")]),
                Some(7.0)
            );
            assert_eq!(f.sample(name, &[("dp_rank", "99")]), None);
            assert_eq!(
                f.sample(name, &[("dp_rank", "none")]).is_some(),
                index >= 2, // The fixture's first two gauges are load metrics; the rest are timing.
            );
        }

        f.observe(&[2], &[2], WorkerGroupState::Ready);
        // Even an availability watcher that has not caught up cannot retain a removed ID.
        for gauge in &f.values {
            gauge.remove_label_values(&["1", "0", "decode"]).unwrap();
            gauge.with_label_values(&["1", "0", "decode"]).set(99);
        }
        for gauge in &f.values {
            let name = &gauge.desc()[0].fq_name;
            assert_eq!(f.sample(name, &[("worker_id", "1")]), None);
            assert_eq!(f.sample(name, &[("worker_id", "2")]), Some(7.0));
        }
        assert_eq!(f.count("discovered"), Some(1.0));
    }

    #[test]
    fn first_wins_metrics_preserve_local_incumbent_and_track_succession() {
        for (incumbent, rejected) in [(1, 2), (2, 1)] {
            let f = Fixture::new();
            let incumbent_id = incumbent.to_string();
            let rejected_id = rejected.to_string();
            *f.available.lock() = HashSet::from([1, 2]);
            f.observe_with_rejections(&[1, 2], &[], &[rejected], WorkerGroupState::Pending);
            assert_eq!(f.count("discovered"), Some(2.0));
            assert_eq!(f.count("available"), Some(0.0));
            assert_eq!(f.count("pending"), Some(1.0));
            assert_eq!(f.count("excluded"), Some(1.0));
            f.observe_with_rejections(&[1, 2], &[incumbent], &[rejected], WorkerGroupState::Ready);
            for gauge in &f.values {
                for id in [&incumbent_id, &rejected_id] {
                    gauge.with_label_values(&[id, "0", "decode"]).set(7);
                }
            }
            assert_eq!(f.count("discovered"), Some(2.0));
            assert_eq!(f.count("available"), Some(1.0));
            assert_eq!(f.count("pending"), Some(0.0));
            assert_eq!(f.count("excluded"), Some(1.0));
            assert_eq!(
                f.sample(
                    "dynamo_frontend_router_worker_state",
                    &[
                        ("router_worker_id", &incumbent_id),
                        ("state", "available"),
                        ("reason", "none")
                    ]
                ),
                Some(1.0)
            );
            assert_eq!(
                f.sample(
                    "dynamo_frontend_router_worker_state",
                    &[
                        ("router_worker_id", &rejected_id),
                        ("state", "excluded"),
                        ("reason", "checksum_mismatch")
                    ]
                ),
                Some(1.0)
            );
            for gauge in &f.values {
                let name = &gauge.desc()[0].fq_name;
                assert_eq!(f.sample(name, &[("worker_id", &incumbent_id)]), Some(7.0));
                assert_eq!(f.sample(name, &[("worker_id", &rejected_id)]), None);
            }

            f.observe(&[rejected], &[], WorkerGroupState::Pending);
            assert_eq!(f.count("pending"), Some(1.0));
            assert_eq!(f.count("excluded"), Some(0.0));
            assert_eq!(
                f.sample(
                    "dynamo_frontend_router_worker_state",
                    &[("reason", "checksum_mismatch")]
                ),
                None
            );
            assert_eq!(
                f.sample(
                    "dynamo_frontend_router_worker_state",
                    &[("router_worker_id", &incumbent_id)]
                ),
                None
            );
            f.observe(&[rejected], &[rejected], WorkerGroupState::Ready);
            assert_eq!(f.count("available"), Some(1.0));
            f.inventory.publish("group".into(), None);
            for state in COUNT_STATES {
                assert_eq!(f.count(state), Some(0.0));
            }
            assert_eq!(f.sample("dynamo_frontend_router_worker_state", &[]), None);
        }
    }

    #[test]
    fn pending_and_hard_unavailable_are_distinct_from_available() {
        let f = Fixture::new();
        assert_eq!(f.count("available"), None);
        f.observe(&[1], &[], WorkerGroupState::Pending);
        *f.available.lock() = HashSet::from([1]);
        assert_eq!(f.count("pending"), Some(1.0));
        assert_eq!(f.count("available"), Some(0.0));
        f.observe(&[1], &[1], WorkerGroupState::Ready);
        f.available.lock().clear();
        assert_eq!(f.count("excluded"), Some(1.0));
        assert_eq!(
            f.sample(
                "dynamo_frontend_router_worker_state",
                &[("state", "excluded"), ("reason", "unavailable")]
            ),
            Some(1.0)
        );
        f.available.lock().insert(1);
        assert_eq!(f.count("available"), Some(1.0));
    }
}
