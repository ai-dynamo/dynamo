// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Weak};

use dynamo_runtime::metrics::PrometheusMetric;
use parking_lot::Mutex;
use prometheus::{
    IntGaugeVec, Opts,
    core::{Collector, Desc},
    proto::MetricFamily,
};
use tokio_util::sync::CancellationToken;

use super::{ROUTER_WORKER_ID_LABEL, labels};
use crate::discovery::RuntimeConfigWatch;
use dynamo_kv_router::scheduling::WorkerAvailabilityProvider;

const LABELS: &[&str] = &[ROUTER_WORKER_ID_LABEL, labels::DP_RANK, labels::WORKER_TYPE];

/// Owned by a scheduler, independently of its request/replica slot state.
pub(crate) struct RouterWorkerRegistration {
    workers: RuntimeConfigWatch,
    worker_type: &'static str,
    cancellation: CancellationToken,
    available_workers: Option<WorkerAvailabilityProvider>,
}

/// Collect from discovery at scrape time so late bookings cannot resurrect removed workers.
#[derive(Clone)]
pub(super) struct RouterWorkerRegistered {
    opts: Opts,
    descriptor: IntGaugeVec,
    sources: Arc<Mutex<Vec<Weak<RouterWorkerRegistration>>>>,
}

impl PrometheusMetric for RouterWorkerRegistered {
    fn with_opts(opts: Opts) -> Result<Self, prometheus::Error> {
        Ok(Self {
            descriptor: IntGaugeVec::new(opts.clone(), LABELS)?,
            opts,
            sources: Arc::new(Mutex::new(Vec::new())),
        })
    }
}

impl RouterWorkerRegistered {
    pub(super) fn watch(
        &self,
        workers: RuntimeConfigWatch,
        worker_type: &'static str,
        cancellation: CancellationToken,
        available_workers: Option<WorkerAvailabilityProvider>,
    ) -> Arc<RouterWorkerRegistration> {
        let source = Arc::new(RouterWorkerRegistration {
            workers,
            worker_type,
            cancellation,
            available_workers,
        });
        let mut sources = self.sources.lock();
        sources.retain(|source| source.strong_count() > 0);
        sources.push(Arc::downgrade(&source));
        source
    }
}

impl Collector for RouterWorkerRegistered {
    fn desc(&self) -> Vec<&Desc> {
        self.descriptor.desc()
    }

    fn collect(&self) -> Vec<MetricFamily> {
        // A fresh vector makes concurrent scrapes independent and unions duplicate labels
        // across scheduler owners without deleting another live owner's registration.
        let gauge = IntGaugeVec::new(self.opts.clone(), LABELS)
            .expect("validated router worker metric options");
        let mut sources = self.sources.lock();
        sources.retain(|source| {
            let Some(source) = source.upgrade() else {
                return false;
            };
            if source.cancellation.is_cancelled() || source.workers.has_changed().is_err() {
                return false;
            }
            // Discovery can retain workers rejected by MCD admission or local fault
            // inhibition. Use the same hard-availability source as selection, not overload.
            let available = source.available_workers.as_ref().map(|provider| provider());
            for (worker_id, config) in source.workers.borrow().iter() {
                if let Some(available) = &available
                    && available
                        .as_ref()
                        .is_none_or(|ids| !ids.contains(worker_id))
                {
                    continue;
                }
                let Ok(ranks) = config.data_parallel_rank_range() else {
                    continue;
                };
                let worker_id = worker_id.to_string();
                for rank in ranks {
                    gauge
                        .with_label_values(&[&worker_id, &rank.to_string(), source.worker_type])
                        .set(1);
                }
            }
            true
        });
        gauge.collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_model::runtime_config::ModelRuntimeConfig;
    use prometheus::{Registry, TextEncoder};
    use std::collections::HashMap;
    use tokio::sync::watch;

    fn setup() -> (RouterWorkerRegistered, Registry) {
        let metric = RouterWorkerRegistered::with_opts(
            Opts::new(
                "dynamo_component_router_worker_registered",
                "Registered workers",
            )
            .const_label("dynamo_component", "frontend"),
        )
        .unwrap();
        let registry = Registry::new();
        registry.register(Box::new(metric.clone())).unwrap();
        (metric, registry)
    }

    fn output(registry: &Registry) -> String {
        TextEncoder::new()
            .encode_to_string(&registry.gather())
            .unwrap()
    }

    #[test]
    fn removed_worker_stays_absent_after_discovery_removal() {
        let (metric, registry) = setup();
        let (tx, rx) = watch::channel(HashMap::from([
            (1, ModelRuntimeConfig::default()),
            (2, ModelRuntimeConfig::default()),
        ]));
        let _owner = metric.watch(rx, "decode", CancellationToken::new(), None);
        assert!(output(&registry).contains("router_worker_id=\"1\""));
        tx.send(HashMap::from([(2, ModelRuntimeConfig::default())]))
            .unwrap();
        // No scheduler/watch task needs to run before the next scrape excludes worker 1.
        assert!(!output(&registry).contains("router_worker_id=\"1\""));
        for _ in 0..2 {
            let text = output(&registry);
            assert!(!text.contains("router_worker_id=\"1\""), "{text}");
            assert!(text.contains("router_worker_id=\"2\""), "{text}");
            assert!(text.contains("dynamo_component=\"frontend\""));
        }
    }

    #[test]
    fn admission_rejection_hides_discovered_workers_and_allows_recovery() {
        use std::collections::HashSet;

        let (metric, registry) = setup();
        let (_discovery_tx, discovery_rx) = watch::channel(HashMap::from([
            (1, ModelRuntimeConfig::default()),
            (2, ModelRuntimeConfig::default()),
        ]));
        let (available_tx, available_rx) = watch::channel(None::<Arc<HashSet<u64>>>);
        let provider: WorkerAvailabilityProvider = Arc::new(move || available_rx.borrow().clone());
        let _owner = metric.watch(
            discovery_rx,
            "decode",
            CancellationToken::new(),
            Some(provider),
        );
        assert!(output(&registry).is_empty());

        available_tx
            .send(Some(Arc::new(HashSet::from([1, 2]))))
            .unwrap();
        assert_eq!(output(&registry).matches("router_worker_id=").count(), 2);
        available_tx
            .send(Some(Arc::new(HashSet::from([2]))))
            .unwrap();
        let text = output(&registry);
        assert!(!text.contains("router_worker_id=\"1\""));
        assert!(text.contains("router_worker_id=\"2\""));

        // MCD conflicts fail the whole admission group closed while discovery remains.
        available_tx.send(Some(Arc::new(HashSet::new()))).unwrap();
        assert!(output(&registry).is_empty());
        available_tx
            .send(Some(Arc::new(HashSet::from([1]))))
            .unwrap();
        let text = output(&registry);
        assert!(text.contains("router_worker_id=\"1\""));
        assert!(!text.contains("router_worker_id=\"2\""));
    }

    #[test]
    fn shared_owners_deduplicate_and_cleanup_independently() {
        let (metric, registry) = setup();
        let (_tx, rx) = watch::channel(HashMap::from([(
            7,
            ModelRuntimeConfig {
                data_parallel_start_rank: 2,
                data_parallel_size: 2,
                ..Default::default()
            },
        )]));
        let cancel = CancellationToken::new();
        let _first = metric.watch(rx.clone(), "decode", cancel.clone(), None);
        let second = metric.watch(rx.clone(), "decode", CancellationToken::new(), None);
        let prefill = metric.watch(rx, "prefill", CancellationToken::new(), None);
        let text = output(&registry);
        assert_eq!(text.matches("worker_type=\"decode\"").count(), 2);
        assert!(text.contains("dp_rank=\"2\""));
        assert!(text.contains("dp_rank=\"3\""));
        cancel.cancel();
        assert_eq!(
            output(&registry).matches("worker_type=\"decode\"").count(),
            2
        );
        drop(second);
        let text = output(&registry);
        assert!(!text.contains("worker_type=\"decode\""));
        assert_eq!(text.matches("worker_type=\"prefill\"").count(), 2);
        drop(prefill);
        assert!(output(&registry).is_empty());
    }

    #[test]
    fn rank_updates_invalid_configs_and_closed_watch_do_not_retain_series() {
        let (metric, registry) = setup();
        let (tx, rx) = watch::channel(HashMap::from([(
            3,
            ModelRuntimeConfig {
                data_parallel_size: 2,
                ..Default::default()
            },
        )]));
        let _owner = metric.watch(rx, "decode", CancellationToken::new(), None);
        assert!(output(&registry).contains("dp_rank=\"1\""));
        tx.send(HashMap::from([(3, ModelRuntimeConfig::default())]))
            .unwrap();
        assert!(!output(&registry).contains("dp_rank=\"1\""));
        tx.send(HashMap::from([(
            3,
            ModelRuntimeConfig {
                data_parallel_size: 0,
                ..Default::default()
            },
        )]))
        .unwrap();
        assert!(output(&registry).is_empty());
        tx.send(HashMap::from([(3, ModelRuntimeConfig::default())]))
            .unwrap();
        assert!(output(&registry).contains("router_worker_id=\"3\""));
        drop(tx);
        assert!(output(&registry).is_empty());
    }
}
