// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use dynamo_global_common::PoolId;
use dynamo_global_router::{LoadReport, PoolSignals, RelaySource, ReportReceipt};
use dynamo_llm::pool_relay::wan::grpc::protocol::{
    KvPoolCatalogUpdate, KvPoolLoadUpdate, ProducerKey, ServingLoadUpdate, ServingReadinessState,
    ServingReadinessUpdate, validate_protocol_envelope,
};
use std::time::Duration;

/// State from one Relay used by readiness and load selection.
pub(crate) struct RelayPoolState {
    catalog: Arc<KvPoolCatalogUpdate>,
    serving_load: Option<Arc<ServingLoadUpdate>>,
    kv_load: Option<Arc<KvPoolLoadUpdate>>,
    ready_models: HashSet<String>,
    source: RelaySource,
    serving_sequence: Option<u64>,
    kv_sequence: Option<u64>,
}

impl RelayPoolState {
    pub(crate) fn from_updates(
        catalog: Arc<KvPoolCatalogUpdate>,
        readiness: &ServingReadinessUpdate,
        mut serving_load: Option<Arc<ServingLoadUpdate>>,
        mut kv_load: Option<Arc<KvPoolLoadUpdate>>,
        expected: RelaySource,
    ) -> anyhow::Result<Self> {
        validate_protocol_envelope(catalog.protocol_version, catalog.contract_marker)?;
        validate_protocol_envelope(readiness.protocol_version, readiness.contract_marker)?;
        let identity = catalog
            .relay
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing Relay identity"))?;
        anyhow::ensure!(
            identity.drt_instance_id == expected.runtime_id
                && identity.relay_incarnation == expected.incarnation
                && expected.incarnation != 0,
            "state does not belong to the stream's Relay"
        );
        anyhow::ensure!(
            readiness.relay.as_ref() == Some(identity),
            "readiness is from another Relay"
        );
        if let Some(load) = &serving_load {
            validate_protocol_envelope(load.protocol_version, load.contract_marker)?;
            anyhow::ensure!(
                load.relay.as_ref() == Some(identity),
                "serving load is from another Relay"
            );
        }
        if let Some(load) = &kv_load {
            validate_protocol_envelope(load.protocol_version, load.contract_marker)?;
            anyhow::ensure!(
                load.relay.as_ref() == Some(identity),
                "KV load is from another Relay"
            );
        }
        let producers = &catalog
            .snapshot
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("complete catalog snapshot required"))?
            .pools;
        let producer_keys = producers
            .iter()
            .map(|pool| {
                let producer = pool
                    .producer
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing catalog producer"))?;
                Ok(ProducerKey::try_from(producer)?)
            })
            .collect::<anyhow::Result<HashSet<_>>>()?;
        let serving_sequence = serving_load.as_ref().map(|load| load.window_sequence);
        let kv_sequence = kv_load.as_ref().map(|load| load.window_sequence);
        // Catalog cutover may precede the next load window. Keep that
        // interval unknown rather than applying an old layout's load.
        let producer_matches = |producer: &Option<_>| {
            producer
                .as_ref()
                .and_then(|producer| ProducerKey::try_from(producer).ok())
                .is_some_and(|producer| producer_keys.contains(&producer))
        };
        if serving_load.as_ref().is_some_and(|load| {
            load.pools.len() != producers.len()
                || load
                    .pools
                    .iter()
                    .any(|pool| !producer_matches(&pool.producer))
        }) {
            serving_load = None;
        }
        if kv_load.as_ref().is_some_and(|load| {
            load.pools.len() != producers.len()
                || load
                    .pools
                    .iter()
                    .any(|pool| !producer_matches(&pool.producer))
        }) {
            kv_load = None;
        }
        // Topology names canonical models; callers may use aliases advertised
        // by the catalog. Resolve them once when state arrives.
        let mut ready_models = HashSet::new();
        for entry in &readiness.entries {
            if entry.state == ServingReadinessState::Ready as i32 {
                ready_models.insert(entry.canonical_model_id.clone());
                ready_models.extend(
                    entry
                        .adapters
                        .iter()
                        .filter(|adapter| adapter.state == ServingReadinessState::Ready as i32)
                        .map(|adapter| adapter.canonical_model_id.clone()),
                );
            }
        }
        for registration in producers.iter().flat_map(|pool| &pool.registrations) {
            if ready_models.contains(&registration.canonical_model_id)
                && dynamo_llm::pool_relay::wan::grpc::protocol::validate_model_registration(
                    registration,
                )
                .is_ok()
            {
                ready_models.extend(registration.aliases.iter().cloned());
            }
        }
        let source = RelaySource {
            runtime_id: identity.drt_instance_id,
            incarnation: identity.relay_incarnation,
        };
        Ok(Self {
            catalog,
            serving_load,
            kv_load,
            ready_models,
            source,
            serving_sequence,
            kv_sequence,
        })
    }

    pub(crate) fn source(&self) -> RelaySource {
        self.source
    }

    pub(crate) fn ready_for(&self, model: &str) -> bool {
        // PoolKey names the Kubernetes namespace; topology entries name
        // Dynamo runtime namespaces. The Relay's source scope supplies the
        // DGD boundary, so these namespace strings must not be equated.
        self.ready_models.contains(model)
    }

    pub(crate) fn ready_models(&self) -> &HashSet<String> {
        &self.ready_models
    }

    fn canonical_model<'a>(&'a self, model: &'a str) -> &'a str {
        self.catalog
            .snapshot
            .as_ref()
            .into_iter()
            .flat_map(|catalog| &catalog.pools)
            .flat_map(|pool| &pool.registrations)
            .find(|registration| {
                self.ready_models.contains(&registration.canonical_model_id)
                    && registration.aliases.iter().any(|alias| alias == model)
                    && dynamo_llm::pool_relay::wan::grpc::protocol::validate_model_registration(
                        registration,
                    )
                    .is_ok()
            })
            .map_or(model, |registration| {
                registration.canonical_model_id.as_str()
            })
    }

    pub(crate) fn window_sequences(&self) -> (Option<u64>, Option<u64>) {
        (self.serving_sequence, self.kv_sequence)
    }

    pub(crate) fn project(
        &self,
        pool: &PoolId,
        model: &str,
        signals: &PoolSignals,
        serving_receipt: Option<ReportReceipt>,
        kv_receipt: Option<ReportReceipt>,
    ) {
        if !self.ready_for(model) {
            signals.forget_source(pool, self.source);
            return;
        }
        let load = self
            .serving_load
            .as_ref()
            .zip(serving_receipt)
            .and_then(|(load, receipt)| {
                pool_report(load, self.canonical_model(model)).map(|report| (report, receipt))
            });
        let kv = self
            .kv_load
            .as_ref()
            .zip(kv_receipt)
            .and_then(|(load, receipt)| {
                pool_kv(load)
                    .map(|(used, total)| (used, total, report_window(load.window_ms), receipt))
            });
        signals.project_source(pool, self.source, load, kv);
    }
}

/// The Relay report window when an update leaves it unset.
const DEFAULT_WINDOW: Duration = Duration::from_secs(1);

fn report_window(window_ms: u64) -> Duration {
    match window_ms {
        0 => DEFAULT_WINDOW,
        window_ms => Duration::from_millis(window_ms),
    }
}

/// The KV blocks the pool's workers hold for running requests, and their
/// total. `None` unless every producer has heard from every rank and knows
/// its capacity: the Relay marks an incomplete view rather than leaving it
/// out, and an incomplete view is not zero load.
fn pool_kv(update: &KvPoolLoadUpdate) -> Option<(u64, u64)> {
    if update.pools.is_empty() {
        return None;
    }
    update
        .pools
        .iter()
        .try_fold((0_u64, 0_u64), |(used, total), entry| {
            let complete = entry.kv_expected_ranks > 0
                && entry.kv_observed_ranks >= entry.kv_expected_ranks
                && entry.total_kv_blocks > 0;
            complete.then(|| {
                (
                    used.saturating_add(entry.kv_used_blocks),
                    total.saturating_add(entry.total_kv_blocks),
                )
            })
        })
}

/// The pool's requests in flight (waiting for a first token or generating)
/// for `model`, those generating, and how many its workers can run at once.
/// `None` unless the request counts are reported: the Relay leaves them
/// unset when its view is incomplete, and unset does not mean zero. A Relay
/// watching one pool reports one model; a model it names differently from
/// the catalog still counts.
fn pool_report(update: &ServingLoadUpdate, model: &str) -> Option<LoadReport> {
    let named = update
        .models
        .iter()
        .filter(|entry| entry.canonical_model_id == model)
        .collect::<Vec<_>>();
    let models = if named.is_empty() {
        update.models.iter().collect()
    } else {
        named
    };
    if models.is_empty() {
        return None;
    }
    let (mut in_flight, mut generating) = (0, 0);
    for entry in models {
        let requests = entry.load.as_ref()?.requests.as_ref()?;
        let running = requests.requests_generating?;
        in_flight += requests.requests_awaiting_first_token? + running;
        generating += running;
    }
    let max_concurrency = update
        .pools
        .iter()
        .map(|pool| pool.deployment.as_ref()?.max_concurrency)
        .sum::<Option<u64>>();
    Some(LoadReport {
        in_flight,
        generating,
        max_concurrency,
        window: report_window(update.window_ms),
    })
}

#[cfg(test)]
mod tests {
    use dynamo_llm::pool_relay::wan::grpc::protocol::{
        KvPoolLoadEntry, LoadView, ModelServingLoad, PoolDeploymentStatus, PoolServingLoad,
        RequestLifecycleStats,
    };

    use super::*;

    /// An update over a 2 s window from a pool of 8 workers that can run
    /// `capacity` requests in all, with each model's requests waiting and
    /// generating.
    fn update(
        capacity: Option<u64>,
        models: &[(&str, Option<u64>, Option<u64>)],
    ) -> ServingLoadUpdate {
        ServingLoadUpdate {
            window_ms: 2000,
            pools: vec![PoolServingLoad {
                deployment: Some(PoolDeploymentStatus {
                    live_workers: Some(8),
                    max_concurrency: capacity,
                }),
                ..Default::default()
            }],
            models: models
                .iter()
                .map(|(name, awaiting, generating)| ModelServingLoad {
                    canonical_model_id: (*name).to_owned(),
                    load: Some(LoadView {
                        requests: Some(RequestLifecycleStats {
                            requests_awaiting_first_token: *awaiting,
                            requests_generating: *generating,
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn counts_waiting_and_generating_requests_against_capacity() {
        let report = pool_report(&update(Some(128), &[("qwen", Some(3), Some(20))]), "qwen");
        assert_eq!(
            report,
            Some(LoadReport {
                in_flight: 23,
                generating: 20,
                max_concurrency: Some(128),
                window: Duration::from_secs(2),
            })
        );
        // A Relay naming the model differently still reports its pool.
        let report = pool_report(
            &update(Some(128), &[("Qwen/Qwen3", Some(1), Some(1))]),
            "qwen",
        );
        assert_eq!(report.map(|report| report.in_flight), Some(2));
        // Without the workers' capacity the counts still arrive.
        let report = pool_report(&update(None, &[("qwen", Some(1), Some(1))]), "qwen");
        assert_eq!(report.map(|report| report.max_concurrency), Some(None));
    }

    #[test]
    fn incomplete_views_are_unknown_not_zero() {
        assert_eq!(
            pool_report(&update(Some(128), &[("qwen", None, Some(1))]), "qwen"),
            None
        );
        assert_eq!(
            pool_report(&update(Some(128), &[("qwen", Some(1), None)]), "qwen"),
            None
        );
        assert_eq!(pool_report(&update(Some(128), &[]), "qwen"), None);
    }

    #[test]
    fn kv_load_sums_producers_and_is_unknown_when_incomplete() {
        let entry = |used, total, observed, expected| KvPoolLoadEntry {
            kv_used_blocks: used,
            total_kv_blocks: total,
            kv_observed_ranks: observed,
            kv_expected_ranks: expected,
            ..Default::default()
        };
        let update = |pools| KvPoolLoadUpdate {
            pools,
            ..Default::default()
        };
        assert_eq!(
            pool_kv(&update(vec![entry(10, 100, 4, 4), entry(5, 100, 4, 4)])),
            Some((15, 200))
        );
        // A rank not yet heard from, or no capacity, is not zero load.
        assert_eq!(pool_kv(&update(vec![entry(10, 100, 3, 4)])), None);
        assert_eq!(pool_kv(&update(vec![entry(0, 0, 4, 4)])), None);
        assert_eq!(pool_kv(&update(vec![])), None);
    }
}
