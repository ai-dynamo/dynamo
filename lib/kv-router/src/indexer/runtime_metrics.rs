// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Builds the indexer's counters from a runtime `Component`. Lives here, not in the stage,
//! because the stage must not depend on `dynamo-runtime`.

use std::sync::Arc;
#[cfg(feature = "metrics")]
use std::sync::OnceLock;

use dynamo_kv_indexer::metrics::KvIndexerMetrics;
use dynamo_runtime::component::Component;
#[cfg(feature = "metrics")]
use dynamo_runtime::metrics::MetricsHierarchy;

#[cfg(feature = "metrics")]
const KV_CACHE_EVENTS_APPLIED_SUFFIX: &str = "kv_cache_events_applied";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENT_WARNINGS_SUFFIX: &str = "kv_cache_event_warnings";
#[cfg(feature = "metrics")]
const CKF_MUTATION_SUFFIX: &str = "ckf_mutation_total";

#[cfg(feature = "metrics")]
static KV_INDEXER_METRICS: OnceLock<Arc<KvIndexerMetrics>> = OnceLock::new();

/// Host-side constructor for [`KvIndexerMetrics`] from a runtime component.
pub trait KvIndexerMetricsExt {
    /// Creates a shared metrics instance from a Component, memoizing the result so repeated
    /// construction does not register the same series twice.
    fn from_component(component: &Component) -> Arc<KvIndexerMetrics>;
}

impl KvIndexerMetricsExt for KvIndexerMetrics {
    fn from_component(component: &Component) -> Arc<KvIndexerMetrics> {
        #[cfg(feature = "metrics")]
        {
            use dynamo_kv_indexer::metrics::{
                CKF_MUTATION_HELP, CKF_MUTATION_LABELS, KV_CACHE_EVENT_WARNINGS_HELP,
                KV_CACHE_EVENT_WARNINGS_LABELS, KV_CACHE_EVENTS_APPLIED_HELP,
                KV_CACHE_EVENTS_APPLIED_LABELS,
            };
            KV_INDEXER_METRICS
                .get_or_init(|| {
                    match (
                        component.metrics().create_intcountervec(
                            KV_CACHE_EVENTS_APPLIED_SUFFIX,
                            KV_CACHE_EVENTS_APPLIED_HELP,
                            KV_CACHE_EVENTS_APPLIED_LABELS,
                            &[],
                        ),
                        component.metrics().create_intcountervec(
                            KV_CACHE_EVENT_WARNINGS_SUFFIX,
                            KV_CACHE_EVENT_WARNINGS_HELP,
                            KV_CACHE_EVENT_WARNINGS_LABELS,
                            &[],
                        ),
                        component.metrics().create_intcountervec(
                            CKF_MUTATION_SUFFIX,
                            CKF_MUTATION_HELP,
                            CKF_MUTATION_LABELS,
                            &[],
                        ),
                    ) {
                        (
                            Ok(kv_cache_events_applied),
                            Ok(kv_cache_event_warnings),
                            Ok(ckf_mutation),
                        ) => Arc::new(KvIndexerMetrics::from_counters(
                            kv_cache_events_applied,
                            kv_cache_event_warnings,
                            ckf_mutation,
                        )),
                        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                            tracing::warn!("Failed to create kv indexer metrics from component: {}. Using unregistered metrics as fallback.", e);
                            Arc::new(KvIndexerMetrics::new_unregistered())
                        }
                    }
                })
                .clone()
        }

        #[cfg(not(feature = "metrics"))]
        {
            let _ = component;
            Arc::new(KvIndexerMetrics::new_unregistered())
        }
    }
}
