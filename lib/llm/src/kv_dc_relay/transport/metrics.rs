// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, Weak};

use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use parking_lot::{Mutex, RwLock};
use prometheus::{IntCounter, IntGauge};

use super::super::identity::DcRelayIdentity;
use super::super::pool_registry::PoolRegistry;
use super::super::topology::{TopologyPublisher, TopologyReadinessState};
use super::source::WanPublicationSource;

#[derive(Debug, Clone, Copy)]
pub(crate) enum StreamKind {
    Catalog,
    Pool,
    Readiness,
    Load,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum SubscriberLimitScope {
    Total,
    PerPool,
    InitializedHub,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MetricsOwnerKey {
    registry_address: usize,
    drt_instance_id: u64,
    namespace: String,
    component: String,
}

struct ActiveMetricsSource {
    identity: DcRelayIdentity,
    pools: Weak<PoolRegistry>,
    topology: Weak<TopologyPublisher>,
    last_terminal_failures: u64,
}

pub(crate) struct TransportMetrics {
    active: RwLock<Option<ActiveMetricsSource>>,
    catalog_subscribers: IntGauge,
    pool_subscribers: IntGauge,
    readiness_subscribers: IntGauge,
    load_subscribers: IntGauge,
    catalog_lagged_total: IntCounter,
    pool_lagged_total: IntCounter,
    readiness_lagged_total: IntCounter,
    load_lagged_total: IntCounter,
    catalog_limit_rejected_total: IntCounter,
    pool_limit_rejected_total: IntCounter,
    pool_per_pool_limit_rejected_total: IntCounter,
    pool_initialized_hub_limit_rejected_total: IntCounter,
    readiness_limit_rejected_total: IntCounter,
    load_limit_rejected_total: IntCounter,
    pub(crate) pool_heartbeats_total: IntCounter,
    pub(crate) load_updates_total: IntCounter,
    catalog_pool_count: IntGauge,
    requested_publication_hub_count: IntGauge,
    initialized_publication_hub_count: IntGauge,
    ready_publication_hub_count: IntGauge,
    idle_publication_hub_count: IntGauge,
    terminal_publication_failures_total: IntCounter,
    degraded_load_coverage_pool_count: IntGauge,
    readiness_unknown_count: IntGauge,
    readiness_unavailable_count: IntGauge,
    readiness_ready_count: IntGauge,
    tls_server_cert_expiry: IntGauge,
    tls_client_ca_expiry: IntGauge,
}

pub(crate) struct MetricsLease {
    metrics: Arc<TransportMetrics>,
    identity: DcRelayIdentity,
}

static METRIC_OWNERS: OnceLock<Mutex<HashMap<MetricsOwnerKey, Weak<TransportMetrics>>>> =
    OnceLock::new();

impl TransportMetrics {
    pub(crate) fn acquire(
        source: &WanPublicationSource,
        server_cert_not_after: Option<i64>,
        client_ca_not_after: Option<i64>,
    ) -> anyhow::Result<(Arc<Self>, MetricsLease)> {
        let component = source.component();
        let registry = component.get_metrics_registry();
        let key = MetricsOwnerKey {
            registry_address: Arc::as_ptr(&registry.prometheus_registry) as usize,
            drt_instance_id: source.relay_identity().drt_instance_id(),
            namespace: component.namespace().name().to_string(),
            component: component.name().to_string(),
        };
        let owners = METRIC_OWNERS.get_or_init(|| Mutex::new(HashMap::new()));
        let metrics = {
            let mut owners = owners.lock();
            if let Some(metrics) = owners.get(&key).and_then(Weak::upgrade) {
                metrics
            } else {
                let metrics = Self::register(component)?;
                owners.insert(key, Arc::downgrade(&metrics));
                metrics
            }
        };
        metrics.attach(source, server_cert_not_after, client_ca_not_after)?;
        let lease = MetricsLease {
            metrics: metrics.clone(),
            identity: source.relay_identity(),
        };
        Ok((metrics, lease))
    }

    fn register(component: &Component) -> anyhow::Result<Arc<Self>> {
        let metrics = component.metrics();
        let subscribers = metrics.create_intgaugevec(
            "kv_dc_relay_subscribers",
            "Current Relay stream subscriber count by finite stream type.",
            &["stream"],
            &[],
        )?;
        let lagged = metrics.create_intcountervec(
            "kv_dc_relay_subscriber_lagged_total",
            "Relay streams terminated after bounded-queue lag by finite stream type.",
            &["stream"],
            &[],
        )?;
        let rejected = metrics.create_intcountervec(
            "kv_dc_relay_subscriber_limit_rejected_total",
            "Relay subscriptions rejected at a configured resource limit.",
            &["stream", "scope"],
            &[],
        )?;
        let readiness = metrics.create_intgaugevec(
            "kv_dc_relay_serving_topology_count",
            "Current namespace model topologies by serving-readiness state.",
            &["state"],
            &[],
        )?;
        let tls_expiry = metrics.create_intgaugevec(
            "kv_dc_relay_tls_expiry_timestamp_seconds",
            "Earliest Relay TLS certificate expiry by material.",
            &["material"],
            &[],
        )?;
        let owner = Arc::new(Self {
            active: RwLock::new(None),
            catalog_subscribers: subscribers.with_label_values(&["catalog"]),
            pool_subscribers: subscribers.with_label_values(&["pool"]),
            readiness_subscribers: subscribers.with_label_values(&["readiness"]),
            load_subscribers: subscribers.with_label_values(&["load"]),
            catalog_lagged_total: lagged.with_label_values(&["catalog"]),
            pool_lagged_total: lagged.with_label_values(&["pool"]),
            readiness_lagged_total: lagged.with_label_values(&["readiness"]),
            load_lagged_total: lagged.with_label_values(&["load"]),
            catalog_limit_rejected_total: rejected.with_label_values(&["catalog", "total"]),
            pool_limit_rejected_total: rejected.with_label_values(&["pool", "total"]),
            pool_per_pool_limit_rejected_total: rejected.with_label_values(&["pool", "per_pool"]),
            pool_initialized_hub_limit_rejected_total: rejected
                .with_label_values(&["pool", "initialized_hub"]),
            readiness_limit_rejected_total: rejected.with_label_values(&["readiness", "total"]),
            load_limit_rejected_total: rejected.with_label_values(&["load", "total"]),
            pool_heartbeats_total: metrics.create_intcounter(
                "kv_dc_relay_pool_heartbeats_total",
                "Application-level heartbeats emitted across pool streams.",
                &[],
            )?,
            load_updates_total: metrics.create_intcounter(
                "kv_dc_relay_load_updates_total",
                "Atomic complete pool-load windows emitted.",
                &[],
            )?,
            catalog_pool_count: metrics.create_intgauge(
                "kv_dc_relay_catalog_pool_count",
                "Current pools in the Relay producer catalog.",
                &[],
            )?,
            requested_publication_hub_count: metrics.create_intgauge(
                "kv_dc_relay_requested_publication_hub_count",
                "Current pools whose lazy publication hub has been requested.",
                &[],
            )?,
            initialized_publication_hub_count: metrics.create_intgauge(
                "kv_dc_relay_initialized_publication_hub_count",
                "Current per-pool publication hubs holding an initialized CKF mirror.",
                &[],
            )?,
            ready_publication_hub_count: metrics.create_intgauge(
                "kv_dc_relay_ready_publication_hub_count",
                "Current publication hubs with a contiguous ready CKF mirror.",
                &[],
            )?,
            idle_publication_hub_count: metrics.create_intgauge(
                "kv_dc_relay_idle_publication_hub_count",
                "Current ready publication hubs with no active subscribers.",
                &[],
            )?,
            terminal_publication_failures_total: metrics.create_intcounter(
                "kv_dc_relay_terminal_publication_failures_total",
                "Pool generations fenced after a terminal publication failure.",
                &[],
            )?,
            degraded_load_coverage_pool_count: metrics.create_intgauge(
                "kv_dc_relay_degraded_load_coverage_pool_count",
                "Pools where at least one load signal lacks complete rank coverage.",
                &[],
            )?,
            readiness_unknown_count: readiness.with_label_values(&["unknown"]),
            readiness_unavailable_count: readiness.with_label_values(&["unavailable"]),
            readiness_ready_count: readiness.with_label_values(&["ready"]),
            tls_server_cert_expiry: tls_expiry.with_label_values(&["server_cert"]),
            tls_client_ca_expiry: tls_expiry.with_label_values(&["client_ca"]),
        });
        let callback_owner = owner.clone();
        component
            .get_metrics_registry()
            .add_update_callback(Arc::new(move || {
                callback_owner.sample();
                Ok(())
            }));
        Ok(owner)
    }

    fn attach(
        &self,
        source: &WanPublicationSource,
        server_cert_not_after: Option<i64>,
        client_ca_not_after: Option<i64>,
    ) -> anyhow::Result<()> {
        let mut active = self.active.write();
        if let Some(current) = active.as_ref()
            && current.pools.upgrade().is_some()
        {
            anyhow::bail!(
                "a WAN Relay is already attached to this component metrics owner (incarnation {})",
                current.identity.relay_incarnation()
            );
        }
        *active = Some(ActiveMetricsSource {
            identity: source.relay_identity(),
            pools: Arc::downgrade(source.pools()),
            topology: Arc::downgrade(source.topology()),
            last_terminal_failures: 0,
        });
        self.tls_server_cert_expiry
            .set(server_cert_not_after.unwrap_or_default());
        self.tls_client_ca_expiry
            .set(client_ca_not_after.unwrap_or_default());
        drop(active);
        self.sample();
        Ok(())
    }

    fn detach(&self, identity: DcRelayIdentity) {
        let mut active = self.active.write();
        if active
            .as_ref()
            .is_some_and(|current| current.identity == identity)
        {
            *active = None;
            self.reset_gauges();
        }
    }

    fn sample(&self) {
        let mut active_guard = self.active.write();
        let Some(active) = active_guard.as_mut() else {
            self.reset_gauges();
            return;
        };
        let Some(pools) = active.pools.upgrade() else {
            *active_guard = None;
            self.reset_gauges();
            return;
        };
        let Some(topology) = active.topology.upgrade() else {
            *active_guard = None;
            self.reset_gauges();
            return;
        };
        let catalog = pools.catalog();
        let publication = pools.publication_metrics();
        let readiness = topology.snapshot();
        let load = pools.load_snapshots();
        self.catalog_pool_count
            .set(saturating_i64(catalog.pools().len()));
        self.requested_publication_hub_count
            .set(saturating_i64(publication.requested_hubs));
        self.initialized_publication_hub_count
            .set(saturating_i64(publication.initialized_hubs));
        self.ready_publication_hub_count
            .set(saturating_i64(publication.ready_hubs));
        self.idle_publication_hub_count
            .set(saturating_i64(publication.idle_hubs));
        if publication.terminal_failures >= active.last_terminal_failures {
            self.terminal_publication_failures_total.inc_by(
                publication
                    .terminal_failures
                    .saturating_sub(active.last_terminal_failures),
            );
        }
        active.last_terminal_failures = publication.terminal_failures;
        self.observe_load_window(&load);
        let mut unknown = 0usize;
        let mut unavailable = 0usize;
        let mut ready = 0usize;
        for entry in &readiness.entries {
            match entry.state {
                TopologyReadinessState::Unknown => unknown += 1,
                TopologyReadinessState::Unavailable => unavailable += 1,
                TopologyReadinessState::Ready => ready += 1,
            }
        }
        self.readiness_unknown_count.set(saturating_i64(unknown));
        self.readiness_unavailable_count
            .set(saturating_i64(unavailable));
        self.readiness_ready_count.set(saturating_i64(ready));
    }

    fn reset_gauges(&self) {
        self.catalog_subscribers.set(0);
        self.pool_subscribers.set(0);
        self.readiness_subscribers.set(0);
        self.load_subscribers.set(0);
        self.catalog_pool_count.set(0);
        self.requested_publication_hub_count.set(0);
        self.initialized_publication_hub_count.set(0);
        self.ready_publication_hub_count.set(0);
        self.idle_publication_hub_count.set(0);
        self.degraded_load_coverage_pool_count.set(0);
        self.readiness_unknown_count.set(0);
        self.readiness_unavailable_count.set(0);
        self.readiness_ready_count.set(0);
        self.tls_server_cert_expiry.set(0);
        self.tls_client_ca_expiry.set(0);
    }

    pub(crate) fn subscriber_guard(&self, stream: StreamKind) -> SubscriberGuard {
        let gauge = self.subscriber_gauge(stream);
        gauge.inc();
        SubscriberGuard(gauge.clone())
    }

    pub(crate) fn subscriber_lagged(&self, stream: StreamKind) {
        self.stream_counter(
            stream,
            &self.catalog_lagged_total,
            &self.pool_lagged_total,
            &self.readiness_lagged_total,
            &self.load_lagged_total,
        )
        .inc();
    }

    pub(crate) fn subscriber_limit_rejected(
        &self,
        stream: StreamKind,
        scope: SubscriberLimitScope,
    ) {
        match scope {
            SubscriberLimitScope::Total => self.stream_counter(
                stream,
                &self.catalog_limit_rejected_total,
                &self.pool_limit_rejected_total,
                &self.readiness_limit_rejected_total,
                &self.load_limit_rejected_total,
            ),
            SubscriberLimitScope::PerPool => {
                debug_assert!(matches!(stream, StreamKind::Pool));
                &self.pool_per_pool_limit_rejected_total
            }
            SubscriberLimitScope::InitializedHub => {
                debug_assert!(matches!(stream, StreamKind::Pool));
                &self.pool_initialized_hub_limit_rejected_total
            }
        }
        .inc();
    }

    pub(crate) fn observe_load_window(&self, snapshots: &[super::super::load::PoolLoadSnapshot]) {
        self.degraded_load_coverage_pool_count.set(saturating_i64(
            snapshots
                .iter()
                .filter(|snapshot| snapshot.has_degraded_coverage())
                .count(),
        ));
    }

    fn subscriber_gauge(&self, stream: StreamKind) -> &IntGauge {
        match stream {
            StreamKind::Catalog => &self.catalog_subscribers,
            StreamKind::Pool => &self.pool_subscribers,
            StreamKind::Readiness => &self.readiness_subscribers,
            StreamKind::Load => &self.load_subscribers,
        }
    }

    fn stream_counter<'a>(
        &'a self,
        stream: StreamKind,
        catalog: &'a IntCounter,
        pool: &'a IntCounter,
        readiness: &'a IntCounter,
        load: &'a IntCounter,
    ) -> &'a IntCounter {
        match stream {
            StreamKind::Catalog => catalog,
            StreamKind::Pool => pool,
            StreamKind::Readiness => readiness,
            StreamKind::Load => load,
        }
    }
}

impl Drop for MetricsLease {
    fn drop(&mut self) {
        self.metrics.detach(self.identity);
    }
}

pub(crate) struct SubscriberGuard(IntGauge);

impl Drop for SubscriberGuard {
    fn drop(&mut self) {
        self.0.dec();
    }
}

fn saturating_i64(value: impl TryInto<i64>) -> i64 {
    value.try_into().unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_runtime::{
        DistributedRuntime, Runtime, distributed::DistributedConfig,
        traits::DistributedRuntimeProvider,
    };
    use tokio_util::sync::CancellationToken;

    use super::super::super::pool_registry::PoolActorConfig;
    use super::*;

    async fn test_component() -> Component {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        drt.namespace("kv-dc-relay-metrics-restart")
            .unwrap()
            .component("relay")
            .unwrap()
    }

    fn source(
        component: Component,
        identity: DcRelayIdentity,
        lifecycle: CancellationToken,
    ) -> WanPublicationSource {
        let pools = Arc::new(PoolRegistry::new(
            identity,
            PoolActorConfig {
                expected_unique_blocks: 32,
                publication_threshold: 1,
                publication_delay: Duration::from_millis(1),
            },
        ));
        let topology = Arc::new(TopologyPublisher::new(Default::default(), &pools.catalog()));
        WanPublicationSource::new(component, pools, topology, identity, lifecycle)
    }

    #[tokio::test]
    async fn same_component_restart_reuses_metrics_registration_and_replaces_lease() {
        let component = test_component().await;
        let first = source(
            component.clone(),
            DcRelayIdentity::new(component.drt().connection_id(), 11),
            CancellationToken::new(),
        );
        let (first_metrics, first_lease) =
            TransportMetrics::acquire(&first, Some(100), Some(200)).unwrap();

        let second = source(
            component,
            DcRelayIdentity::new(first.relay_identity().drt_instance_id(), 12),
            CancellationToken::new(),
        );
        assert!(TransportMetrics::acquire(&second, Some(300), Some(400)).is_err());

        drop(first_lease);
        let (second_metrics, second_lease) =
            TransportMetrics::acquire(&second, Some(300), Some(400)).unwrap();
        assert!(Arc::ptr_eq(&first_metrics, &second_metrics));
        assert_eq!(second_metrics.tls_server_cert_expiry.get(), 300);
        assert_eq!(second_metrics.tls_client_ca_expiry.get(), 400);
        drop(second_lease);
        assert_eq!(second_metrics.tls_server_cert_expiry.get(), 0);
        assert_eq!(second_metrics.tls_client_ca_expiry.get(), 0);
    }
}
