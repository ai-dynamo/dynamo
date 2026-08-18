// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! System health monitoring and health check management

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
    time::Instant,
};
use tokio::sync::mpsc;

use crate::component;
use crate::config::HealthStatus;
use crate::metrics::{MetricsHierarchy, prometheus_names::distributed_runtime};

/// Health check target containing instance info and payload
#[derive(Clone, Debug)]
pub struct HealthCheckTarget {
    pub instance: component::Instance,
    pub payload: serde_json::Value,
}

/// Current Health Status
/// If use_endpoint_health_status is set then
/// initialize the endpoint_health hashmap to the
/// starting health status
#[derive(Clone)]
pub struct SystemHealth {
    system_health: HealthStatus,
    endpoint_health: Arc<std::sync::RwLock<HashMap<String, HealthStatus>>>,
    /// Maps endpoint subject to health check target (instance + payload)
    health_check_targets: Arc<std::sync::RwLock<HashMap<String, HealthCheckTarget>>>,
    /// Maps endpoint subject to its specific health check notifier
    health_check_notifiers: Arc<std::sync::RwLock<HashMap<String, Arc<tokio::sync::Notify>>>>,
    /// Channel for new endpoint registrations
    /// This solves the race condition where HealthCheckManager starts before endpoints are registered
    /// Using a channel ensures no registrations are lost.
    new_endpoint_tx: mpsc::UnboundedSender<String>,
    new_endpoint_rx: Arc<parking_lot::Mutex<Option<mpsc::UnboundedReceiver<String>>>>,
    use_endpoint_health_status: Vec<String>,
    health_check_enabled: bool,
    health_path: String,
    live_path: String,
    start_time: Instant,
    uptime_gauge: OnceLock<prometheus::Gauge>,
}

impl SystemHealth {
    pub fn new(
        starting_health_status: HealthStatus,
        use_endpoint_health_status: Vec<String>,
        health_check_enabled: bool,
        health_path: String,
        live_path: String,
    ) -> Self {
        // Force NotReady when canary is enabled — canary verifies before marking Ready.
        let initial_endpoint_status = if health_check_enabled {
            HealthStatus::NotReady
        } else {
            starting_health_status.clone()
        };
        let mut endpoint_health = HashMap::new();
        for endpoint in &use_endpoint_health_status {
            endpoint_health.insert(endpoint.clone(), initial_endpoint_status.clone());
        }

        // Create the channel for endpoint registration notifications
        let (tx, rx) = mpsc::unbounded_channel();

        SystemHealth {
            system_health: starting_health_status,
            endpoint_health: Arc::new(std::sync::RwLock::new(endpoint_health)),
            health_check_targets: Arc::new(std::sync::RwLock::new(HashMap::new())),
            health_check_notifiers: Arc::new(std::sync::RwLock::new(HashMap::new())),
            new_endpoint_tx: tx,
            new_endpoint_rx: Arc::new(parking_lot::Mutex::new(Some(rx))),
            use_endpoint_health_status,
            health_check_enabled,
            health_path,
            live_path,
            start_time: Instant::now(),
            uptime_gauge: OnceLock::new(),
        }
    }

    pub fn health_check_enabled(&self) -> bool {
        self.health_check_enabled
    }

    /// Signal endpoint transport registration. Sets Ready when canary is disabled;
    /// no-op when canary is enabled (canary will set Ready after verification).
    pub fn set_endpoint_registered(&self, endpoint: &str) {
        if !self.health_check_enabled {
            self.set_endpoint_health_status(endpoint, HealthStatus::Ready);
        }
    }

    pub fn set_health_status(&mut self, status: HealthStatus) {
        self.system_health = status;
    }

    pub fn set_endpoint_health_status(&self, endpoint: &str, status: HealthStatus) {
        let mut endpoint_health = self.endpoint_health.write().unwrap();
        endpoint_health.insert(endpoint.to_string(), status);
    }

    /// Returns the overall health status and endpoint health statuses
    /// System health is determined by ALL endpoints that have registered health checks
    pub fn get_health_status(&self) -> (bool, HashMap<String, String>) {
        let health_check_targets = self.health_check_targets.read().unwrap();
        let endpoint_health = self.endpoint_health.read().unwrap();
        let mut endpoints: HashMap<String, String> = HashMap::new();

        for (endpoint, status) in endpoint_health.iter() {
            endpoints.insert(
                endpoint.clone(),
                if *status == HealthStatus::Ready {
                    "ready".to_string()
                } else {
                    "notready".to_string()
                },
            );
        }

        let healthy = if !self.use_endpoint_health_status.is_empty() {
            self.use_endpoint_health_status.iter().all(|endpoint| {
                endpoint_health
                    .get(endpoint)
                    .is_some_and(|status| *status == HealthStatus::Ready)
            })
        } else {
            // If we have registered health check targets, use them to determine health
            if !health_check_targets.is_empty() {
                health_check_targets
                    .iter()
                    .all(|(endpoint_subject, _target)| {
                        endpoint_health
                            .get(endpoint_subject)
                            .is_some_and(|status| *status == HealthStatus::Ready)
                    })
            } else {
                // No health check targets registered, use simple system health
                self.system_health == HealthStatus::Ready
            }
        };

        (healthy, endpoints)
    }

    /// Register a health check target for an endpoint
    ///
    /// A repeat registration under the same subject is a restart, not an error: the new
    /// incarnation replaces the previous target and is announced to the health check
    /// manager so the canary re-arms against it. Its health resets to `NotReady`, since
    /// the earlier incarnation's verdict says nothing about the process now serving.
    pub fn register_health_check_target(
        &self,
        endpoint_subject: &str,
        instance: component::Instance,
        payload: serde_json::Value,
    ) {
        let key = endpoint_subject.to_owned();

        // Atomically replace under a single write lock to avoid races.
        let replaced = {
            let mut targets = self.health_check_targets.write().unwrap();
            targets
                .insert(key.clone(), HealthCheckTarget { instance, payload })
                .is_some()
        };

        if replaced {
            tracing::debug!("Re-registering health check for endpoint '{key}'; replacing target.");
        }

        // Create and store a unique notifier for this endpoint (idempotent). The existing
        // notifier is kept on replace so an outgoing monitor is not left holding a handle
        // nobody signals.
        {
            let mut notifiers = self.health_check_notifiers.write().unwrap();
            notifiers
                .entry(key.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Notify::new()));
        }

        // Initialize endpoint health status conservatively to NotReady.
        {
            let mut endpoint_health = self.endpoint_health.write().unwrap();
            if replaced {
                endpoint_health.insert(key.clone(), HealthStatus::NotReady);
            } else {
                endpoint_health
                    .entry(key.clone())
                    .or_insert(HealthStatus::NotReady);
            }
        }

        if let Err(e) = self.new_endpoint_tx.send(key.clone()) {
            tracing::error!(
                "Failed to send endpoint '{}' registration to health check manager: {}. \
                 Health checks will not be performed for this endpoint.",
                key,
                e
            );
        }
    }

    /// Deregister an endpoint's health check target
    ///
    /// Clears the target, its notifier, and its health status together. Worker health is
    /// derived from the registered targets, so a target left behind for an endpoint that
    /// never finished starting — or that has shut down — holds the whole worker unhealthy
    /// against an endpoint nobody can reach.
    pub fn deregister_health_check_target(&self, endpoint_subject: &str) {
        let removed = {
            let mut targets = self.health_check_targets.write().unwrap();
            targets.remove(endpoint_subject).is_some()
        };
        {
            let mut notifiers = self.health_check_notifiers.write().unwrap();
            notifiers.remove(endpoint_subject);
        }
        {
            let mut endpoint_health = self.endpoint_health.write().unwrap();
            endpoint_health.remove(endpoint_subject);
        }

        if removed {
            tracing::debug!("Deregistered health check target for endpoint '{endpoint_subject}'");
        }
    }

    /// Get all health check targets
    pub fn get_health_check_targets(&self) -> Vec<(String, HealthCheckTarget)> {
        let targets = self.health_check_targets.read().unwrap();
        targets
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Check if any health check targets are registered
    pub fn has_health_check_targets(&self) -> bool {
        let targets = self.health_check_targets.read().unwrap();
        !targets.is_empty()
    }

    /// Get list of endpoints with health check targets
    pub fn get_health_check_endpoints(&self) -> Vec<String> {
        let targets = self.health_check_targets.read().unwrap();
        targets.keys().cloned().collect()
    }

    /// Get health check target for a specific endpoint
    pub fn get_health_check_target(&self, endpoint: &str) -> Option<HealthCheckTarget> {
        let targets = self.health_check_targets.read().unwrap();
        targets.get(endpoint).cloned()
    }

    /// Get the endpoint health status (Ready/NotReady)
    pub fn get_endpoint_health_status(&self, endpoint: &str) -> Option<HealthStatus> {
        let endpoint_health = self.endpoint_health.read().unwrap();
        endpoint_health.get(endpoint).cloned()
    }

    /// Get the endpoint-specific health check notifier
    pub fn get_endpoint_health_check_notifier(
        &self,
        endpoint_subject: &str,
    ) -> Option<Arc<tokio::sync::Notify>> {
        let notifiers = self.health_check_notifiers.read().unwrap();
        notifiers.get(endpoint_subject).cloned()
    }

    /// Take the receiver for new endpoint registrations (can only be called once)
    /// This is used by HealthCheckManager to receive notifications of new endpoints
    pub fn take_new_endpoint_receiver(&self) -> Option<mpsc::UnboundedReceiver<String>> {
        self.new_endpoint_rx.lock().take()
    }

    /// Initialize the uptime gauge using the provided metrics registry
    pub fn initialize_uptime_gauge<T: MetricsHierarchy>(&self, registry: &T) -> anyhow::Result<()> {
        let gauge = registry.metrics().create_gauge(
            distributed_runtime::UPTIME_SECONDS,
            "Total uptime of the DistributedRuntime in seconds",
            &[],
        )?;
        self.uptime_gauge
            .set(gauge)
            .map_err(|_| anyhow::anyhow!("uptime_gauge already initialized"))?;
        Ok(())
    }

    /// Get the current uptime as a Duration
    pub fn uptime(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Update the uptime gauge with the current uptime value
    pub fn update_uptime_gauge(&self) {
        if let Some(gauge) = self.uptime_gauge.get() {
            gauge.set(self.uptime().as_secs_f64());
        }
    }

    /// Get the health check path
    pub fn health_path(&self) -> &str {
        &self.health_path
    }

    /// Get the liveness check path
    pub fn live_path(&self) -> &str {
        &self.live_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::{Instance, TransportType};

    const ENDPOINT: &str = "generate";

    fn system_health(health_check_enabled: bool) -> SystemHealth {
        SystemHealth::new(
            HealthStatus::NotReady,
            // Deprecated and ignored in practice (see RuntimeConfig::from_settings),
            // so the realistic case is an empty vector.
            Vec::new(),
            health_check_enabled,
            "/health".to_string(),
            "/live".to_string(),
        )
    }

    fn instance() -> Instance {
        Instance {
            component: "backend".to_string(),
            endpoint: ENDPOINT.to_string(),
            namespace: "dynamo".to_string(),
            instance_id: 1,
            transport: TransportType::Tcp("127.0.0.1:0".to_string()),
            device_type: None,
            request_plane_codec: None,
        }
    }

    /// A worker that registers a health-check payload reports ready once its
    /// endpoint is registered, with the canary off.
    #[test]
    fn registered_target_makes_the_worker_ready_with_canary_off() {
        let health = system_health(false);
        health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        health.set_endpoint_registered(ENDPOINT);

        let (healthy, endpoints) = health.get_health_status();
        assert!(healthy, "a registered, ready endpoint must report healthy");
        assert_eq!(endpoints.get(ENDPOINT).map(String::as_str), Some("ready"));
    }

    /// Regression guard for the push-egress health-check bug.
    ///
    /// `health_check_targets` is populated ONLY by passing a
    /// `health_check_payload` to `serve_endpoint`. An earlier revision of the
    /// push-egress path skipped that payload to avoid the `start_with_registration`
    /// bail, on the theory that it merely disabled the canary. It does not: with
    /// the map empty, `get_health_status` stops consulting endpoint status at all
    /// and falls through to the process-wide `system_health`, which starts
    /// `NotReady` and which the TRT-LLM worker never sets. The endpoint is marked
    /// ready and the worker still reports 503 — on default settings, since this
    /// path does not depend on the canary being enabled.
    #[test]
    fn ready_endpoint_without_a_registered_target_still_reports_unhealthy() {
        let health = system_health(false);
        // No register_health_check_target: this is the "skip the payload" case.
        health.set_endpoint_registered(ENDPOINT);

        let (healthy, endpoints) = health.get_health_status();
        assert_eq!(
            endpoints.get(ENDPOINT).map(String::as_str),
            Some("ready"),
            "the endpoint itself is ready"
        );
        assert!(
            !healthy,
            "with no health-check target the endpoint's readiness is ignored and \
             the worker falls back to system_health (NotReady) — this is the 503"
        );
    }

    /// The fallthrough is only escapable by setting system health directly,
    /// which in this repo only the vLLM worker does.
    #[test]
    fn without_targets_health_tracks_system_health_only() {
        let mut health = system_health(false);
        health.set_endpoint_registered(ENDPOINT);
        assert!(!health.get_health_status().0);

        health.set_health_status(HealthStatus::Ready);
        assert!(
            health.get_health_status().0,
            "with no targets, system_health alone decides"
        );
    }

    /// With the canary on, endpoint registration deliberately does NOT mark the
    /// endpoint ready — the canary does, after verifying a real generation. A
    /// push endpoint therefore needs a locally registered engine for the canary
    /// to dispatch to, which is why `serve_endpoint` registers a pull engine
    /// alongside the push ingress.
    #[test]
    fn canary_enabled_withholds_ready_until_verified() {
        let health = system_health(true);
        health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        health.set_endpoint_registered(ENDPOINT);

        assert!(
            !health.get_health_status().0,
            "canary must verify before the worker reports ready"
        );

        health.set_endpoint_health_status(ENDPOINT, HealthStatus::Ready);
        assert!(
            health.get_health_status().0,
            "after the canary marks it ready the worker is healthy"
        );
    }

    /// An endpoint that restarts under the same subject re-registers. The second
    /// registration must take effect — the canary has to probe the process that is
    /// actually serving now, with the payload that incarnation asked for.
    #[test]
    fn re_registration_installs_the_restarts_target_and_withholds_ready() {
        let health = system_health(true);
        health.register_health_check_target(
            ENDPOINT,
            instance(),
            serde_json::json!({"generation": "first"}),
        );
        health.set_endpoint_health_status(ENDPOINT, HealthStatus::Ready);
        assert!(health.get_health_status().0);

        let mut restarted = instance();
        restarted.instance_id = 2;
        health.register_health_check_target(
            ENDPOINT,
            restarted,
            serde_json::json!({"generation": "second"}),
        );

        let target = health
            .get_health_check_target(ENDPOINT)
            .expect("the restart is the registered target");
        assert_eq!(target.payload, serde_json::json!({"generation": "second"}));
        assert_eq!(target.instance.instance_id, 2);
        assert!(
            !health.get_health_status().0,
            "the previous incarnation's canary verdict must not carry over to the restart"
        );
    }

    /// The health check manager learns about endpoints through the registration
    /// channel, so a restart has to be announced there too — otherwise the canary
    /// keeps probing on behalf of an endpoint that is gone and never re-arms.
    #[test]
    fn re_registration_is_announced_to_the_health_check_manager() {
        let health = system_health(true);
        let mut rx = health
            .take_new_endpoint_receiver()
            .expect("the receiver is available before the manager takes it");

        health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));

        assert_eq!(rx.try_recv().ok().as_deref(), Some(ENDPOINT));
        assert_eq!(
            rx.try_recv().ok().as_deref(),
            Some(ENDPOINT),
            "the restart must reach the manager as well as the first registration"
        );
    }

    /// A target left behind for an endpoint that stopped keeps the worker unhealthy
    /// forever: it is never probed, so it never becomes ready, and worker health is
    /// the conjunction over registered targets.
    #[test]
    fn deregistering_a_stopped_endpoint_releases_the_worker() {
        let mut health = system_health(true);
        health.set_health_status(HealthStatus::Ready);
        health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        assert!(
            !health.get_health_status().0,
            "an unverified target holds the worker unhealthy"
        );

        health.deregister_health_check_target(ENDPOINT);

        assert!(health.get_health_check_target(ENDPOINT).is_none());
        assert!(
            health
                .get_endpoint_health_check_notifier(ENDPOINT)
                .is_none(),
            "the notifier is endpoint-scoped state and goes with the target"
        );
        assert!(health.get_endpoint_health_status(ENDPOINT).is_none());
        let (healthy, endpoints) = health.get_health_status();
        assert!(
            healthy,
            "with the stopped endpoint's target gone the worker is judged on what remains"
        );
        assert!(!endpoints.contains_key(ENDPOINT));
    }
}
