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

#[derive(Clone, Debug)]
struct RegisteredHealthCheckTarget {
    target: HealthCheckTarget,
    registration: u64,
    notifier: Arc<tokio::sync::Notify>,
}

/// The target, notifier, and identity for one canary registration.
#[derive(Clone)]
pub struct CanaryHandles {
    pub target: HealthCheckTarget,
    pub notifier: Arc<tokio::sync::Notify>,
    pub registration: ProbedRegistration,
}

/// Identifies the registration a canary verdict describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbedRegistration(u64);

/// Identifies one health-check registration for later release.
#[derive(Debug)]
pub struct HealthCheckRegistration {
    subject: String,
    registration: u64,
}

/// Current Health Status
/// If use_endpoint_health_status is set then
/// initialize the endpoint_health hashmap to the
/// starting health status
#[derive(Clone)]
pub struct SystemHealth {
    system_health: HealthStatus,
    endpoint_health: Arc<std::sync::RwLock<HashMap<String, HealthStatus>>>,
    /// Registrations by subject in insertion order; the last entry is current.
    health_check_targets: Arc<std::sync::RwLock<HashMap<String, Vec<RegisteredHealthCheckTarget>>>>,
    /// Supplies unique registration identities.
    next_registration: Arc<std::sync::atomic::AtomicU64>,
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
            next_registration: Arc::new(std::sync::atomic::AtomicU64::new(0)),
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

    /// Record a verdict only if the probed registration is still current.
    pub fn set_endpoint_health_status_for(
        &self,
        endpoint: &str,
        probed: ProbedRegistration,
        status: HealthStatus,
    ) -> bool {
        // Hold both locks so release cannot race the status write.
        let targets = self.health_check_targets.read().unwrap();
        let current = targets
            .get(endpoint)
            .and_then(|outstanding| outstanding.last())
            .map(|registered| registered.registration);
        if current != Some(probed.0) {
            tracing::debug!(
                "Discarding health check verdict for '{endpoint}': the registration it was \
                 issued against is no longer serving the subject"
            );
            return false;
        }
        self.endpoint_health
            .write()
            .unwrap()
            .insert(endpoint.to_string(), status);
        true
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
                    .all(|(endpoint_subject, _outstanding)| {
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

    /// Register a target and return the receipt required to release it.
    pub fn register_health_check_target(
        &self,
        endpoint_subject: &str,
        instance: component::Instance,
        payload: serde_json::Value,
    ) -> HealthCheckRegistration {
        let key = endpoint_subject.to_owned();
        let registration = self
            .next_registration
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let replaced = {
            let mut targets = self.health_check_targets.write().unwrap();
            let outstanding = targets.entry(key.clone()).or_default();
            let replaced = !outstanding.is_empty();
            outstanding.push(RegisteredHealthCheckTarget {
                target: HealthCheckTarget { instance, payload },
                registration,
                notifier: Arc::new(tokio::sync::Notify::new()),
            });
            replaced
        };

        if replaced {
            tracing::debug!("Re-registering health check for endpoint '{key}'; replacing target.");
        }

        // Without a canary, no later probe can lift a re-exposed target from NotReady.
        {
            let mut endpoint_health = self.endpoint_health.write().unwrap();
            if replaced && self.health_check_enabled {
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

        HealthCheckRegistration {
            subject: key,
            registration,
        }
    }

    /// Release one target registration without disturbing newer registrations.
    pub fn release_health_check_target(&self, registration: HealthCheckRegistration) {
        let HealthCheckRegistration {
            subject,
            registration,
        } = registration;

        let (subject_vacated, re_exposed) = {
            let mut targets = self.health_check_targets.write().unwrap();
            let Some(outstanding) = targets.get_mut(&subject) else {
                return;
            };
            let Some(position) = outstanding
                .iter()
                .position(|registered| registered.registration == registration)
            else {
                return;
            };
            let was_current = position + 1 == outstanding.len();
            outstanding.remove(position);
            if outstanding.is_empty() {
                targets.remove(&subject);
                (true, false)
            } else {
                (false, was_current)
            }
        };

        if !subject_vacated {
            if re_exposed && self.health_check_enabled {
                self.endpoint_health
                    .write()
                    .unwrap()
                    .insert(subject.clone(), HealthStatus::NotReady);
            }
            tracing::debug!(
                "Released one health check registration for endpoint '{subject}'; other \
                 registrations still hold the subject."
            );
            return;
        }

        // Configured endpoint names remain visible after their final target is released.
        {
            let mut endpoint_health = self.endpoint_health.write().unwrap();
            if self
                .use_endpoint_health_status
                .iter()
                .any(|configured| configured == &subject)
            {
                endpoint_health.insert(subject.clone(), HealthStatus::NotReady);
            } else {
                endpoint_health.remove(&subject);
            }
        }
        tracing::debug!("Deregistered health check target for endpoint '{subject}'");
    }

    /// Get all health check targets
    pub fn get_health_check_targets(&self) -> Vec<(String, HealthCheckTarget)> {
        let targets = self.health_check_targets.read().unwrap();
        targets
            .iter()
            .filter_map(|(subject, outstanding)| {
                outstanding
                    .last()
                    .map(|registered| (subject.clone(), registered.target.clone()))
            })
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

    /// Get the current health-check target for an endpoint.
    pub fn get_health_check_target(&self, endpoint: &str) -> Option<HealthCheckTarget> {
        let targets = self.health_check_targets.read().unwrap();
        targets
            .get(endpoint)
            .and_then(|outstanding| outstanding.last())
            .map(|registered| registered.target.clone())
    }

    /// Get the endpoint health status (Ready/NotReady)
    pub fn get_endpoint_health_status(&self, endpoint: &str) -> Option<HealthStatus> {
        let endpoint_health = self.endpoint_health.read().unwrap();
        endpoint_health.get(endpoint).cloned()
    }

    /// Get the notifier belonging to the current target registration.
    pub fn get_endpoint_health_check_notifier(
        &self,
        endpoint_subject: &str,
    ) -> Option<Arc<tokio::sync::Notify>> {
        let targets = self.health_check_targets.read().unwrap();
        targets
            .get(endpoint_subject)
            .and_then(|outstanding| outstanding.last())
            .map(|registered| registered.notifier.clone())
    }

    /// Read the current target, notifier, and identity under one lock.
    pub fn get_canary_handles(&self, endpoint_subject: &str) -> Option<CanaryHandles> {
        let targets = self.health_check_targets.read().unwrap();
        targets
            .get(endpoint_subject)
            .and_then(|outstanding| outstanding.last())
            .map(|registered| CanaryHandles {
                target: registered.target.clone(),
                notifier: registered.notifier.clone(),
                registration: ProbedRegistration(registered.registration),
            })
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
        configured_system_health(health_check_enabled, Vec::new())
    }

    fn configured_system_health(
        health_check_enabled: bool,
        use_endpoint_health_status: Vec<String>,
    ) -> SystemHealth {
        SystemHealth::new(
            HealthStatus::NotReady,
            use_endpoint_health_status,
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

    #[test]
    fn releasing_a_stopped_endpoints_registration_releases_the_worker() {
        let mut health = system_health(true);
        health.set_health_status(HealthStatus::Ready);
        let registration =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        assert!(
            !health.get_health_status().0,
            "an unverified target holds the worker unhealthy"
        );

        health.release_health_check_target(registration);

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
        assert!(!health.has_health_check_targets());
        assert!(health.get_health_check_targets().is_empty());
    }

    #[test]
    fn releasing_a_displaced_registration_leaves_the_live_endpoint_alone() {
        let health = system_health(true);
        let displaced = health.register_health_check_target(
            ENDPOINT,
            instance(),
            serde_json::json!({"generation": "first"}),
        );
        let mut live_instance = instance();
        live_instance.instance_id = 2;
        health.register_health_check_target(
            ENDPOINT,
            live_instance,
            serde_json::json!({"generation": "second"}),
        );
        health.set_endpoint_health_status(ENDPOINT, HealthStatus::Ready);

        health.release_health_check_target(displaced);

        let target = health
            .get_health_check_target(ENDPOINT)
            .expect("the live registration still holds the subject");
        assert_eq!(target.payload, serde_json::json!({"generation": "second"}));
        assert_eq!(target.instance.instance_id, 2);
        assert!(
            health
                .get_endpoint_health_check_notifier(ENDPOINT)
                .is_some(),
            "the live endpoint's handler still signals through this notifier"
        );
        assert_eq!(
            health.get_endpoint_health_status(ENDPOINT),
            Some(HealthStatus::Ready),
            "the live endpoint's canary verdict must survive the other one's release"
        );
        assert!(health.get_health_status().0);
    }

    #[test]
    fn releasing_the_probed_registration_withholds_ready_from_the_one_it_re_exposes() {
        let mut health = system_health(true);
        health.set_health_status(HealthStatus::Ready);
        let earlier = health.register_health_check_target(
            ENDPOINT,
            instance(),
            serde_json::json!({"generation": "first"}),
        );
        let mut probed_instance = instance();
        probed_instance.instance_id = 2;
        let probed = health.register_health_check_target(
            ENDPOINT,
            probed_instance,
            serde_json::json!({"generation": "second"}),
        );
        health.set_endpoint_health_status(ENDPOINT, HealthStatus::Ready);
        assert!(health.get_health_status().0);

        health.release_health_check_target(probed);

        let target = health
            .get_health_check_target(ENDPOINT)
            .expect("the earlier registration is outstanding and takes the subject back");
        assert_eq!(target.payload, serde_json::json!({"generation": "first"}));
        assert_eq!(
            health.get_endpoint_health_status(ENDPOINT),
            Some(HealthStatus::NotReady),
            "the departing registration's verdict says nothing about the re-exposed target"
        );
        assert!(
            !health.get_health_status().0,
            "the worker must not report ready over a target no canary has probed"
        );

        health.set_endpoint_health_status(ENDPOINT, HealthStatus::Ready);
        assert!(health.get_health_status().0);
        health.release_health_check_target(earlier);
        assert!(health.get_health_check_target(ENDPOINT).is_none());
    }

    #[test]
    fn releasing_out_of_order_cannot_reinstate_a_released_registration() {
        let mut health = system_health(true);
        health.set_health_status(HealthStatus::Ready);

        let first = health.register_health_check_target(
            ENDPOINT,
            instance(),
            serde_json::json!({"generation": "first"}),
        );
        let mut second_instance = instance();
        second_instance.instance_id = 2;
        let second = health.register_health_check_target(
            ENDPOINT,
            second_instance,
            serde_json::json!({"generation": "second"}),
        );

        health.release_health_check_target(first);
        health.release_health_check_target(second);

        assert!(
            health.get_health_check_target(ENDPOINT).is_none(),
            "both registrations are released, so nothing may hold the subject"
        );
        assert!(
            health
                .get_endpoint_health_check_notifier(ENDPOINT)
                .is_none()
        );
        assert!(health.get_endpoint_health_status(ENDPOINT).is_none());
        assert!(
            health.get_health_status().0,
            "a phantom target would hold the worker unhealthy forever"
        );
    }

    #[test]
    fn releasing_in_order_also_clears_the_subject() {
        let health = system_health(true);
        let first =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        let second =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));

        health.release_health_check_target(second);
        assert!(
            health.get_health_check_target(ENDPOINT).is_some(),
            "the older registration is still outstanding and takes the subject back"
        );

        health.release_health_check_target(first);
        assert!(health.get_health_check_target(ENDPOINT).is_none());
        assert!(
            health
                .get_endpoint_health_check_notifier(ENDPOINT)
                .is_none()
        );
        assert!(health.get_endpoint_health_status(ENDPOINT).is_none());
    }

    #[test]
    fn a_repeated_release_does_not_disturb_a_later_registration() {
        let health = system_health(true);
        let registration =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        let subject = registration.subject.clone();
        let id = registration.registration;
        health.release_health_check_target(registration);

        health.register_health_check_target(
            ENDPOINT,
            instance(),
            serde_json::json!({"generation": "restart"}),
        );
        health.release_health_check_target(HealthCheckRegistration {
            subject,
            registration: id,
        });

        let target = health
            .get_health_check_target(ENDPOINT)
            .expect("the restart's registration is untouched by the stale release");
        assert_eq!(target.payload, serde_json::json!({"generation": "restart"}));
    }

    #[test]
    fn a_re_exposed_registration_keeps_its_readiness_when_the_canary_is_off() {
        let health = system_health(false);
        let live = health.register_health_check_target(
            ENDPOINT,
            instance(),
            serde_json::json!({"generation": "live"}),
        );
        health.set_endpoint_registered(ENDPOINT);
        let mut departing_instance = instance();
        departing_instance.instance_id = 2;
        let departing = health.register_health_check_target(
            ENDPOINT,
            departing_instance,
            serde_json::json!({"generation": "departing"}),
        );
        health.set_endpoint_registered(ENDPOINT);
        assert!(health.get_health_status().0);

        health.release_health_check_target(departing);

        assert_eq!(
            health.get_endpoint_health_status(ENDPOINT),
            Some(HealthStatus::Ready),
            "no canary will ever lift a reset, so the still-serving endpoint keeps its verdict"
        );
        assert!(
            health.get_health_status().0,
            "a worker must not be pulled from rotation by another endpoint's departure"
        );

        health.release_health_check_target(live);
    }

    #[test]
    fn a_rolled_back_start_leaves_the_endpoint_it_displaced_ready_when_the_canary_is_off() {
        let health = system_health(false);
        let live = health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        health.set_endpoint_registered(ENDPOINT);

        let rolled_back =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        health.release_health_check_target(rolled_back);

        assert_eq!(
            health.get_endpoint_health_status(ENDPOINT),
            Some(HealthStatus::Ready),
            "the rollback must leave the process as it found it"
        );
        assert!(health.get_health_status().0);

        health.release_health_check_target(live);
    }

    #[test]
    fn a_re_exposed_registration_is_still_reset_when_the_canary_is_on() {
        let health = system_health(true);
        let live = health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        let departing =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        health.set_endpoint_health_status(ENDPOINT, HealthStatus::Ready);

        health.release_health_check_target(departing);

        assert_eq!(
            health.get_endpoint_health_status(ENDPOINT),
            Some(HealthStatus::NotReady)
        );
        health.release_health_check_target(live);
    }

    #[test]
    fn a_configured_endpoint_stays_in_the_status_map_after_its_last_release() {
        let health = configured_system_health(false, vec![ENDPOINT.to_string()]);
        let registration =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        health.set_endpoint_registered(ENDPOINT);
        assert!(health.get_health_status().0);

        health.release_health_check_target(registration);

        assert_eq!(
            health.get_endpoint_health_status(ENDPOINT),
            Some(HealthStatus::NotReady),
            "a configured endpoint whose target is gone is not ready, not unknown"
        );
        let (healthy, endpoints) = health.get_health_status();
        assert!(
            !healthy,
            "a configured endpoint with no target cannot be ready"
        );
        assert_eq!(
            endpoints.get(ENDPOINT).map(String::as_str),
            Some("notready")
        );
    }

    #[test]
    fn an_unconfigured_endpoint_leaves_the_status_map_after_its_last_release() {
        let health = system_health(false);
        let registration =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));

        health.release_health_check_target(registration);

        assert_eq!(health.get_endpoint_health_status(ENDPOINT), None);
        assert!(!health.get_health_status().1.contains_key(ENDPOINT));
    }

    #[test]
    fn overlapping_registrations_get_their_own_notifiers() {
        let health = system_health(true);
        let earlier =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        let earlier_notifier = health
            .get_endpoint_health_check_notifier(ENDPOINT)
            .expect("the first registration's handler needs a notifier");

        let later =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        let later_notifier = health
            .get_endpoint_health_check_notifier(ENDPOINT)
            .expect("the displacing registration's handler needs one of its own");
        assert!(
            !Arc::ptr_eq(&earlier_notifier, &later_notifier),
            "a restart's handler must not be handed the outgoing incarnation's notifier"
        );

        health.release_health_check_target(later);
        let re_exposed = health
            .get_endpoint_health_check_notifier(ENDPOINT)
            .expect("the re-exposed registration still has its notifier");
        assert!(
            Arc::ptr_eq(&re_exposed, &earlier_notifier),
            "the re-exposed registration is signalled on the notifier its own handler holds"
        );

        health.release_health_check_target(earlier);
        assert!(
            health
                .get_endpoint_health_check_notifier(ENDPOINT)
                .is_none()
        );
    }

    #[test]
    fn a_verdict_for_a_released_registration_is_discarded() {
        let health = system_health(true);
        let probed =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        let handles = health
            .get_canary_handles(ENDPOINT)
            .expect("the registration was just installed");

        health.release_health_check_target(probed);
        let successor =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));

        assert!(
            !health.set_endpoint_health_status_for(
                ENDPOINT,
                handles.registration,
                HealthStatus::Ready
            ),
            "a verdict about a released registration must not be filed"
        );
        assert!(
            !health.get_health_status().0,
            "the successor must still wait for a canary of its own"
        );

        let current = health
            .get_canary_handles(ENDPOINT)
            .expect("the successor is serving");
        assert!(health.set_endpoint_health_status_for(
            ENDPOINT,
            current.registration,
            HealthStatus::Ready
        ));
        assert!(health.get_health_status().0);
        health.release_health_check_target(successor);
    }

    #[test]
    fn a_verdict_for_the_serving_registration_is_filed() {
        let health = system_health(true);
        let displaced =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        let serving =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));

        let handles = health
            .get_canary_handles(ENDPOINT)
            .expect("the displacing registration is serving");
        assert!(health.set_endpoint_health_status_for(
            ENDPOINT,
            handles.registration,
            HealthStatus::Ready
        ));
        assert!(health.get_health_status().0);

        health.release_health_check_target(serving);
        health.release_health_check_target(displaced);
    }

    #[test]
    fn a_verdict_for_a_vacated_subject_is_discarded() {
        let health = system_health(true);
        let registration =
            health.register_health_check_target(ENDPOINT, instance(), serde_json::json!({}));
        let handles = health
            .get_canary_handles(ENDPOINT)
            .expect("the registration was just installed");
        health.release_health_check_target(registration);

        assert!(!health.set_endpoint_health_status_for(
            ENDPOINT,
            handles.registration,
            HealthStatus::Ready
        ));
        assert!(health.get_canary_handles(ENDPOINT).is_none());
    }
}
