// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{DistributedRuntime, HealthStatus, SystemHealth};
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::{MissedTickBehavior, interval};
use tracing::{debug, error, info, warn};

/// Configuration for health check behavior
pub struct HealthCheckConfig {
    /// Wait time before sending canary health checks (when no activity)
    pub canary_wait_time: Duration,
    /// How long since last response before considering unhealthy
    pub respond_stale_threshold: Duration,
    /// Timeout for health check requests
    pub request_timeout: Duration,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            canary_wait_time: Duration::from_secs(crate::config::DEFAULT_CANARY_WAIT_TIME_SECS),
            respond_stale_threshold: Duration::from_secs(
                crate::config::DEFAULT_HEALTH_CHECK_RESPOND_STALE_THRESHOLD_SECS,
            ),
            request_timeout: Duration::from_secs(
                crate::config::DEFAULT_HEALTH_CHECK_REQUEST_TIMEOUT_SECS,
            ),
        }
    }
}

/// Health check manager that monitors endpoint health
pub struct HealthCheckManager {
    drt: DistributedRuntime,
    config: HealthCheckConfig,
    /// NATS reply inbox for health check responses
    health_check_inbox: String,
    /// Track pending health check requests
    /// Maps: reply_subject -> (endpoint_subject, sent_time)
    pending_health_checks: Arc<Mutex<HashMap<String, (String, Instant)>>>,
}

impl HealthCheckManager {
    pub fn new(drt: DistributedRuntime, config: HealthCheckConfig) -> Self {
        // Generate unique inbox for this health check manager
        let health_check_inbox = format!("_INBOX.health_check.{}", uuid::Uuid::new_v4());

        Self {
            drt,
            config,
            health_check_inbox,
            pending_health_checks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start the health check with timed task and activity-based reset
    pub async fn start(self: Arc<Self>, activity_notifier: Arc<tokio::sync::Notify>) {
        // Start the response listener in a separate task
        let _listener_handle = self.clone().start_response_listener();

        // Start the timeout monitor in a separate task
        let _timeout_monitor_handle = self.clone().start_timeout_monitor();

        info!(
            "Health check manager started with canary_wait_time: {:?}, respond_stale_threshold: {:?}",
            self.config.canary_wait_time, self.config.respond_stale_threshold
        );

        loop {
            // Wait for either timeout or activity notification
            tokio::select! {
                _ = tokio::time::sleep(self.config.canary_wait_time) => {
                    // Timer expired - send canary health check.
                    debug!("No activity detected, sending health check");
                    if let Err(e) = self.check_all_endpoints().await {
                        error!("Health check error: {}", e);
                    }
                }
                _ = activity_notifier.notified() => {
                    // Activity detected, reset timer
                    debug!("Activity detected, resetting health check timer");
                }
            }
        }
    }

    /// Start a separate async loop that monitors for timeouts
    fn start_timeout_monitor(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            // Check for timeouts every 100ms to ensure we catch them quickly
            let mut interval = tokio::time::interval(Duration::from_millis(100));

            loop {
                interval.tick().await;

                // Check all pending health checks for timeouts
                let mut timed_out = Vec::new();
                {
                    let pending = self.pending_health_checks.lock().unwrap();
                    for (reply_subject, (endpoint, sent_at)) in pending.iter() {
                        if sent_at.elapsed() > self.config.request_timeout {
                            timed_out.push((reply_subject.clone(), endpoint.clone()));
                        }
                    }
                }

                // Process timeouts
                for (reply_subject, endpoint) in timed_out {
                    // Remove from pending
                    {
                        let mut pending = self.pending_health_checks.lock().unwrap();
                        pending.remove(&reply_subject);
                    }

                    // Mark endpoint as unhealthy
                    warn!(
                        "Health check timeout for endpoint {}, marking as NotReady",
                        endpoint
                    );
                    self.drt
                        .system_health
                        .lock()
                        .unwrap()
                        .set_endpoint_health_status(&endpoint, HealthStatus::NotReady);
                }
            }
        })
    }

    /// Start a separate async loop that listens for health check responses
    fn start_response_listener(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            // Subscribe to our inbox pattern
            let nats_client = self.drt.nats_client();
            let inbox_pattern = format!("{}.>", self.health_check_inbox);

            match nats_client.client().subscribe(inbox_pattern.clone()).await {
                Ok(mut subscriber) => {
                    // Listen for responses indefinitely
                    while let Some(msg) = subscriber.next().await {
                        // Extract endpoint from our tracking map using the full subject
                        let reply_subject = msg.subject.to_string();

                        let endpoint_info = {
                            let mut pending = self.pending_health_checks.lock().unwrap();
                            pending.remove(&reply_subject)
                        };

                        if let Some((endpoint, sent_time)) = endpoint_info {
                            let elapsed = sent_time.elapsed();
                            info!(
                                "Received health check response for endpoint {} ({} bytes, {:?} elapsed, reply_subject: {})",
                                endpoint,
                                msg.payload.len(),
                                elapsed,
                                reply_subject
                            );

                            // Mark endpoint as healthy
                            self.drt
                                .system_health
                                .lock()
                                .unwrap()
                                .set_endpoint_health_status(&endpoint, crate::HealthStatus::Ready);
                        } else {
                            warn!(
                                "Received LATE/UNKNOWN response on subject: {} (not in pending_health_checks)",
                                reply_subject
                            );
                        }
                    }

                    warn!("Response listener subscription ended");
                }
                Err(e) => {
                    error!("Failed to subscribe to health check responses: {}", e);
                }
            }
        })
    }

    /// Check health of all registered endpoints
    pub async fn check_all_endpoints(&self) -> anyhow::Result<()> {
        let endpoints_to_check: Vec<(String, serde_json::Value)> = {
            let system_health = self.drt.system_health.lock().unwrap();
            system_health.get_health_check_payloads()
        };

        for (endpoint_subject, health_check_payload) in endpoints_to_check {
            self.check_endpoint_health(&endpoint_subject, &health_check_payload)
                .await?;
        }

        Ok(())
    }

    /// Check health of a single endpoint
    ///
    /// - If health check already pending -> skip (timeout monitor will handle it)
    /// - If no pending request -> send a new health check
    pub async fn check_endpoint_health(
        &self,
        endpoint_subject: &str,
        health_check_payload: &serde_json::Value,
    ) -> anyhow::Result<()> {
        // Check if we have a pending health check for this endpoint
        let has_pending = {
            let pending = self.pending_health_checks.lock().unwrap();
            pending
                .iter()
                .any(|(_, (endpoint, _))| endpoint == endpoint_subject)
        };

        if has_pending {
            // Already have a pending request - the timeout monitor will handle it
            debug!("Health check already pending for {}", endpoint_subject);
        } else {
            // No pending request - send a new health check
            match self
                .send_health_check_request(endpoint_subject, health_check_payload)
                .await
            {
                Ok(()) => {
                    info!("Health check request sent to {}", endpoint_subject);
                }
                Err(e) => {
                    error!("Failed to send health check to {}: {}", endpoint_subject, e);
                    // Mark as unhealthy if we can't even send the health check
                    let system_health = self.drt.system_health.lock().unwrap();
                    system_health
                        .set_endpoint_health_status(endpoint_subject, HealthStatus::NotReady);
                }
            }
        }

        Ok(())
    }

    /// Send a health check request to an endpoint (fire and forget)
    async fn send_health_check_request(
        &self,
        endpoint_subject: &str,
        payload: &serde_json::Value,
    ) -> anyhow::Result<()> {
        // Create a unique reply subject for this health check
        let reply_subject = format!("{}.{}", self.health_check_inbox, uuid::Uuid::new_v4());

        debug!(
            "Sending health check to {} with reply_subject: {} and payload: {}",
            endpoint_subject, reply_subject, payload
        );

        // Track this pending health check with timestamp
        {
            let mut pending = self.pending_health_checks.lock().unwrap();
            pending.insert(
                reply_subject.clone(),
                (endpoint_subject.to_string(), Instant::now()),
            );
        }

        // Send the request with our reply-to address (fire and forget)
        let nats_client = self.drt.nats_client();
        let payload_bytes = serde_json::to_vec(payload)?;
        let client = nats_client.client().clone();
        let subject = endpoint_subject.to_string();

        tokio::spawn(async move {
            // Publish the message with a reply-to subject
            // The endpoint will send its response to our reply_subject
            if let Err(e) = client
                .publish_with_reply(subject.clone(), reply_subject.clone(), payload_bytes.into())
                .await
            {
                error!("Failed to publish health check to {}: {}", subject, e);
            }
        });

        Ok(())
    }
}

/// Start health check manager for the distributed runtime
/// Returns the join handle and the activity notifier
pub fn start_health_check_manager(
    drt: DistributedRuntime,
    config: Option<HealthCheckConfig>,
) -> (tokio::task::JoinHandle<()>, Arc<tokio::sync::Notify>) {
    let config = config.unwrap_or_default();
    let manager = Arc::new(HealthCheckManager::new(drt, config));
    let activity_notifier = Arc::new(tokio::sync::Notify::new());
    let notifier_clone = activity_notifier.clone();

    // Spawn the health check loop
    let handle = tokio::spawn(async move {
        manager.start(notifier_clone).await;
    });

    (handle, activity_notifier)
}

/// Get health check status for all endpoints
pub async fn get_health_check_status(
    drt: &DistributedRuntime,
    threshold: Duration,
) -> anyhow::Result<serde_json::Value> {
    // Get endpoints list from SystemHealth
    let endpoint_subjects: Vec<String> = {
        let system_health = drt.system_health.lock().unwrap();
        system_health.get_health_check_endpoints()
    };

    let mut endpoint_statuses = HashMap::new();

    // Check each endpoint
    {
        let system_health = drt.system_health.lock().unwrap();
        for endpoint_subject in &endpoint_subjects {
            let has_recent_response =
                system_health.has_responded_recently(endpoint_subject, threshold);

            let last_response = system_health
                .get_last_response_time(endpoint_subject)
                .map(|t| t.elapsed().as_secs());

            endpoint_statuses.insert(
                endpoint_subject.clone(),
                serde_json::json!({
                    "healthy": has_recent_response,
                    "seconds_since_last_response": last_response,
                }),
            );
        }
    }

    let overall_healthy = endpoint_statuses
        .values()
        .all(|v| v["healthy"].as_bool().unwrap_or(false));

    Ok(serde_json::json!({
        "status": if overall_healthy { "ready" } else { "notready" },
        "endpoints_checked": endpoint_subjects.len(),
        "endpoint_statuses": endpoint_statuses,
        "check_threshold_seconds": threshold.as_secs(),
    }))
}

// ===============================
// Integration Tests (require DRT)
// ===============================
#[cfg(all(test, feature = "integration"))]
mod integration_tests {
    use super::*;
    use crate::HealthStatus;
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn test_initialization() {
        let drt = create_test_drt_async().await;

        let canary_wait_time = Duration::from_secs(5);
        let respond_stale_threshold = Duration::from_secs(2);
        let request_timeout = Duration::from_secs(3);

        let config = HealthCheckConfig {
            canary_wait_time: canary_wait_time,
            respond_stale_threshold: respond_stale_threshold,
            request_timeout: request_timeout,
        };

        let manager = HealthCheckManager::new(drt.clone(), config);

        assert_eq!(manager.config.canary_wait_time, canary_wait_time);
        assert_eq!(
            manager.config.respond_stale_threshold,
            respond_stale_threshold
        );
        assert_eq!(manager.config.request_timeout, request_timeout);

        assert!(Arc::ptr_eq(&manager.drt.system_health, &drt.system_health));
    }

    #[tokio::test]
    async fn test_payload_registration() {
        let drt = create_test_drt_async().await;

        let endpoint = "test.endpoint";
        let payload = serde_json::json!({
            "prompt": "test",
            "_health_check": true
        });

        drt.system_health
            .lock()
            .unwrap()
            .register_health_check_payload(endpoint, payload.clone());

        let retrieved = drt
            .system_health
            .lock()
            .unwrap()
            .get_health_check_payload(endpoint);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), payload);

        // Verify endpoint appears in the list
        let endpoints = drt
            .system_health
            .lock()
            .unwrap()
            .get_health_check_endpoints();
        assert!(endpoints.contains(&endpoint.to_string()));
    }

    #[tokio::test]
    async fn test_poll_all_endpoints() {
        let drt = create_test_drt_async().await;

        for i in 0..3 {
            let endpoint = format!("test.endpoint.{}", i);
            let payload = serde_json::json!({
                "prompt": format!("test{}", i),
                "_health_check": true
            });
            drt.system_health
                .lock()
                .unwrap()
                .register_health_check_payload(&endpoint, payload);
        }

        let config = HealthCheckConfig {
            canary_wait_time: Duration::from_secs(5),
            respond_stale_threshold: Duration::from_secs(2),
            request_timeout: Duration::from_secs(1),
        };

        let manager = HealthCheckManager::new(drt.clone(), config);
        manager.check_all_endpoints().await.unwrap();

        // Verify all endpoints have pending health checks
        let pending = manager.pending_health_checks.lock().unwrap();
        // Should have 3 pending requests (one for each endpoint)
        assert_eq!(pending.len(), 3);
        // Check that all endpoints are represented in pending requests
        let endpoints: Vec<String> = pending
            .values()
            .map(|(endpoint, _)| endpoint.clone())
            .collect();
        assert!(endpoints.contains(&"test.endpoint.0".to_string()));
        assert!(endpoints.contains(&"test.endpoint.1".to_string()));
        assert!(endpoints.contains(&"test.endpoint.2".to_string()));
    }

    #[tokio::test]
    async fn test_response_handling() {
        let drt = create_test_drt_async().await;

        let endpoint = "test.endpoint";

        // Register endpoint
        let payload = serde_json::json!({
            "prompt": "test",
            "_health_check": true
        });
        drt.system_health
            .lock()
            .unwrap()
            .register_health_check_payload(endpoint, payload.clone());
        // Simulate a response by updating the last response time
        drt.system_health
            .lock()
            .unwrap()
            .update_last_response_time(endpoint);

        // Status should be Ready
        let status = drt
            .system_health
            .lock()
            .unwrap()
            .get_endpoint_health_status(endpoint);
        assert_eq!(status, Some(HealthStatus::Ready));
    }

    #[tokio::test]
    async fn test_request_timeout() {
        let drt = create_test_drt_async().await;

        let endpoint = "test.endpoint.timeout";
        let payload = serde_json::json!({
            "prompt": "test",
            "_health_check": true
        });
        drt.system_health
            .lock()
            .unwrap()
            .register_health_check_payload(endpoint, payload.clone());

        // Initially, the endpoint should be Ready (default after registration)
        let initial_status = drt
            .system_health
            .lock()
            .unwrap()
            .get_endpoint_health_status(endpoint);
        assert_eq!(initial_status, Some(HealthStatus::Ready));

        let config = HealthCheckConfig {
            canary_wait_time: Duration::from_secs(25),
            respond_stale_threshold: Duration::from_secs(2),
            request_timeout: Duration::from_secs(1),
        };

        let manager = HealthCheckManager::new(drt.clone(), config);

        // First check - will send health check since no pending request exists
        manager
            .check_endpoint_health(endpoint, &payload)
            .await
            .unwrap();
        {
            let pending = manager.pending_health_checks.lock().unwrap();
            let has_pending = pending.values().any(|(ep, _)| ep == endpoint);
            assert!(
                has_pending,
                "Should send health check when no pending request exists"
            );
        }
        let status = drt
            .system_health
            .lock()
            .unwrap()
            .get_endpoint_health_status(endpoint);
        assert_eq!(
            status,
            Some(HealthStatus::Ready),
            "Should be Ready initially"
        );

        // Second check while request is pending - should NOT send another
        manager
            .check_endpoint_health(endpoint, &payload)
            .await
            .unwrap();

        // Verify no duplicate health check was sent
        {
            let pending = manager.pending_health_checks.lock().unwrap();
            // Should still have exactly 1 pending request
            let count = pending.values().filter(|(ep, _)| ep == endpoint).count();
            assert_eq!(
                count, 1,
                "Should not send duplicate health check while one is pending"
            );
        }
    }
}
