// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::DistributedRuntime;
use crate::config::HealthStatus;
use crate::engine::AsyncEngine;
use crate::pipeline::SingleIn;
use crate::protocols::maybe_error::MaybeError;
use futures::StreamExt;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Configuration for health check behavior
pub struct HealthCheckConfig {
    /// Wait time before sending canary health checks (when no activity)
    pub canary_wait_time: Duration,
    /// Timeout for health check requests
    pub request_timeout: Duration,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            canary_wait_time: Duration::from_secs(crate::config::DEFAULT_CANARY_WAIT_TIME_SECS),
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
    /// Track per-endpoint health check tasks
    /// Maps: endpoint_subject -> task_handle
    endpoint_tasks: Arc<Mutex<HashMap<String, JoinHandle<()>>>>,
}

impl HealthCheckManager {
    pub fn new(drt: DistributedRuntime, config: HealthCheckConfig) -> Self {
        Self {
            drt,
            config,
            endpoint_tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start the health check manager by spawning per-endpoint monitoring tasks
    pub async fn start(self: Arc<Self>) -> anyhow::Result<()> {
        // Get all registered endpoints at startup
        let targets = self.drt.system_health().lock().get_health_check_targets();

        info!(
            "Starting health check tasks for {} endpoints with canary_wait_time: {:?}",
            targets.len(),
            self.config.canary_wait_time
        );

        // Spawn a health check task for each registered endpoint
        for (endpoint_subject, _target) in targets {
            self.spawn_endpoint_health_check_task(endpoint_subject);
        }

        // CRITICAL: Spawn a task to monitor for NEW endpoints registered after startup
        // This uses a channel-based approach to guarantee no lost notifications
        // Will return an error if the receiver has already been taken
        self.spawn_new_endpoint_monitor().await?;

        info!("HealthCheckManager started successfully with channel-based endpoint discovery");
        Ok(())
    }

    /// Spawn a dedicated health check task for a specific endpoint
    fn spawn_endpoint_health_check_task(self: &Arc<Self>, endpoint_subject: String) {
        let manager = self.clone();
        let canary_wait = self.config.canary_wait_time;
        let endpoint_subject_clone = endpoint_subject.clone();

        // Get the endpoint-specific notifier
        let notifier = self
            .drt
            .system_health()
            .lock()
            .get_endpoint_health_check_notifier(&endpoint_subject)
            .expect("Notifier should exist for registered endpoint");

        let task = tokio::spawn(async move {
            let endpoint_subject = endpoint_subject_clone;
            info!("Health check task started for: {}", endpoint_subject);

            loop {
                // Wait for either timeout or activity notification
                tokio::select! {
                    _ = tokio::time::sleep(canary_wait) => {
                        // Timeout - send health check for this specific endpoint
                        debug!("Canary timer expired for {}, sending health check", endpoint_subject);

                        // Get the health check payload for this endpoint
                        let target = manager.drt.system_health().lock().get_health_check_target(&endpoint_subject);

                        if let Some(target) = target {
                            if let Err(e) = manager.send_health_check_request(&endpoint_subject, &target.payload).await {
                                error!("Failed to send health check for {}: {}", endpoint_subject, e);
                            }
                        } else {
                            // This should never happen - targets are registered at startup and never removed
                            error!(
                                "CRITICAL: Health check target for {} disappeared unexpectedly! This indicates a bug. Stopping health check task.",
                                endpoint_subject
                            );
                            break;
                        }
                    }

                    _ = notifier.notified() => {
                        // Activity detected - reset timer for this endpoint only.
                        // A notification means push_handler successfully streamed
                        // a non-error response chunk, proving the engine is healthy.
                        debug!("Activity detected for {}, resetting health check timer", endpoint_subject);
                        manager.drt.system_health().lock().set_endpoint_health_status(
                            &endpoint_subject,
                            crate::config::HealthStatus::Ready,
                        );
                    }
                }
            }

            info!("Health check task for {} exiting", endpoint_subject);
        });

        // Store the task handle
        self.endpoint_tasks
            .lock()
            .insert(endpoint_subject.clone(), task);

        info!(
            "Spawned health check task for endpoint: {}",
            endpoint_subject
        );
    }

    /// Spawn a task to monitor for newly registered endpoints
    /// Returns an error if duplicate endpoints are detected, indicating a bug in the system
    async fn spawn_new_endpoint_monitor(self: &Arc<Self>) -> anyhow::Result<()> {
        let manager = self.clone();

        // Get the receiver (can only be taken once)
        let mut rx = manager
            .drt
            .system_health()
            .lock()
            .take_new_endpoint_receiver()
            .ok_or_else(|| {
                anyhow::anyhow!("Endpoint receiver already taken - this should only be called once")
            })?;

        tokio::spawn(async move {
            info!("Starting dynamic endpoint discovery monitor with channel-based notifications");

            while let Some(endpoint_subject) = rx.recv().await {
                debug!(
                    "Received endpoint registration via channel: {}",
                    endpoint_subject
                );

                let already_exists = {
                    let tasks = manager.endpoint_tasks.lock();
                    tasks.contains_key(&endpoint_subject)
                };

                if already_exists {
                    error!(
                        "CRITICAL: Received registration for endpoint '{}' that already has a health check task!",
                        endpoint_subject
                    );
                    break;
                }

                info!(
                    "Spawning health check task for new endpoint: {}",
                    endpoint_subject
                );
                manager.spawn_endpoint_health_check_task(endpoint_subject);
            }

            info!("Endpoint discovery monitor exiting - no new endpoints will be monitored!");
        });

        info!("Dynamic endpoint discovery monitor started");
        Ok(())
    }

    /// Send a health check request via the local endpoint registry (in-process).
    async fn send_health_check_request(
        &self,
        endpoint_subject: &str,
        payload: &serde_json::Value,
    ) -> anyhow::Result<()> {
        debug!(
            "Sending health check to {} via local registry",
            endpoint_subject
        );

        let engine = self
            .drt
            .local_endpoint_registry()
            .get(endpoint_subject)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Endpoint '{}' not found in local registry, engine may still be initializing",
                    endpoint_subject
                )
            })?;

        // Clone what we need for the spawned task
        let system_health = self.drt.system_health().clone();
        let endpoint_subject_owned = endpoint_subject.to_string();
        let payload = payload.clone();
        let timeout = self.config.request_timeout;

        // Spawn task to send health check and wait for response
        tokio::spawn(async move {
            let result = tokio::time::timeout(timeout, async {
                let request = SingleIn::new(payload);
                match engine.generate(request).await {
                    Ok(mut response_stream) => {
                        // Get the first response to verify endpoint is alive.
                        // Check for errors
                        let is_healthy = if let Some(response) = response_stream.next().await {
                            if let Some(error) = response.err() {
                                warn!(
                                    "Health check error response from {}: {:?}",
                                    endpoint_subject_owned, error
                                );
                                false
                            } else {
                                debug!("Health check successful for {}", endpoint_subject_owned);
                                true
                            }
                        } else {
                            warn!(
                                "Health check got no response from {}",
                                endpoint_subject_owned
                            );
                            false
                        };

                        tokio::spawn(async move {
                            // We need to consume the rest of the stream to avoid warnings on the frontend.
                            response_stream.for_each(|_| async {}).await;
                        });

                        // Update health status based on response
                        system_health.lock().set_endpoint_health_status(
                            &endpoint_subject_owned,
                            if is_healthy {
                                HealthStatus::Ready
                            } else {
                                HealthStatus::NotReady
                            },
                        );
                    }
                    Err(e) => {
                        error!(
                            "Health check request failed for {}: {}",
                            endpoint_subject_owned, e
                        );
                        system_health.lock().set_endpoint_health_status(
                            &endpoint_subject_owned,
                            HealthStatus::NotReady,
                        );
                    }
                }
            })
            .await;

            // Handle timeout
            if result.is_err() {
                warn!("Health check timeout for {}", endpoint_subject_owned);
                system_health
                    .lock()
                    .set_endpoint_health_status(&endpoint_subject_owned, HealthStatus::NotReady);
            }

            debug!("Health check completed for {}", endpoint_subject_owned);
        });

        Ok(())
    }
}

/// Start health check manager for the distributed runtime
pub async fn start_health_check_manager(
    drt: DistributedRuntime,
    config: Option<HealthCheckConfig>,
) -> anyhow::Result<()> {
    let config = config.unwrap_or_default();
    let manager = Arc::new(HealthCheckManager::new(drt, config));

    // Start the health check manager (this spawns per-endpoint tasks internally)
    manager.start().await?;

    Ok(())
}

/// Get health check status for all endpoints
pub async fn get_health_check_status(
    drt: &DistributedRuntime,
) -> anyhow::Result<serde_json::Value> {
    // Get endpoints list from SystemHealth
    let endpoint_subjects: Vec<String> = drt.system_health().lock().get_health_check_endpoints();

    let mut endpoint_statuses = HashMap::new();

    // Check each endpoint's health status
    {
        let system_health = drt.system_health();
        let system_health_lock = system_health.lock();
        for endpoint_subject in &endpoint_subjects {
            let health_status = system_health_lock
                .get_endpoint_health_status(endpoint_subject)
                .unwrap_or(HealthStatus::NotReady);

            let is_healthy = matches!(health_status, HealthStatus::Ready);

            endpoint_statuses.insert(
                endpoint_subject.clone(),
                serde_json::json!({
                    "healthy": is_healthy,
                    "status": format!("{:?}", health_status),
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
    }))
}

// ============================================================
// Full pipeline tests: push_handler → notify → HealthCheckManager
// These tests use the real HealthCheckManager (spawn_endpoint_health_check_task)
// and the real push_handler pipeline (TwoPartCodec + TCP + engine.generate()).
// ============================================================
#[cfg(all(test, feature = "integration"))]
mod push_handler_notify_tests {
    use super::*;
    use crate::component::{Instance, TransportType};
    use crate::config::HealthStatus;
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use crate::engine::{AsyncEngine, AsyncEngineContextProvider};
    use crate::local_endpoint_registry::LocalAsyncEngine;
    use crate::pipeline::network::codec::{TwoPartCodec, TwoPartMessage};
    use crate::pipeline::network::tcp::server::{ServerOptions, TcpStreamServer};
    use crate::pipeline::network::{
        ConnectionInfo, Ingress, PushWorkHandler, ResponseService, StreamOptions,
    };
    use crate::pipeline::{ManyOut, ResponseStream, SingleIn};
    use crate::protocols::annotated::Annotated;
    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::stream;
    use std::sync::Arc;
    use std::time::Duration;

    type TestRequest = serde_json::Value;
    type TestResponse = Annotated<serde_json::Value>;

    /// A mock engine that streams a configurable sequence of success/error chunks.
    /// Used both as the push_handler pipeline engine and registered in
    /// the local endpoint registry for health check requests.
    struct MockStreamingEngine {
        num_chunks: usize,
        /// If set, chunks at these indices will be error responses.
        error_indices: Vec<usize>,
    }

    impl MockStreamingEngine {
        fn success(num_chunks: usize) -> Arc<Self> {
            Arc::new(Self {
                num_chunks,
                error_indices: vec![],
            })
        }

        fn all_errors(num_chunks: usize) -> Arc<Self> {
            Arc::new(Self {
                num_chunks,
                error_indices: (0..num_chunks).collect(),
            })
        }

        fn with_error_at(num_chunks: usize, error_indices: Vec<usize>) -> Arc<Self> {
            Arc::new(Self {
                num_chunks,
                error_indices,
            })
        }
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<TestRequest>, ManyOut<TestResponse>, anyhow::Error>
        for MockStreamingEngine
    {
        async fn generate(
            &self,
            input: SingleIn<TestRequest>,
        ) -> anyhow::Result<ManyOut<TestResponse>> {
            let (_data, ctx) = input.into_parts();
            let chunks: Vec<TestResponse> = (0..self.num_chunks)
                .map(|i| {
                    if self.error_indices.contains(&i) {
                        Annotated::from_error(format!("mock error at chunk {i}"))
                    } else {
                        Annotated::from_data(serde_json::json!({"token": i}))
                    }
                })
                .collect();
            Ok(ResponseStream::new(
                Box::pin(stream::iter(chunks)),
                ctx.context(),
            ))
        }
    }

    /// Encodes a request as a TwoPartCodec payload with the given connection info.
    fn encode_request(
        request_id: &str,
        connection_info: ConnectionInfo,
        request_body: &serde_json::Value,
    ) -> Bytes {
        let control = serde_json::json!({
            "id": request_id,
            "request_type": "single_in",
            "response_type": "many_out",
            "connection_info": connection_info,
        });
        let header = serde_json::to_vec(&control).unwrap();
        let data = serde_json::to_vec(request_body).unwrap();
        let msg = TwoPartMessage::new(Bytes::from(header), Bytes::from(data));
        TwoPartCodec::default().encode_message(msg).unwrap()
    }

    /// Sets up a TCP server and registers a response stream for push_handler
    /// to connect back to.
    async fn setup_tcp_receiver(request_id: &str) -> (Arc<TcpStreamServer>, ConnectionInfo) {
        let options = ServerOptions::builder().port(0).build().unwrap();
        let server = TcpStreamServer::new(options).await.unwrap();

        let context = crate::pipeline::Context::with_id((), request_id.to_string());
        let stream_options = StreamOptions::builder()
            .context(context.context())
            .enable_request_stream(false)
            .enable_response_stream(true)
            .build()
            .unwrap();

        let pending = server.register(stream_options).await;
        let connection_info = pending
            .recv_stream
            .as_ref()
            .unwrap()
            .connection_info
            .clone();

        (server, connection_info)
    }

    /// Registers an endpoint in the DRT with the given engine in local registry.
    /// Returns the notifier that the real HealthCheckManager will listen on.
    fn register_endpoint(
        drt: &crate::DistributedRuntime,
        endpoint_name: &str,
        local_engine: LocalAsyncEngine,
    ) -> Arc<tokio::sync::Notify> {
        let payload = serde_json::json!({
            "prompt": "health",
            "_health_check": true
        });
        drt.system_health().lock().register_health_check_target(
            endpoint_name,
            Instance {
                component: "test_component".to_string(),
                endpoint: endpoint_name.to_string(),
                namespace: "test_namespace".to_string(),
                instance_id: 0,
                transport: TransportType::Nats(endpoint_name.to_string()),
                device_type: None,
            },
            payload,
        );

        drt.local_endpoint_registry()
            .register(endpoint_name.to_string(), local_engine);

        drt.system_health()
            .lock()
            .get_endpoint_health_check_notifier(endpoint_name)
            .expect("Notifier should exist for registered endpoint")
    }

    /// Helper: send a request through the ingress pipeline.
    async fn send_request(ingress: &Ingress<SingleIn<TestRequest>, ManyOut<TestResponse>>) {
        let request_id = uuid::Uuid::new_v4().to_string();
        let (_server, connection_info) = setup_tcp_receiver(&request_id).await;
        let payload = encode_request(
            &request_id,
            connection_info,
            &serde_json::json!({"prompt": "test"}),
        );
        let result = ingress.handle_payload(payload, Some(request_id)).await;
        assert!(result.is_ok(), "handle_payload should succeed");
    }

    /// Helper: assert endpoint health status.
    fn assert_status(
        drt: &crate::DistributedRuntime,
        endpoint_name: &str,
        expected: HealthStatus,
        msg: &str,
    ) {
        let status = drt
            .system_health()
            .lock()
            .get_endpoint_health_status(endpoint_name);
        assert_eq!(status, Some(expected), "{msg}");
    }

    /// Helper: create ingress pipeline with given engine and notifier.
    fn create_ingress(
        engine: Arc<MockStreamingEngine>,
        notifier: Arc<tokio::sync::Notify>,
    ) -> Arc<Ingress<SingleIn<TestRequest>, ManyOut<TestResponse>>> {
        let ingress =
            Ingress::<SingleIn<TestRequest>, ManyOut<TestResponse>>::for_engine(engine).unwrap();
        ingress
            .set_endpoint_health_check_notifier(notifier)
            .unwrap();
        ingress
    }

    /// Helper: start HealthCheckManager with given canary wait.
    async fn start_manager(drt: &crate::DistributedRuntime, canary_wait_ms: u64) {
        let config = HealthCheckConfig {
            canary_wait_time: Duration::from_millis(canary_wait_ms),
            request_timeout: Duration::from_secs(1),
        };
        let manager = Arc::new(HealthCheckManager::new(drt.clone(), config));
        manager.start().await.unwrap();
    }

    // =================================================================
    // Test 1: Successful streaming → notification → Ready
    // Canary engine returns errors, so Ready can only come from notify.
    // =================================================================
    #[tokio::test]
    async fn test_successful_streaming_sets_ready() {
        let drt = create_test_drt_async().await;
        let endpoint = "test.successful_streaming";

        let notifier = register_endpoint(&drt, endpoint, MockStreamingEngine::all_errors(1));
        assert_status(&drt, endpoint, HealthStatus::NotReady, "initial");

        let ingress = create_ingress(MockStreamingEngine::success(5), notifier);
        start_manager(&drt, 500).await;

        send_request(&ingress).await;
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Ready can only come from notification (canary engine errors)
        assert_status(
            &drt,
            endpoint,
            HealthStatus::Ready,
            "successful streaming should set Ready via notification path",
        );
    }

    // =================================================================
    // Test 2: Idle engine → canary fires → successful health check → Ready
    // =================================================================
    #[tokio::test]
    async fn test_canary_fires_on_idle_engine() {
        let drt = create_test_drt_async().await;
        let endpoint = "test.canary_idle";

        let _notifier = register_endpoint(&drt, endpoint, MockStreamingEngine::success(1));
        assert_status(&drt, endpoint, HealthStatus::NotReady, "initial");

        start_manager(&drt, 50).await;
        tokio::time::sleep(Duration::from_millis(300)).await;

        // No requests sent — canary fired and succeeded
        assert_status(
            &drt,
            endpoint,
            HealthStatus::Ready,
            "canary should fire and set Ready on idle engine",
        );
    }

    // =================================================================
    // Test 3: Error streaming → no notification → canary errors → NotReady
    // =================================================================
    #[tokio::test]
    async fn test_error_streaming_stays_not_ready() {
        let drt = create_test_drt_async().await;
        let endpoint = "test.error_streaming";

        let notifier = register_endpoint(&drt, endpoint, MockStreamingEngine::all_errors(1));
        assert_status(&drt, endpoint, HealthStatus::NotReady, "initial");

        // Pipeline streams only errors — no notifications sent
        let ingress = create_ingress(MockStreamingEngine::all_errors(3), notifier);
        start_manager(&drt, 50).await;

        send_request(&ingress).await;
        // Wait for canary to fire (50ms wait + margin)
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Error streaming didn't notify, canary fired but engine also errored
        assert_status(
            &drt,
            endpoint,
            HealthStatus::NotReady,
            "error streaming should not notify, canary also errors — stays NotReady",
        );
    }

    // =================================================================
    // Test 4: Idle engine → canary fires → failing health check → NotReady
    // =================================================================
    #[tokio::test]
    async fn test_idle_engine_with_failing_canary() {
        let drt = create_test_drt_async().await;
        let endpoint = "test.canary_fails";

        let _notifier = register_endpoint(&drt, endpoint, MockStreamingEngine::all_errors(1));
        assert_status(&drt, endpoint, HealthStatus::NotReady, "initial");

        start_manager(&drt, 50).await;
        tokio::time::sleep(Duration::from_millis(300)).await;

        // No requests sent, canary fired but engine returned error
        assert_status(
            &drt,
            endpoint,
            HealthStatus::NotReady,
            "canary fired but engine errored, status stays NotReady",
        );
    }

    // =================================================================
    // Test 5: Mixed streaming (success + trailing error) → Ready
    // Successful chunks notify before the error, so status becomes Ready.
    // Canary engine errors, proving Ready came from notification path.
    // =================================================================
    #[tokio::test]
    async fn test_mixed_streaming_sets_ready() {
        let drt = create_test_drt_async().await;
        let endpoint = "test.mixed_streaming";

        let notifier = register_endpoint(&drt, endpoint, MockStreamingEngine::all_errors(1));
        assert_status(&drt, endpoint, HealthStatus::NotReady, "initial");

        // 5 chunks: 4 success + error at index 4
        let ingress = create_ingress(MockStreamingEngine::with_error_at(5, vec![4]), notifier);
        start_manager(&drt, 500).await;

        send_request(&ingress).await;
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Successful chunks notified before the error chunk
        assert_status(
            &drt,
            endpoint,
            HealthStatus::Ready,
            "successful chunks should set Ready despite trailing error",
        );
    }
}

// ===============================
// Integration Tests (require DRT)
// ===============================
#[cfg(all(test, feature = "integration"))]
mod integration_tests {
    use super::*;
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn test_initialization() {
        let drt = create_test_drt_async().await;

        let canary_wait_time = Duration::from_secs(5);
        let request_timeout = Duration::from_secs(3);

        let config = HealthCheckConfig {
            canary_wait_time,
            request_timeout,
        };

        let manager = HealthCheckManager::new(drt.clone(), config);

        assert_eq!(manager.config.canary_wait_time, canary_wait_time);
        assert_eq!(manager.config.request_timeout, request_timeout);
    }

    #[tokio::test]
    async fn test_payload_registration() {
        let drt = create_test_drt_async().await;

        let endpoint = "test.endpoint";
        let payload = serde_json::json!({
            "prompt": "test",
            "_health_check": true
        });

        drt.system_health().lock().register_health_check_target(
            endpoint,
            crate::component::Instance {
                component: "test_component".to_string(),
                endpoint: "test_endpoint".to_string(),
                namespace: "test_namespace".to_string(),
                instance_id: 12345,
                transport: crate::component::TransportType::Nats(endpoint.to_string()),
                device_type: None,
            },
            payload.clone(),
        );

        let retrieved = drt
            .system_health()
            .lock()
            .get_health_check_target(endpoint)
            .map(|t| t.payload);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), payload);

        // Verify endpoint appears in the list
        let endpoints = drt.system_health().lock().get_health_check_endpoints();
        assert!(endpoints.contains(&endpoint.to_string()));
    }

    #[tokio::test]
    async fn test_spawn_per_endpoint_tasks() {
        let drt = create_test_drt_async().await;

        for i in 0..3 {
            let endpoint = format!("test.endpoint.{}", i);
            let payload = serde_json::json!({
                "prompt": format!("test{}", i),
                "_health_check": true
            });
            drt.system_health().lock().register_health_check_target(
                &endpoint,
                crate::component::Instance {
                    component: "test_component".to_string(),
                    endpoint: format!("test_endpoint_{}", i),
                    namespace: "test_namespace".to_string(),
                    instance_id: i,
                    transport: crate::component::TransportType::Nats(endpoint.clone()),
                    device_type: None,
                },
                payload,
            );
        }

        let config = HealthCheckConfig {
            canary_wait_time: Duration::from_secs(5),
            request_timeout: Duration::from_secs(1),
        };

        let manager = Arc::new(HealthCheckManager::new(drt.clone(), config));
        manager.clone().start().await.unwrap();

        // Verify all endpoints have their own health check tasks
        let tasks = manager.endpoint_tasks.lock();
        // Should have 3 tasks (one for each endpoint)
        assert_eq!(tasks.len(), 3);
        // Check that all endpoints are represented in tasks
        let endpoints: Vec<String> = tasks.keys().cloned().collect();
        assert!(endpoints.contains(&"test.endpoint.0".to_string()));
        assert!(endpoints.contains(&"test.endpoint.1".to_string()));
        assert!(endpoints.contains(&"test.endpoint.2".to_string()));
    }

    #[tokio::test]
    async fn test_endpoint_health_check_notifier_created() {
        let drt = create_test_drt_async().await;

        let endpoint = "test.endpoint.notifier";
        let payload = serde_json::json!({
            "prompt": "test",
            "_health_check": true
        });

        // Register the endpoint
        drt.system_health().lock().register_health_check_target(
            endpoint,
            crate::component::Instance {
                component: "test_component".to_string(),
                endpoint: "test_endpoint_notifier".to_string(),
                namespace: "test_namespace".to_string(),
                instance_id: 999,
                transport: crate::component::TransportType::Nats(endpoint.to_string()),
                device_type: None,
            },
            payload.clone(),
        );

        // Verify that a notifier was created for this endpoint
        let notifier = drt
            .system_health()
            .lock()
            .get_endpoint_health_check_notifier(endpoint);

        assert!(
            notifier.is_some(),
            "Endpoint should have a notifier created"
        );

        // Verify we can notify it without panicking
        if let Some(notifier) = notifier {
            notifier.notify_one();
        }

        // Initially, the endpoint should be Ready (default after registration)
        let status = drt
            .system_health()
            .lock()
            .get_endpoint_health_status(endpoint);
        assert_eq!(status, Some(HealthStatus::NotReady));
    }
}
