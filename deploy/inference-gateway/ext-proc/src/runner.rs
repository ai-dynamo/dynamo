// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standard Rust EPP process bootstrap.
//!
//! A native Rust binary implementing the Envoy ext_proc gRPC service, using
//! Dynamo's KV-aware router for endpoint selection.
//!
//! The ext-proc port (9002) serves TLS (self-signed cert). The health port
//! (9003) is plaintext (K8s probes don't need TLS).

use std::sync::Arc;

use anyhow::Result;
use dynamo_kv_router::services::selection::RouterPluginRegistry;
use dynamo_runtime::namespace::{NamespaceFilter, NamespacePrefixMode};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_rustls::TlsAcceptor;
use tokio_util::sync::CancellationToken;

use crate::{EppMode, EppStandaloneConfig, ExtProcServer, Router, metrics};

/// Populate a Runtime EPP catalog with Dynamo's shipped named providers.
///
/// Starts from the caller's registry (or an empty one) and registers the shipped
/// types with [`dynamo_router_policies::register`]. The process installer then
/// applies the required builtin default factory and routing-mode providers.
///
/// Callers must not pre-register Dynamo's shipped type names; a duplicate is a
/// startup error rather than a silent override. Custom worker-selection and
/// request-classifier providers already in `policy_registry` are preserved.
fn prepare_runtime_plugin_registry(
    policy_registry: Option<RouterPluginRegistry>,
) -> Result<RouterPluginRegistry> {
    let mut registry = policy_registry.unwrap_or_default();
    dynamo_router_policies::register(&mut registry).map_err(|error| {
        anyhow::anyhow!(
            "Dynamo shipped router plugin registration failed \
             (supplied registries must not pre-register shipped type names): {error}"
        )
    })?;
    Ok(registry)
}

/// Install the Runtime catalog once before discovery or router construction.
fn install_runtime_plugin_registry(registry: RouterPluginRegistry) -> Result<()> {
    if !dynamo_llm::kv_router::plugins::install_router_plugin_registry(registry) {
        anyhow::bail!("router plugin registry already installed");
    }
    Ok(())
}

/// Mode-dependent registry bootstrap for [`run`].
///
/// Standalone returns the supplied (or empty) registry unchanged and does not
/// install. Runtime prepares shipped providers, installs once (fatal if already
/// installed), and returns an unused empty registry for [`run_inner`].
fn bootstrap_plugin_registry(
    mode: EppMode,
    policy_registry: Option<RouterPluginRegistry>,
) -> Result<RouterPluginRegistry> {
    match mode {
        EppMode::Standalone => Ok(policy_registry.unwrap_or_default()),
        EppMode::DynamoRuntime => {
            let registry = prepare_runtime_plugin_registry(policy_registry)?;
            install_runtime_plugin_registry(registry)?;
            Ok(RouterPluginRegistry::default())
        }
    }
}

const GRPC_PORT: u16 = 9002;
const HEALTH_PORT: u16 = 9003;
const HEALTH_SERVICE_NAME: &str = "inference-extension";
/// Cap concurrent in-flight TLS handshakes + active gRPC streams. Prevents a
/// connection flood from exhausting fds / memory. Tuned for an inference EPP
/// where a single Envoy upstream typically holds <100 concurrent streams.
const MAX_CONCURRENT_CONNECTIONS: usize = 1024;
/// Propagation window after a shutdown signal before the server stops
/// accepting new connections. The gateway stops routing to this EPP as soon
/// as health flips to NOT_SERVING. Configurable via
/// `DYN_EPP_GRACEFUL_SHUTDOWN_PROPAGATION_SECS`.
const DEFAULT_GRACEFUL_SHUTDOWN_PROPAGATION_SECS: u64 = 5;
const GRACEFUL_SHUTDOWN_PROPAGATION_ENV: &str = "DYN_EPP_GRACEFUL_SHUTDOWN_PROPAGATION_SECS";
/// Max time to wait for the TLS handshake to complete before dropping the
/// connection. Without this, a client that finishes the TCP connect but
/// stalls the TLS handshake holds a connection-limit permit indefinitely;
/// enough such stalls exhaust all permits and starve legitimate ext_proc
/// traffic (slowloris-style). Only the handshake is bounded — established
/// connections may stay open for the lifetime of their bidi stream.
const TLS_HANDSHAKE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

struct Config {
    namespace_filter: NamespaceFilter,
    namespace_prefix_mode: NamespacePrefixMode,
    component: String,
}

impl Config {
    fn from_env() -> Self {
        let namespace = env_or("DYN_NAMESPACE", "").unwrap_or_else(|| "vllm-agg".to_string());
        let prefix = env_or("DYN_NAMESPACE_PREFIX", "");
        let namespace_filter =
            NamespaceFilter::from_namespace_and_prefix(Some(&namespace), prefix.as_deref());

        if parse_env("DYN_ENFORCE_DISAGG", false) {
            tracing::warn!(
                "DYN_ENFORCE_DISAGG is deprecated and ignored; routing topology and readiness are determined from registered worker types"
            );
        }

        Self {
            namespace_filter,
            namespace_prefix_mode: NamespacePrefixMode::from_env(),
            component: env_or("DYN_COMPONENT_NAME", "").unwrap_or_else(|| "backend".to_string()),
        }
    }
}

fn env_or(key: &str, empty_means_unset: &str) -> Option<String> {
    std::env::var(key).ok().and_then(|v| {
        let trimmed = v.trim();
        if trimmed.is_empty() || trimmed == empty_means_unset {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn parse_env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Generate a self-signed TLS acceptor for the ext-proc gRPC server.
fn create_tls_acceptor() -> Result<TlsAcceptor> {
    use rcgen::{CertificateParams, KeyPair};
    use rustls::ServerConfig;
    use tokio_rustls::rustls;

    let key_pair = KeyPair::generate()?;
    let mut params = CertificateParams::new(vec!["localhost".to_string()])?;
    params
        .subject_alt_names
        .push(rcgen::SanType::IpAddress(std::net::IpAddr::V4(
            std::net::Ipv4Addr::UNSPECIFIED,
        )));
    let cert = params.self_signed(&key_pair)?;

    let cert_pem = cert.pem();
    let key_pem = key_pair.serialize_pem();

    let certs = rustls_pemfile::certs(&mut cert_pem.as_bytes())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let key = rustls_pemfile::private_key(&mut key_pem.as_bytes())?
        .ok_or_else(|| anyhow::anyhow!("No private key found in PEM"))?;

    // Build with an explicit crypto provider. This crate compiles in BOTH
    // rustls providers via feature unification (our direct `ring` feature plus
    // `aws-lc-rs` pulled in transitively by `kube`), so the parameterless
    // `ServerConfig::builder()` cannot auto-select a process-default provider
    // and would panic. Pin to `ring`, matching the rustls feature we enable
    // for our own serving path.
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut tls_config = ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()?
        .with_no_client_auth()
        .with_single_cert(certs, key)?;
    tls_config.alpn_protocols = vec![b"h2".to_vec()];

    tracing::info!("Generated self-signed TLS certificate for ext-proc server");
    Ok(TlsAcceptor::from(Arc::new(tls_config)))
}

/// Run EPP until it exits.
///
/// Runtime mode installs Dynamo's shipped named providers plus any supplied
/// catalog into the process registry ModelManager uses. Standalone keeps the
/// supplied registry on the selector path without process installation.
pub async fn run(policy_registry: Option<RouterPluginRegistry>) -> Result<()> {
    init_tracing();
    let mode = EppMode::from_env()?;
    let registry = bootstrap_plugin_registry(mode, policy_registry)?;
    run_inner(mode, registry).await
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
}

/// Wait for SIGTERM (kubelet pod termination) or SIGINT (Ctrl-C).
async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Tasks started before router initialization that must not outlive a failed run.
struct BackgroundTasks {
    health_task: JoinHandle<()>,
    metrics_task: Option<JoinHandle<()>>,
    shutdown_task: JoinHandle<()>,
}

impl BackgroundTasks {
    async fn shutdown(self) {
        let Self {
            health_task,
            metrics_task,
            shutdown_task,
        } = self;

        health_task.abort();
        if let Some(task) = &metrics_task {
            task.abort();
        }
        shutdown_task.abort();

        let _ = health_task.await;
        if let Some(task) = metrics_task {
            let _ = task.await;
        }
        let _ = shutdown_task.await;
    }
}

async fn run_inner(mode: EppMode, policy_registry: RouterPluginRegistry) -> Result<()> {
    let standalone = matches!(mode, EppMode::Standalone);

    let config = Config::from_env();

    tracing::info!(
        port = GRPC_PORT,
        health_port = HEALTH_PORT,
        namespace_filter = ?config.namespace_filter,
        namespace_prefix_mode = ?config.namespace_prefix_mode,
        component = %config.component,
        standalone,
        "Starting Dynamo Rust EPP"
    );

    // Start plaintext gRPC health server immediately (NOT_SERVING until router ready).
    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_service_status(HEALTH_SERVICE_NAME, tonic_health::ServingStatus::NotServing)
        .await;

    let health_addr = format!("0.0.0.0:{HEALTH_PORT}").parse()?;
    tracing::info!(%health_addr, "Starting gRPC health server (plaintext)");
    let health_task = tokio::spawn(async move {
        if let Err(error) = tonic::transport::Server::builder()
            .add_service(health_service)
            .serve(health_addr)
            .await
        {
            tracing::error!(%error, "Health server exited");
        }
    });

    // Start before router init so scrapes during a slow discovery bootstrap
    // return an empty exposition rather than a connection refusal.
    let metrics_port = parse_env(metrics::METRICS_PORT_ENV, metrics::DEFAULT_METRICS_PORT);
    let metrics_task = if metrics_port == 0 {
        tracing::info!("Metrics server disabled");
        None
    } else {
        Some(tokio::spawn(async move {
            if let Err(e) = metrics::serve(metrics_port).await {
                tracing::error!(error = %e, port = metrics_port, "Metrics server exited");
            }
        }))
    };

    // Shutdown coordination: on SIGTERM/SIGINT, flip health to NOT_SERVING
    // (the gateway stops routing new requests to this EPP), allow the endpoint
    // propagation window to elapse, then stop accepting connections. The
    // protocol-correct drain deadline and forced close of long-lived HTTP/2
    // connections are handled by the follow-up connection-lifecycle work.
    let draining = CancellationToken::new();
    let shutdown = CancellationToken::new();
    let shutdown_task = {
        let draining = draining.clone();
        let shutdown = shutdown.clone();
        tokio::spawn(async move {
            wait_for_shutdown_signal().await;
            tracing::info!("Shutdown signal received; starting endpoint withdrawal");
            // The readiness mirror owns the health transition so it cannot
            // race with a final SERVING update. If initialization has not
            // reached `serve` yet, health is already NOT_SERVING and the
            // draining cancellation below makes initialization return.
            draining.cancel();
            let propagation_secs = parse_env(
                GRACEFUL_SHUTDOWN_PROPAGATION_ENV,
                DEFAULT_GRACEFUL_SHUTDOWN_PROPAGATION_SECS,
            );
            tracing::info!(
                propagation_secs,
                "EPP health set to NOT_SERVING; allowing endpoint propagation"
            );
            tokio::time::sleep(std::time::Duration::from_secs(propagation_secs)).await;
            shutdown.cancel();
            tracing::info!("EPP endpoint propagation complete; stopping accepts");
        })
    };
    let background_tasks = BackgroundTasks {
        health_task,
        metrics_task,
        shutdown_task,
    };

    let result = async {
        if standalone {
            let selector_cfg = EppStandaloneConfig::from_env()?;
            tracing::info!(
                inference_pool_name = %selector_cfg.inference_pool_name,
                model_name = %selector_cfg.model_name,
                block_size = selector_cfg.block_size,
                "Initializing standalone selector mode (no Dynamo runtime)..."
            );
            metrics::set_served_model(&selector_cfg.model_name);
            let router = tokio::select! {
                _ = draining.cancelled() => {
                    tracing::info!("Shutdown received during standalone EPP initialization");
                    return Ok(());
                }
                router = crate::EppRouter::from_selector(selector_cfg, policy_registry) => Arc::new(router?),
            };
            if draining.is_cancelled() {
                tracing::info!("Shutdown received before standalone EPP serving started");
                return Ok(());
            }
            let ready_router = router.clone();
            serve(
                router,
                move || ready_router.is_ready(),
                health_reporter,
                draining,
                shutdown,
            )
            .await
        } else {
            tracing::info!("Initializing KV-aware router from discovery...");
            let router = tokio::select! {
                _ = draining.cancelled() => {
                    tracing::info!("Shutdown received during Dynamo discovery initialization");
                    return Ok(());
                }
                router = Router::from_discovery(config.namespace_filter, config.namespace_prefix_mode, &config.component) => router?,
            };
            if draining.is_cancelled() {
                tracing::info!("Shutdown received before Dynamo discovery serving started");
                return Ok(());
            }
            metrics::set_served_model(router.served_model());
            let ready = router.pod_store_ready();
            serve(
                Arc::new(router),
                move || ready.load(std::sync::atomic::Ordering::Acquire),
                health_reporter,
                draining,
                shutdown,
            )
            .await
        }
    }
    .await;
    background_tasks.shutdown().await;
    result
}

/// Mirror the picker's live readiness onto the gRPC health status, then serve
/// the ext_proc endpoint. Shared by both Dynamo-discovery and standalone modes.
async fn serve<P: crate::EndpointPicker>(
    picker: Arc<P>,
    is_ready: impl Fn() -> bool + Send + 'static,
    health_reporter: tonic_health::server::HealthReporter,
    draining: CancellationToken,
    shutdown: CancellationToken,
) -> Result<()> {
    // Continuously mirror readiness onto the health status. `is_ready()` is a
    // *live* signal that can flip both ways — standalone discovery clears it when
    // the InferencePool is deleted/invalid (nothing routable), and in replicated
    // mode it stays false until the peer set finishes its initial sync. A latch
    // (set SERVING once) would strand those states, so a background task tracks
    // transitions and moves the health status in lock-step, dropping out of
    // SERVING when readiness drops and recovering when it returns. Health starts
    // NOT_SERVING (set during startup); the mirror polls a cheap closure (atomic loads).
    let readiness_task = {
        let health_reporter = health_reporter.clone();
        let draining = draining.clone();
        tokio::spawn(async move {
            let mut last: Option<bool> = None;
            loop {
                if draining.is_cancelled() {
                    health_reporter
                        .set_service_status(
                            HEALTH_SERVICE_NAME,
                            tonic_health::ServingStatus::NotServing,
                        )
                        .await;
                    tracing::info!("EPP readiness mirror stopped; health set to NOT_SERVING");
                    break;
                }

                let now = is_ready();
                if !draining.is_cancelled() && last != Some(now) {
                    let status = if now {
                        tonic_health::ServingStatus::Serving
                    } else {
                        tonic_health::ServingStatus::NotServing
                    };
                    health_reporter
                        .set_service_status(HEALTH_SERVICE_NAME, status)
                        .await;
                    if draining.is_cancelled() {
                        health_reporter
                            .set_service_status(
                                HEALTH_SERVICE_NAME,
                                tonic_health::ServingStatus::NotServing,
                            )
                            .await;
                        tracing::info!("EPP readiness mirror stopped; health set to NOT_SERVING");
                        break;
                    }
                    tracing::info!(ready = now, "EPP readiness changed; health status updated");
                    last = Some(now);
                }

                tokio::select! {
                    _ = draining.cancelled() => {}
                    _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {}
                }
            }
        })
    };

    let server = ExtProcServer::new(picker);
    // Default to TLS. Verified working with kGateway (`appProtocol: http2`
    // upstreams negotiate h2 over TLS via ALPN when the cert is presented).
    // Set DYN_SECURE_SERVING=false to fall back to plaintext h2c, e.g. for
    // local debugging or non-TLS gateways.
    let secure_serving = parse_env("DYN_SECURE_SERVING", true);
    let addr: std::net::SocketAddr = format!("0.0.0.0:{GRPC_PORT}").parse()?;

    if secure_serving {
        let tls_acceptor = create_tls_acceptor()?;
        let svc = server.into_service();
        let listener = TcpListener::bind(addr).await?;
        let conn_semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_CONNECTIONS));
        tracing::info!(
            %addr,
            max_connections = MAX_CONCURRENT_CONNECTIONS,
            "Listening for ext_proc connections (TLS)"
        );

        let result: Result<()> = loop {
            // Acquire permit before accept() so we backpressure the listener
            // instead of accepting and immediately dropping connections. Stop
            // accepting once the endpoint propagation window has elapsed.
            let permit = tokio::select! {
                _ = shutdown.cancelled() => break Ok(()),
                permit = conn_semaphore.clone().acquire_owned() => match permit {
                    Ok(permit) => permit,
                    Err(error) => break Err(error.into()),
                },
            };
            let (tcp_stream, remote_addr) = tokio::select! {
                _ = shutdown.cancelled() => break Ok(()),
                accepted = listener.accept() => match accepted {
                    Ok(accepted) => accepted,
                    Err(error) => break Err(error.into()),
                },
            };
            // tonic's own listener (the plaintext path) sets TCP_NODELAY; this
            // hand-built one has to as well.
            if let Err(error) = tcp_stream.set_nodelay(true) {
                tracing::debug!(%remote_addr, %error, "Failed to set TCP_NODELAY");
            }
            let tls_acceptor = tls_acceptor.clone();
            let svc = svc.clone();

            tokio::spawn(async move {
                let _permit = permit; // released when this task exits (incl. handshake timeout)
                let tls_stream = match tokio::time::timeout(
                    TLS_HANDSHAKE_TIMEOUT,
                    tls_acceptor.accept(tcp_stream),
                )
                .await
                {
                    Ok(Ok(s)) => s,
                    Ok(Err(e)) => {
                        tracing::debug!(%remote_addr, error = %e, "TLS handshake failed");
                        return;
                    }
                    Err(_) => {
                        tracing::debug!(
                            %remote_addr,
                            timeout_secs = TLS_HANDSHAKE_TIMEOUT.as_secs(),
                            "TLS handshake timed out; dropping connection"
                        );
                        return;
                    }
                };

                let io = hyper_util::rt::TokioIo::new(tls_stream);
                let hyper_svc = hyper_util::service::TowerToHyperService::new(svc);
                if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                    hyper_util::rt::TokioExecutor::new(),
                )
                .serve_connection(io, hyper_svc)
                .await
                {
                    tracing::debug!(%remote_addr, error = %e, "Connection ended");
                }
            });
        };
        readiness_task.abort();
        let _ = readiness_task.await;
        result
    } else {
        tracing::info!(%addr, "Listening for ext_proc connections (plaintext h2)");
        tonic::transport::Server::builder()
            .add_service(server.into_service())
            .serve_with_shutdown(addr, shutdown.cancelled_owned())
            .await?;
        readiness_task.abort();
        let _ = readiness_task.await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use dynamo_kv_router::plugins::WorkerSelectionPolicyRegistryError;
    use dynamo_kv_router::plugins::worker_selection::{
        WorkerSelectionPolicy, WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
        WorkerSelectionPolicyProviderError,
    };
    use dynamo_kv_router::{
        DEFAULT_ROUTING_GROUP, KvRouterConfig, RoutingPartitionId, RoutingPartitionRef, WorkerType,
    };
    use dynamo_llm::discovery::{ModelManager, WORKER_TYPE_PREFILL};
    use dynamo_llm::kv_router::plugins::router_plugin_registry;
    use dynamo_runtime::distributed::DistributedConfig;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use uuid::Uuid;

    use super::*;

    const RUNTIME_BOOTSTRAP_CHILD: &str = "DYN_EPP_RUNTIME_BOOTSTRAP_CHILD";

    fn policy_config(yaml: &str) -> (tempfile::NamedTempFile, KvRouterConfig) {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), yaml).unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(file.path().display().to_string()),
            ..Default::default()
        };
        (file, config)
    }

    fn assert_exact_one_passing_child(output: &std::process::Output) {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "child failed:\n{stdout}\n{stderr}"
        );
        assert!(
            stdout
                .lines()
                .any(|line| line.starts_with("test result: ok. 1 passed; 0 failed; 0 ignored;")),
            "child must execute exactly one passing test:\n{stdout}\n{stderr}"
        );
    }

    struct AlwaysFirstPicker;

    impl dynamo_kv_router::plugins::worker_selection::WorkerPicker for AlwaysFirstPicker {
        fn pick(
            &self,
            _context: &dynamo_kv_router::plugins::worker_selection::WorkerSelectionContext<'_>,
            input: dynamo_kv_router::plugins::worker_selection::WorkerInputView<'_>,
        ) -> Result<usize, dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicyError>
        {
            if input.candidates().is_empty() {
                Err(
                    dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicyError::failed(
                        "no eligible worker",
                    ),
                )
            } else {
                Ok(0)
            }
        }
    }

    #[test]
    fn namespace_environment_preserves_exact_and_prefix_intent() {
        for (prefix, namespace, strict, generation, sibling) in [
            (None, Some("default-foo"), None, false, false),
            (None, Some("default-foo"), Some("true"), false, false),
            (Some("default-foo"), Some("ignored"), None, true, true),
            (
                Some("default-foo"),
                Some("ignored"),
                Some("true"),
                true,
                false,
            ),
            (Some(" "), Some("default-foo"), Some("true"), false, false),
            (Some(" dynamo "), Some("ignored"), Some("true"), true, true),
            (None, Some("dynamo"), Some("true"), true, true),
        ] {
            temp_env::with_vars(
                [
                    ("DYN_NAMESPACE_PREFIX", prefix),
                    ("DYN_NAMESPACE", namespace),
                    ("DYN_NAMESPACE_PREFIX_STRICT", strict),
                ],
                || {
                    let config = Config::from_env();
                    let filter = config.namespace_filter;
                    let mode = config.namespace_prefix_mode;
                    assert!(filter.matches_with_prefix_mode("default-foo", mode));
                    assert_eq!(
                        filter.matches_with_prefix_mode("default-foo-1a2b3c4d", mode),
                        generation,
                        "{filter:?}"
                    );
                    assert_eq!(
                        filter.matches_with_prefix_mode("default-foo-bar", mode),
                        sibling,
                        "{filter:?}"
                    );
                },
            );
        }
    }

    /// Stock Runtime preparation resolves the named default with real parameters and keeps
    /// the builtin default plus routing modes when no instance is selected.
    #[test]
    fn stock_runtime_registry_resolves_named_default_and_required_modes() {
        let registry =
            dynamo_router_policies::with_defaults(prepare_runtime_plugin_registry(None).unwrap());
        let partition = RoutingPartitionRef::new("model", "default");

        let (_file, named) = policy_config(
            r#"
worker_selection:
  aggregated: tuned
  prefill: tuned
  decode: tuned
  instances:
    - name: tuned
      type: dynamo-default-cost-fn
      parameters:
        router_temperature: 0.25
"#,
        );
        let named_factory = registry.resolve(&named).unwrap().unwrap();
        for role in [
            WorkerType::Aggregated,
            WorkerType::Prefill,
            WorkerType::Decode,
        ] {
            let policy = named_factory(&named, role, partition);
            let snapshot = policy.runtime_config().unwrap();
            assert_eq!(
                snapshot.parameters().get("router_temperature"),
                Some(&serde_json::json!(0.25)),
                "{role:?}"
            );
        }

        let default_factory = registry
            .resolve(&KvRouterConfig::default())
            .unwrap()
            .unwrap();
        for role in [
            WorkerType::Aggregated,
            WorkerType::Prefill,
            WorkerType::Decode,
        ] {
            let policy = default_factory(&KvRouterConfig::default(), role, partition);
            let snapshot = policy.runtime_config().unwrap();
            assert_ne!(
                snapshot.parameters().get("router_temperature"),
                Some(&serde_json::json!(0.25)),
                "no selected instance must retain the named-default parameters ({role:?})"
            );
        }

        for mode in [
            dynamo_router_policies::router_modes::RANDOM,
            dynamo_router_policies::router_modes::ROUND_ROBIN,
            dynamo_router_policies::router_modes::LEAST_LOADED,
        ] {
            let (_file, mode_config) = policy_config(&format!(
                r#"
worker_selection:
  aggregated: mode
  instances:
    - name: mode
      type: {mode}
"#
            ));
            assert!(
                registry.resolve(&mode_config).unwrap().is_some(),
                "{mode} must remain available"
            );
        }

        let (_file, unknown) = policy_config(
            r#"
worker_selection:
  aggregated: missing
  instances:
    - name: missing
      type: not-a-registered-policy
"#,
        );
        assert!(matches!(
            registry.resolve(&unknown),
            Err(WorkerSelectionPolicyRegistryError::UnknownType { name, .. })
                if name == "not-a-registered-policy"
        ));

        let (_file, invalid) = policy_config(
            r#"
worker_selection:
  aggregated: tuned
  instances:
    - name: tuned
      type: dynamo-default-cost-fn
      parameters:
        router_temperature: -1.0
"#,
        );
        assert!(
            matches!(
                registry.resolve(&invalid),
                Err(WorkerSelectionPolicyRegistryError::Provider { .. })
            ),
            "invalid named-default parameters must fail resolution"
        );
    }

    #[test]
    fn duplicate_shipped_provider_name_fails_preparation() {
        let mut registry = RouterPluginRegistry::default();
        registry
            .register_worker_selection(
                "dynamo-default-cost-fn",
                Arc::new(
                    |_: &WorkerSelectionPolicyParameters| -> std::result::Result<
                        WorkerSelectionPolicyFactory,
                        WorkerSelectionPolicyProviderError,
                    > {
                        Err(WorkerSelectionPolicyProviderError::new("should not run"))
                    },
                ),
            )
            .unwrap();
        let error = prepare_runtime_plugin_registry(Some(registry))
            .err()
            .expect("duplicate shipped name must fail preparation");
        assert!(
            error.to_string().contains("dynamo-default-cost-fn"),
            "duplicate error should name the shipped type: {error}"
        );
    }

    #[test]
    fn preparation_preserves_custom_provider_alongside_shipped_types() {
        let mut registry = RouterPluginRegistry::default();
        registry
            .register_worker_selection(
                "custom-sentinel",
                Arc::new(|_: &WorkerSelectionPolicyParameters| {
                    Ok(Arc::new(
                        |config: &KvRouterConfig, worker_type, _partition| {
                            WorkerSelectionPolicy::new(
                                config.clone(),
                                worker_type.as_str(),
                                Vec::new(),
                                Box::new(AlwaysFirstPicker),
                            )
                        },
                    ) as WorkerSelectionPolicyFactory)
                }),
            )
            .unwrap();
        let registry = dynamo_router_policies::with_defaults(
            prepare_runtime_plugin_registry(Some(registry)).unwrap(),
        );
        let (_file, config) = policy_config(
            r#"
worker_selection:
  aggregated: custom-sentinel
  instances:
    - name: custom-sentinel
      type: custom-sentinel
"#,
        );
        assert!(registry.resolve(&config).unwrap().is_some());
        let (_file, shipped) = policy_config(
            r#"
worker_selection:
  aggregated: tuned
  instances:
    - name: tuned
      type: dynamo-default-cost-fn
"#,
        );
        assert!(registry.resolve(&shipped).unwrap().is_some());
    }

    #[test]
    fn standalone_bootstrap_passes_through_without_process_install() {
        let mut registry = RouterPluginRegistry::default();
        registry
            .register_worker_selection(
                "standalone-sentinel",
                Arc::new(|_: &WorkerSelectionPolicyParameters| {
                    Ok(Arc::new(
                        |config: &KvRouterConfig, worker_type, _partition| {
                            WorkerSelectionPolicy::new(
                                config.clone(),
                                worker_type.as_str(),
                                Vec::new(),
                                Box::new(AlwaysFirstPicker),
                            )
                        },
                    ) as WorkerSelectionPolicyFactory)
                }),
            )
            .unwrap();
        let (_file, config) = policy_config(
            r#"
worker_selection:
  aggregated: standalone-sentinel
  instances:
    - name: standalone-sentinel
      type: standalone-sentinel
"#,
        );
        let returned =
            bootstrap_plugin_registry(EppMode::Standalone, Some(registry)).unwrap();
        assert!(
            dynamo_router_policies::with_defaults(returned)
                .resolve(&config)
                .unwrap()
                .is_some()
        );
        assert!(
            matches!(
                router_plugin_registry().resolve_for_worker_type(&config, WorkerType::Aggregated),
                Err(WorkerSelectionPolicyRegistryError::UnknownType { name, .. })
                    if name == "standalone-sentinel"
            ),
            "Standalone bootstrap must not install into the process registry"
        );
    }

    /// Fresh-process Runtime bootstrap: process install + ModelManager typed role.
    #[tokio::test]
    async fn runtime_bootstrap_installs_sentinel_for_managed_kv_router() {
        const TEST: &str = concat!(
            module_path!(),
            "::runtime_bootstrap_installs_sentinel_for_managed_kv_router"
        );
        let test_name = TEST.split_once("::").unwrap().1;
        if std::env::var(RUNTIME_BOOTSTRAP_CHILD).as_deref() != Ok(test_name) {
            let mut child = tokio::process::Command::new(std::env::current_exe().unwrap());
            child
                .args(["--exact", test_name, "--nocapture"])
                .env(RUNTIME_BOOTSTRAP_CHILD, test_name)
                .kill_on_drop(true);
            let output = tokio::time::timeout(Duration::from_secs(60), child.output())
                .await
                .expect("runtime bootstrap child must finish within its deadline")
                .expect("spawn runtime bootstrap child");
            assert_exact_one_passing_child(&output);
            return;
        }

        const MODEL: &str = "epp-runtime-catalog";
        let sentinel_yaml = r#"
worker_selection:
  prefill: recording-sentinel
  instances:
    - name: recording-sentinel
      type: recording-sentinel
"#;
        let (_file, config) = policy_config(sentinel_yaml);
        assert!(
            matches!(
                router_plugin_registry()
                    .resolve_for_worker_type(&config, WorkerType::Prefill),
                Err(WorkerSelectionPolicyRegistryError::UnknownType { name, .. })
                    if name == "recording-sentinel"
            ),
            "uninstalled process registry must reject the sentinel as UnknownType"
        );

        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut registry = RouterPluginRegistry::default();
        registry
            .register_worker_selection(
                "recording-sentinel",
                Arc::new({
                    let seen = Arc::clone(&seen);
                    move |_parameters: &WorkerSelectionPolicyParameters| {
                        let seen = Arc::clone(&seen);
                        let factory: WorkerSelectionPolicyFactory =
                            Arc::new(move |config, worker_type, partition| {
                                seen.lock()
                                    .unwrap()
                                    .push((worker_type, partition.into_owned()));
                                WorkerSelectionPolicy::new(
                                    config.clone(),
                                    worker_type.as_str(),
                                    Vec::new(),
                                    Box::new(AlwaysFirstPicker),
                                )
                            });
                        Ok(factory)
                    }
                }),
            )
            .expect("register sentinel");

        let for_inner =
            bootstrap_plugin_registry(EppMode::DynamoRuntime, Some(registry)).unwrap();
        assert!(for_inner.is_empty());

        let runtime_config = KvRouterConfig {
            use_kv_events: false,
            router_track_active_blocks: false,
            skip_initial_worker_wait: true,
            router_policy_config: config.router_policy_config.clone(),
            ..Default::default()
        };
        assert!(
            router_plugin_registry()
                .resolve_for_worker_type(&runtime_config, WorkerType::Prefill)
                .unwrap()
                .is_some()
        );

        let runtime = Runtime::from_current().unwrap();
        let drt = tokio::time::timeout(
            Duration::from_secs(10),
            DistributedRuntime::new(runtime, DistributedConfig::process_local()),
        )
        .await
        .expect("process-local runtime construction timed out")
        .unwrap();
        let endpoint = drt
            .namespace(format!("epp-runtime-catalog-{}", Uuid::new_v4()))
            .unwrap()
            .component("backend")
            .unwrap()
            .endpoint("generate");
        let model_manager = Arc::new(ModelManager::new());
        let managed = tokio::time::timeout(
            Duration::from_secs(20),
            model_manager.managed_kv_router_for_with_worker_role(
                &endpoint,
                16,
                Some(runtime_config),
                None,
                Some(WorkerType::Prefill),
                WORKER_TYPE_PREFILL,
                Some(MODEL.to_string()),
                false,
            ),
        )
        .await
        .expect("managed KvRouter construction timed out")
        .expect("managed KvRouter construction");

        let expected = RoutingPartitionId::new(MODEL, DEFAULT_ROUTING_GROUP);
        assert_eq!(
            *seen.lock().unwrap(),
            vec![(WorkerType::Prefill, expected)],
            "Runtime bootstrap install must reach ModelManager with the real role and partition"
        );

        let repeat = bootstrap_plugin_registry(EppMode::DynamoRuntime, None)
            .err()
            .expect("second Runtime bootstrap must fail");
        assert!(
            repeat
                .to_string()
                .contains("router plugin registry already installed"),
            "{repeat}"
        );

        drop(managed);
        drop(model_manager);
        drt.shutdown();
    }
}
