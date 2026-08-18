// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use parking_lot::{Mutex, RwLock};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::codec::CompressionEncoding;
use tonic::transport::{Certificate, Identity, Server, ServerTlsConfig};
use x509_parser::prelude::FromDer as _;

use super::super::protocol::{FILE_DESCRIPTOR_SET, KvEventRelayServer};
use super::super::transport_config::KvDcRelayTransportConfig;
use super::grpc::{KvEventRelayService, KvEventRelayServiceConfig, SubscriberLimits};
use super::load::{LoadUpdateHub, run_load_publisher};
use super::source::WanPublicationSource;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct KvDcRelayTransportHealth {
    pub(crate) enabled: bool,
    pub(crate) serving: bool,
    pub(crate) bound_address: Option<SocketAddr>,
    pub(crate) server_cert_not_after: Option<i64>,
    pub(crate) client_ca_not_after: Option<i64>,
    pub(crate) last_error: Option<String>,
}

pub(crate) struct KvDcRelayTransport {
    cancel: CancellationToken,
    health: Arc<RwLock<KvDcRelayTransportHealth>>,
    task: Mutex<Option<JoinHandle<()>>>,
}

impl KvDcRelayTransport {
    pub(crate) async fn start(
        source: WanPublicationSource,
        config: KvDcRelayTransportConfig,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        let (tls, server_cert_not_after, client_ca_not_after) = build_tls_config(&config)?;
        let reflection = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
            .build_v1()
            .context("building KV Relay reflection service")?;
        let listener = TcpListener::bind(config.bind)
            .await
            .with_context(|| format!("binding KV Relay gRPC listener at {}", config.bind))?;
        let bound_address = listener
            .local_addr()
            .context("reading bound KV Relay gRPC listener address")?;
        let cancel = source.lifecycle().child_token();
        let fatal_cancel = source.lifecycle().clone();
        let health = Arc::new(RwLock::new(KvDcRelayTransportHealth {
            enabled: true,
            serving: false,
            bound_address: Some(bound_address),
            server_cert_not_after,
            client_ca_not_after,
            last_error: None,
        }));
        let load_window = Duration::from_millis(config.load_window_ms);
        let load_updates = LoadUpdateHub::new(&source, load_window, config.load_fanout_capacity);
        let limits = SubscriberLimits::new(
            config.max_catalog_subscribers,
            config.max_pool_streams_total,
            config.max_readiness_subscribers,
            config.max_load_subscribers,
        );
        let service = KvEventRelayServer::new(KvEventRelayService::new(
            source.clone(),
            cancel.clone(),
            KvEventRelayServiceConfig {
                pool_heartbeat_interval: Duration::from_millis(config.pool_heartbeat_interval_ms),
                readiness_heartbeat_interval: Duration::from_millis(
                    config.readiness_heartbeat_interval_ms,
                ),
                load_updates: load_updates.clone(),
                limits,
                snapshot_encoding_permits: Arc::new(Semaphore::new(
                    config.publication_encoding_concurrency,
                )),
            },
        ))
        .accept_compressed(CompressionEncoding::Zstd)
        .send_compressed(CompressionEncoding::Zstd)
        .max_encoding_message_size(config.max_message_bytes)
        .max_decoding_message_size(config.max_message_bytes);
        let (health_reporter, health_service) = tonic_health::server::health_reporter();
        let router = Server::builder()
            .http2_keepalive_interval(Some(Duration::from_millis(config.keepalive_interval_ms)))
            .http2_keepalive_timeout(Some(Duration::from_millis(config.keepalive_timeout_ms)))
            .tls_config(tls)
            .context("configuring KV Relay mTLS")?
            .add_service(service)
            .add_service(health_service)
            .add_service(reflection);

        health_reporter
            .set_serving::<KvEventRelayServer<KvEventRelayService>>()
            .await;
        health.write().serving = true;
        tracing::info!(
            bind = %bound_address,
            ?server_cert_not_after,
            ?client_ca_not_after,
            "Started KV DC Relay WAN transport with mandatory mTLS"
        );

        let server_cancel = cancel.clone();
        let server_health = health.clone();
        let server_task = tokio::spawn(async move {
            let shutdown = async move {
                server_cancel.cancelled().await;
                health_reporter
                    .set_not_serving::<KvEventRelayServer<KvEventRelayService>>()
                    .await;
                server_health.write().serving = false;
            };
            router
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown)
                .await
                .context("KV Relay gRPC server failed")
        });
        let load_task = tokio::spawn(run_load_publisher(
            source,
            load_window,
            load_updates,
            cancel.clone(),
        ));
        let supervisor_cancel = cancel.clone();
        let supervisor_health = health.clone();
        let task = tokio::spawn(supervise_transport(
            server_task,
            load_task,
            supervisor_cancel,
            fatal_cancel,
            supervisor_health,
        ));
        Ok(Self {
            cancel,
            health,
            task: Mutex::new(Some(task)),
        })
    }

    pub(crate) fn health(&self) -> KvDcRelayTransportHealth {
        self.health.read().clone()
    }

    pub(crate) async fn shutdown(&self) {
        self.cancel.cancel();
        let task = self.task.lock().take();
        if let Some(task) = task
            && let Err(error) = task.await
            && !error.is_cancelled()
        {
            tracing::warn!(%error, "KV Relay transport supervisor failed during shutdown");
        }
        self.health.write().serving = false;
    }
}

impl Drop for KvDcRelayTransport {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

async fn supervise_transport(
    mut server: JoinHandle<anyhow::Result<()>>,
    mut load: JoinHandle<anyhow::Result<()>>,
    cancel: CancellationToken,
    fatal_cancel: CancellationToken,
    health: Arc<RwLock<KvDcRelayTransportHealth>>,
) {
    enum Exit {
        Cancelled,
        Server(Result<anyhow::Result<()>, tokio::task::JoinError>),
        Load(Result<anyhow::Result<()>, tokio::task::JoinError>),
    }
    let exit = tokio::select! {
        biased;
        _ = cancel.cancelled() => Exit::Cancelled,
        result = &mut server => Exit::Server(result),
        result = &mut load => Exit::Load(result),
    };
    let fatal = match &exit {
        Exit::Cancelled => None,
        Exit::Server(Ok(Ok(()))) => Some("KV Relay gRPC server stopped unexpectedly".to_string()),
        Exit::Server(Ok(Err(error))) => Some(error.to_string()),
        Exit::Server(Err(error)) => Some(format!("KV Relay gRPC server task failed: {error}")),
        Exit::Load(Ok(Ok(()))) => Some("KV Relay load publisher stopped unexpectedly".to_string()),
        Exit::Load(Ok(Err(error))) => Some(error.to_string()),
        Exit::Load(Err(error)) => Some(format!("KV Relay load publisher task failed: {error}")),
    };
    if let Some(reason) = fatal {
        tracing::error!(error = %reason, "KV DC Relay transport failed");
        health.write().last_error = Some(reason);
        fatal_cancel.cancel();
    }
    cancel.cancel();
    match exit {
        Exit::Cancelled => {
            let _ = tokio::join!(server, load);
        }
        Exit::Server(_) => {
            let _ = load.await;
        }
        Exit::Load(_) => {
            let _ = server.await;
        }
    }
    health.write().serving = false;
}

fn build_tls_config(
    config: &KvDcRelayTransportConfig,
) -> anyhow::Result<(ServerTlsConfig, Option<i64>, Option<i64>)> {
    ensure_rustls_crypto_provider()?;
    let cert = fs::read(&config.tls_server_cert).with_context(|| {
        format!(
            "reading KV Relay TLS server certificate {}",
            config.tls_server_cert.display()
        )
    })?;
    let key = fs::read(&config.tls_server_key).with_context(|| {
        format!(
            "reading KV Relay TLS server key {}",
            config.tls_server_key.display()
        )
    })?;
    let client_ca = fs::read(&config.tls_client_ca).with_context(|| {
        format!(
            "reading KV Relay TLS client CA {}",
            config.tls_client_ca.display()
        )
    })?;
    let server_cert_not_after = earliest_not_after(&cert);
    let client_ca_not_after = earliest_not_after(&client_ca);
    if server_cert_not_after.is_none() {
        tracing::warn!("Could not parse KV Relay server certificate expiry");
    }
    if client_ca_not_after.is_none() {
        tracing::warn!("Could not parse KV Relay client CA expiry");
    }
    Ok((
        ServerTlsConfig::new()
            .identity(Identity::from_pem(cert, key))
            .client_ca_root(Certificate::from_pem(client_ca)),
        server_cert_not_after,
        client_ca_not_after,
    ))
}

fn ensure_rustls_crypto_provider() -> anyhow::Result<()> {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    }
    anyhow::ensure!(
        rustls::crypto::CryptoProvider::get_default().is_some(),
        "failed to install a process-level rustls crypto provider"
    );
    Ok(())
}

fn earliest_not_after(pem: &[u8]) -> Option<i64> {
    x509_parser::pem::Pem::iter_from_buffer(pem)
        .filter_map(|pem| {
            let pem = pem.ok()?;
            let (_, certificate) =
                x509_parser::certificate::X509Certificate::from_der(&pem.contents).ok()?;
            Some(certificate.validity().not_after.timestamp())
        })
        .min()
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::super::test_support::{test_pki, tls_test_config};
    use super::*;

    const TEST_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----\nMIIDBzCCAe+gAwIBAgIUPlwUBjTCvHFKr288WGaDmjFA5egwDQYJKoZIhvcNAQEL\nBQAwEzERMA8GA1UEAwwIc21va2UtY2EwHhcNMjYwNzE0MTcxNTA1WhcNMjYwNzE2\nMTcxNTA1WjATMREwDwYDVQQDDAhzbW9rZS1jYTCCASIwDQYJKoZIhvcNAQEBBQAD\nggEPADCCAQoCggEBANJzBlZQz7UysbejNPzBMEsVzZHCRx1Eu0TXzj1/FANYjIKz\nxjN+v49q2jwsvl7HMNK7jMWJ2V1oBv0W7ZNmSUYA1qyERN2j8c8Z257Cf+c0tGS7\nI/wbtDX/g9knCrUdelKYxub8oRhyGMP+iIMR5w1LUQvmAULoaszLRab4+GOV8ijU\nIYVZvxTsMFY0ztdG6pP/H7gIJXkuwdfqC+BRCXoO/ppWa2MdGz3zz+uaG8nPhX7u\nsBqohwHug8DPnBHCZT2jJisNcV3zylVNnGtPS/TnV288mgJbKecP2IDzS5GP35XZ\nrYToMZ9k7IDU97+BzCdLIpqc9ZFsWD92ANcaxA0CAwEAAaNTMFEwHQYDVR0OBBYE\nFEi1N71BRzWnbtAqVF3D3OQQLkOtMB8GA1UdIwQYMBaAFEi1N71BRzWnbtAqVF3D\n3OQQLkOtMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQELBQADggEBAIY8H/6Y\nR3E+hB24zGRInHAWP5HZQRPpPg8kR+eMvMN5xoJ5ShtmBXSUPERTqP8Y32qVGCvZ\nVFu026QlQf2itEXeVjXH/Uj60m3lBnu7/oEK08miNtIXC1fDQof3zcEa25794Tyr\nZygnb7ujYQjJDHAoP0DG0XPfGt08iP3BNCFLPmomz1CBSpXRri8W20/Enbv7XfRW\nY1DzvndIwXiastq4lcR02EoP3rDX1WQpodnt+8bVl35Knb//1dxlyAr4V2y6YZrG\nEdu8xGIZutL0KaA15LGy07BVivS3sy5uvXZaghtvHeYHf7up8g2Il3eHLuaiOjbB\nWAQz2ykKZ+IVQoM=\n-----END CERTIFICATE-----\n";

    #[test]
    fn certificate_expiry_uses_earliest_certificate_in_bundle() {
        assert_eq!(
            earliest_not_after(TEST_CERT_PEM.as_bytes()),
            Some(1_784_222_105)
        );
        let bundle = format!("{TEST_CERT_PEM}{TEST_CERT_PEM}");
        assert_eq!(earliest_not_after(bundle.as_bytes()), Some(1_784_222_105));
        assert_eq!(earliest_not_after(b"not a certificate"), None);
    }

    #[test]
    fn invalid_tls_material_is_rejected_while_building_the_server() {
        let temp = TempDir::new().unwrap();
        let pki = test_pki();
        let config = tls_test_config(&temp, &pki);
        fs::write(&config.tls_server_key, "not a private key").unwrap();
        let (tls, _, _) = build_tls_config(&config).unwrap();
        assert!(Server::builder().tls_config(tls).is_err());
    }

    #[tokio::test]
    async fn supervised_load_task_failure_cancels_the_relay() {
        let cancel = CancellationToken::new();
        let fatal = CancellationToken::new();
        let health = Arc::new(RwLock::new(KvDcRelayTransportHealth {
            enabled: true,
            serving: true,
            ..KvDcRelayTransportHealth::default()
        }));
        let server_cancel = cancel.clone();
        let server = tokio::spawn(async move {
            server_cancel.cancelled().await;
            Ok(())
        });
        let load = tokio::spawn(async { Ok(()) });
        supervise_transport(server, load, cancel, fatal.clone(), health.clone()).await;
        assert!(fatal.is_cancelled());
        assert!(!health.read().serving);
        assert!(
            health
                .read()
                .last_error
                .as_deref()
                .is_some_and(|error| error.contains("load publisher stopped unexpectedly"))
        );
    }

    #[tokio::test]
    async fn relay_root_cancellation_marks_wan_transport_not_serving() {
        let relay_cancel = CancellationToken::new();
        let transport_cancel = relay_cancel.child_token();
        let health = Arc::new(RwLock::new(KvDcRelayTransportHealth {
            enabled: true,
            serving: true,
            ..KvDcRelayTransportHealth::default()
        }));
        let server_cancel = transport_cancel.clone();
        let load_cancel = transport_cancel.clone();
        let server = tokio::spawn(async move {
            server_cancel.cancelled().await;
            Ok(())
        });
        let load = tokio::spawn(async move {
            load_cancel.cancelled().await;
            Ok(())
        });
        let supervisor = tokio::spawn(supervise_transport(
            server,
            load,
            transport_cancel,
            relay_cancel.clone(),
            health.clone(),
        ));

        relay_cancel.cancel();
        supervisor.await.unwrap();

        assert!(!health.read().serving);
        assert!(health.read().last_error.is_none());
    }
}
