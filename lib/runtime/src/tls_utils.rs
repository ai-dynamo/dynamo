// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared TLS utilities for the Dynamo runtime.
//!
//! Provides helpers for loading PEM certificates and building rustls
//! `ServerConfig` / `ClientConfig` objects for transport-layer security.

use std::{path::Path, sync::Arc};

use anyhow::{Context, Result};
use rustls::{ClientConfig, RootCertStore, ServerConfig};
use rustls_pemfile::{certs, private_key};

/// TLS handshake timeout, configurable via `DYN_TCP_TLS_HANDSHAKE_TIMEOUT_SECS` (default: 3s).
pub fn handshake_timeout() -> std::time::Duration {
    use crate::config::environment_names::tcp_response_stream::tls as env;
    let secs = std::env::var(env::DYN_TCP_TLS_HANDSHAKE_TIMEOUT_SECS)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(3);
    std::time::Duration::from_secs(secs)
}

/// Build a rustls `ServerConfig` from PEM certificate and key files.
pub fn server_tls_config(cert_path: &Path, key_path: &Path) -> Result<ServerConfig> {
    let cert_pem = std::fs::read(cert_path)
        .with_context(|| format!("reading cert: {}", cert_path.display()))?;
    let key_pem =
        std::fs::read(key_path).with_context(|| format!("reading key: {}", key_path.display()))?;

    let cert_chain = certs(&mut cert_pem.as_slice())
        .collect::<Result<Vec<_>, _>>()
        .context("parsing certificate PEM")?;

    let key = private_key(&mut key_pem.as_slice())
        .context("parsing private key PEM")?
        .context("no private key found in PEM")?;

    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let config = ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .context("configuring TLS protocol versions")?
        .with_no_client_auth()
        .with_single_cert(cert_chain, key)
        .context("building ServerConfig")?;

    Ok(config)
}

/// Build a rustls `ClientConfig` for outbound TLS connections.
///
/// - `ca_cert_path`: trust this CA for verifying the server certificate.
///   When `None`, the root store is empty — supply a CA cert or use `insecure`.
/// - `insecure`: skip certificate verification entirely. **Dev/test only.**
pub fn client_tls_config(ca_cert_path: Option<&Path>, insecure: bool) -> Result<ClientConfig> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());

    if insecure {
        tracing::info!("TCP TLS: certificate verification disabled (insecure mode)");
        let config = ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .context("configuring TLS protocol versions")?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoVerifier))
            .with_no_client_auth();
        return Ok(config);
    }

    let mut root_store = RootCertStore::empty();
    if let Some(ca_path) = ca_cert_path {
        let ca_pem = std::fs::read(ca_path)
            .with_context(|| format!("reading CA cert: {}", ca_path.display()))?;
        let ca_certs = certs(&mut ca_pem.as_slice())
            .collect::<Result<Vec<_>, _>>()
            .context("parsing CA certificate PEM")?;
        for cert in ca_certs {
            root_store
                .add(cert)
                .context("adding CA certificate to root store")?;
        }
        if root_store.is_empty() {
            anyhow::bail!(
                "CA certificate store is empty after parsing {}; \
                 ensure the file contains at least one valid PEM certificate",
                ca_path.display()
            );
        }
    }
    // When no CA cert is provided, the root store is empty — the caller must
    // supply a CA cert or use `insecure = true`. This is intentional: in
    // cluster deployments, certs are issued by an internal CA and system roots
    // are not relevant.

    let config = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .context("configuring TLS protocol versions")?
        .with_root_certificates(root_store)
        .with_no_client_auth();

    Ok(config)
}

/// Certificate verifier that accepts any certificate.
/// **Only for development/testing. Never use in production.**
#[derive(Debug)]
struct NoVerifier;

impl rustls::client::danger::ServerCertVerifier for NoVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_cert_files() -> (NamedTempFile, NamedTempFile) {
        let key_pair = rcgen::KeyPair::generate().unwrap();
        let cert = rcgen::CertificateParams::new(vec!["localhost".to_string()])
            .unwrap()
            .self_signed(&key_pair)
            .unwrap();
        let mut cert_file = NamedTempFile::new().unwrap();
        cert_file.write_all(cert.pem().as_bytes()).unwrap();
        let mut key_file = NamedTempFile::new().unwrap();
        key_file
            .write_all(key_pair.serialize_pem().as_bytes())
            .unwrap();
        (cert_file, key_file)
    }

    #[test]
    fn server_config_roundtrip() {
        let (cert, key) = make_cert_files();
        server_tls_config(cert.path(), key.path()).unwrap();
    }

    #[test]
    fn server_config_bad_paths() {
        let missing = std::path::Path::new("/nonexistent/x.pem");
        assert!(
            server_tls_config(missing, missing)
                .unwrap_err()
                .to_string()
                .contains("reading cert")
        );
        let (cert, _) = make_cert_files();
        assert!(
            server_tls_config(cert.path(), missing)
                .unwrap_err()
                .to_string()
                .contains("reading key")
        );
    }

    #[test]
    fn client_config_insecure() {
        client_tls_config(None, true).unwrap();
    }

    #[test]
    fn client_config_with_ca() {
        let (cert, _) = make_cert_files();
        client_tls_config(Some(cert.path()), false).unwrap();
    }

    #[test]
    fn client_config_empty_ca_errors() {
        let empty = NamedTempFile::new().unwrap();
        assert!(
            client_tls_config(Some(empty.path()), false)
                .unwrap_err()
                .to_string()
                .contains("CA certificate store is empty")
        );
    }

    #[test]
    fn client_config_missing_ca_errors() {
        assert!(
            client_tls_config(Some(std::path::Path::new("/nonexistent/ca.pem")), false)
                .unwrap_err()
                .to_string()
                .contains("reading CA cert")
        );
    }
}
