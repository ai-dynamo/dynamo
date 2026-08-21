// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared TLS utilities for the Dynamo runtime.
//!
//! Provides helpers for loading PEM certificates and building rustls
//! `ServerConfig` / `ClientConfig` objects for transport-layer security.

use std::{
    fmt,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use rustls::server::{ClientHello, ResolvesServerCert};
use rustls::sign::CertifiedKey;
use rustls::{ClientConfig, RootCertStore, ServerConfig, SignatureScheme};
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
///
/// The certificate is served through a `ReloadingCertifiedKey`, so a rotated
/// cert/key on disk (in-place rewrite or an atomic symlink swap) is picked up
/// automatically on the next handshake without restarting the process. The
/// initial load is validated eagerly: an invalid cert/key path
/// fails here rather than starting a server that cannot serve TLS.
///
/// When `client_ca_cert_path` is `Some`, the server requires clients to present
/// a certificate signed by that CA (mutual TLS); an unauthenticated client is
/// rejected at the handshake. When `None`, client certificates are not
/// requested.
pub fn server_tls_config(
    cert_path: &Path,
    key_path: &Path,
    client_ca_cert_path: Option<&Path>,
) -> Result<ServerConfig> {
    let resolver = Arc::new(ReloadingCertifiedKey::new(cert_path, key_path)?);

    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let builder = ServerConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .context("configuring TLS protocol versions")?;

    let config = if let Some(ca_path) = client_ca_cert_path {
        let ca_pem = std::fs::read(ca_path)
            .with_context(|| format!("reading client CA cert: {}", ca_path.display()))?;
        let ca_certs = certs(&mut ca_pem.as_slice())
            .collect::<Result<Vec<_>, _>>()
            .context("parsing client CA certificate PEM")?;
        let mut client_roots = RootCertStore::empty();
        for cert in ca_certs {
            client_roots
                .add(cert)
                .context("adding client CA certificate to root store")?;
        }
        if client_roots.is_empty() {
            anyhow::bail!(
                "client CA certificate store is empty after parsing {}; \
                 ensure the file contains at least one valid PEM certificate",
                ca_path.display()
            );
        }
        let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
            Arc::new(client_roots),
            provider,
        )
        .build()
        .context("building client certificate verifier")?;
        builder
            .with_client_cert_verifier(verifier)
            .with_cert_resolver(resolver)
    } else {
        builder.with_no_client_auth().with_cert_resolver(resolver)
    };

    Ok(config)
}

/// Build a server `ServerConfig` for a TCP plane from optional cert/key/client-CA
/// paths, with the validation and misconfiguration diagnostics shared by the
/// request-plane and response-stream servers. `plane` labels the log lines
/// (e.g. `"TCP request plane"` / `"TCP server"`).
///
/// Returns `Ok(None)` for the plaintext case and fails closed on partial or
/// invalid configuration (cert without key, a client CA without a server
/// cert/key). When a client CA is supplied, the resulting config enforces mTLS.
pub fn server_tls_acceptor_config(
    plane: &str,
    cert: Option<&Path>,
    key: Option<&Path>,
    client_ca: Option<&Path>,
) -> Result<Option<ServerConfig>> {
    use crate::config::environment_names::tcp_response_stream::tls as env;
    match (cert, key) {
        (Some(cert), Some(key)) => {
            let config = server_tls_config(cert, key, client_ca)
                .with_context(|| format!("building {plane} TLS config from cert/key/client CA"))?;
            if client_ca.is_some() {
                tracing::info!(
                    plane,
                    "TLS enabled with mutual authentication (client certificates required)"
                );
                // Every component also dials peers as a client. Enforcing client
                // certs here while presenting no identity of our own means our
                // outbound handshakes to other mTLS peers would fail.
                let has_client_identity = std::env::var(env::DYN_TCP_TLS_CLIENT_CERT_PATH).is_ok()
                    && std::env::var(env::DYN_TCP_TLS_CLIENT_KEY_PATH).is_ok();
                if !has_client_identity {
                    tracing::warn!(
                        plane,
                        client_ca_var = env::DYN_TCP_TLS_CLIENT_CA_CERT_PATH,
                        client_cert_var = env::DYN_TCP_TLS_CLIENT_CERT_PATH,
                        client_key_var = env::DYN_TCP_TLS_CLIENT_KEY_PATH,
                        "server enforces client certificates but no client identity is configured; outbound connections to peers that also enforce mTLS will fail the handshake",
                    );
                }
            } else {
                tracing::info!(plane, "TLS enabled");
            }
            // Applies to both TLS and mTLS: if the client side has no way to
            // verify this server, peers dialing it fail the handshake with an
            // opaque error.
            let client_trust_set = std::env::var(env::DYN_TCP_TLS_CA_CERT_PATH).is_ok()
                || crate::config::env_is_truthy(env::DYN_TCP_TLS_INSECURE);
            if !client_trust_set {
                tracing::warn!(
                    plane,
                    ca_var = env::DYN_TCP_TLS_CA_CERT_PATH,
                    insecure_var = env::DYN_TCP_TLS_INSECURE,
                    "server has TLS enabled but no client trust is configured; peers cannot verify this server",
                );
            }
            Ok(Some(config))
        }
        (Some(_), None) | (None, Some(_)) => anyhow::bail!(
            "both {} and {} must be set to enable {plane} TLS",
            env::DYN_TCP_TLS_CERT_PATH,
            env::DYN_TCP_TLS_KEY_PATH,
        ),
        (None, None) if client_ca.is_some() => anyhow::bail!(
            "{} requires {} and {} to also be set",
            env::DYN_TCP_TLS_CLIENT_CA_CERT_PATH,
            env::DYN_TCP_TLS_CERT_PATH,
            env::DYN_TCP_TLS_KEY_PATH,
        ),
        (None, None) => {
            let client_trust_set = std::env::var(env::DYN_TCP_TLS_CA_CERT_PATH).is_ok()
                || crate::config::env_is_truthy(env::DYN_TCP_TLS_INSECURE);
            if client_trust_set {
                tracing::warn!(
                    plane,
                    cert_var = env::DYN_TCP_TLS_CERT_PATH,
                    key_var = env::DYN_TCP_TLS_KEY_PATH,
                    "server is running in plaintext but client TLS env vars are set; set the server cert/key to enable TLS, or unset the client vars",
                );
            }
            Ok(None)
        }
    }
}

/// Build a rustls `ClientConfig` for outbound TLS connections.
///
/// - `ca_cert_path`: trust this CA for verifying the server certificate.
///   When `None`, the root store is empty — supply a CA cert or use `insecure`.
/// - `insecure`: skip certificate verification entirely. **Dev/test only.**
/// - `client_cert_path` + `client_key_path`: when both are `Some`, the client
///   presents this certificate to the server (mutual TLS). The identity is
///   served through a `ReloadingCertifiedKey`, so a rotated client cert/key on
///   disk is picked up without a process restart. Both must be set together.
pub fn client_tls_config(
    ca_cert_path: Option<&Path>,
    insecure: bool,
    client_cert_path: Option<&Path>,
    client_key_path: Option<&Path>,
) -> Result<ClientConfig> {
    if client_cert_path.is_some() != client_key_path.is_some() {
        anyhow::bail!("client cert and key paths must both be set or both be unset");
    }

    let provider = Arc::new(rustls::crypto::ring::default_provider());

    if insecure {
        tracing::info!("TLS: certificate verification disabled (insecure mode)");
        let builder = ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .context("configuring TLS protocol versions")?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoVerifier));
        let config = match (client_cert_path, client_key_path) {
            (Some(cp), Some(kp)) => {
                builder.with_client_cert_resolver(Arc::new(ReloadingCertifiedKey::new(cp, kp)?))
            }
            _ => builder.with_no_client_auth(),
        };
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

    let builder = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .context("configuring TLS protocol versions")?
        .with_root_certificates(root_store);
    let config = match (client_cert_path, client_key_path) {
        (Some(cp), Some(kp)) => {
            builder.with_client_cert_resolver(Arc::new(ReloadingCertifiedKey::new(cp, kp)?))
        }
        _ => builder.with_no_client_auth(),
    };

    Ok(config)
}

/// Load a leaf certificate chain + private key from PEM bytes into a rustls
/// [`CertifiedKey`], validating that the certificate and key match.
fn load_certified_key(cert_pem: &[u8], key_pem: &[u8]) -> Result<CertifiedKey> {
    let mut cert_reader = cert_pem;
    let cert_chain = certs(&mut cert_reader)
        .collect::<Result<Vec<_>, _>>()
        .context("parsing certificate PEM")?;
    if cert_chain.is_empty() {
        anyhow::bail!("no certificates found in PEM");
    }

    let mut key_reader = key_pem;
    let key = private_key(&mut key_reader)
        .context("parsing private key PEM")?
        .context("no private key found in PEM")?;
    let signing_key =
        rustls::crypto::ring::sign::any_supported_type(&key).context("loading TLS private key")?;

    let certified_key = CertifiedKey::new(cert_chain, signing_key);
    certified_key
        .keys_match()
        .context("TLS certificate and private key do not match")?;
    Ok(certified_key)
}

/// Content fingerprint of a loaded identity. Change is detected by hashing the
/// file *contents* (blake3) rather than mtime, so atomic symlink swaps (where a
/// mounted directory of certs is rotated by relinking) are handled reliably.
#[derive(Debug, Eq, PartialEq)]
struct IdentityFingerprint {
    content_hash: [u8; 32],
}

impl IdentityFingerprint {
    fn from_loaded(cert_pem: &[u8], key_pem: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(cert_pem);
        hasher.update(&[0]); // domain separator between cert and key
        hasher.update(key_pem);
        Self {
            content_hash: *hasher.finalize().as_bytes(),
        }
    }
}

struct LoadedIdentity {
    fingerprint: IdentityFingerprint,
    certified_key: Arc<CertifiedKey>,
}

/// Shared reloadable identity. The current identity lives in an [`ArcSwap`]
/// read lock-free on the handshake path; a background thread owns the filesystem
/// reads and swaps in a new identity when the on-disk contents change.
struct ReloadingState {
    cert_path: PathBuf,
    key_path: PathBuf,
    current: ArcSwap<CertifiedKey>,
    /// Fingerprint of the last successfully loaded identity. Only the background
    /// reloader (and tests) touch this, so it never contends with handshakes.
    fingerprint: Mutex<IdentityFingerprint>,
}

impl ReloadingState {
    fn load(cert_path: &Path, key_path: &Path) -> Result<LoadedIdentity> {
        let cert_pem = std::fs::read(cert_path)
            .with_context(|| format!("reading cert: {}", cert_path.display()))?;
        let key_pem = std::fs::read(key_path)
            .with_context(|| format!("reading key: {}", key_path.display()))?;
        let certified_key = load_certified_key(&cert_pem, &key_pem)?;
        let fingerprint = IdentityFingerprint::from_loaded(&cert_pem, &key_pem);
        Ok(LoadedIdentity {
            fingerprint,
            certified_key: Arc::new(certified_key),
        })
    }

    /// Re-read the identity from disk and swap it in if the contents changed.
    /// Runs off the handshake path (background thread / tests). A failed read
    /// leaves the last valid identity in place and propagates the error so the
    /// caller can back off.
    fn refresh(&self) -> Result<()> {
        let reloaded = Self::load(&self.cert_path, &self.key_path)?;
        let mut fingerprint = self
            .fingerprint
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if *fingerprint != reloaded.fingerprint {
            let cert_count = reloaded.certified_key.cert.len();
            self.current.store(reloaded.certified_key);
            *fingerprint = reloaded.fingerprint;
            tracing::info!(
                cert_path = %self.cert_path.display(),
                cert_count,
                "Reloaded rotated TLS certificate and key from disk"
            );
        }
        Ok(())
    }

    /// Spawn a background thread that periodically refreshes the identity. It
    /// holds only a `Weak` reference, so it exits once the resolver is dropped.
    /// A plain OS thread (not a Tokio task) keeps the filesystem reads off every
    /// async runtime worker and avoids depending on a runtime being present when
    /// the resolver is built.
    fn spawn_reloader(state: &Arc<Self>) {
        let weak = Arc::downgrade(state);
        let spawned = std::thread::Builder::new()
            .name("tls-cert-reloader".to_string())
            .spawn(move || {
                let mut interval = ReloadingCertifiedKey::RELOAD_CHECK_INTERVAL;
                loop {
                    std::thread::sleep(interval);
                    let Some(state) = weak.upgrade() else {
                        break; // resolver dropped; stop reloading
                    };
                    match state.refresh() {
                        Ok(()) => interval = ReloadingCertifiedKey::RELOAD_CHECK_INTERVAL,
                        Err(error) => {
                            interval = ReloadingCertifiedKey::FAILURE_RETRY_INTERVAL;
                            tracing::warn!(
                                cert_path = %state.cert_path.display(),
                                error = %format!("{error:#}"),
                                "Failed to reload rotated TLS certificate; keeping the last valid identity"
                            );
                        }
                    }
                }
            });
        if let Err(error) = spawned {
            tracing::warn!(
                error = %error,
                "Failed to spawn TLS certificate reloader thread; certificate hot-reload is disabled for this identity"
            );
        }
    }
}

/// A rustls certificate resolver whose served identity is refreshed from disk by
/// a background thread, so a rotated cert/key (in-place rewrite or atomic
/// symlink swap) is picked up without a process restart.
///
/// `resolve()` only reads the current identity from an [`ArcSwap`] — it performs
/// no filesystem I/O and never blocks, so it is safe to call from the
/// `tokio-rustls` handshake poll. A failed reload keeps the last valid identity.
///
/// The same type serves as both a [`ResolvesServerCert`] (server leaf cert) and
/// a [`rustls::client::ResolvesClientCert`] (mTLS client identity).
pub(crate) struct ReloadingCertifiedKey {
    state: Arc<ReloadingState>,
}

impl fmt::Debug for ReloadingCertifiedKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReloadingCertifiedKey")
            .field("cert_path", &self.state.cert_path)
            .field("key_path", &self.state.key_path)
            .finish_non_exhaustive()
    }
}

impl ReloadingCertifiedKey {
    const RELOAD_CHECK_INTERVAL: Duration = Duration::from_secs(30);
    const FAILURE_RETRY_INTERVAL: Duration = Duration::from_secs(1);

    fn new(cert_path: &Path, key_path: &Path) -> Result<Self> {
        let loaded = ReloadingState::load(cert_path, key_path)?;
        let state = Arc::new(ReloadingState {
            cert_path: cert_path.to_path_buf(),
            key_path: key_path.to_path_buf(),
            current: ArcSwap::from(loaded.certified_key),
            fingerprint: Mutex::new(loaded.fingerprint),
        });
        ReloadingState::spawn_reloader(&state);
        Ok(Self { state })
    }

    fn resolve_key(&self) -> Arc<CertifiedKey> {
        self.state.current.load_full()
    }

    /// Test-only: perform one synchronous reload cycle (what the background
    /// thread does on each tick) so tests can drive rotation deterministically.
    #[cfg(test)]
    fn reload_now(&self) -> Result<()> {
        self.state.refresh()
    }
}

impl ResolvesServerCert for ReloadingCertifiedKey {
    fn resolve(&self, _client_hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        Some(self.resolve_key())
    }
}

impl rustls::client::ResolvesClientCert for ReloadingCertifiedKey {
    fn resolve(
        &self,
        _root_hint_subjects: &[&[u8]],
        _sigschemes: &[SignatureScheme],
    ) -> Option<Arc<CertifiedKey>> {
        Some(self.resolve_key())
    }

    fn has_certs(&self) -> bool {
        true
    }
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
        server_tls_config(cert.path(), key.path(), None).unwrap();
    }

    #[test]
    fn server_config_with_mtls() {
        // A client CA turns on client-certificate verification (mTLS).
        let (cert, key) = make_cert_files();
        server_tls_config(cert.path(), key.path(), Some(cert.path())).unwrap();
    }

    #[test]
    fn server_config_mtls_empty_client_ca_errors() {
        let (cert, key) = make_cert_files();
        let empty = NamedTempFile::new().unwrap();
        assert!(
            server_tls_config(cert.path(), key.path(), Some(empty.path()))
                .unwrap_err()
                .to_string()
                .contains("client CA certificate store is empty")
        );
    }

    #[test]
    fn server_config_bad_paths() {
        let missing = std::path::Path::new("/nonexistent/x.pem");
        assert!(
            server_tls_config(missing, missing, None)
                .unwrap_err()
                .to_string()
                .contains("reading cert")
        );
        let (cert, _) = make_cert_files();
        assert!(
            server_tls_config(cert.path(), missing, None)
                .unwrap_err()
                .to_string()
                .contains("reading key")
        );
    }

    fn make_cert_pem() -> (String, String) {
        let key_pair = rcgen::KeyPair::generate().unwrap();
        let cert = rcgen::CertificateParams::new(vec!["localhost".to_string()])
            .unwrap()
            .self_signed(&key_pair)
            .unwrap();
        (cert.pem(), key_pair.serialize_pem())
    }

    #[test]
    fn certified_key_reloads_rotated_files() {
        let (cert1, key1) = make_cert_pem();
        let cert_file = NamedTempFile::new().unwrap();
        let key_file = NamedTempFile::new().unwrap();
        std::fs::write(cert_file.path(), &cert1).unwrap();
        std::fs::write(key_file.path(), &key1).unwrap();

        let resolver = ReloadingCertifiedKey::new(cert_file.path(), key_file.path()).unwrap();
        let before = resolver.resolve_key().cert[0].clone();

        // Rotate the file contents in place and force a re-check.
        let (cert2, key2) = make_cert_pem();
        std::fs::write(cert_file.path(), &cert2).unwrap();
        std::fs::write(key_file.path(), &key2).unwrap();
        resolver.reload_now().unwrap();

        let after = resolver.resolve_key().cert[0].clone();
        assert_ne!(
            before, after,
            "resolver should serve the rotated certificate after the contents change"
        );
    }

    #[test]
    fn certified_key_reloads_symlinked_generation() {
        use std::os::unix::fs::symlink;

        // Mimic a symlink-based cert rotation: the mounted paths are symlinks
        // into a per-generation directory, rotated by an atomic rename over the
        // link.
        let dir = tempfile::tempdir().unwrap();
        let (c1, k1) = make_cert_pem();
        let gen1 = dir.path().join("gen1");
        std::fs::create_dir(&gen1).unwrap();
        std::fs::write(gen1.join("tls.crt"), &c1).unwrap();
        std::fs::write(gen1.join("tls.key"), &k1).unwrap();

        let cert_link = dir.path().join("tls.crt");
        let key_link = dir.path().join("tls.key");
        symlink(gen1.join("tls.crt"), &cert_link).unwrap();
        symlink(gen1.join("tls.key"), &key_link).unwrap();

        let resolver = ReloadingCertifiedKey::new(&cert_link, &key_link).unwrap();
        let before = resolver.resolve_key().cert[0].clone();

        let (c2, k2) = make_cert_pem();
        let gen2 = dir.path().join("gen2");
        std::fs::create_dir(&gen2).unwrap();
        std::fs::write(gen2.join("tls.crt"), &c2).unwrap();
        std::fs::write(gen2.join("tls.key"), &k2).unwrap();
        // Atomic symlink swap: create new links then rename over the live ones.
        let cert_tmp = dir.path().join("tls.crt.tmp");
        let key_tmp = dir.path().join("tls.key.tmp");
        symlink(gen2.join("tls.crt"), &cert_tmp).unwrap();
        symlink(gen2.join("tls.key"), &key_tmp).unwrap();
        std::fs::rename(&cert_tmp, &cert_link).unwrap();
        std::fs::rename(&key_tmp, &key_link).unwrap();
        resolver.reload_now().unwrap();

        let after = resolver.resolve_key().cert[0].clone();
        assert_ne!(
            before, after,
            "resolver should follow the swapped symlink to the new generation"
        );
    }

    #[test]
    fn certified_key_keeps_previous_on_corrupt_reload() {
        let (c1, k1) = make_cert_pem();
        let cert_file = NamedTempFile::new().unwrap();
        let key_file = NamedTempFile::new().unwrap();
        std::fs::write(cert_file.path(), &c1).unwrap();
        std::fs::write(key_file.path(), &k1).unwrap();
        let resolver = ReloadingCertifiedKey::new(cert_file.path(), key_file.path()).unwrap();
        let before = resolver.resolve_key().cert[0].clone();

        // Simulate a partial write mid-rotation.
        std::fs::write(cert_file.path(), b"not a valid pem").unwrap();
        let reload_result = resolver.reload_now();

        assert!(
            reload_result.is_err(),
            "a corrupt reload should surface an error to the caller"
        );
        let after = resolver.resolve_key().cert[0].clone();
        assert_eq!(
            before, after,
            "a failed reload must keep serving the previously loaded certificate"
        );
    }

    #[test]
    fn client_config_insecure() {
        client_tls_config(None, true, None, None).unwrap();
    }

    #[test]
    fn client_config_with_ca() {
        let (cert, _) = make_cert_files();
        client_tls_config(Some(cert.path()), false, None, None).unwrap();
    }

    #[test]
    fn client_config_with_mtls() {
        // A client cert/key pair is presented as the client identity (mTLS).
        let (cert, key) = make_cert_files();
        client_tls_config(
            Some(cert.path()),
            false,
            Some(cert.path()),
            Some(key.path()),
        )
        .unwrap();
    }

    #[test]
    fn client_config_mtls_insecure() {
        // Client identity is also honored in insecure (no server verification) mode.
        let (cert, key) = make_cert_files();
        client_tls_config(None, true, Some(cert.path()), Some(key.path())).unwrap();
    }

    #[test]
    fn client_config_partial_mtls_errors() {
        // Cert without key (or vice versa) is rejected.
        let (cert, _) = make_cert_files();
        assert!(client_tls_config(Some(cert.path()), false, Some(cert.path()), None).is_err());
    }

    #[test]
    fn client_config_empty_ca_errors() {
        let empty = NamedTempFile::new().unwrap();
        assert!(
            client_tls_config(Some(empty.path()), false, None, None)
                .unwrap_err()
                .to_string()
                .contains("CA certificate store is empty")
        );
    }

    #[test]
    fn client_config_missing_ca_errors() {
        assert!(
            client_tls_config(
                Some(std::path::Path::new("/nonexistent/ca.pem")),
                false,
                None,
                None
            )
            .unwrap_err()
            .to_string()
            .contains("reading CA cert")
        );
    }
}
