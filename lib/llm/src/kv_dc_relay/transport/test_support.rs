// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;

use rcgen::{
    BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, KeyPair, KeyUsagePurpose,
};
use tempfile::TempDir;

use super::super::transport_config::KvDcRelayTransportConfig;

pub(super) struct TestPki {
    pub(super) ca_pem: String,
    pub(super) server_cert_pem: String,
    pub(super) server_key_pem: String,
    pub(super) client_cert_pem: String,
    pub(super) client_key_pem: String,
    pub(super) wrong_client_cert_pem: String,
    pub(super) wrong_client_key_pem: String,
}

pub(super) fn test_pki() -> TestPki {
    fn make_ca(common_name: &str) -> (rcgen::Certificate, KeyPair) {
        let mut params = CertificateParams::new(Vec::<String>::new()).unwrap();
        params
            .distinguished_name
            .push(rcgen::DnType::CommonName, common_name);
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params.key_usages = vec![
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::DigitalSignature,
            KeyUsagePurpose::CrlSign,
        ];
        let key = KeyPair::generate().unwrap();
        let cert = params.self_signed(&key).unwrap();
        (cert, key)
    }

    fn leaf(
        names: Vec<String>,
        usage: ExtendedKeyUsagePurpose,
        ca: &rcgen::Certificate,
        ca_key: &KeyPair,
    ) -> (String, String) {
        let mut params = CertificateParams::new(names).unwrap();
        params.extended_key_usages = vec![usage];
        let key = KeyPair::generate().unwrap();
        let cert = params.signed_by(&key, ca, ca_key).unwrap();
        (cert.pem(), key.serialize_pem())
    }

    let (ca, ca_key) = make_ca("relay-test-ca");
    let (wrong_ca, wrong_ca_key) = make_ca("relay-wrong-ca");
    let (server_cert_pem, server_key_pem) = leaf(
        vec!["localhost".to_string()],
        ExtendedKeyUsagePurpose::ServerAuth,
        &ca,
        &ca_key,
    );
    let (client_cert_pem, client_key_pem) = leaf(
        Vec::new(),
        ExtendedKeyUsagePurpose::ClientAuth,
        &ca,
        &ca_key,
    );
    let (wrong_client_cert_pem, wrong_client_key_pem) = leaf(
        Vec::new(),
        ExtendedKeyUsagePurpose::ClientAuth,
        &wrong_ca,
        &wrong_ca_key,
    );
    TestPki {
        ca_pem: ca.pem(),
        server_cert_pem,
        server_key_pem,
        client_cert_pem,
        client_key_pem,
        wrong_client_cert_pem,
        wrong_client_key_pem,
    }
}

pub(super) fn tls_test_config(temp: &TempDir, pki: &TestPki) -> KvDcRelayTransportConfig {
    let server_cert = temp.path().join("server.crt");
    let server_key = temp.path().join("server.key");
    let client_ca = temp.path().join("client-ca.crt");
    fs::write(&server_cert, &pki.server_cert_pem).unwrap();
    fs::write(&server_key, &pki.server_key_pem).unwrap();
    fs::write(&client_ca, &pki.ca_pem).unwrap();
    KvDcRelayTransportConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        tls_server_cert: server_cert,
        tls_server_key: server_key,
        tls_client_ca: client_ca,
        max_message_bytes: 8 * 1024 * 1024,
        keepalive_interval_ms: 20_000,
        keepalive_timeout_ms: 10_000,
        pool_heartbeat_interval_ms: 10_000,
        readiness_heartbeat_interval_ms: 10_000,
        load_window_ms: 10_000,
        load_fanout_capacity: 4,
        publication_queue_capacity: 4,
        publication_queue_bytes: 8 * 1024 * 1024,
        publication_encoding_concurrency: 1,
        max_catalog_subscribers: 4,
        max_pool_streams_total: 4,
        max_subscribers_per_pool: 4,
        max_initialized_pool_hubs: 4,
        max_readiness_subscribers: 4,
        max_load_subscribers: 4,
    }
}
