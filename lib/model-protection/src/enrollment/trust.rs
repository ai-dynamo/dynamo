// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use rustls_pki_types::{CertificateDer, UnixTime};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Duration;
use webpki::{EndEntityCert, KeyUsage, anchor_from_trusted_cert};
use x509_parser::{
    extensions::GeneralName, parse_x509_certificate, prelude::X509Certificate,
    public_key::PublicKey,
};

use super::format::{ValidatedRequest, hex, identifier, parse};
use super::{EnrollmentError as Error, Result};
use crate::format::has_valid_ed25519_signature;

pub const TRUST_POLICY_DOMAIN: &[u8] = b"model-protection-enrollment-trust-policy-v1\0";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustPolicy {
    pub format: String,
    pub format_version: u16,
    pub policy_id: String,
    pub sequence: u64,
    pub valid_from: u64,
    pub expires_at: u64,
    pub root_certificates: Vec<String>,
    pub allowed_manufacturers: Vec<String>,
    pub revoked_certificate_sha256: Vec<String>,
    pub revocation_mode: String,
}

pub struct VerifiedTrustPolicy {
    policy: TrustPolicy,
    digest: [u8; 32],
}

pub struct VerifiedEk {
    request_digest: [u8; 32],
    policy_digest: [u8; 32],
}

impl VerifiedTrustPolicy {
    pub fn policy_id(&self) -> &str {
        &self.policy.policy_id
    }
    pub fn sequence(&self) -> u64 {
        self.policy.sequence
    }
    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
}

impl VerifiedEk {
    pub fn request_digest(&self) -> &[u8; 32] {
        &self.request_digest
    }
    pub fn policy_digest(&self) -> &[u8; 32] {
        &self.policy_digest
    }
}

/// `minimum_sequence` must come from rollback-resistant operator state, not the bundle.
pub fn verify_trust_policy(
    bytes: &[u8],
    envelope: &[u8],
    key_id: &str,
    key: &[u8; 32],
    minimum_sequence: u64,
    now: u64,
) -> Result<VerifiedTrustPolicy> {
    if !has_valid_ed25519_signature(TRUST_POLICY_DOMAIN, bytes, envelope, key_id, key) {
        return Err(Error::Signature);
    }
    let policy: TrustPolicy = parse(bytes)?;
    if policy.format != "model-protection-enrollment-trust-policy"
        || policy.format_version != 1
        || !identifier(&policy.policy_id)
        || policy.sequence == 0
        || policy.sequence < minimum_sequence
        || policy.revocation_mode != "operator-fingerprint-snapshot-v1"
        || policy.root_certificates.is_empty()
        || policy.root_certificates.len() > 16
        || policy.allowed_manufacturers.is_empty()
        || policy.allowed_manufacturers.len() > 16
        || policy.revoked_certificate_sha256.len() > 1024
        || policy.valid_from > now
        || policy.expires_at <= now
        || policy.expires_at <= policy.valid_from
        || policy.expires_at - policy.valid_from > 7 * 24 * 3600
    {
        return Err(Error::Trust);
    }
    for fingerprint in &policy.revoked_certificate_sha256 {
        crate::format::decode_lower_hex::<32>(fingerprint, "revoked fingerprint")
            .map_err(|_| Error::Trust)?;
    }
    for manufacturer in &policy.allowed_manufacturers {
        if manufacturer.len() != 11
            || !manufacturer.starts_with("id:")
            || !manufacturer[3..].bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(Error::Trust);
        }
    }
    Ok(VerifiedTrustPolicy {
        policy,
        digest: Sha256::digest(bytes).into(),
    })
}

fn certificate(value: &str) -> Result<Vec<u8>> {
    if value.len() > 22 * 1024 {
        return Err(Error::Trust);
    }
    let der = BASE64.decode(value).map_err(|_| Error::Trust)?;
    if der.is_empty() || der.len() > 16 * 1024 || BASE64.encode(&der) != value {
        return Err(Error::Trust);
    }
    let (remaining, _) = parse_x509_certificate(&der).map_err(|_| Error::Trust)?;
    if !remaining.is_empty() {
        return Err(Error::Trust);
    }
    Ok(der)
}

fn parse_certificate(der: &[u8]) -> Result<X509Certificate<'_>> {
    let (remaining, parsed) = parse_x509_certificate(der).map_err(|_| Error::Trust)?;
    if !remaining.is_empty() {
        return Err(Error::Trust);
    }
    Ok(parsed)
}

fn validate_trust_anchor(der: &[u8], now: u64) -> Result<()> {
    let root = parse_certificate(der)?;
    let not_before = root.validity().not_before.timestamp();
    let not_after = root.validity().not_after.timestamp();
    if root.subject() != root.issuer()
        || !root.is_ca()
        || root
            .key_usage()
            .map_err(|_| Error::Trust)?
            .map_or(true, |usage| !usage.value.key_cert_sign())
        || root.verify_signature(None).is_err()
        || now < u64::try_from(not_before).map_err(|_| Error::Trust)?
        || now >= u64::try_from(not_after).map_err(|_| Error::Trust)?
    {
        return Err(Error::Trust);
    }
    Ok(())
}

/// No AIA, CRL URL or other customer-controlled network resource is fetched.
/// Revocation freshness is the operator-signed snapshot's responsibility.
pub fn verify_ek(
    request: &ValidatedRequest,
    policy: &VerifiedTrustPolicy,
    now: u64,
) -> Result<VerifiedEk> {
    if now < policy.policy.valid_from || now >= policy.policy.expires_at {
        return Err(Error::Trust);
    }
    let chain = &request.request().ek_certificate_chain;
    if chain.is_empty() || chain.len() > 6 {
        return Err(Error::Trust);
    }
    let leaf_der = certificate(chain.first().ok_or(Error::Trust)?)?;
    let parsed = parse_certificate(&leaf_der)?;
    if parsed.version().0 != 2 || parsed.is_ca() {
        return Err(Error::Trust);
    }
    let ku = parsed
        .key_usage()
        .map_err(|_| Error::Trust)?
        .ok_or(Error::Trust)?;
    if !ku.value.key_encipherment() || ku.value.key_cert_sign() || ku.value.digital_signature() {
        return Err(Error::Trust);
    }
    let eku = parsed
        .extended_key_usage()
        .map_err(|_| Error::Trust)?
        .ok_or(Error::Trust)?;
    if !eku
        .value
        .other
        .iter()
        .any(|oid| oid.to_id_string() == "2.23.133.8.1")
    {
        return Err(Error::Trust);
    }
    let san = parsed
        .subject_alternative_name()
        .map_err(|_| Error::Trust)?
        .ok_or(Error::Trust)?;
    if parsed.subject().iter().next().is_none() && !san.critical {
        return Err(Error::Trust);
    }
    let mut manufacturers = Vec::new();
    for name in &san.value.general_names {
        if let GeneralName::DirectoryName(directory) = name {
            for attribute in directory.iter_attributes() {
                if attribute.attr_type().to_id_string() == "2.23.133.2.1" {
                    manufacturers.push(attribute.as_str().map_err(|_| Error::Trust)?);
                }
            }
        }
    }
    if manufacturers.len() != 1
        || !policy
            .policy
            .allowed_manufacturers
            .iter()
            .any(|m| m == manufacturers[0])
    {
        return Err(Error::Trust);
    }
    let PublicKey::RSA(rsa) = parsed.public_key().parsed().map_err(|_| Error::Trust)? else {
        return Err(Error::Trust);
    };
    let ek_public = BASE64
        .decode(&request.request().ek_public)
        .map_err(|_| Error::Trust)?;
    let modulus = rsa.modulus.strip_prefix(&[0]).unwrap_or(rsa.modulus);
    let exponent = rsa.exponent.strip_prefix(&[0]).unwrap_or(rsa.exponent);
    if ek_public.len() < 58
        || rsa.key_size() != 2048
        || exponent != [1, 0, 1]
        || modulus != &ek_public[58..]
    {
        return Err(Error::Trust);
    }
    let mut chain_der = Vec::with_capacity(chain.len());
    chain_der.push(leaf_der);
    for cert in &chain[1..] {
        chain_der.push(certificate(cert)?);
    }
    let mut root_der = Vec::with_capacity(policy.policy.root_certificates.len());
    for root in &policy.policy.root_certificates {
        let der = certificate(root)?;
        validate_trust_anchor(&der, now)?;
        root_der.push(der);
    }
    for der in chain_der.iter().chain(&root_der) {
        let fingerprint = hex(&Sha256::digest(der));
        if policy
            .policy
            .revoked_certificate_sha256
            .contains(&fingerprint)
        {
            return Err(Error::Trust);
        }
    }

    let trust_certificates: Vec<_> = root_der
        .iter()
        .map(|der| CertificateDer::from(der.as_slice()))
        .collect();
    let trust_anchors: Vec<_> = trust_certificates
        .iter()
        .map(anchor_from_trusted_cert)
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| Error::Trust)?;
    let intermediate_certificates: Vec<_> = chain_der[1..]
        .iter()
        .map(|der| CertificateDer::from(der.as_slice()))
        .collect();
    let end_entity_der = CertificateDer::from(chain_der[0].as_slice());
    let end_entity = EndEntityCert::try_from(&end_entity_der).map_err(|_| Error::Trust)?;
    end_entity
        .verify_for_usage(
            webpki::ALL_VERIFICATION_ALGS,
            &trust_anchors,
            &intermediate_certificates,
            UnixTime::since_unix_epoch(Duration::from_secs(now)),
            KeyUsage::required(&[0x67, 0x81, 0x05, 0x08, 0x01]),
            None,
            None,
        )
        .map_err(|_| Error::Trust)?;
    Ok(VerifiedEk {
        request_digest: *request.digest(),
        policy_digest: policy.digest,
    })
}

#[cfg(test)]
mod tests {
    use super::super::format::{test_request, validate_request};
    use super::*;
    use pkcs8::EncodePrivateKey;
    use rcgen::{
        BasicConstraints, CertificateParams, CustomExtension, DistinguishedName, DnType, IsCa,
        KeyPair, KeyUsagePurpose, PKCS_ECDSA_P256_SHA256, PKCS_RSA_SHA256,
    };
    use ring::signature::{Ed25519KeyPair, KeyPair as RingKeyPair};
    use rsa::{
        RsaPrivateKey,
        pkcs1v15::SigningKey,
        rand_core::OsRng,
        signature::{SignatureEncoding, Signer},
        traits::PublicKeyParts,
    };
    use sha2::Sha256;

    fn test_certificates(now: u64) -> (Vec<u8>, Vec<u8>, RsaPrivateKey, RsaPrivateKey) {
        let ca_key = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
        let mut ca_params = CertificateParams::default();
        ca_params.distinguished_name = DistinguishedName::new();
        ca_params
            .distinguished_name
            .push(DnType::CommonName, "Test TPM root");
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        ca_params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        ca_params.not_before = time::OffsetDateTime::from_unix_timestamp(now as i64 - 100).unwrap();
        ca_params.not_after = time::OffsetDateTime::from_unix_timestamp(now as i64 + 1000).unwrap();
        let ca = ca_params.self_signed(&ca_key).unwrap();

        let ek_key = RsaPrivateKey::new(&mut OsRng, 2048).unwrap();
        let ek_pkcs8 = ek_key.to_pkcs8_der().unwrap();
        let ek_signing_key = KeyPair::from_pkcs8_der_and_sign_algo(
            &rustls_pki_types::PrivatePkcs8KeyDer::from(ek_pkcs8.as_bytes()),
            &PKCS_RSA_SHA256,
        )
        .unwrap();
        let mut leaf_params = CertificateParams::default();
        leaf_params.distinguished_name = DistinguishedName::new();
        leaf_params
            .distinguished_name
            .push(DnType::CommonName, "Test EK");
        leaf_params.key_usages = vec![KeyUsagePurpose::KeyEncipherment];
        leaf_params.not_before =
            time::OffsetDateTime::from_unix_timestamp(now as i64 - 100).unwrap();
        leaf_params.not_after =
            time::OffsetDateTime::from_unix_timestamp(now as i64 + 1000).unwrap();
        let mut eku = CustomExtension::from_oid_content(
            &[2, 5, 29, 37],
            vec![0x30, 7, 6, 5, 0x67, 0x81, 0x05, 0x08, 0x01],
        );
        eku.set_criticality(false);
        leaf_params.custom_extensions.push(eku);
        let directory_name = vec![
            0x30, 0x18, 0x31, 0x16, 0x30, 0x14, 0x06, 0x05, 0x67, 0x81, 0x05, 0x02, 0x01, 0x13,
            0x0b, b'i', b'd', b':', b'4', b'9', b'4', b'E', b'5', b'4', b'4', b'3',
        ];
        let mut san = vec![0x30, 0x1c, 0xa4, 0x1a];
        san.extend_from_slice(&directory_name);
        leaf_params
            .custom_extensions
            .push(CustomExtension::from_oid_content(&[2, 5, 29, 17], san));
        let leaf = leaf_params
            .signed_by(&ek_signing_key, &ca, &ca_key)
            .unwrap();
        let ak_key = RsaPrivateKey::new(&mut OsRng, 2048).unwrap();
        (ca.der().to_vec(), leaf.der().to_vec(), ek_key, ak_key)
    }

    fn signed_policy(policy: &TrustPolicy) -> (Vec<u8>, Vec<u8>, [u8; 32]) {
        let key = Ed25519KeyPair::from_seed_unchecked(&[0x55; 32]).unwrap();
        let bytes = serde_json::to_vec(policy).unwrap();
        let signature = key.sign(&crate::format::signature_payload(
            TRUST_POLICY_DOMAIN,
            &bytes,
        ));
        let envelope = serde_json::to_vec(&crate::SignatureEnvelope {
            algorithm: "Ed25519".into(),
            key_id: "trust-policy".into(),
            signature: BASE64.encode(signature.as_ref()),
        })
        .unwrap();
        (
            bytes,
            envelope,
            key.public_key().as_ref().try_into().unwrap(),
        )
    }

    #[test]
    fn pinned_ek_chain_rejects_revocation_mismatch_expiration_and_unknown_root() {
        let now = 1_790_000_000;
        let (ca_der, leaf_der, ek_key, ak_key) = test_certificates(now as u64);
        let mut request = test_request();
        let mut ek = BASE64.decode(&request.ek_public).unwrap();
        ek[58..].copy_from_slice(&ek_key.n().to_bytes_be());
        request.ek_public = BASE64.encode(&ek);
        let mut ak = BASE64.decode(&request.ak_public).unwrap();
        ak[24..].copy_from_slice(&ak_key.n().to_bytes_be());
        request.ak_public = BASE64.encode(ak);
        request.ek_certificate_chain = vec![BASE64.encode(&leaf_der)];
        let request = validate_request(request).unwrap();
        let mut policy = TrustPolicy {
            format: "model-protection-enrollment-trust-policy".into(),
            format_version: 1,
            policy_id: "test-policy".into(),
            sequence: 1,
            valid_from: now as u64 - 10,
            expires_at: now as u64 + 2000,
            root_certificates: vec![BASE64.encode(&ca_der)],
            allowed_manufacturers: vec!["id:494E5443".into()],
            revoked_certificate_sha256: vec![],
            revocation_mode: "operator-fingerprint-snapshot-v1".into(),
        };
        let (bytes, envelope, key) = signed_policy(&policy);
        let verified =
            verify_trust_policy(&bytes, &envelope, "trust-policy", &key, 1, now as u64).unwrap();
        let verified_ek = verify_ek(&request, &verified, now as u64).unwrap();
        let activation =
            super::super::credential::make_challenge(&request, &verified_ek, now as u64, 600)
                .unwrap();
        let transcript =
            super::super::format::transcript_digest(&request, &activation.challenge).unwrap();
        let mut attest = vec![0xff, 0x54, 0x43, 0x47, 0x80, 0x1a];
        let signer_name = crate::format::decode_lower_hex::<34>(
            &request.request().ak_qualified_name,
            "test name",
        )
        .unwrap();
        for value in [signer_name.as_slice(), transcript.as_slice()] {
            attest.extend_from_slice(&(value.len() as u16).to_be_bytes());
            attest.extend_from_slice(value);
        }
        attest.extend_from_slice(&[0; 25]);
        let creation_hash = crate::format::decode_lower_hex::<32>(
            &request.request().duk_creation_hash,
            "test hash",
        )
        .unwrap();
        for value in [request.duk_name().as_slice(), creation_hash.as_slice()] {
            attest.extend_from_slice(&(value.len() as u16).to_be_bytes());
            attest.extend_from_slice(value);
        }
        let ak_signature = SigningKey::<Sha256>::new(ak_key.clone())
            .sign(&attest)
            .to_vec();
        let mut response = super::super::format::Response {
            format: "model-protection-enrollment-response".into(),
            format_version: 1,
            challenge_id: activation.challenge.challenge_id.clone(),
            transcript_sha256: hex(&transcript),
            activation_proof: hex(&super::super::format::activation_proof(
                activation.activation_secret(),
                &transcript,
            )),
            attestation: BASE64.encode(&attest),
            ak_signature: BASE64.encode(ak_signature),
        };
        let response_bytes = serde_json::to_vec(&response).unwrap();
        let proof = super::super::format::verify_response(
            &request,
            &activation.challenge,
            &response_bytes,
            activation.activation_secret(),
            now as u64,
        )
        .unwrap();
        assert!(
            super::super::format::verify_response(
                &request,
                &activation.challenge,
                &response_bytes,
                &[0; 32],
                now as u64
            )
            .is_err()
        );
        response.activation_proof = hex(&[0; 32]);
        assert!(
            super::super::format::verify_response(
                &request,
                &activation.challenge,
                &serde_json::to_vec(&response).unwrap(),
                activation.activation_secret(),
                now as u64
            )
            .is_err()
        );
        let directory = tempfile::tempdir().unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let mut registry =
            super::super::registry::Registry::open(&directory.path().join("registry.sqlite"), true)
                .unwrap();
        registry
            .record_challenge(&request, &activation.challenge, now as u64)
            .unwrap();
        let certified = registry
            .certify(
                &request,
                &verified_ek,
                &proof,
                "certification-1",
                1,
                now as u64,
            )
            .unwrap();
        assert_eq!(
            registry.certification_bytes("certification-1").unwrap(),
            certified
        );
        assert!(
            registry
                .certify(
                    &request,
                    &verified_ek,
                    &proof,
                    "certification-2",
                    1,
                    now as u64
                )
                .is_err()
        );
        registry
            .reserve_issuance(
                "certification-1",
                &certified,
                &request.request().binding,
                "license-1",
                1,
            )
            .unwrap();
        assert!(
            registry
                .reserve_issuance(
                    "certification-1",
                    &certified,
                    &request.request().binding,
                    "license-2",
                    2
                )
                .is_err()
        );
        assert!(
            verify_trust_policy(&bytes, &envelope, "trust-policy", &key, 2, now as u64).is_err()
        );
        assert!(verify_ek(&request, &verified, now as u64 + 1001).is_err());
        policy.revoked_certificate_sha256 = vec![hex(&Sha256::digest(&leaf_der))];
        let (bytes, envelope, key) = signed_policy(&policy);
        let verified =
            verify_trust_policy(&bytes, &envelope, "trust-policy", &key, 1, now as u64).unwrap();
        assert!(verify_ek(&request, &verified, now as u64).is_err());
        policy.revoked_certificate_sha256.clear();
        policy.root_certificates = vec![BASE64.encode(&leaf_der)];
        let (bytes, envelope, key) = signed_policy(&policy);
        let verified =
            verify_trust_policy(&bytes, &envelope, "trust-policy", &key, 1, now as u64).unwrap();
        assert!(verify_ek(&request, &verified, now as u64).is_err());
        let mut mismatch = request.request().clone();
        ek[59] ^= 1;
        mismatch.ek_public = BASE64.encode(ek);
        assert!(verify_ek(&validate_request(mismatch).unwrap(), &verified, now as u64).is_err());
    }
}
