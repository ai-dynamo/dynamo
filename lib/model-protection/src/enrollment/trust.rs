// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use openssl::stack::Stack;
use openssl::x509::{
    X509, X509StoreContext,
    store::X509StoreBuilder,
    verify::{X509VerifyFlags, X509VerifyParam},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use x509_parser::{extensions::GeneralName, parse_x509_certificate};

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

fn certificate(value: &str) -> Result<X509> {
    if value.len() > 22 * 1024 {
        return Err(Error::Trust);
    }
    let der = BASE64.decode(value).map_err(|_| Error::Trust)?;
    if der.is_empty() || der.len() > 16 * 1024 || BASE64.encode(&der) != value {
        return Err(Error::Trust);
    }
    let cert = X509::from_der(&der).map_err(|_| Error::Trust)?;
    if cert.to_der().map_err(|_| Error::Trust)? != der {
        return Err(Error::Trust);
    }
    Ok(cert)
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
    let leaf = certificate(chain.first().ok_or(Error::Trust)?)?;
    let der = leaf.to_der().map_err(|_| Error::Trust)?;
    let (remaining, parsed) = parse_x509_certificate(&der).map_err(|_| Error::Trust)?;
    if !remaining.is_empty() || parsed.version().0 != 2 || parsed.is_ca() {
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
    let rsa = leaf
        .public_key()
        .map_err(|_| Error::Trust)?
        .rsa()
        .map_err(|_| Error::Trust)?;
    let ek_public = BASE64
        .decode(&request.request().ek_public)
        .map_err(|_| Error::Trust)?;
    if rsa.n().num_bits() != 2048
        || rsa.e().to_vec() != [1, 0, 1]
        || rsa.n().to_vec() != ek_public[58..]
    {
        return Err(Error::Trust);
    }
    let mut store = X509StoreBuilder::new().map_err(|_| Error::Trust)?;
    for root in &policy.policy.root_certificates {
        store
            .add_cert(certificate(root)?)
            .map_err(|_| Error::Trust)?;
    }
    let mut parameters = X509VerifyParam::new().map_err(|_| Error::Trust)?;
    parameters.set_time(i64::try_from(now).map_err(|_| Error::Trust)?);
    parameters.set_auth_level(2);
    parameters.set_depth(5);
    parameters
        .set_flags(X509VerifyFlags::X509_STRICT | X509VerifyFlags::CHECK_SS_SIGNATURE)
        .map_err(|_| Error::Trust)?;
    store.set_param(&parameters).map_err(|_| Error::Trust)?;
    let mut intermediates = Stack::new().map_err(|_| Error::Trust)?;
    for cert in &chain[1..] {
        intermediates
            .push(certificate(cert)?)
            .map_err(|_| Error::Trust)?;
    }
    let mut context = X509StoreContext::new().map_err(|_| Error::Trust)?;
    let is_valid = context
        .init(&store.build(), &leaf, &intermediates, |context| {
            if !context.verify_cert()? {
                return Ok(false);
            }
            let Some(chain) = context.chain() else {
                return Ok(false);
            };
            for certificate in chain {
                let fingerprint = hex(&Sha256::digest(certificate.to_der()?));
                if policy
                    .policy
                    .revoked_certificate_sha256
                    .contains(&fingerprint)
                {
                    return Ok(false);
                }
            }
            Ok(true)
        })
        .map_err(|_| Error::Trust)?;
    if !is_valid {
        return Err(Error::Trust);
    }
    Ok(VerifiedEk {
        request_digest: *request.digest(),
        policy_digest: policy.digest,
    })
}

#[cfg(test)]
mod tests {
    use super::super::format::{test_request, validate_request};
    use super::*;
    use openssl::{
        asn1::{Asn1Integer, Asn1Object, Asn1OctetString, Asn1Time},
        bn::BigNum,
        hash::MessageDigest,
        pkey::PKey,
        rsa::Rsa,
        x509::{
            X509Extension, X509NameBuilder,
            extension::{AuthorityKeyIdentifier, BasicConstraints, KeyUsage, SubjectKeyIdentifier},
        },
    };
    use ring::signature::{Ed25519KeyPair, KeyPair};

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
        let ca_key = PKey::from_rsa(Rsa::generate(2048).unwrap()).unwrap();
        let ek_key = PKey::from_rsa(Rsa::generate(2048).unwrap()).unwrap();
        let ak_key = PKey::from_rsa(Rsa::generate(2048).unwrap()).unwrap();
        let mut ca_name = X509NameBuilder::new().unwrap();
        ca_name.append_entry_by_text("CN", "Test TPM root").unwrap();
        let ca_name = ca_name.build();
        let mut ca = X509::builder().unwrap();
        ca.set_version(2).unwrap();
        ca.set_serial_number(&Asn1Integer::from_bn(&BigNum::from_u32(1).unwrap()).unwrap())
            .unwrap();
        ca.set_subject_name(&ca_name).unwrap();
        ca.set_issuer_name(&ca_name).unwrap();
        ca.set_pubkey(&ca_key).unwrap();
        ca.set_not_before(&Asn1Time::from_unix(now - 100).unwrap())
            .unwrap();
        ca.set_not_after(&Asn1Time::from_unix(now + 1000).unwrap())
            .unwrap();
        ca.append_extension(BasicConstraints::new().critical().ca().build().unwrap())
            .unwrap();
        ca.append_extension(
            KeyUsage::new()
                .critical()
                .key_cert_sign()
                .crl_sign()
                .build()
                .unwrap(),
        )
        .unwrap();
        ca.append_extension(
            SubjectKeyIdentifier::new()
                .build(&ca.x509v3_context(None, None))
                .unwrap(),
        )
        .unwrap();
        ca.sign(&ca_key, MessageDigest::sha256()).unwrap();
        let ca = ca.build();
        let mut subject = X509NameBuilder::new().unwrap();
        subject.append_entry_by_text("CN", "Test EK").unwrap();
        let subject = subject.build();
        let mut leaf = X509::builder().unwrap();
        leaf.set_version(2).unwrap();
        leaf.set_serial_number(&Asn1Integer::from_bn(&BigNum::from_u32(2).unwrap()).unwrap())
            .unwrap();
        leaf.set_subject_name(&subject).unwrap();
        leaf.set_issuer_name(ca.subject_name()).unwrap();
        leaf.set_pubkey(&ek_key).unwrap();
        leaf.set_not_before(&Asn1Time::from_unix(now - 100).unwrap())
            .unwrap();
        leaf.set_not_after(&Asn1Time::from_unix(now + 1000).unwrap())
            .unwrap();
        leaf.append_extension(BasicConstraints::new().critical().build().unwrap())
            .unwrap();
        leaf.append_extension(
            KeyUsage::new()
                .critical()
                .key_encipherment()
                .build()
                .unwrap(),
        )
        .unwrap();
        leaf.append_extension(
            SubjectKeyIdentifier::new()
                .build(&leaf.x509v3_context(Some(&ca), None))
                .unwrap(),
        )
        .unwrap();
        leaf.append_extension(
            AuthorityKeyIdentifier::new()
                .keyid(true)
                .build(&leaf.x509v3_context(Some(&ca), None))
                .unwrap(),
        )
        .unwrap();
        let eku = Asn1OctetString::new_from_bytes(&[0x30, 7, 6, 5, 0x67, 0x81, 5, 8, 1]).unwrap();
        leaf.append_extension(
            X509Extension::new_from_der(&Asn1Object::from_str("2.5.29.37").unwrap(), false, &eku)
                .unwrap(),
        )
        .unwrap();
        let mut directory = X509NameBuilder::new().unwrap();
        directory
            .append_entry_by_text("2.23.133.2.1", "id:494E5443")
            .unwrap();
        let directory = directory.build().to_der().unwrap();
        let mut san = vec![
            0x30,
            (directory.len() + 2) as u8,
            0xa4,
            directory.len() as u8,
        ];
        san.extend_from_slice(&directory);
        leaf.append_extension(
            X509Extension::new_from_der(
                &Asn1Object::from_str("2.5.29.17").unwrap(),
                false,
                &Asn1OctetString::new_from_bytes(&san).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        leaf.sign(&ca_key, MessageDigest::sha256()).unwrap();
        let leaf = leaf.build();
        let mut request = test_request();
        let mut ek = BASE64.decode(&request.ek_public).unwrap();
        ek[58..].copy_from_slice(&ek_key.rsa().unwrap().n().to_vec());
        request.ek_public = BASE64.encode(&ek);
        let mut ak = BASE64.decode(&request.ak_public).unwrap();
        ak[24..].copy_from_slice(&ak_key.rsa().unwrap().n().to_vec());
        request.ak_public = BASE64.encode(ak);
        request.ek_certificate_chain = vec![BASE64.encode(leaf.to_der().unwrap())];
        let request = validate_request(request).unwrap();
        let mut policy = TrustPolicy {
            format: "model-protection-enrollment-trust-policy".into(),
            format_version: 1,
            policy_id: "test-policy".into(),
            sequence: 1,
            valid_from: now as u64 - 10,
            expires_at: now as u64 + 2000,
            root_certificates: vec![BASE64.encode(ca.to_der().unwrap())],
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
        let mut signer = openssl::sign::Signer::new(MessageDigest::sha256(), &ak_key).unwrap();
        signer.update(&attest).unwrap();
        let ak_signature = signer.sign_to_vec().unwrap();
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
        policy.revoked_certificate_sha256 = vec![hex(&Sha256::digest(leaf.to_der().unwrap()))];
        let (bytes, envelope, key) = signed_policy(&policy);
        let verified =
            verify_trust_policy(&bytes, &envelope, "trust-policy", &key, 1, now as u64).unwrap();
        assert!(verify_ek(&request, &verified, now as u64).is_err());
        policy.revoked_certificate_sha256.clear();
        policy.root_certificates = vec![BASE64.encode(leaf.to_der().unwrap())];
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
