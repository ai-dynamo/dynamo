// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};

use crate::format::{decode_lower_hex, has_valid_ed25519_signature, signature_payload};
use crate::{
    ProtectionError, Result, TPM_DEVICE_PUBLIC_BYTES, TpmDevicePublic, validate_tpm_device_public,
};

pub const CERTIFIED_DEVICE_FORMAT: &str = "model-protection-certified-device";
pub const CERTIFIED_DEVICE_VERSION: u16 = 1;
pub const CERTIFIED_DEVICE_SIGNATURE_DOMAIN: &[u8] =
    b"model-protection-certified-device-signature-v1\0";
pub const MAX_CERTIFIED_DEVICE_BYTES: usize = 64 * 1024;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CertifiedDevice {
    pub format: String,
    pub format_version: u16,
    pub certification_id: String,
    pub customer_scope_id: String,
    pub artifact_id: String,
    pub tpm_public: String,
    pub policy_authority_name: String,
}

/// Verified issuer input, not independent EK/AK hardware attestation.
pub struct VerifiedCertifiedDevice {
    certification_id: String,
    public: TpmDevicePublic,
    policy_authority_name: [u8; 34],
}

impl VerifiedCertifiedDevice {
    pub fn certification_id(&self) -> &str {
        &self.certification_id
    }

    pub fn public(&self) -> &TpmDevicePublic {
        &self.public
    }

    pub fn policy_authority_name(&self) -> &[u8; 34] {
        &self.policy_authority_name
    }
}

pub fn certified_device_signature_payload(bytes: &[u8]) -> Vec<u8> {
    signature_payload(CERTIFIED_DEVICE_SIGNATURE_DOMAIN, bytes)
}

pub fn verify_certified_device(
    bytes: &[u8],
    envelope: &[u8],
    expected_key_id: &str,
    verifying_key: &[u8; 32],
    customer_scope_id: &str,
    artifact_id: &str,
) -> Result<VerifiedCertifiedDevice> {
    if bytes.is_empty()
        || bytes.len() > MAX_CERTIFIED_DEVICE_BYTES
        || !has_valid_ed25519_signature(
            CERTIFIED_DEVICE_SIGNATURE_DOMAIN,
            bytes,
            envelope,
            expected_key_id,
            verifying_key,
        )
    {
        return Err(ProtectionError::CertifiedDeviceInvalid);
    }
    let certified: CertifiedDevice =
        serde_json::from_slice(bytes).map_err(|_| ProtectionError::CertifiedDeviceInvalid)?;
    if certified.format != CERTIFIED_DEVICE_FORMAT
        || certified.format_version != CERTIFIED_DEVICE_VERSION
        || certified.certification_id.is_empty()
        || certified.certification_id.len() > 128
        || certified.certification_id.chars().any(char::is_control)
        || certified.customer_scope_id != customer_scope_id
        || certified.artifact_id != artifact_id
        || certified.tpm_public.len() != TPM_DEVICE_PUBLIC_BYTES.div_ceil(3) * 4
    {
        return Err(ProtectionError::CertifiedDeviceInvalid);
    }
    decode_lower_hex::<16>(&certified.artifact_id, "certified artifact id")
        .map_err(|_| ProtectionError::CertifiedDeviceInvalid)?;
    let policy_authority_name = decode_lower_hex(&certified.policy_authority_name, "policy name")
        .map_err(|_| ProtectionError::CertifiedDeviceInvalid)?;
    let public = BASE64
        .decode(certified.tpm_public.as_bytes())
        .map_err(|_| ProtectionError::CertifiedDeviceInvalid)?;
    let public = validate_tpm_device_public(&public, &policy_authority_name)?;
    Ok(VerifiedCertifiedDevice {
        certification_id: certified.certification_id,
        public,
        policy_authority_name,
    })
}

#[cfg(test)]
mod tests {
    use ring::signature::{Ed25519KeyPair, KeyPair};

    use super::*;
    use crate::SignatureEnvelope;
    use crate::device::test_device_public;

    const ARTIFACT: &str = "00112233445566778899aabbccddeeff";

    fn fixture() -> (Ed25519KeyPair, Vec<u8>) {
        let key = Ed25519KeyPair::from_seed_unchecked(&[0x42; 32]).unwrap();
        let mut name = [0x22; 34];
        name[..2].copy_from_slice(&[0, 0x0b]);
        let record = CertifiedDevice {
            format: CERTIFIED_DEVICE_FORMAT.into(),
            format_version: 1,
            certification_id: "certification-1".into(),
            customer_scope_id: "customer-1".into(),
            artifact_id: ARTIFACT.into(),
            tpm_public: BASE64.encode(test_device_public(&name)),
            policy_authority_name: name.iter().map(|byte| format!("{byte:02x}")).collect(),
        };
        (key, serde_json::to_vec(&record).unwrap())
    }

    fn envelope(key: &Ed25519KeyPair, bytes: &[u8]) -> Vec<u8> {
        serde_json::to_vec(&SignatureEnvelope {
            algorithm: "Ed25519".into(),
            key_id: "enrollment-v1".into(),
            signature: BASE64.encode(
                key.sign(&certified_device_signature_payload(bytes))
                    .as_ref(),
            ),
        })
        .unwrap()
    }

    fn verify(
        key: &Ed25519KeyPair,
        bytes: &[u8],
        signature: &[u8],
    ) -> Result<VerifiedCertifiedDevice> {
        verify_certified_device(
            bytes,
            signature,
            "enrollment-v1",
            key.public_key().as_ref().try_into().unwrap(),
            "customer-1",
            ARTIFACT,
        )
    }

    #[test]
    fn verifies_existing_format_and_exact_bytes() {
        let (key, bytes) = fixture();
        let signature = envelope(&key, &bytes);
        let verified = verify(&key, &bytes, &signature).unwrap();
        assert_eq!(verified.certification_id(), "certification-1");
        assert_eq!(verified.public().modulus(), &[0x81; 256]);
        let mut edited = bytes;
        edited.push(b'\n');
        assert!(verify(&key, &edited, &signature).is_err());
        assert!(verify(&key, &edited, &envelope(&key, &edited)).is_ok());
    }

    #[test]
    fn rejects_signed_schema_profile_and_binding_errors() {
        let (key, bytes) = fixture();
        let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        for (field, value) in [
            ("format", serde_json::json!("other")),
            ("format_version", serde_json::json!(2)),
            ("certification_id", serde_json::json!("\n")),
            ("customer_scope_id", serde_json::json!("another-customer")),
            (
                "artifact_id",
                serde_json::json!("ffeeddccbbaa99887766554433221100"),
            ),
            (
                "policy_authority_name",
                serde_json::json!("000b".to_owned() + &"33".repeat(32)),
            ),
            (
                "tpm_public",
                serde_json::json!(BASE64.encode([0; TPM_DEVICE_PUBLIC_BYTES])),
            ),
            ("unknown_field", serde_json::json!(true)),
        ] {
            let mut record = original.clone();
            record[field] = value;
            let invalid = serde_json::to_vec(&record).unwrap();
            assert!(
                verify(&key, &invalid, &envelope(&key, &invalid)).is_err(),
                "{field}"
            );
        }
        let duplicate = format!(
            "{{\"format_version\":1,{}",
            std::str::from_utf8(&bytes[1..]).unwrap()
        )
        .into_bytes();
        assert!(verify(&key, &duplicate, &envelope(&key, &duplicate)).is_err());
        assert!(verify(&key, &vec![b' '; MAX_CERTIFIED_DEVICE_BYTES + 1], b"{}").is_err());
    }

    #[test]
    fn rejects_wrong_key_algorithm_domain_and_envelope_schema() {
        let (key, bytes) = fixture();
        let mut signature: serde_json::Value =
            serde_json::from_slice(&envelope(&key, &bytes)).unwrap();
        for (field, value) in [
            ("algorithm", serde_json::json!("RSA")),
            ("key_id", serde_json::json!("package-v1")),
            (
                "signature",
                serde_json::json!(BASE64.encode(key.sign(&bytes).as_ref())),
            ),
            ("extra", serde_json::json!(true)),
        ] {
            let original = signature.clone();
            signature[field] = value;
            assert!(verify(&key, &bytes, &serde_json::to_vec(&signature).unwrap()).is_err());
            signature = original;
        }
        let wrong_key = Ed25519KeyPair::from_seed_unchecked(&[0x43; 32]).unwrap();
        assert!(verify(&wrong_key, &bytes, &envelope(&key, &bytes)).is_err());
    }
}
