// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};

use crate::format::{SignatureEnvelope, has_valid_ed25519_signature, signature_payload};
use crate::{ProtectionError, Result};

pub const ISSUER_RECORD_FORMAT: &str = "model-protection-issuer-dek";
pub const ISSUER_RECORD_VERSION: u16 = 2;
pub const ISSUER_RECORD_SIGNATURE_DOMAIN: &[u8] = b"model-protection-issuer-dek-signature-v2\0";
pub const MAX_ISSUER_RECORD_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IssuerDekRecord {
    pub format: String,
    pub format_version: u16,
    pub artifact_id: String,
    pub customer_scope_id: String,
    pub model_id: String,
    pub model_version: String,
    pub manifest_sha256: String,
    pub kek_key_id: String,
    pub kek_key_version: String,
    pub wrapped_dek: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedIssuerDekRecord {
    pub payload: String,
    pub signature: SignatureEnvelope,
}

pub fn issuer_record_signature_payload(bytes: &[u8]) -> Vec<u8> {
    signature_payload(ISSUER_RECORD_SIGNATURE_DOMAIN, bytes)
}

pub fn verify_issuer_record(
    envelope_bytes: &[u8],
    expected_key_id: &str,
    verifying_key: &[u8; 32],
) -> Result<IssuerDekRecord> {
    if envelope_bytes.is_empty() || envelope_bytes.len() > MAX_ISSUER_RECORD_BYTES {
        return Err(ProtectionError::IssuerRecordInvalid);
    }
    let envelope: SignedIssuerDekRecord =
        serde_json::from_slice(envelope_bytes).map_err(|_| ProtectionError::IssuerRecordInvalid)?;
    let payload = BASE64
        .decode(envelope.payload.as_bytes())
        .map_err(|_| ProtectionError::IssuerRecordInvalid)?;
    if payload.is_empty()
        || payload.len() > MAX_ISSUER_RECORD_BYTES
        || !has_valid_ed25519_signature(
            ISSUER_RECORD_SIGNATURE_DOMAIN,
            &payload,
            &serde_json::to_vec(&envelope.signature)
                .map_err(|_| ProtectionError::IssuerRecordInvalid)?,
            expected_key_id,
            verifying_key,
        )
    {
        return Err(ProtectionError::IssuerRecordInvalid);
    }
    let record: IssuerDekRecord =
        serde_json::from_slice(&payload).map_err(|_| ProtectionError::IssuerRecordInvalid)?;
    if record.format != ISSUER_RECORD_FORMAT || record.format_version != ISSUER_RECORD_VERSION {
        return Err(ProtectionError::IssuerRecordInvalid);
    }
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::signature::{Ed25519KeyPair, KeyPair};

    #[test]
    fn rejects_tampered_issuer_record_association() {
        let key = Ed25519KeyPair::from_seed_unchecked(&[4; 32]).unwrap();
        let record = IssuerDekRecord {
            format: ISSUER_RECORD_FORMAT.into(),
            format_version: ISSUER_RECORD_VERSION,
            artifact_id: "00".repeat(16),
            customer_scope_id: "customer-a".into(),
            model_id: "model-a".into(),
            model_version: "1".into(),
            manifest_sha256: "11".repeat(32),
            kek_key_id: "kek-a".into(),
            kek_key_version: "1".into(),
            wrapped_dek: BASE64.encode([7; 40]),
        };
        let payload = serde_json::to_vec(&record).unwrap();
        let envelope = SignedIssuerDekRecord {
            payload: BASE64.encode(&payload),
            signature: SignatureEnvelope {
                algorithm: "Ed25519".into(),
                key_id: "package-a".into(),
                signature: BASE64.encode(
                    key.sign(&issuer_record_signature_payload(&payload))
                        .as_ref(),
                ),
            },
        };
        let bytes = serde_json::to_vec(&envelope).unwrap();
        assert!(
            verify_issuer_record(
                &bytes,
                "package-a",
                key.public_key().as_ref().try_into().unwrap()
            )
            .is_ok()
        );

        let mut tampered: SignedIssuerDekRecord = serde_json::from_slice(&bytes).unwrap();
        let mut record: IssuerDekRecord =
            serde_json::from_slice(&BASE64.decode(&tampered.payload).unwrap()).unwrap();
        record.artifact_id = "ff".repeat(16);
        tampered.payload = BASE64.encode(serde_json::to_vec(&record).unwrap());
        assert!(matches!(
            verify_issuer_record(
                &serde_json::to_vec(&tampered).unwrap(),
                "package-a",
                key.public_key().as_ref().try_into().unwrap()
            ),
            Err(ProtectionError::IssuerRecordInvalid)
        ));
    }
}
