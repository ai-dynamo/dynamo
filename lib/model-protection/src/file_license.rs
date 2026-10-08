// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Signed offline entitlement for the explicitly software-key package profile.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::format::{decode_lower_hex, has_valid_ed25519_signature, signature_payload};
use crate::{Entitlement, ProtectionError, Result, VerifiedManifest};

pub const FILE_LICENSE_FORMAT: &str = "model-protection-file-license";
pub const FILE_LICENSE_SIGNATURE_DOMAIN: &[u8] = b"model-protection-file-license-signature-v1\0";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileLicense {
    pub format: String,
    pub format_version: u16,
    pub license_id: String,
    pub artifact_id: String,
    pub manifest_sha256: String,
    pub customer_scope_id: String,
    pub model_id: String,
    pub model_version: String,
    pub key_sha256: String,
    pub entitlement: Entitlement,
}

pub fn file_license_signature_payload(bytes: &[u8]) -> Vec<u8> {
    signature_payload(FILE_LICENSE_SIGNATURE_DOMAIN, bytes)
}

pub fn verify_file_license(
    bytes: &[u8],
    signature: &[u8],
    key_id: &str,
    key: &[u8; 32],
    verified: &VerifiedManifest,
) -> Result<FileLicense> {
    if bytes.is_empty() || bytes.len() > crate::license::MAX_LICENSE_BYTES {
        return Err(ProtectionError::LicenseInvalid("file license size"));
    }
    if !has_valid_ed25519_signature(FILE_LICENSE_SIGNATURE_DOMAIN, bytes, signature, key_id, key) {
        return Err(ProtectionError::LicenseSignatureInvalid);
    }
    let license: FileLicense = serde_json::from_slice(bytes)
        .map_err(|_| ProtectionError::LicenseInvalid("file license json"))?;
    let manifest = verified.manifest();
    if license.format != FILE_LICENSE_FORMAT
        || license.format_version != 1
        || license.license_id.is_empty()
        || license.license_id.len() > 128
        || license.license_id.chars().any(char::is_control)
        || license.entitlement.mode != "offline-perpetual"
        || license.entitlement.generation == 0
    {
        return Err(ProtectionError::LicenseInvalid("file license format"));
    }
    decode_lower_hex::<32>(&license.key_sha256, "key digest")?;
    if manifest.runtime.protection_profile.as_deref() != Some("encrypted-file-license")
        || license.artifact_id != manifest.artifact_id
        || license.customer_scope_id != manifest.customer_scope_id
        || license.model_id != manifest.model.model_id
        || license.model_version != manifest.model.model_version
        || decode_lower_hex::<32>(&license.manifest_sha256, "manifest digest")?
            != *verified.digest()
    {
        return Err(ProtectionError::LicenseBindingMismatch);
    }
    Ok(license)
}

impl FileLicense {
    pub fn verify_key(&self, key: &[u8; 32]) -> Result<()> {
        let digest: [u8; 32] = Sha256::digest(key).into();
        if decode_lower_hex::<32>(&self.key_sha256, "key digest")? != digest {
            return Err(ProtectionError::LicenseBindingMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
    use ring::signature::{Ed25519KeyPair, KeyPair};

    #[test]
    fn software_license_requires_exact_signed_package_and_key() {
        let mut manifest =
            crate::parse_manifest(&crate::format::tests::valid_manifest_json()).unwrap();
        manifest.format_version = 2;
        manifest.runtime.protection_profile = Some("encrypted-file-license".into());
        let package_key = Ed25519KeyPair::from_seed_unchecked(&[7; 32]).unwrap();
        let license_key = Ed25519KeyPair::from_seed_unchecked(&[8; 32]).unwrap();
        let sign = |signer: &Ed25519KeyPair, payload: &[u8], key_id: &str| {
            serde_json::to_vec(&crate::SignatureEnvelope {
                algorithm: "Ed25519".into(),
                key_id: key_id.into(),
                signature: BASE64.encode(signer.sign(payload).as_ref()),
            })
            .unwrap()
        };
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let verified = crate::verify_manifest(
            &bytes,
            &sign(
                &package_key,
                &crate::manifest_signature_payload(&bytes),
                "package",
            ),
            "package",
            package_key.public_key().as_ref().try_into().unwrap(),
        )
        .unwrap();
        let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let dek = [3; 32];
        let mut license = FileLicense {
            format: FILE_LICENSE_FORMAT.into(),
            format_version: 1,
            license_id: "test".into(),
            artifact_id: manifest.artifact_id.clone(),
            manifest_sha256: hex(verified.digest()),
            customer_scope_id: manifest.customer_scope_id.clone(),
            model_id: manifest.model.model_id.clone(),
            model_version: manifest.model.model_version.clone(),
            key_sha256: hex(&Sha256::digest(dek)),
            entitlement: Entitlement {
                mode: "offline-perpetual".into(),
                generation: 1,
            },
        };
        let bytes = serde_json::to_vec(&license).unwrap();
        let signature = sign(
            &license_key,
            &file_license_signature_payload(&bytes),
            "license",
        );
        let public = license_key.public_key().as_ref().try_into().unwrap();
        let checked =
            verify_file_license(&bytes, &signature, "license", public, &verified).unwrap();
        checked.verify_key(&dek).unwrap();
        assert!(checked.verify_key(&[4; 32]).is_err());
        assert!(
            verify_file_license(&bytes, &signature, "wrong-key-id", public, &verified).is_err()
        );
        license.artifact_id = "ff".repeat(16);
        let altered = serde_json::to_vec(&license).unwrap();
        assert!(matches!(
            verify_file_license(&altered, &signature, "license", public, &verified),
            Err(ProtectionError::LicenseSignatureInvalid)
        ));
        assert!(matches!(
            verify_file_license(
                &altered,
                &sign(
                    &license_key,
                    &file_license_signature_payload(&altered),
                    "license"
                ),
                "license",
                public,
                &verified
            ),
            Err(ProtectionError::LicenseBindingMismatch)
        ));
    }
}
