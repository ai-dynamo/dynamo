// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::format::{
    VerifiedManifest, decode_lower_hex, has_valid_ed25519_signature, signature_payload,
};
use crate::{ProtectionError, Result};

pub const LICENSE_FORMAT: &str = "model-protection-license";
pub const LICENSE_FORMAT_VERSION: u16 = 1;
pub const LICENSE_SIGNATURE_DOMAIN: &[u8] = b"model-protection-license-signature-v1\0";
pub const TPM_PROFILE: &str = "tpm2-offline-rsa2048-oaep-sha256-v1";
pub const TPM_POLICY_SIGNATURE: &str = "ecdsa-p256-sha256-p1363";
pub const TPM_POLICY_REF_HEX: &str =
    "30d14d0ec65c3ccbf79f0cddddc9db09a3676a43910fde63e3889f65a9a371a6";
pub const TPM_OAEP_LABEL: &[u8] = b"model-protection-dek-v1\0";
pub const MAX_LICENSE_BYTES: usize = 64 * 1024;
const RSA_2048_CIPHERTEXT_BYTES: usize = 256;
const P256_SIGNATURE_BYTES: usize = 64;
const TPM_CC_RSA_DECRYPT: u32 = 0x0000_0159;
const TPM_CC_POLICY_COMMAND_CODE: u32 = 0x0000_016c;
const TPM_CC_POLICY_CP_HASH: u32 = 0x0000_016e;
const TPM_CC_POLICY_AUTHORIZE: u32 = 0x0000_016a;
const TPM_ALG_OAEP: u16 = 0x0017;
const TPM_ALG_SHA256: u16 = 0x000b;

pub fn license_signature_payload(bytes: &[u8]) -> Vec<u8> {
    signature_payload(LICENSE_SIGNATURE_DOMAIN, bytes)
}

pub fn rsa_decrypt_cp_hash(device_name: &[u8; 34], ciphertext: &[u8; 256]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(TPM_CC_RSA_DECRYPT.to_be_bytes());
    digest.update(device_name);
    digest.update((ciphertext.len() as u16).to_be_bytes());
    digest.update(ciphertext);
    digest.update(TPM_ALG_OAEP.to_be_bytes());
    digest.update(TPM_ALG_SHA256.to_be_bytes());
    digest.update((TPM_OAEP_LABEL.len() as u16).to_be_bytes());
    digest.update(TPM_OAEP_LABEL);
    digest.finalize().into()
}

pub fn approved_rsa_decrypt_policy(command_parameters_hash: &[u8; 32]) -> [u8; 32] {
    let mut first = Sha256::new();
    first.update([0_u8; 32]);
    first.update(TPM_CC_POLICY_COMMAND_CODE.to_be_bytes());
    first.update(TPM_CC_RSA_DECRYPT.to_be_bytes());

    let mut second = Sha256::new();
    second.update(first.finalize());
    second.update(TPM_CC_POLICY_CP_HASH.to_be_bytes());
    second.update(command_parameters_hash);
    second.finalize().into()
}

pub fn policy_authorize_auth_policy(
    policy_authority_name: &[u8],
    policy_ref: &[u8; 32],
) -> [u8; 32] {
    let mut first = Sha256::new();
    first.update([0_u8; 32]);
    first.update(TPM_CC_POLICY_AUTHORIZE.to_be_bytes());
    first.update(policy_authority_name);

    let mut second = Sha256::new();
    second.update(first.finalize());
    second.update(policy_ref);
    second.finalize().into()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct License {
    pub format: String,
    pub format_version: u16,
    pub license_id: String,
    pub customer_scope_id: String,
    pub artifact_id: String,
    pub manifest_sha256: String,
    pub model_id: String,
    pub model_version: String,
    pub entitlement: Entitlement,
    pub recipient: TpmRecipient,
    pub wrapped_dek: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Entitlement {
    pub mode: String,
    pub generation: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TpmRecipient {
    pub kind: String,
    pub profile: String,
    pub device_key_name: String,
    pub device_public_key_sha256: String,
    pub command_parameters_hash: String,
    pub approved_policy_digest: String,
    pub policy_ref: String,
    pub policy_authority_key_id: String,
    pub policy_signature_algorithm: String,
    pub policy_signature: String,
}

#[derive(Clone, Debug)]
pub struct AuthorizedModel {
    license: License,
    manifest: VerifiedManifest,
}

impl AuthorizedModel {
    #[cfg(test)]
    pub(crate) fn new(license: License, manifest: VerifiedManifest) -> Self {
        Self { license, manifest }
    }

    pub fn license(&self) -> &License {
        &self.license
    }

    pub fn manifest(&self) -> &VerifiedManifest {
        &self.manifest
    }
}

pub(crate) fn verify_license(
    license_bytes: &[u8],
    signature_bytes: &[u8],
    expected_key_id: &str,
    verifying_key: &[u8; 32],
    manifest: VerifiedManifest,
) -> Result<AuthorizedModel> {
    if license_bytes.is_empty() || license_bytes.len() > MAX_LICENSE_BYTES {
        return Err(ProtectionError::LicenseInvalid("license size"));
    }
    if !has_valid_ed25519_signature(
        LICENSE_SIGNATURE_DOMAIN,
        license_bytes,
        signature_bytes,
        expected_key_id,
        verifying_key,
    ) {
        return Err(ProtectionError::LicenseSignatureInvalid);
    }
    let license: License = serde_json::from_slice(license_bytes)
        .map_err(|_| ProtectionError::LicenseInvalid("license json"))?;
    license.validate()?;
    license.bind(&manifest)?;
    Ok(AuthorizedModel { license, manifest })
}

impl License {
    fn validate(&self) -> Result<()> {
        if self.format != LICENSE_FORMAT || self.format_version != LICENSE_FORMAT_VERSION {
            return Err(ProtectionError::LicenseInvalid("license format"));
        }
        for value in [
            &self.license_id,
            &self.customer_scope_id,
            &self.model_id,
            &self.model_version,
            &self.recipient.policy_authority_key_id,
        ] {
            if value.is_empty() || value.len() > 128 || value.chars().any(char::is_control) {
                return Err(ProtectionError::LicenseInvalid("license identifier"));
            }
        }
        decode_license_hex::<16>(&self.artifact_id, "license artifact id")?;
        decode_license_hex::<32>(&self.manifest_sha256, "license manifest digest")?;
        let device_name =
            decode_license_hex::<34>(&self.recipient.device_key_name, "license device key name")?;
        if device_name[..2] != [0, 0x0b] {
            return Err(ProtectionError::LicenseInvalid("license device key name"));
        }
        let device_public_digest = decode_license_hex::<32>(
            &self.recipient.device_public_key_sha256,
            "license device key digest",
        )?;
        if device_name[2..] != device_public_digest {
            return Err(ProtectionError::LicenseInvalid("license device key digest"));
        }
        decode_license_hex::<32>(
            &self.recipient.command_parameters_hash,
            "license command parameters hash",
        )?;
        decode_license_hex::<32>(
            &self.recipient.approved_policy_digest,
            "license approved policy digest",
        )?;
        decode_license_hex::<32>(&self.recipient.policy_ref, "license policy ref")?;
        if self.entitlement.mode != "offline-perpetual"
            || self.entitlement.generation == 0
            || self.recipient.kind != "tpm2"
            || self.recipient.profile != TPM_PROFILE
            || self.recipient.policy_ref != TPM_POLICY_REF_HEX
            || self.recipient.policy_signature_algorithm != TPM_POLICY_SIGNATURE
        {
            return Err(ProtectionError::LicenseInvalid("license policy"));
        }
        let wrapped = BASE64
            .decode(self.wrapped_dek.as_bytes())
            .map_err(|_| ProtectionError::LicenseInvalid("wrapped dek"))?;
        if wrapped.len() != RSA_2048_CIPHERTEXT_BYTES {
            return Err(ProtectionError::LicenseInvalid("wrapped dek"));
        }
        let policy_signature = BASE64
            .decode(self.recipient.policy_signature.as_bytes())
            .map_err(|_| ProtectionError::LicenseInvalid("policy signature"))?;
        if policy_signature.len() != P256_SIGNATURE_BYTES {
            return Err(ProtectionError::LicenseInvalid("policy signature"));
        }
        Ok(())
    }

    fn bind(&self, verified: &VerifiedManifest) -> Result<()> {
        let manifest = verified.manifest();
        let expected_digest = decode_license_hex::<32>(&self.manifest_sha256, "license digest")?;
        if self.artifact_id != manifest.artifact_id
            || self.customer_scope_id != manifest.customer_scope_id
            || self.model_id != manifest.model.model_id
            || self.model_version != manifest.model.model_version
            || expected_digest != *verified.digest()
        {
            return Err(ProtectionError::LicenseBindingMismatch);
        }
        Ok(())
    }
}

pub(crate) fn decode_license_hex<const N: usize>(
    value: &str,
    label: &'static str,
) -> Result<[u8; N]> {
    decode_lower_hex(value, label).map_err(|_| ProtectionError::LicenseInvalid(label))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{SignatureEnvelope, verify_manifest};
    use ring::signature::{Ed25519KeyPair, KeyPair};

    fn lower_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn computes_frozen_rsa_decrypt_policy_encoding() {
        let device_name: [u8; 34] =
            decode_license_hex(&format!("000b{}", "11".repeat(32)), "test device name").unwrap();
        let ciphertext: [u8; 256] = std::array::from_fn(|index| index as u8);
        let cp_hash = rsa_decrypt_cp_hash(&device_name, &ciphertext);
        assert_eq!(
            lower_hex(&cp_hash),
            "92ea10bbcf39b63e1180809cd93eafce50623aeca2d518b605779b134378e134"
        );
        assert_eq!(
            lower_hex(&approved_rsa_decrypt_policy(&cp_hash)),
            "2a05643d8fde3f8a8145aec73370b0b81448c2dde9f1ca8fc2b2adb50accb2bb"
        );
        let policy_name: [u8; 34] =
            decode_license_hex(&format!("000b{}", "22".repeat(32)), "test policy name").unwrap();
        let policy_ref = decode_license_hex(TPM_POLICY_REF_HEX, "test policy ref").unwrap();
        assert_eq!(
            lower_hex(&policy_authorize_auth_policy(&policy_name, &policy_ref)),
            "1a33fe07b06b12c79e9a5b3d06f81e84beead674e685c1afadc2db4fd65dd69b"
        );
    }

    #[test]
    fn verifies_signature_and_exact_package_binding() {
        let package_key = Ed25519KeyPair::from_seed_unchecked(&[1; 32]).unwrap();
        let manifest_bytes = br#"{"format":"secure-model-package","format_version":1,"artifact_id":"00112233445566778899aabbccddeeff","customer_scope_id":"customer-a","model":{"model_id":"tiny","model_version":"1","framework":"safetensors"},"encryption":{"algorithm":"AES-256-GCM","nonce_prefix":"01020304","tag_bits":128,"record_plaintext_limit":1024},"protected_files":[{"file_id":1,"container_path":"weights/model.safetensors.protected","output_path":"model.safetensors","container_size":84,"container_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","plaintext_size":32,"plaintext_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","record_count":1,"first_global_record_counter":0}],"public_files":[],"runtime":{"minimum_runtime_version":"1.5.0","required_load_format":"safetensors"}}"#;
        let package_signature = serde_json::to_vec(&SignatureEnvelope {
            algorithm: "Ed25519".into(),
            key_id: "package-test".into(),
            signature: BASE64.encode(
                package_key
                    .sign(&crate::manifest_signature_payload(manifest_bytes))
                    .as_ref(),
            ),
        })
        .unwrap();
        let manifest = verify_manifest(
            manifest_bytes,
            &package_signature,
            "package-test",
            package_key.public_key().as_ref().try_into().unwrap(),
        )
        .unwrap();
        let license_key = Ed25519KeyPair::from_seed_unchecked(&[2; 32]).unwrap();
        let license = License {
            format: LICENSE_FORMAT.into(),
            format_version: LICENSE_FORMAT_VERSION,
            license_id: "license-a".into(),
            customer_scope_id: "customer-a".into(),
            artifact_id: "00112233445566778899aabbccddeeff".into(),
            manifest_sha256: lower_hex(manifest.digest()),
            model_id: "tiny".into(),
            model_version: "1".into(),
            entitlement: Entitlement {
                mode: "offline-perpetual".into(),
                generation: 1,
            },
            recipient: TpmRecipient {
                kind: "tpm2".into(),
                profile: TPM_PROFILE.into(),
                device_key_name: "000b".to_string() + &"33".repeat(32),
                device_public_key_sha256: "33".repeat(32),
                command_parameters_hash: "44".repeat(32),
                approved_policy_digest: "22".repeat(32),
                policy_ref: TPM_POLICY_REF_HEX.into(),
                policy_authority_key_id: "tpm-policy-v1".into(),
                policy_signature_algorithm: TPM_POLICY_SIGNATURE.into(),
                policy_signature: BASE64.encode([5; P256_SIGNATURE_BYTES]),
            },
            wrapped_dek: BASE64.encode([3; RSA_2048_CIPHERTEXT_BYTES]),
        };
        let license_bytes = serde_json::to_vec(&license).unwrap();
        let signature = serde_json::to_vec(&SignatureEnvelope {
            algorithm: "Ed25519".into(),
            key_id: "license-test".into(),
            signature: BASE64.encode(
                license_key
                    .sign(&license_signature_payload(&license_bytes))
                    .as_ref(),
            ),
        })
        .unwrap();
        assert!(
            verify_license(
                &license_bytes,
                &signature,
                "license-test",
                license_key.public_key().as_ref().try_into().unwrap(),
                manifest.clone(),
            )
            .is_ok()
        );

        let mut wrong_profile = license.clone();
        wrong_profile.recipient.profile = "tpm2".into();
        assert!(matches!(
            wrong_profile.validate(),
            Err(ProtectionError::LicenseInvalid("license policy"))
        ));
        let mut wrong_wrapping = license.clone();
        wrong_wrapping.wrapped_dek = BASE64.encode([3; 255]);
        assert!(matches!(
            wrong_wrapping.validate(),
            Err(ProtectionError::LicenseInvalid("wrapped dek"))
        ));
        let mut wrong_cp_hash = license.clone();
        wrong_cp_hash.recipient.command_parameters_hash = "00".repeat(31);
        assert!(matches!(
            wrong_cp_hash.validate(),
            Err(ProtectionError::LicenseInvalid(
                "license command parameters hash"
            ))
        ));
        let mut wrong_device_digest = license.clone();
        wrong_device_digest.recipient.device_public_key_sha256 = "11".repeat(32);
        assert!(matches!(
            wrong_device_digest.validate(),
            Err(ProtectionError::LicenseInvalid("license device key digest"))
        ));

        let mut wrong = license;
        wrong.artifact_id = "ff".repeat(16);
        let wrong_bytes = serde_json::to_vec(&wrong).unwrap();
        let wrong_signature = serde_json::to_vec(&SignatureEnvelope {
            algorithm: "Ed25519".into(),
            key_id: "license-test".into(),
            signature: BASE64.encode(
                license_key
                    .sign(&license_signature_payload(&wrong_bytes))
                    .as_ref(),
            ),
        })
        .unwrap();
        assert!(matches!(
            verify_license(
                &wrong_bytes,
                &wrong_signature,
                "license-test",
                license_key.public_key().as_ref().try_into().unwrap(),
                manifest,
            ),
            Err(ProtectionError::LicenseBindingMismatch)
        ));
    }
}
