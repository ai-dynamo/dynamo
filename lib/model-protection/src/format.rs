// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use ring::signature;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{ProtectionError, Result};

pub const FORMAT_NAME: &str = "secure-model-package";
pub const FORMAT_VERSION: u16 = 1;
pub const MANIFEST_DIGEST_DOMAIN: &[u8] = b"model-protection-manifest-v1\0";
pub const MANIFEST_SIGNATURE_DOMAIN: &[u8] = b"model-protection-manifest-signature-v1\0";
pub const MAX_MANIFEST_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_FILES: usize = 4096;
pub const MAX_PATH_BYTES: usize = 1024;
pub const MAX_RECORD_PLAINTEXT: u32 = 64 * 1024 * 1024;
pub const MAX_FILE_BYTES: u64 = 1 << 40;
pub const MAX_RECORDS_PER_ARTIFACT: u64 = 1 << 20;
pub const MAX_SAFETENSORS_INDEX_BYTES: u64 = 4 * 1024 * 1024;
pub(crate) const MAX_SIGNATURE_BYTES: usize = 4096;
const RECORD_OVERHEAD_BYTES: u64 = 36 + 16;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format: String,
    pub format_version: u16,
    pub artifact_id: String,
    pub customer_scope_id: String,
    pub model: ModelIdentity,
    pub encryption: Encryption,
    pub protected_files: Vec<ProtectedFile>,
    pub public_files: Vec<PublicFile>,
    pub runtime: RuntimeRequirements,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    pub model_id: String,
    pub model_version: String,
    pub framework: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Encryption {
    pub algorithm: String,
    pub nonce_prefix: String,
    pub tag_bits: u16,
    pub record_plaintext_limit: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedFile {
    pub file_id: u32,
    pub container_path: String,
    pub output_path: String,
    pub container_size: u64,
    pub container_sha256: String,
    pub plaintext_size: u64,
    pub plaintext_sha256: String,
    pub record_count: u32,
    pub first_global_record_counter: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicFile {
    pub source_path: String,
    pub output_path: String,
    pub size: u64,
    pub sha256: String,
    pub publish_to_model_card: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeRequirements {
    pub minimum_runtime_version: String,
    pub required_load_format: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignatureEnvelope {
    pub algorithm: String,
    pub key_id: String,
    pub signature: String,
}

/// A parsed manifest whose signature was checked over the exact input bytes.
#[derive(Clone, Debug)]
pub struct VerifiedManifest {
    manifest: Manifest,
    digest: [u8; 32],
}

impl VerifiedManifest {
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
}

pub fn parse_manifest(bytes: &[u8]) -> Result<Manifest> {
    if bytes.is_empty() || bytes.len() > MAX_MANIFEST_BYTES {
        return Err(ProtectionError::InvalidPackage("manifest size"));
    }
    let manifest: Manifest = serde_json::from_slice(bytes)
        .map_err(|_| ProtectionError::InvalidPackage("manifest json"))?;
    manifest.validate()?;
    Ok(manifest)
}

pub fn manifest_digest(bytes: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(MANIFEST_DIGEST_DOMAIN);
    digest.update(bytes);
    digest.finalize().into()
}

pub fn manifest_signature_payload(bytes: &[u8]) -> Vec<u8> {
    signature_payload(MANIFEST_SIGNATURE_DOMAIN, bytes)
}

pub fn verify_manifest(
    manifest_bytes: &[u8],
    envelope_bytes: &[u8],
    expected_key_id: &str,
    verifying_key: &[u8; 32],
) -> Result<VerifiedManifest> {
    if manifest_bytes.is_empty() || manifest_bytes.len() > MAX_MANIFEST_BYTES {
        return Err(ProtectionError::InvalidPackage("manifest size"));
    }
    if !has_valid_ed25519_signature(
        MANIFEST_SIGNATURE_DOMAIN,
        manifest_bytes,
        envelope_bytes,
        expected_key_id,
        verifying_key,
    ) {
        return Err(ProtectionError::ManifestSignatureInvalid);
    }
    Ok(VerifiedManifest {
        manifest: parse_manifest(manifest_bytes)?,
        digest: manifest_digest(manifest_bytes),
    })
}

pub(crate) fn has_valid_ed25519_signature(
    domain: &[u8],
    payload: &[u8],
    envelope_bytes: &[u8],
    expected_key_id: &str,
    verifying_key: &[u8; 32],
) -> bool {
    if envelope_bytes.is_empty() || envelope_bytes.len() > MAX_SIGNATURE_BYTES {
        return false;
    }
    let Ok(envelope) = serde_json::from_slice::<SignatureEnvelope>(envelope_bytes) else {
        return false;
    };
    if envelope.algorithm != "Ed25519" || envelope.key_id != expected_key_id {
        return false;
    }
    let Ok(signature_bytes) = BASE64.decode(envelope.signature.as_bytes()) else {
        return false;
    };
    let signed = signature_payload(domain, payload);
    signature::UnparsedPublicKey::new(&signature::ED25519, verifying_key)
        .verify(&signed, &signature_bytes)
        .is_ok()
}

pub(crate) fn signature_payload(domain: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut signed = Vec::with_capacity(domain.len() + payload.len());
    signed.extend_from_slice(domain);
    signed.extend_from_slice(payload);
    signed
}

impl Manifest {
    pub fn artifact_id_bytes(&self) -> Result<[u8; 16]> {
        decode_lower_hex::<16>(&self.artifact_id, "artifact id")
    }

    pub fn nonce_prefix_bytes(&self) -> Result<[u8; 4]> {
        decode_lower_hex::<4>(&self.encryption.nonce_prefix, "nonce prefix")
    }

    pub fn validate(&self) -> Result<()> {
        if self.format != FORMAT_NAME || self.format_version != FORMAT_VERSION {
            return Err(ProtectionError::InvalidPackage("format version"));
        }
        self.artifact_id_bytes()?;
        validate_identifier(&self.customer_scope_id)?;
        validate_identifier(&self.model.model_id)?;
        validate_identifier(&self.model.model_version)?;
        semver::Version::parse(&self.runtime.minimum_runtime_version)
            .map_err(|_| ProtectionError::InvalidPackage("minimum runtime version"))?;
        if self.model.framework != "safetensors"
            || self.runtime.required_load_format != "safetensors"
        {
            return Err(ProtectionError::InvalidPackage("model format"));
        }
        if self.encryption.algorithm != "AES-256-GCM" || self.encryption.tag_bits != 128 {
            return Err(ProtectionError::InvalidPackage("encryption parameters"));
        }
        self.nonce_prefix_bytes()?;
        if self.encryption.record_plaintext_limit == 0
            || self.encryption.record_plaintext_limit > MAX_RECORD_PLAINTEXT
        {
            return Err(ProtectionError::InvalidPackage("record limit"));
        }
        let file_count = self
            .protected_files
            .len()
            .checked_add(self.public_files.len())
            .ok_or(ProtectionError::InvalidPackage("file count"))?;
        if self.protected_files.is_empty() || file_count > MAX_FILES {
            return Err(ProtectionError::InvalidPackage("file count"));
        }

        let mut file_ids = BTreeSet::new();
        let mut source_paths = BTreeSet::new();
        let mut output_paths = BTreeSet::new();
        let mut counter_ranges = Vec::with_capacity(self.protected_files.len());
        let mut total_records = 0_u64;
        for file in &self.protected_files {
            if file.file_id == 0 || !file_ids.insert(file.file_id) {
                return Err(ProtectionError::InvalidPackage("file id"));
            }
            validate_relative_path(&file.container_path)?;
            validate_relative_path(&file.output_path)?;
            validate_output_path(&file.output_path)?;
            let Some(container_name) = file.container_path.strip_prefix("weights/") else {
                return Err(ProtectionError::InvalidPackage("protected file path"));
            };
            if container_name.contains('/')
                || file.output_path.contains('/')
                || !container_name.ends_with(".safetensors.protected")
                || !file.output_path.ends_with(".safetensors")
            {
                return Err(ProtectionError::InvalidPackage("protected file path"));
            }
            if !source_paths.insert(file.container_path.as_str())
                || !output_paths.insert(file.output_path.as_str())
            {
                return Err(ProtectionError::InvalidPackage("duplicate path"));
            }
            validate_file_sizes(file.container_size, file.plaintext_size)?;
            validate_sha256(&file.container_sha256)?;
            validate_sha256(&file.plaintext_sha256)?;
            if file.record_count == 0 {
                return Err(ProtectionError::InvalidPackage("record count"));
            }
            let expected_record_count = file
                .plaintext_size
                .div_ceil(u64::from(self.encryption.record_plaintext_limit));
            if u64::from(file.record_count) != expected_record_count {
                return Err(ProtectionError::InvalidPackage("record count"));
            }
            total_records = total_records
                .checked_add(u64::from(file.record_count))
                .ok_or(ProtectionError::InvalidPackage("record count"))?;
            let expected_container_size = file
                .plaintext_size
                .checked_add(
                    u64::from(file.record_count)
                        .checked_mul(RECORD_OVERHEAD_BYTES)
                        .ok_or(ProtectionError::InvalidPackage("container size"))?,
                )
                .ok_or(ProtectionError::InvalidPackage("container size"))?;
            if file.container_size != expected_container_size {
                return Err(ProtectionError::InvalidPackage("container size"));
            }
            let end = file
                .first_global_record_counter
                .checked_add(u64::from(file.record_count))
                .ok_or(ProtectionError::InvalidPackage("record counter"))?;
            counter_ranges.push((file.first_global_record_counter, end));
        }
        if total_records > MAX_RECORDS_PER_ARTIFACT {
            return Err(ProtectionError::InvalidPackage("record count"));
        }
        counter_ranges.sort_unstable();
        if counter_ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(ProtectionError::InvalidPackage("record counter overlap"));
        }
        if counter_ranges.first().is_none_or(|range| range.0 != 0)
            || counter_ranges.windows(2).any(|pair| pair[0].1 != pair[1].0)
        {
            return Err(ProtectionError::InvalidPackage("record counter sequence"));
        }

        for file in &self.public_files {
            validate_relative_path(&file.source_path)?;
            validate_relative_path(&file.output_path)?;
            validate_output_path(&file.output_path)?;
            if !file.source_path.starts_with("public/")
                || file.source_path.strip_prefix("public/") != Some(file.output_path.as_str())
                || !source_paths.insert(file.source_path.as_str())
                || !output_paths.insert(file.output_path.as_str())
            {
                return Err(ProtectionError::InvalidPackage("public file path"));
            }
            if !file.publish_to_model_card {
                return Err(ProtectionError::InvalidPackage("public file policy"));
            }
            if file.size == 0 || file.size > MAX_FILE_BYTES {
                return Err(ProtectionError::InvalidPackage("public file size"));
            }
            if !is_allowed_public_metadata(&file.output_path)
                || (file.output_path == "model.safetensors.index.json"
                    && file.size > MAX_SAFETENSORS_INDEX_BYTES)
            {
                return Err(ProtectionError::InvalidPackage("public file policy"));
            }
            validate_sha256(&file.sha256)?;
        }
        for output in &output_paths {
            let mut parent = Path::new(output).parent();
            while let Some(path) = parent.filter(|path| !path.as_os_str().is_empty()) {
                let path = path
                    .to_str()
                    .ok_or(ProtectionError::InvalidPackage("output path"))?;
                if output_paths.contains(path) {
                    return Err(ProtectionError::InvalidPackage("output path collision"));
                }
                parent = Path::new(path).parent();
            }
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SafetensorsIndex {
    #[serde(default, rename = "metadata")]
    _metadata: Option<serde_json::Value>,
    weight_map: BTreeMap<String, String>,
}

/// Validates the exact safetensors layout shared by package producers and consumers.
pub fn validate_safetensors_index(manifest: &Manifest, index_bytes: Option<&[u8]>) -> Result<()> {
    let protected: BTreeSet<&str> = manifest
        .protected_files
        .iter()
        .map(|file| file.output_path.as_str())
        .collect();
    let Some(bytes) = index_bytes else {
        if protected.len() > 1 {
            return Err(ProtectionError::InvalidPackage(
                "safetensors index required",
            ));
        }
        return Ok(());
    };
    let index: SafetensorsIndex = serde_json::from_slice(bytes)
        .map_err(|_| ProtectionError::InvalidPackage("safetensors index"))?;
    if index.weight_map.is_empty() {
        return Err(ProtectionError::InvalidPackage("safetensors index"));
    }
    let mut referenced = BTreeSet::new();
    for output in index.weight_map.values() {
        validate_relative_path(output)?;
        if !protected.contains(output.as_str()) {
            return Err(ProtectionError::InvalidPackage("safetensors index"));
        }
        referenced.insert(output.as_str());
    }
    if referenced != protected {
        return Err(ProtectionError::InvalidPackage("safetensors index"));
    }
    Ok(())
}

pub fn enforce_runtime_version(manifest: &VerifiedManifest, current: &str) -> Result<()> {
    let minimum = semver::Version::parse(&manifest.manifest.runtime.minimum_runtime_version)
        .map_err(|_| ProtectionError::InvalidPackage("minimum runtime version"))?;
    let current =
        semver::Version::parse(current).map_err(|_| ProtectionError::RuntimeUnsupported)?;
    if current < minimum {
        return Err(ProtectionError::RuntimeUnsupported);
    }
    Ok(())
}

fn validate_output_path(value: &str) -> Result<()> {
    if Path::new(value).components().any(|component| {
        matches!(component, Component::Normal(name) if
            name == ".owner.lock"
                || name == "state"
                || name.to_str().is_some_and(|name| name.ends_with(".partial")))
    }) {
        return Err(ProtectionError::InvalidPackage("reserved output path"));
    }
    Ok(())
}

pub fn is_allowed_public_metadata(path: &str) -> bool {
    matches!(
        path,
        "config.json"
            | "generation_config.json"
            | "tokenizer.json"
            | "tokenizer_config.json"
            | "special_tokens_map.json"
            | "added_tokens.json"
            | "vocab.json"
            | "vocab.txt"
            | "merges.txt"
            | "tokenizer.model"
            | "sentencepiece.bpe.model"
            | "spiece.model"
            | "preprocessor_config.json"
            | "processor_config.json"
            | "chat_template.json"
            | "chat_template.jinja"
            | "model.safetensors.index.json"
    )
}

fn validate_identifier(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 128 || value.chars().any(char::is_control) {
        return Err(ProtectionError::InvalidPackage("identifier"));
    }
    Ok(())
}

fn validate_file_sizes(container_size: u64, plaintext_size: u64) -> Result<()> {
    if container_size == 0
        || plaintext_size == 0
        || container_size > MAX_FILE_BYTES
        || plaintext_size > MAX_FILE_BYTES
    {
        return Err(ProtectionError::InvalidPackage("file size"));
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<()> {
    decode_lower_hex::<32>(value, "sha256").map(|_| ())
}

pub(crate) fn validate_relative_path(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > MAX_PATH_BYTES
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_alphanumeric() && !matches!(byte, b'/' | b'.' | b'_' | b'-'))
    {
        return Err(ProtectionError::InvalidPackage("path"));
    }
    if value.ends_with('/')
        || value
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."))
    {
        return Err(ProtectionError::InvalidPackage("path"));
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path.components().any(|component| {
            !matches!(component, Component::Normal(_))
                || component.as_os_str().to_str().is_none()
                || component.as_os_str().len() > 240
        })
    {
        return Err(ProtectionError::InvalidPackage("path"));
    }
    Ok(())
}

pub(crate) fn decode_lower_hex<const N: usize>(
    value: &str,
    label: &'static str,
) -> Result<[u8; N]> {
    if value.len() != N * 2
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
    {
        return Err(ProtectionError::InvalidPackage(label));
    }
    let mut output = [0_u8; N];
    for (index, slot) in output.iter_mut().enumerate() {
        let high = hex_nibble(value.as_bytes()[index * 2]);
        let low = hex_nibble(value.as_bytes()[index * 2 + 1]);
        *slot = (high << 4) | low;
    }
    Ok(output)
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => unreachable!("validated hex digit"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::signature::{Ed25519KeyPair, KeyPair};

    fn valid_manifest_json() -> Vec<u8> {
        br#"{
          "format":"secure-model-package",
          "format_version":1,
          "artifact_id":"00112233445566778899aabbccddeeff",
          "customer_scope_id":"customer-a",
          "model":{"model_id":"tiny","model_version":"1","framework":"safetensors"},
          "encryption":{"algorithm":"AES-256-GCM","nonce_prefix":"01020304","tag_bits":128,"record_plaintext_limit":1024},
          "protected_files":[{"file_id":1,"container_path":"weights/model.safetensors.protected","output_path":"model.safetensors","container_size":84,"container_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","plaintext_size":32,"plaintext_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","record_count":1,"first_global_record_counter":0}],
          "public_files":[{"source_path":"public/config.json","output_path":"config.json","size":2,"sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","publish_to_model_card":true}],
          "runtime":{"minimum_runtime_version":"1.5.0","required_load_format":"safetensors"}
        }"#
        .to_vec()
    }

    #[test]
    fn parses_valid_manifest_and_rejects_duplicate_fields() {
        assert!(parse_manifest(&valid_manifest_json()).is_ok());
        let duplicate = String::from_utf8(valid_manifest_json()).unwrap().replacen(
            "\"format_version\":1,",
            "\"format_version\":1,\"format_version\":1,",
            1,
        );
        assert!(matches!(
            parse_manifest(duplicate.as_bytes()),
            Err(ProtectionError::InvalidPackage("manifest json"))
        ));
    }

    #[test]
    fn rejects_traversal_and_overlapping_counters() {
        let mut manifest = parse_manifest(&valid_manifest_json()).unwrap();
        manifest.public_files[0].output_path = "../config.json".to_string();
        assert!(manifest.validate().is_err());

        let mut manifest = parse_manifest(&valid_manifest_json()).unwrap();
        let mut duplicate = manifest.protected_files[0].clone();
        duplicate.file_id = 2;
        duplicate.container_path = "weights/other.safetensors.protected".to_string();
        duplicate.output_path = "other.safetensors".to_string();
        manifest.protected_files.push(duplicate);
        assert!(matches!(
            manifest.validate(),
            Err(ProtectionError::InvalidPackage("record counter overlap"))
        ));

        let mut manifest = parse_manifest(&valid_manifest_json()).unwrap();
        manifest.protected_files[0].first_global_record_counter = 1;
        assert!(matches!(
            manifest.validate(),
            Err(ProtectionError::InvalidPackage("record counter sequence"))
        ));

        for alias in [
            "nested//config.json",
            "nested/./config.json",
            "nested/config.json/",
        ] {
            assert!(validate_relative_path(alias).is_err(), "accepted {alias:?}");
            let mut manifest = parse_manifest(&valid_manifest_json()).unwrap();
            manifest.public_files[0].source_path = format!("public/{alias}");
            manifest.public_files[0].output_path = alias.to_string();
            assert!(manifest.validate().is_err(), "accepted {alias:?}");
        }

        let mut manifest = parse_manifest(&valid_manifest_json()).unwrap();
        manifest.public_files[0].source_path = "public/plaintext.safetensors".to_string();
        manifest.public_files[0].output_path = "plaintext.safetensors".to_string();
        assert!(matches!(
            manifest.validate(),
            Err(ProtectionError::InvalidPackage("public file policy"))
        ));

        let mut manifest = parse_manifest(&valid_manifest_json()).unwrap();
        manifest.protected_files[0].output_path = "nested/model.safetensors".to_string();
        assert!(matches!(
            manifest.validate(),
            Err(ProtectionError::InvalidPackage("protected file path"))
        ));

        let mut manifest = parse_manifest(&valid_manifest_json()).unwrap();
        manifest.public_files[0].publish_to_model_card = false;
        assert!(matches!(
            manifest.validate(),
            Err(ProtectionError::InvalidPackage("public file policy"))
        ));
    }

    #[test]
    fn validates_sharded_safetensors_layout() {
        let mut manifest = parse_manifest(&valid_manifest_json()).unwrap();
        let mut second = manifest.protected_files[0].clone();
        second.file_id = 2;
        second.container_path = "weights/model-00002.safetensors.protected".to_string();
        second.output_path = "model-00002.safetensors".to_string();
        second.first_global_record_counter = 1;
        manifest.protected_files[0].container_path =
            "weights/model-00001.safetensors.protected".to_string();
        manifest.protected_files[0].output_path = "model-00001.safetensors".to_string();
        manifest.protected_files.push(second);
        assert!(matches!(
            validate_safetensors_index(&manifest, None),
            Err(ProtectionError::InvalidPackage(
                "safetensors index required"
            ))
        ));
        let valid =
            br#"{"weight_map":{"a":"model-00001.safetensors","b":"model-00002.safetensors"}}"#;
        assert!(validate_safetensors_index(&manifest, Some(valid)).is_ok());
        let incomplete = br#"{"weight_map":{"a":"model-00001.safetensors"}}"#;
        assert!(validate_safetensors_index(&manifest, Some(incomplete)).is_err());
    }

    #[test]
    fn verifies_signature_over_exact_manifest_bytes() {
        let manifest = valid_manifest_json();
        let signing_key = Ed25519KeyPair::from_seed_unchecked(&[7_u8; 32]).unwrap();
        let envelope = SignatureEnvelope {
            algorithm: "Ed25519".to_string(),
            key_id: "package-test".to_string(),
            signature: BASE64.encode(
                signing_key
                    .sign(&manifest_signature_payload(&manifest))
                    .as_ref(),
            ),
        };
        let envelope = serde_json::to_vec(&envelope).unwrap();
        let verified = verify_manifest(
            &manifest,
            &envelope,
            "package-test",
            signing_key.public_key().as_ref().try_into().unwrap(),
        )
        .unwrap();
        assert!(enforce_runtime_version(&verified, "1.5.0").is_ok());
        assert!(matches!(
            enforce_runtime_version(&verified, "1.4.9"),
            Err(ProtectionError::RuntimeUnsupported)
        ));
        let mut changed = manifest;
        changed.push(b' ');
        assert!(matches!(
            verify_manifest(
                &changed,
                &envelope,
                "package-test",
                signing_key.public_key().as_ref().try_into().unwrap(),
            ),
            Err(ProtectionError::ManifestSignatureInvalid)
        ));
    }
}
