// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Layer configuration and key-provider dispatch shared by all backend adapters.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use rustix::fs::{CWD, FileType, Mode, OFlags, ResolveFlags, fstat, openat2};
use rustix::process::geteuid;
use serde::Deserialize;
use zeroize::Zeroizing;

use crate::{
    AuthorizedModel, CancellationToken, FileLicense, ProtectionError, Result, SecretDek,
    SecureModelSession, VerifiedManifest, enforce_process_persistence_policy,
    load_verified_manifest, verify_file_license,
};

pub const MAX_CONFIG_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProtectionLayers {
    pub package_verification: bool,
    pub license_verification: bool,
    pub tpm_binding: bool,
    pub secure_materialization: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustKey {
    pub key_id: String,
    pub public_key_file: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TpmConfig {
    pub device_key_handle: String,
    #[serde(default)]
    pub policy_authority_key_handle: Option<String>,
    #[serde(default)]
    pub policy_authority_public_file: Option<PathBuf>,
    pub policy_authority_key_id: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyProviderConfig {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub key_file: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    #[serde(default = "schema_version")]
    pub schema_version: u16,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub layers: ProtectionLayers,
    #[serde(default)]
    pub package_trust: Option<TrustKey>,
    #[serde(default)]
    pub license_root: Option<PathBuf>,
    #[serde(default)]
    pub license_trust: Option<TrustKey>,
    #[serde(default)]
    pub key_provider: Option<KeyProviderConfig>,
    #[serde(default)]
    pub tpm: Option<TpmConfig>,
    #[serde(default)]
    pub process_memory_margin_bytes: u64,
}

const fn schema_version() -> u16 {
    2
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyConfig {
    license_root: PathBuf,
    package_trust: TrustKey,
    license_trust: TrustKey,
    tpm: TpmConfig,
    process_memory_margin_bytes: u64,
}

impl RuntimeConfig {
    /// Existing configuration files retain their fixed TPM profile. Inline JSON
    /// uses V2 and false defaults; no new flag is implicitly enabled.
    pub fn parse(bytes: &[u8], allow_legacy: bool) -> Result<Self> {
        if bytes.is_empty() || bytes.len() > MAX_CONFIG_BYTES {
            return Err(ProtectionError::RuntimeConfigInvalid);
        }
        let value: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|_| ProtectionError::RuntimeConfigInvalid)?;
        if allow_legacy
            && value.get("schema_version").is_none()
            && value.get("layers").is_none()
            && value.get("license_root").is_some()
        {
            let old: LegacyConfig =
                serde_json::from_slice(bytes).map_err(|_| ProtectionError::RuntimeConfigInvalid)?;
            return Ok(Self {
                schema_version: 2,
                profile: Some("encrypted-tpm".into()),
                layers: ProtectionLayers {
                    package_verification: true,
                    license_verification: true,
                    tpm_binding: true,
                    secure_materialization: true,
                },
                package_trust: Some(old.package_trust),
                license_root: Some(old.license_root),
                license_trust: Some(old.license_trust),
                tpm: Some(old.tpm),
                key_provider: Some(KeyProviderConfig {
                    kind: "tpm".into(),
                    key_file: None,
                }),
                process_memory_margin_bytes: old.process_memory_margin_bytes,
            });
        }
        serde_json::from_slice(bytes).map_err(|_| ProtectionError::RuntimeConfigInvalid)
    }

    pub fn profile(&self) -> &str {
        self.profile
            .as_deref()
            .unwrap_or(if self.layers.tpm_binding {
                "encrypted-tpm"
            } else if self.layers.license_verification {
                "encrypted-file-license"
            } else {
                "encrypted-file"
            })
    }

    pub fn validate(&self) -> Result<()> {
        let invalid = || ProtectionError::RuntimeConfigInvalid;
        if self.schema_version != 2 || self.process_memory_margin_bytes > 1 << 40 {
            return Err(invalid());
        }
        for (enabled, name) in [
            (self.layers.package_verification, "package_verification"),
            (self.layers.secure_materialization, "secure_materialization"),
        ] {
            if !enabled {
                return Err(ProtectionError::ProtectionLayerDisabled(name));
            }
        }
        match self.profile() {
            "encrypted-tpm" => {
                if !self.layers.tpm_binding {
                    return Err(ProtectionError::ProtectionLayerDisabled("tpm_binding"));
                }
                if !self.layers.license_verification {
                    return Err(ProtectionError::ProtectionLayerDisabled(
                        "license_verification",
                    ));
                }
                let tpm = self.tpm.as_ref().ok_or_else(invalid)?;
                parse_handle(&tpm.device_key_handle).ok_or_else(invalid)?;
                if !valid_identifier(&tpm.policy_authority_key_id)
                    || (tpm.policy_authority_key_handle.is_some()
                        == tpm.policy_authority_public_file.is_some())
                    || tpm
                        .policy_authority_key_handle
                        .as_ref()
                        .is_some_and(|h| parse_handle(h).is_none())
                    || tpm
                        .policy_authority_public_file
                        .as_ref()
                        .is_some_and(|p| !p.is_absolute())
                {
                    return Err(invalid());
                }
                if self
                    .key_provider
                    .as_ref()
                    .is_some_and(|k| k.kind != "tpm" || k.key_file.is_some())
                {
                    return Err(invalid());
                }
            }
            "encrypted-file" | "encrypted-file-license" => {
                if self.layers.tpm_binding {
                    return Err(invalid());
                }
                if self.profile() == "encrypted-file-license" && !self.layers.license_verification {
                    return Err(ProtectionError::ProtectionLayerDisabled(
                        "license_verification",
                    ));
                }
                if self.profile() == "encrypted-file" && self.layers.license_verification {
                    return Err(invalid());
                }
                let provider = self.key_provider.as_ref().ok_or_else(invalid)?;
                if provider.kind != "file"
                    || !provider.key_file.as_ref().is_some_and(|p| p.is_absolute())
                {
                    return Err(invalid());
                }
            }
            _ => return Err(invalid()),
        }
        validate_trust(self.package_trust.as_ref().ok_or_else(invalid)?)?;
        if self.layers.license_verification {
            let trust = self.license_trust.as_ref().ok_or_else(invalid)?;
            validate_trust(trust)?;
            if !self.license_root.as_ref().is_some_and(|p| p.is_absolute()) {
                return Err(invalid());
            }
            if self
                .package_trust
                .as_ref()
                .is_some_and(|p| p.key_id == trust.key_id)
            {
                return Err(invalid());
            }
        }
        Ok(())
    }
}

fn validate_trust(key: &TrustKey) -> Result<()> {
    if !valid_identifier(&key.key_id) || !key.public_key_file.is_absolute() {
        return Err(ProtectionError::RuntimeConfigInvalid);
    }
    Ok(())
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 128 && !value.chars().any(char::is_control)
}

pub fn parse_handle(value: &str) -> Option<u32> {
    value
        .strip_prefix("0x")
        .filter(|digits| digits.len() == 8)
        .and_then(|digits| u32::from_str_radix(digits, 16).ok())
        .filter(|h| (0x81000000..=0x81ffffff).contains(h))
}

/// No symlinks, special files, shared permissions or unbounded reads.
pub fn read_private_file(path: &Path, maximum: usize) -> Result<Zeroizing<Vec<u8>>> {
    if !path.is_absolute() {
        return Err(ProtectionError::RuntimeConfigInvalid);
    }
    let fd = openat2(
        CWD,
        path,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
        ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
    )
    .map_err(|_| ProtectionError::RuntimeConfigInvalid)?;
    let stat = fstat(&fd).map_err(|_| ProtectionError::RuntimeConfigInvalid)?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || stat.st_nlink != 1
        || stat.st_uid != geteuid().as_raw()
        || stat.st_size <= 0
        || stat.st_size as u64 > maximum as u64
        || Mode::from_raw_mode(stat.st_mode).intersects(Mode::RWXG | Mode::RWXO)
    {
        return Err(ProtectionError::RuntimeConfigInvalid);
    }
    let mut bytes = Zeroizing::new(Vec::with_capacity(stat.st_size as usize));
    File::from(fd)
        .take(maximum as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| ProtectionError::RuntimeConfigInvalid)?;
    if bytes.len() != stat.st_size as usize || bytes.len() > maximum {
        return Err(ProtectionError::RuntimeConfigInvalid);
    }
    Ok(bytes)
}

fn read_public_key(path: &Path) -> Result<[u8; 32]> {
    read_private_file(path, 32)?
        .as_slice()
        .try_into()
        .map_err(|_| ProtectionError::RuntimeConfigInvalid)
}

pub struct PreparedRuntime {
    session: SecureModelSession,
    verified: VerifiedManifest,
    #[cfg_attr(not(feature = "tpm2"), allow(dead_code))]
    authorized: Option<AuthorizedModel>,
    file_license: Option<FileLicense>,
    package_root: PathBuf,
    config: RuntimeConfig,
    weights_ready: bool,
}

impl PreparedRuntime {
    pub fn prepare(package_root: &Path, namespace: &str, input: &str) -> Result<Self> {
        let inline = input.trim_start().starts_with('{');
        if inline && input.len() > MAX_CONFIG_BYTES {
            return Err(ProtectionError::RuntimeConfigInvalid);
        }
        let bytes = if inline {
            Zeroizing::new(input.as_bytes().to_vec())
        } else {
            read_private_file(Path::new(input), MAX_CONFIG_BYTES)?
        };
        let config = RuntimeConfig::parse(&bytes, !inline)?;
        config.validate()?;
        #[cfg(not(feature = "tpm2"))]
        if config.layers.tpm_binding {
            return Err(ProtectionError::TpmUnavailable);
        }
        let trust = config
            .package_trust
            .as_ref()
            .ok_or(ProtectionError::RuntimeConfigInvalid)?;
        let package_key = read_public_key(&trust.public_key_file)?;
        let verified = load_verified_manifest(package_root, &trust.key_id, &package_key)?;
        let signed_profile = verified
            .manifest()
            .runtime
            .protection_profile
            .as_deref()
            .unwrap_or("encrypted-tpm");
        if signed_profile != config.profile() {
            return Err(ProtectionError::LicenseBindingMismatch);
        }
        let mut authorized = None;
        let mut file_license = None;
        if config.layers.license_verification {
            let license_trust = config
                .license_trust
                .as_ref()
                .ok_or(ProtectionError::RuntimeConfigInvalid)?;
            let license_key = read_public_key(&license_trust.public_key_file)?;
            if package_key == license_key {
                return Err(ProtectionError::LicenseInvalid("trust domain"));
            }
            let root = config
                .license_root
                .as_ref()
                .ok_or(ProtectionError::RuntimeConfigInvalid)?;
            if config.layers.tpm_binding {
                authorized = Some(crate::materialize::load_authorized_verified(
                    root,
                    &license_trust.key_id,
                    &license_key,
                    verified.clone(),
                )?);
            } else {
                file_license = Some(verify_file_license(
                    &read_private_file(
                        &root.join("model.protection.license.json"),
                        MAX_CONFIG_BYTES,
                    )?,
                    &read_private_file(
                        &root.join("model.protection.license.sig"),
                        crate::format::MAX_SIGNATURE_BYTES,
                    )?,
                    &license_trust.key_id,
                    &license_key,
                    &verified,
                )?);
            }
        }
        enforce_process_persistence_policy()?;
        let mut session = SecureModelSession::prepare_verified(
            namespace,
            &verified,
            config.process_memory_margin_bytes,
        )?;
        session.stage_verified_metadata(package_root, &verified)?;
        Ok(Self {
            session,
            verified,
            authorized,
            file_license,
            package_root: package_root.to_path_buf(),
            config,
            weights_ready: false,
        })
    }

    pub fn model_path(&self) -> &Path {
        self.session.model_path()
    }

    pub fn materialize(&mut self, cancellation: &CancellationToken) -> Result<()> {
        if self.weights_ready {
            return Err(ProtectionError::SessionConflict);
        }
        if cancellation.is_cancelled() {
            return Err(ProtectionError::MaterializationCancelled);
        }
        if self.config.layers.tpm_binding {
            #[cfg(not(feature = "tpm2"))]
            return Err(ProtectionError::TpmUnavailable);
            #[cfg(feature = "tpm2")]
            {
                let tpm = self
                    .config
                    .tpm
                    .as_ref()
                    .ok_or(ProtectionError::RuntimeConfigInvalid)?;
                let policy = match (
                    &tpm.policy_authority_key_handle,
                    &tpm.policy_authority_public_file,
                ) {
                    (Some(handle), None) => crate::TpmPolicyAuthority::Persistent(
                        parse_handle(handle).ok_or(ProtectionError::RuntimeConfigInvalid)?,
                    ),
                    (None, Some(path)) => {
                        crate::TpmPolicyAuthority::Public(read_private_file(path, 88)?.to_vec())
                    }
                    _ => return Err(ProtectionError::RuntimeConfigInvalid),
                };
                crate::materialize_tpm_model_with_policy(
                    &mut self.session,
                    &self.package_root,
                    self.authorized
                        .as_ref()
                        .ok_or(ProtectionError::RuntimeConfigInvalid)?,
                    &tpm.policy_authority_key_id,
                    "device:/dev/tpmrm0",
                    parse_handle(&tpm.device_key_handle)
                        .ok_or(ProtectionError::RuntimeConfigInvalid)?,
                    &policy,
                    cancellation,
                )?;
            }
        } else {
            let path = self
                .config
                .key_provider
                .as_ref()
                .and_then(|p| p.key_file.as_ref())
                .ok_or(ProtectionError::RuntimeConfigInvalid)?;
            let bytes =
                read_private_file(path, 32).map_err(|_| ProtectionError::KeyProviderInvalid)?;
            let key_bytes = Zeroizing::new(
                <[u8; 32]>::try_from(bytes.as_slice())
                    .map_err(|_| ProtectionError::KeyProviderInvalid)?,
            );
            if let Some(license) = &self.file_license {
                license.verify_key(&key_bytes)?;
            }
            let key = SecretDek::from_file(*key_bytes)?;
            self.session.materialize_verified_cancellable(
                &self.package_root,
                &self.verified,
                key,
                cancellation,
            )?;
        }
        self.weights_ready = true;
        Ok(())
    }

    pub fn cleanup(self) -> Result<()> {
        self.session.cleanup()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};

    fn file_config() -> serde_json::Value {
        serde_json::json!({"schema_version":2,"profile":"encrypted-file",
            "layers":{"package_verification":true,"secure_materialization":true},
            "package_trust":{"key_id":"package-v1","public_key_file":"/runtime/trust/package.key"},
            "key_provider":{"type":"file","key_file":"/run/secrets/model.key"}})
    }

    #[test]
    fn defaults_do_not_enable_any_layer() {
        let config = RuntimeConfig::parse(b"{}", false).unwrap();
        assert!(
            !config.layers.package_verification
                && !config.layers.license_verification
                && !config.layers.tpm_binding
                && !config.layers.secure_materialization
        );
        assert!(matches!(
            config.validate(),
            Err(ProtectionError::ProtectionLayerDisabled(
                "package_verification"
            ))
        ));
    }

    #[test]
    fn file_mode_has_no_license_or_tpm_requirements() {
        let config =
            RuntimeConfig::parse(&serde_json::to_vec(&file_config()).unwrap(), false).unwrap();
        config.validate().unwrap();
        assert!(config.tpm.is_none() && config.license_root.is_none());
        let mut value = file_config();
        value["layers"]["tpm_binding"] = true.into();
        assert!(
            RuntimeConfig::parse(&serde_json::to_vec(&value).unwrap(), false)
                .unwrap()
                .validate()
                .is_err()
        );
    }

    #[test]
    fn unknown_removed_flags_duplicates_and_strings_are_rejected() {
        for input in [
            br#"{"layers":{"audit":false}}"#.as_slice(),
            br#"{"layers":{"tpm_binding":"false"}}"#,
            br#"{"layers":{"tpm_binding":false,"tpm_binding":true}}"#,
            br#"{"schema_version":2,"schema_version":2}"#,
        ] {
            assert!(RuntimeConfig::parse(input, false).is_err());
        }
    }

    #[test]
    fn private_file_reader_rejects_shared_permissions_and_links() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("key");
        std::fs::write(&path, [1; 32]).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(read_private_file(&path, 32).unwrap().len(), 32);
        let link = root.path().join("link");
        symlink(&path, &link).unwrap();
        assert!(read_private_file(&link, 32).is_err());
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(read_private_file(&path, 32).is_err());
    }

    #[test]
    fn legacy_file_requires_exactly_one_policy_source_and_inline_cannot_enable_it() {
        let mut value = serde_json::json!({
            "license_root":"/runtime/license",
            "package_trust":{"key_id":"package","public_key_file":"/runtime/package.pub"},
            "license_trust":{"key_id":"license","public_key_file":"/runtime/license.pub"},
            "tpm":{"device_key_handle":"0x81012003","policy_authority_key_id":"policy","policy_authority_public_file":"/runtime/policy.tpmt-public"},
            "process_memory_margin_bytes":0
        });
        let bytes = serde_json::to_vec(&value).unwrap();
        RuntimeConfig::parse(&bytes, true)
            .unwrap()
            .validate()
            .unwrap();
        assert!(
            RuntimeConfig::parse(&bytes, false)
                .unwrap()
                .validate()
                .is_err()
        );
        value["tpm"]["policy_authority_key_handle"] = "0x81012004".into();
        assert!(
            RuntimeConfig::parse(&serde_json::to_vec(&value).unwrap(), true)
                .unwrap()
                .validate()
                .is_err()
        );
        assert_eq!(parse_handle("0x81012003"), Some(0x81012003));
        for handle in ["81012003", "0x1", "0x80000001", "0x01000000"] {
            assert_eq!(parse_handle(handle), None);
        }
    }

    #[test]
    fn disabled_layer_is_rejected_before_opening_package_or_key() {
        assert!(matches!(
            PreparedRuntime::prepare(Path::new("/does-not-exist"), "test", "{}"),
            Err(ProtectionError::ProtectionLayerDisabled(
                "package_verification"
            ))
        ));
        let mut value = file_config();
        value["layers"]["secure_materialization"] = false.into();
        assert!(matches!(
            PreparedRuntime::prepare(Path::new("/does-not-exist"), "test", &value.to_string()),
            Err(ProtectionError::ProtectionLayerDisabled(
                "secure_materialization"
            ))
        ));
    }

    #[test]
    #[ignore = "requires isolated no-swap container and /run/protected-test-models tmpfs mount"]
    fn file_provider_lifecycle_in_secure_container() {
        use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
        use ring::signature::{Ed25519KeyPair, KeyPair};
        use sha2::Digest;
        let directory = tempfile::tempdir().unwrap();
        let package = directory.path().join("package");
        std::fs::create_dir(&package).unwrap();
        std::fs::create_dir(package.join("weights")).unwrap();
        let key_path = directory.path().join("model-dek.bin");
        let public_path = directory.path().join("package.pub");
        let dek = [0x42; 32];
        let plaintext = b"synthetic weights; no customer model";
        let mut manifest =
            crate::parse_manifest(&crate::format::tests::valid_manifest_json()).unwrap();
        manifest.format_version = 2;
        manifest.runtime.protection_profile = Some("encrypted-file".into());
        manifest.public_files.clear();
        let mut ciphertext = Vec::new();
        let encrypted = crate::encrypt_records(
            plaintext.as_slice(),
            &mut ciphertext,
            &dek,
            &manifest.artifact_id_bytes().unwrap(),
            &manifest.nonce_prefix_bytes().unwrap(),
            1,
            0,
            1024,
        )
        .unwrap();
        let file = &mut manifest.protected_files[0];
        file.container_size = encrypted.container_size;
        file.container_sha256 = encrypted
            .container_sha256
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        file.plaintext_size = encrypted.plaintext_size;
        file.plaintext_sha256 = encrypted
            .plaintext_sha256
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        file.record_count = encrypted.record_count;
        std::fs::write(package.join(&file.container_path), ciphertext).unwrap();
        let signer = Ed25519KeyPair::from_seed_unchecked(&[7; 32]).unwrap();
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let signature = serde_json::to_vec(&crate::SignatureEnvelope {
            algorithm: "Ed25519".into(),
            key_id: "package-v1".into(),
            signature: BASE64.encode(
                signer
                    .sign(&crate::manifest_signature_payload(&bytes))
                    .as_ref(),
            ),
        })
        .unwrap();
        std::fs::write(package.join("model.protection.json"), &bytes).unwrap();
        std::fs::write(package.join("model.protection.sig"), &signature).unwrap();
        std::fs::write(&key_path, dek).unwrap();
        std::fs::write(&public_path, signer.public_key().as_ref()).unwrap();
        for path in [&key_path, &public_path] {
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let mut config = file_config();
        config["key_provider"]["key_file"] = key_path.to_str().unwrap().into();
        config["package_trust"]["public_key_file"] = public_path.to_str().unwrap().into();
        let input = config.to_string();
        let mut runtime = PreparedRuntime::prepare(&package, "protected-test", &input).unwrap();
        let path = runtime.model_path().to_path_buf();
        runtime.materialize(&CancellationToken::default()).unwrap();
        assert_eq!(
            std::fs::read(path.join("model.safetensors")).unwrap(),
            plaintext
        );
        assert!(matches!(
            runtime.materialize(&CancellationToken::default()),
            Err(ProtectionError::SessionConflict)
        ));
        runtime.cleanup().unwrap();
        assert!(!path.exists());
        std::fs::write(&key_path, [0x43; 32]).unwrap();
        let mut wrong_key = PreparedRuntime::prepare(&package, "protected-test", &input).unwrap();
        let wrong_path = wrong_key.model_path().to_path_buf();
        assert!(
            wrong_key
                .materialize(&CancellationToken::default())
                .is_err()
        );
        wrong_key.cleanup().unwrap();
        assert!(!wrong_path.exists());
        let canceled = CancellationToken::default();
        canceled.cancel();
        std::fs::write(&key_path, dek).unwrap();
        let mut runtime = PreparedRuntime::prepare(&package, "protected-test", &input).unwrap();
        let path = runtime.model_path().to_path_buf();
        assert!(matches!(
            runtime.materialize(&canceled),
            Err(ProtectionError::MaterializationCancelled)
        ));
        runtime.cleanup().unwrap();
        assert!(!path.exists());
        // Exercise the license-enabled software profile through the same runtime.
        manifest.runtime.protection_profile = Some("encrypted-file-license".into());
        let licensed_bytes = serde_json::to_vec(&manifest).unwrap();
        let licensed_signature = serde_json::to_vec(&crate::SignatureEnvelope {
            algorithm: "Ed25519".into(),
            key_id: "package-v1".into(),
            signature: BASE64.encode(
                signer
                    .sign(&crate::manifest_signature_payload(&licensed_bytes))
                    .as_ref(),
            ),
        })
        .unwrap();
        std::fs::write(package.join("model.protection.json"), &licensed_bytes).unwrap();
        std::fs::write(package.join("model.protection.sig"), &licensed_signature).unwrap();
        assert!(matches!(
            PreparedRuntime::prepare(&package, "protected-test", &input),
            Err(ProtectionError::LicenseBindingMismatch)
        ));
        let license_key = Ed25519KeyPair::from_seed_unchecked(&[8; 32]).unwrap();
        let license_public = directory.path().join("license.pub");
        std::fs::write(&license_public, license_key.public_key().as_ref()).unwrap();
        std::fs::set_permissions(&license_public, std::fs::Permissions::from_mode(0o600)).unwrap();
        let license_root = directory.path().join("license");
        std::fs::create_dir(&license_root).unwrap();
        let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let license = crate::FileLicense {
            format: crate::FILE_LICENSE_FORMAT.into(),
            format_version: 1,
            license_id: "test".into(),
            artifact_id: manifest.artifact_id.clone(),
            manifest_sha256: hex(&crate::manifest_digest(&licensed_bytes)),
            customer_scope_id: manifest.customer_scope_id.clone(),
            model_id: manifest.model.model_id.clone(),
            model_version: manifest.model.model_version.clone(),
            key_sha256: hex(&sha2::Sha256::digest(dek)),
            entitlement: crate::Entitlement {
                mode: "offline-perpetual".into(),
                generation: 1,
            },
        };
        let license_bytes = serde_json::to_vec(&license).unwrap();
        let license_signature = serde_json::to_vec(&crate::SignatureEnvelope {
            algorithm: "Ed25519".into(),
            key_id: "license-v1".into(),
            signature: BASE64.encode(
                license_key
                    .sign(&crate::file_license_signature_payload(&license_bytes))
                    .as_ref(),
            ),
        })
        .unwrap();
        for (name, content) in [
            ("model.protection.license.json", license_bytes),
            ("model.protection.license.sig", license_signature),
        ] {
            let file = license_root.join(name);
            std::fs::write(&file, content).unwrap();
            std::fs::set_permissions(file, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        config["profile"] = "encrypted-file-license".into();
        config["layers"]["license_verification"] = true.into();
        config["license_root"] = license_root.to_str().unwrap().into();
        config["license_trust"] =
            serde_json::json!({"key_id":"license-v1","public_key_file":license_public});
        let mut runtime =
            PreparedRuntime::prepare(&package, "protected-test", &config.to_string()).unwrap();
        let path = runtime.model_path().to_path_buf();
        runtime.materialize(&CancellationToken::default()).unwrap();
        assert_eq!(
            std::fs::read(path.join("model.safetensors")).unwrap(),
            plaintext
        );
        runtime.cleanup().unwrap();
        assert!(!path.exists());
        std::fs::write(package.join("model.protection.json"), &bytes).unwrap();
        std::fs::write(package.join("model.protection.sig"), &signature).unwrap();
        config["profile"] = "encrypted-tpm".into();
        config["layers"]["tpm_binding"] = true.into();
        assert!(PreparedRuntime::prepare(&package, "protected-test", &config.to_string()).is_err());
        let mut corrupted = bytes;
        corrupted.push(b' ');
        std::fs::write(package.join("model.protection.json"), corrupted).unwrap();
        assert!(matches!(
            PreparedRuntime::prepare(&package, "protected-test", &input),
            Err(ProtectionError::ManifestSignatureInvalid)
        ));
    }
}
