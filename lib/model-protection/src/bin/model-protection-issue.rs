// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use dynamo_model_protection::{
    Entitlement, License, MAX_CERTIFIED_DEVICE_BYTES, MAX_ISSUER_RECORD_BYTES, TPM_OAEP_LABEL,
    TPM_POLICY_REF_HEX, TPM_POLICY_SIGNATURE, TPM_PROFILE, TpmRecipient,
    approved_rsa_decrypt_policy, license_signature_payload, rsa_decrypt_cp_hash,
    verify_certified_device, verify_issuer_record, verify_manifest,
};
use p256::pkcs8::DecodePublicKey;
use rustix::fs::{CWD, Mode, OFlags, RenameFlags, ResolveFlags, openat, openat2, renameat_with};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[path = "support/atomic_file.rs"]
mod atomic_file;

#[path = "support/file_issue.rs"]
mod file_issue;

#[path = "support/secure_file.rs"]
mod secure_file;
#[path = "support/software_keys.rs"]
#[allow(dead_code)]
mod software_keys;
use secure_file::read_bounded as read_file_bounded;
use software_keys::{
    lock_process_memory, read_kek, read_passphrase, read_policy_key, read_signing_key,
    sign_ed25519, sign_p256_digest, unwrap_dek, wrap_to_tpm,
};

const MAX_CONTROL_BYTES: u64 = 4 * 1024 * 1024;
const USAGE: &str = "Usage: model-protection-issue --package PATH --issuer-record PATH --certified-device PATH --certified-device-signature PATH --output PATH --package-key-id ID --package-public-key PATH --enrollment-key-id ID --enrollment-public-key PATH --license-signing-key PATH --license-key-passphrase-file PATH --license-key-id ID --policy-signing-key PATH --policy-key-passphrase-file PATH --policy-key-id ID --kek-key-file PATH --kek-key-id ID --kek-key-version VERSION --license-id ID --generation NUMBER [--registry PATH]\nProduction admission requires build feature enrollment-authority and --registry.\nPackager-only pilot issuance requires explicit --allow-development-certification true; not production enrollment.";

type Result<T> = std::result::Result<T, IssueError>;

#[derive(Debug)]
struct IssueError(&'static str);

impl std::fmt::Display for IssueError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for IssueError {}

#[derive(Serialize)]
struct LicenseEnvelope {
    algorithm: &'static str,
    key_id: String,
    signature: String,
}

fn main() {
    if env::args().len() == 2 && env::args().nth(1).as_deref() == Some("--help") {
        println!("{USAGE}");
        println!(
            "Software profiles: export-file-key or issue-file-license --package PATH --issuer-record PATH --package-key-id ID --package-public-key PATH --kek-key-file PATH --kek-key-id ID --kek-key-version VERSION --output PATH"
        );
        println!(
            "issue-file-license additionally requires --license-signing-key PATH --license-key-passphrase-file PATH --license-key-id ID --license-id ID --generation NUMBER. Software profiles are not device-bound."
        );
        println!(
            "Public-key export (issuer host): model-protection-issue export-policy-public --policy-public-key PATH_TO_P256_PUBLIC_PEM --output PATH_TO_TPMT_PUBLIC"
        );
        return;
    }
    if let Err(error) = run() {
        eprintln!("model_protection event=license_issue_failed code={error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let command: Vec<String> = env::args().skip(1).collect();
    if let Some(name @ ("export-file-key" | "issue-file-license")) =
        command.first().map(String::as_str)
    {
        return file_issue::run(name, &command[1..]);
    }
    if command
        .first()
        .is_some_and(|value| value == "export-policy-public")
    {
        return export_policy_public(&command[1..]);
    }
    let args = parse_args()?;
    enforce_admission_mode(&args)?;
    lock_process_memory().map_err(IssueError)?;
    #[cfg(feature = "enrollment-authority")]
    let mut registry = dynamo_model_protection::enrollment::registry::Registry::open(
        &required_path(&args, "--registry")?,
        false,
    )
    .map_err(|_| IssueError("ISSUER_REGISTRY_REQUIRED"))?;
    #[cfg(not(feature = "enrollment-authority"))]
    if args.contains_key("--registry") {
        return Err(IssueError("ISSUER_REGISTRY_UNSUPPORTED"));
    }
    let package = required_path(&args, "--package")?;
    let issuer_record_path = required_path(&args, "--issuer-record")?;
    let certified_device_path = required_path(&args, "--certified-device")?;
    let certified_device_signature_path = required_path(&args, "--certified-device-signature")?;
    let output = required_path(&args, "--output")?;
    if [
        &package,
        &issuer_record_path,
        &certified_device_path,
        &output,
    ]
    .iter()
    .any(|path| !path.is_absolute())
        || !certified_device_signature_path.is_absolute()
    {
        return Err(IssueError("ISSUER_CONFIG_INVALID"));
    }

    let package_key_id = required(&args, "--package-key-id")?;
    let enrollment_key_id = required(&args, "--enrollment-key-id")?;
    let license_key_id = required(&args, "--license-key-id")?;
    let policy_key_id = required(&args, "--policy-key-id")?;
    let kek_key_id = required(&args, "--kek-key-id")?;
    let kek_key_version = required(&args, "--kek-key-version")?;
    let key_ids = [
        package_key_id,
        enrollment_key_id,
        license_key_id,
        policy_key_id,
        kek_key_id,
    ];
    for identifier in key_ids
        .into_iter()
        .chain([required(&args, "--license-id")?, kek_key_version])
    {
        if !valid_identifier(identifier) {
            return Err(IssueError("ISSUER_CONFIG_INVALID"));
        }
    }
    if BTreeSet::from(key_ids).len() != key_ids.len() {
        return Err(IssueError("ISSUER_CONFIG_INVALID"));
    }
    let package_key = read_exact_key(&required_path(&args, "--package-public-key")?)?;
    let enrollment_key = read_exact_key(&required_path(&args, "--enrollment-public-key")?)?;
    if package_key == enrollment_key {
        return Err(IssueError("ISSUER_CONFIG_INVALID"));
    }
    let manifest_bytes = read_bounded(&package.join("model.protection.json"), MAX_CONTROL_BYTES)?;
    let manifest_signature =
        read_bounded(&package.join("model.protection.sig"), MAX_CONTROL_BYTES)?;
    let verified = verify_manifest(
        &manifest_bytes,
        &manifest_signature,
        package_key_id,
        &package_key,
    )
    .map_err(|_| IssueError("PACKAGE_VERIFICATION_FAILED"))?;

    let certified_bytes = read_bounded(&certified_device_path, MAX_CERTIFIED_DEVICE_BYTES as u64)?;
    let certified_signature = read_bounded(
        &certified_device_signature_path,
        MAX_CERTIFIED_DEVICE_BYTES as u64,
    )?;
    let manifest = verified.manifest();
    if !matches!(
        manifest.runtime.protection_profile.as_deref(),
        None | Some("encrypted-tpm")
    ) {
        return Err(IssueError("TPM_PROFILE_REQUIRED"));
    }
    let certified = verify_certified_device(
        &certified_bytes,
        &certified_signature,
        enrollment_key_id,
        &enrollment_key,
        &manifest.customer_scope_id,
        &manifest.artifact_id,
    )
    .map_err(|_| IssueError("CERTIFIED_DEVICE_INVALID"))?;
    let expected_policy_authority_name = certified.policy_authority_name();
    let device_name = certified.public().name();
    let device_public_digest = certified.public().digest();
    let modulus = certified.public().modulus();
    let generation = required(&args, "--generation")?
        .parse::<u64>()
        .map_err(|_| IssueError("ISSUER_CONFIG_INVALID"))?;
    if generation == 0 {
        return Err(IssueError("ISSUER_CONFIG_INVALID"));
    }

    let record_bytes = read_bounded(&issuer_record_path, MAX_ISSUER_RECORD_BYTES as u64)?;
    let record = verify_issuer_record(&record_bytes, package_key_id, &package_key)
        .map_err(|_| IssueError("ISSUER_RECORD_INVALID"))?;
    if record.artifact_id != manifest.artifact_id
        || record.customer_scope_id != manifest.customer_scope_id
        || record.model_id != manifest.model.model_id
        || record.model_version != manifest.model.model_version
        || record.manifest_sha256 != lower_hex(verified.digest())
        || record.kek_key_id != kek_key_id
        || record.kek_key_version != kek_key_version
    {
        return Err(IssueError("ISSUER_RECORD_INVALID"));
    }
    let wrapped_issuer_dek = BASE64
        .decode(record.wrapped_dek.as_bytes())
        .map_err(|_| IssueError("ISSUER_RECORD_INVALID"))?;
    if wrapped_issuer_dek.len() != 40 {
        return Err(IssueError("ISSUER_RECORD_INVALID"));
    }

    let license_key_passphrase =
        read_passphrase(&required_path(&args, "--license-key-passphrase-file")?)
            .map_err(IssueError)?;
    let policy_key_passphrase =
        read_passphrase(&required_path(&args, "--policy-key-passphrase-file")?)
            .map_err(IssueError)?;
    let license_signer = read_signing_key(
        &required_path(&args, "--license-signing-key")?,
        &license_key_passphrase,
    )
    .map_err(IssueError)?;
    let policy_signer = read_policy_key(
        &required_path(&args, "--policy-signing-key")?,
        &policy_key_passphrase,
    )
    .map_err(IssueError)?;
    let kek = read_kek(&required_path(&args, "--kek-key-file")?).map_err(IssueError)?;
    let policy_point = policy_signer.verifying_key().to_encoded_point(false);
    if policy_authority_name(
        b"\x06\x08\x2a\x86\x48\xce\x3d\x03\x01\x07",
        policy_point.as_bytes(),
    )
    .is_none_or(|name| &name != expected_policy_authority_name)
    {
        return Err(IssueError("ISSUER_KEY_INVALID"));
    }
    #[cfg(feature = "enrollment-authority")]
    let binding = dynamo_model_protection::enrollment::format::Binding {
        customer_scope_id: manifest.customer_scope_id.clone(),
        artifact_id: manifest.artifact_id.clone(),
        manifest_sha256: lower_hex(verified.digest()),
        policy_authority_name: lower_hex(expected_policy_authority_name),
    };
    #[cfg(feature = "enrollment-authority")]
    let intent_digest: [u8; 32] = Sha256::digest(
        serde_json::to_vec(&(
            "model-protection-issuance-intent-v1",
            &binding,
            Sha256::digest(&record_bytes).as_slice(),
            key_ids,
            kek_key_version,
            package_key,
            enrollment_key,
            license_signer.verifying_key().as_bytes(),
            policy_point.as_bytes(),
        ))
        .map_err(|_| IssueError("ISSUER_CONFIG_INVALID"))?,
    )
    .into();
    #[cfg(feature = "enrollment-authority")]
    let intent = dynamo_model_protection::enrollment::registry::IssuanceIntent {
        certification_id: certified.certification_id(),
        certificate_bytes: &certified_bytes,
        binding: &binding,
        license_id: required(&args, "--license-id")?,
        generation,
        digest: &intent_digest,
    };
    #[cfg(feature = "enrollment-authority")]
    if let Some(bundle) = registry
        .begin_issuance(&intent)
        .map_err(|_| IssueError("ISSUER_ADMISSION_DENIED"))?
    {
        return write_license_bundle(&output, &bundle.license, &bundle.signature);
    }
    #[cfg(not(feature = "enrollment-authority"))]
    if output.exists() {
        return Err(IssueError("ISSUER_CONFIG_INVALID"));
    }
    let dek = unwrap_dek(&kek, &wrapped_issuer_dek).map_err(IssueError)?;
    let wrapped_dek = wrap_to_tpm(modulus, &dek, TPM_OAEP_LABEL).map_err(IssueError)?;

    let operation = (|| {
        let cp_hash = rsa_decrypt_cp_hash(device_name, &wrapped_dek);
        let approved_policy = approved_rsa_decrypt_policy(&cp_hash);
        let policy_ref = decode_hex::<32>(TPM_POLICY_REF_HEX)?;
        let mut policy_digest = Sha256::new();
        policy_digest.update(approved_policy);
        policy_digest.update(policy_ref);
        let policy_digest: [u8; 32] = policy_digest.finalize().into();
        let policy_signature =
            sign_p256_digest(&policy_signer, &policy_digest).map_err(IssueError)?;

        let license = License {
            format: "model-protection-license".to_string(),
            format_version: 1,
            license_id: required(&args, "--license-id")?.to_string(),
            customer_scope_id: manifest.customer_scope_id.clone(),
            artifact_id: manifest.artifact_id.clone(),
            manifest_sha256: lower_hex(verified.digest()),
            model_id: manifest.model.model_id.clone(),
            model_version: manifest.model.model_version.clone(),
            entitlement: Entitlement {
                mode: "offline-perpetual".to_string(),
                generation,
            },
            recipient: TpmRecipient {
                kind: "tpm2".to_string(),
                profile: TPM_PROFILE.to_string(),
                device_key_name: lower_hex(device_name),
                device_public_key_sha256: lower_hex(device_public_digest),
                command_parameters_hash: lower_hex(&cp_hash),
                approved_policy_digest: lower_hex(&approved_policy),
                policy_ref: TPM_POLICY_REF_HEX.to_string(),
                policy_authority_key_id: policy_key_id.to_string(),
                policy_signature_algorithm: TPM_POLICY_SIGNATURE.to_string(),
                policy_signature: BASE64.encode(policy_signature),
            },
            wrapped_dek: BASE64.encode(wrapped_dek),
        };
        let license_bytes =
            serde_json::to_vec_pretty(&license).map_err(|_| IssueError("LICENSE_INVALID"))?;
        let signature = sign_ed25519(&license_signer, &license_signature_payload(&license_bytes));
        Ok((
            license_bytes,
            LicenseEnvelope {
                algorithm: "Ed25519",
                key_id: license_key_id.to_string(),
                signature: BASE64.encode(signature),
            },
        ))
    })();

    let (license, signature) = operation?;
    let signature =
        serde_json::to_vec_pretty(&signature).map_err(|_| IssueError("LICENSE_INVALID"))?;
    #[cfg(feature = "enrollment-authority")]
    let dynamo_model_protection::enrollment::registry::IssuedBundle { license, signature } =
        registry
            .finalize_issuance(
                &intent,
                dynamo_model_protection::enrollment::registry::IssuedBundle { license, signature },
            )
            .map_err(|_| IssueError("ISSUER_ADMISSION_DENIED"))?;
    write_license_bundle(&output, &license, &signature)
}

fn enforce_admission_mode(args: &BTreeMap<String, String>) -> Result<()> {
    #[cfg(feature = "enrollment-authority")]
    if !args.contains_key("--registry") || args.contains_key("--allow-development-certification") {
        return Err(IssueError("ISSUER_REGISTRY_REQUIRED"));
    }
    #[cfg(not(feature = "enrollment-authority"))]
    if args.contains_key("--registry")
        || args
            .get("--allow-development-certification")
            .map(String::as_str)
            != Some("true")
    {
        return Err(IssueError("ISSUER_PRODUCTION_FEATURE_REQUIRED"));
    }
    Ok(())
}

fn policy_authority_name(parameters: &[u8], encoded_point: &[u8]) -> Option<[u8; 34]> {
    const P256_OID: &[u8] = b"\x06\x08\x2a\x86\x48\xce\x3d\x03\x01\x07";
    let point = match encoded_point {
        [0x04, 0x41, point @ ..] if point.len() == 65 => point,
        point if point.len() == 65 => point,
        _ => return None,
    };
    if parameters != P256_OID || point[0] != 0x04 {
        return None;
    }

    let public = dynamo_model_protection::tpm_policy_authority_public(point).ok()?;
    dynamo_model_protection::validate_tpm_policy_authority_public(&public).ok()
}

fn export_policy_public(args: &[String]) -> Result<()> {
    let mut options = BTreeMap::new();
    let mut pairs = args.chunks_exact(2);
    for pair in &mut pairs {
        if !matches!(pair[0].as_str(), "--policy-public-key" | "--output")
            || pair[1].is_empty()
            || options.insert(pair[0].clone(), pair[1].clone()).is_some()
        {
            return Err(IssueError("ISSUER_CONFIG_INVALID"));
        }
    }
    if !pairs.remainder().is_empty() {
        return Err(IssueError("ISSUER_CONFIG_INVALID"));
    }
    let input = read_bounded(&required_path(&options, "--policy-public-key")?, 4096)?;
    let pem = std::str::from_utf8(&input).map_err(|_| IssueError("INPUT_FILE_INVALID"))?;
    let key =
        p256::PublicKey::from_public_key_pem(pem).map_err(|_| IssueError("INPUT_FILE_INVALID"))?;
    use p256::elliptic_curve::sec1::ToEncodedPoint;
    let public = dynamo_model_protection::tpm_policy_authority_public(
        key.to_encoded_point(false).as_bytes(),
    )
    .map_err(|_| IssueError("INPUT_FILE_INVALID"))?;
    atomic_file::publish_new(&required_path(&options, "--output")?, &public)
        .map_err(|_| IssueError("POLICY_PUBLIC_EXPORT_FAILED"))?;
    println!(
        "policy_authority_name={}",
        lower_hex(
            &dynamo_model_protection::validate_tpm_policy_authority_public(&public)
                .map_err(|_| IssueError("INPUT_FILE_INVALID"))?
        )
    );
    Ok(())
}

fn write_license_bundle(output: &Path, license: &[u8], signature: &[u8]) -> Result<()> {
    // All writes/renames are relative to a pinned, owner-only directory descriptor.
    let parent = openat2(
        CWD,
        output.parent().ok_or(IssueError("LICENSE_IO_ERROR"))?,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
        Mode::empty(),
        ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
    )
    .map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
    validate_private_directory(&parent)?;
    let name = output.file_name().ok_or(IssueError("LICENSE_IO_ERROR"))?;
    match openat(
        &parent,
        name,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    ) {
        Ok(existing) => {
            verify_published(&existing, license, signature)?;
            File::from(parent)
                .sync_all()
                .map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
            return Ok(());
        }
        Err(rustix::io::Errno::NOENT) => {}
        Err(_) => return Err(IssueError("LICENSE_IO_ERROR")),
    }
    let temporary = format!(".license-partial-{}", uuid::Uuid::new_v4().simple());
    rustix::fs::mkdirat(&parent, &temporary, Mode::from_raw_mode(0o700))
        .map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
    let directory = openat(
        &parent,
        &temporary,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
    let result = (|| {
        for (name, bytes) in [
            ("model.protection.license.json", license),
            ("model.protection.license.sig", signature),
        ] {
            let fd = openat(
                &directory,
                name,
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::from_raw_mode(0o600),
            )
            .map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
            let mut file = File::from(fd);
            file.write_all(bytes)
                .and_then(|()| file.sync_all())
                .map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
        }
        File::from(
            directory
                .try_clone()
                .map_err(|_| IssueError("LICENSE_IO_ERROR"))?,
        )
        .sync_all()
        .map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
        match renameat_with(&parent, &temporary, &parent, name, RenameFlags::NOREPLACE) {
            Ok(()) => {}
            Err(rustix::io::Errno::EXIST) => {
                let existing = openat(
                    &parent,
                    name,
                    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
                verify_published(&existing, license, signature)?;
            }
            Err(_) => return Err(IssueError("LICENSE_IO_ERROR")),
        }
        File::from(
            parent
                .try_clone()
                .map_err(|_| IssueError("LICENSE_IO_ERROR"))?,
        )
        .sync_all()
        .map_err(|_| IssueError("LICENSE_IO_ERROR"))
    })();
    // Never remove the published bundle, even after a parent fsync failure.
    if openat(
        &parent,
        &temporary,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW,
        Mode::empty(),
    )
    .is_ok()
    {
        for name in [
            "model.protection.license.json",
            "model.protection.license.sig",
        ] {
            let _ = rustix::fs::unlinkat(&directory, name, rustix::fs::AtFlags::empty());
        }
        let _ = rustix::fs::unlinkat(&parent, &temporary, rustix::fs::AtFlags::REMOVEDIR);
    }
    result
}

fn validate_private_directory(fd: &rustix::fd::OwnedFd) -> Result<()> {
    let stat = rustix::fs::fstat(fd).map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
    if stat.st_uid != rustix::process::geteuid().as_raw() || stat.st_mode & 0o077 != 0 {
        return Err(IssueError("LICENSE_IO_ERROR"));
    }
    Ok(())
}

fn verify_published(
    directory: &rustix::fd::OwnedFd,
    license: &[u8],
    signature: &[u8],
) -> Result<()> {
    validate_private_directory(directory)?;
    for (name, expected) in [
        ("model.protection.license.json", license),
        ("model.protection.license.sig", signature),
    ] {
        let fd = openat(
            directory,
            name,
            OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
        let stat = rustix::fs::fstat(&fd).map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
        if rustix::fs::FileType::from_raw_mode(stat.st_mode) != rustix::fs::FileType::RegularFile
            || stat.st_nlink != 1
            || stat.st_uid != rustix::process::geteuid().as_raw()
            || stat.st_mode & 0o077 != 0
            || stat.st_size != expected.len() as i64
        {
            return Err(IssueError("LICENSE_IO_ERROR"));
        }
        let mut bytes = Vec::new();
        File::from(fd)
            .take(expected.len() as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
        if bytes != expected {
            return Err(IssueError("LICENSE_IO_ERROR"));
        }
    }
    File::from(
        directory
            .try_clone()
            .map_err(|_| IssueError("LICENSE_IO_ERROR"))?,
    )
    .sync_all()
    .map_err(|_| IssueError("LICENSE_IO_ERROR"))
}

fn parse_args() -> Result<BTreeMap<String, String>> {
    let mut parsed = BTreeMap::new();
    let mut args = env::args().skip(1);
    while let Some(name) = args.next() {
        if !name.starts_with("--") || !allowed_argument(&name) || parsed.contains_key(&name) {
            return Err(IssueError("ISSUER_CONFIG_INVALID"));
        }
        let value = args.next().ok_or(IssueError("ISSUER_CONFIG_INVALID"))?;
        if value.is_empty() || value.starts_with("--") {
            return Err(IssueError("ISSUER_CONFIG_INVALID"));
        }
        parsed.insert(name, value);
    }
    Ok(parsed)
}

fn allowed_argument(name: &str) -> bool {
    matches!(
        name,
        "--package"
            | "--issuer-record"
            | "--certified-device"
            | "--certified-device-signature"
            | "--output"
            | "--package-key-id"
            | "--package-public-key"
            | "--enrollment-key-id"
            | "--enrollment-public-key"
            | "--license-signing-key"
            | "--license-key-passphrase-file"
            | "--license-key-id"
            | "--policy-signing-key"
            | "--policy-key-passphrase-file"
            | "--policy-key-id"
            | "--kek-key-file"
            | "--kek-key-id"
            | "--kek-key-version"
            | "--license-id"
            | "--generation"
            | "--registry"
            | "--allow-development-certification"
    )
}

fn required<'a>(args: &'a BTreeMap<String, String>, name: &str) -> Result<&'a str> {
    args.get(name)
        .map(String::as_str)
        .ok_or(IssueError("ISSUER_CONFIG_INVALID"))
}

fn required_path(args: &BTreeMap<String, String>, name: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(required(args, name)?))
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 128 && !value.chars().any(char::is_control)
}

fn decode_hex<const N: usize>(value: &str) -> Result<[u8; N]> {
    if value.len() != N * 2
        || value
            .bytes()
            .any(|byte| !matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(IssueError("CERTIFIED_DEVICE_INVALID"));
    }
    let mut decoded = [0_u8; N];
    for (index, output) in decoded.iter_mut().enumerate() {
        *output = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| IssueError("CERTIFIED_DEVICE_INVALID"))?;
    }
    Ok(decoded)
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    read_file_bounded(path, limit, false).map_err(|_| IssueError("INPUT_FILE_INVALID"))
}

fn read_exact_key(path: &Path) -> Result<[u8; 32]> {
    read_bounded(path, 32)?
        .try_into()
        .map_err(|_| IssueError("INPUT_FILE_INVALID"))
}

fn lower_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exports_only_public_p256_template_without_overwrite() {
        use p256::pkcs8::EncodePublicKey;
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let signer = p256::ecdsa::SigningKey::from_slice(&[0x22; 32]).unwrap();
        let pem = signer
            .verifying_key()
            .to_public_key_pem(p256::pkcs8::LineEnding::LF)
            .unwrap();
        let input = directory.path().join("policy.pem");
        let output = directory.path().join("policy.tpmt-public");
        atomic_file::publish_new(&input, pem.as_bytes()).unwrap();
        let args = vec![
            "--policy-public-key".into(),
            input.display().to_string(),
            "--output".into(),
            output.display().to_string(),
        ];
        export_policy_public(&args).unwrap();
        let public = std::fs::read(&output).unwrap();
        assert_eq!(public.len(), 88);
        assert!(dynamo_model_protection::validate_tpm_policy_authority_public(&public).is_ok());
        assert!(export_policy_public(&args).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), public);
    }

    #[test]
    fn publication_retry_is_exact_private_and_never_overwrites() {
        use std::os::unix::fs::PermissionsExt;
        let parent = tempfile::tempdir().unwrap();
        std::fs::set_permissions(parent.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let output = parent.path().join("license");
        write_license_bundle(&output, b"license", b"signature").unwrap();
        write_license_bundle(&output, b"license", b"signature").unwrap();
        assert!(write_license_bundle(&output, b"changed", b"signature").is_err());
        assert_eq!(
            std::fs::read(output.join("model.protection.license.json")).unwrap(),
            b"license"
        );
        let link = parent.path().join("link");
        std::os::unix::fs::symlink(&output, &link).unwrap();
        assert!(write_license_bundle(&link, b"license", b"signature").is_err());
        std::fs::set_permissions(parent.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(write_license_bundle(&output, b"license", b"signature").is_err());
    }

    #[test]
    fn production_admission_cannot_silently_fall_back_to_development() {
        let mut args = BTreeMap::new();
        assert!(enforce_admission_mode(&args).is_err());
        args.insert("--allow-development-certification".into(), "true".into());
        #[cfg(not(feature = "enrollment-authority"))]
        assert!(enforce_admission_mode(&args).is_ok());
        #[cfg(feature = "enrollment-authority")]
        assert!(enforce_admission_mode(&args).is_err());
        args.insert("--registry".into(), "/issuer/registry.sqlite".into());
        assert!(enforce_admission_mode(&args).is_err());
        args.remove("--allow-development-certification");
        #[cfg(feature = "enrollment-authority")]
        assert!(enforce_admission_mode(&args).is_ok());
    }

    #[test]
    fn freezes_policy_authority_tpm_name_template() {
        let point = [vec![0x04], vec![0x11; 32], vec![0x22; 32]].concat();
        let name =
            policy_authority_name(b"\x06\x08\x2a\x86\x48\xce\x3d\x03\x01\x07", &point).unwrap();
        assert_eq!(
            lower_hex(&name),
            "000bf0d3c91a030c41ac66e9fd7554c478455ef9218a69a2dd743349640df8b3dd35"
        );
    }
}
