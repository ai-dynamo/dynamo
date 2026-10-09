// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit software-key issuance. Never exports a TPM-profile package key.

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use dynamo_model_protection::{
    Entitlement, FILE_LICENSE_FORMAT, FileLicense, MAX_ISSUER_RECORD_BYTES, SignatureEnvelope,
    file_license_signature_payload, verify_file_license, verify_issuer_record, verify_manifest,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use super::secure_file::read_bounded;
use super::software_keys::{
    lock_process_memory, read_kek, read_passphrase, read_signing_key, sign_ed25519, unwrap_dek,
};
use super::{
    IssueError, Result, atomic_file, lower_hex, required, required_path, valid_identifier,
};

pub fn run(command: &str, arguments: &[String]) -> Result<()> {
    let mut args = BTreeMap::new();
    let mut pairs = arguments.chunks_exact(2);
    for pair in pairs.by_ref() {
        let allowed = matches!(
            pair[0].as_str(),
            "--package"
                | "--issuer-record"
                | "--output"
                | "--package-key-id"
                | "--package-public-key"
                | "--kek-key-file"
                | "--kek-key-id"
                | "--kek-key-version"
        ) || command == "issue-file-license"
            && matches!(
                pair[0].as_str(),
                "--license-signing-key"
                    | "--license-key-passphrase-file"
                    | "--license-key-id"
                    | "--license-id"
                    | "--generation"
            );
        if !allowed
            || pair[1].is_empty()
            || pair[1].starts_with("--")
            || args.insert(pair[0].clone(), pair[1].clone()).is_some()
        {
            return Err(IssueError("ISSUER_CONFIG_INVALID"));
        }
    }
    if !pairs.remainder().is_empty() {
        return Err(IssueError("ISSUER_CONFIG_INVALID"));
    }
    lock_process_memory().map_err(IssueError)?;
    let package = required_path(&args, "--package")?;
    let key_id = required(&args, "--package-key-id")?;
    if !valid_identifier(key_id) {
        return Err(IssueError("ISSUER_CONFIG_INVALID"));
    }
    let key: [u8; 32] = read_bounded(&required_path(&args, "--package-public-key")?, 32, true)
        .map_err(|_| IssueError("ISSUER_KEY_INVALID"))?
        .try_into()
        .map_err(|_| IssueError("ISSUER_KEY_INVALID"))?;
    let verified = verify_manifest(
        &read_bounded(
            &package.join("model.protection.json"),
            4 * 1024 * 1024,
            true,
        )
        .map_err(|_| IssueError("PACKAGE_VERIFICATION_FAILED"))?,
        &read_bounded(&package.join("model.protection.sig"), 4096, true)
            .map_err(|_| IssueError("PACKAGE_VERIFICATION_FAILED"))?,
        key_id,
        &key,
    )
    .map_err(|_| IssueError("PACKAGE_VERIFICATION_FAILED"))?;
    let manifest = verified.manifest();
    match (command, manifest.runtime.protection_profile.as_deref()) {
        ("export-file-key", Some("encrypted-file" | "encrypted-file-license"))
        | ("issue-file-license", Some("encrypted-file-license")) => {}
        _ => return Err(IssueError("SOFTWARE_PROFILE_REQUIRED")),
    }
    let record = verify_issuer_record(
        &read_bounded(
            &required_path(&args, "--issuer-record")?,
            MAX_ISSUER_RECORD_BYTES as u64,
            true,
        )
        .map_err(|_| IssueError("ISSUER_RECORD_INVALID"))?,
        key_id,
        &key,
    )
    .map_err(|_| IssueError("ISSUER_RECORD_INVALID"))?;
    if record.artifact_id != manifest.artifact_id
        || record.customer_scope_id != manifest.customer_scope_id
        || record.model_id != manifest.model.model_id
        || record.model_version != manifest.model.model_version
        || record.manifest_sha256 != lower_hex(verified.digest())
        || record.kek_key_id != required(&args, "--kek-key-id")?
        || record.kek_key_version != required(&args, "--kek-key-version")?
    {
        return Err(IssueError("ISSUER_RECORD_INVALID"));
    }
    let kek = read_kek(&required_path(&args, "--kek-key-file")?).map_err(IssueError)?;
    let dek = unwrap_dek(
        &kek,
        &BASE64
            .decode(&record.wrapped_dek)
            .map_err(|_| IssueError("ISSUER_RECORD_INVALID"))?,
    )
    .map_err(IssueError)?;
    let output = required_path(&args, "--output")?;
    if command == "export-file-key" {
        atomic_file::publish_new(&output, dek.as_ref())
            .map_err(|_| IssueError("ISSUER_OUTPUT_INVALID"))?;
    } else {
        let license_id = required(&args, "--license-id")?;
        let license_key_id = required(&args, "--license-key-id")?;
        let generation = required(&args, "--generation")?
            .parse::<u64>()
            .map_err(|_| IssueError("ISSUER_CONFIG_INVALID"))?;
        if !valid_identifier(license_id)
            || !valid_identifier(license_key_id)
            || generation == 0
            || license_key_id == key_id
        {
            return Err(IssueError("ISSUER_CONFIG_INVALID"));
        }
        let pass = read_passphrase(&required_path(&args, "--license-key-passphrase-file")?)
            .map_err(IssueError)?;
        let signer = read_signing_key(&required_path(&args, "--license-signing-key")?, &pass)
            .map_err(IssueError)?;
        if signer.verifying_key().as_bytes() == &key {
            return Err(IssueError("ISSUER_CONFIG_INVALID"));
        }
        let license = FileLicense {
            format: FILE_LICENSE_FORMAT.into(),
            format_version: 1,
            license_id: license_id.into(),
            artifact_id: manifest.artifact_id.clone(),
            manifest_sha256: lower_hex(verified.digest()),
            customer_scope_id: manifest.customer_scope_id.clone(),
            model_id: manifest.model.model_id.clone(),
            model_version: manifest.model.model_version.clone(),
            key_sha256: lower_hex(&Sha256::digest(dek.as_ref())),
            entitlement: Entitlement {
                mode: "offline-perpetual".into(),
                generation,
            },
        };
        let bytes =
            serde_json::to_vec(&license).map_err(|_| IssueError("ISSUER_OUTPUT_INVALID"))?;
        let signature = serde_json::to_vec(&SignatureEnvelope {
            algorithm: "Ed25519".into(),
            key_id: license_key_id.into(),
            signature: BASE64.encode(sign_ed25519(
                &signer,
                &file_license_signature_payload(&bytes),
            )),
        })
        .map_err(|_| IssueError("ISSUER_OUTPUT_INVALID"))?;
        verify_file_license(
            &bytes,
            &signature,
            license_key_id,
            signer.verifying_key().as_bytes(),
            &verified,
        )
        .map_err(|_| IssueError("ISSUER_OUTPUT_INVALID"))?;
        super::write_license_bundle(&output, &bytes, &signature)?;
    }
    println!("model_protection event=software_issue_complete");
    Ok(())
}
