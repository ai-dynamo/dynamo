// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(all(target_os = "linux", feature = "packager"))]

use aes_kw::KekAes256;
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use dynamo_model_protection::{
    ISSUER_RECORD_FORMAT, ISSUER_RECORD_VERSION, IssuerDekRecord, SignatureEnvelope,
    SignedIssuerDekRecord, issuer_record_signature_payload, manifest_signature_payload,
    verify_file_license, verify_manifest,
};
use ed25519_dalek::{Signer, SigningKey, pkcs8::EncodePrivateKey};
use pkcs8::LineEnding;
use std::{fs, os::unix::fs::PermissionsExt, path::Path, process::Command};

fn private_file(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
#[ignore = "requires isolated issuer container with 256 MiB memlock; only synthetic keys"]
fn export_and_software_license_bind_exact_signed_package_without_overwrite() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::set_permissions(root, fs::Permissions::from_mode(0o700)).unwrap();
    let package = root.join("package");
    fs::create_dir(&package).unwrap();
    let signer = SigningKey::from_bytes(&[7; 32]);
    let license_signer = SigningKey::from_bytes(&[8; 32]);
    let dek = [3; 32];
    let kek = [4; 32];
    let manifest = serde_json::json!({"format":"secure-model-package","format_version":2,
        "artifact_id":"00112233445566778899aabbccddeeff","customer_scope_id":"test-scope",
        "model":{"model_id":"tiny","model_version":"1","framework":"safetensors"},
        "encryption":{"algorithm":"AES-256-GCM","nonce_prefix":"01020304","tag_bits":128,"record_plaintext_limit":1024},
        "protected_files":[{"file_id":1,"container_path":"weights/model.safetensors.protected","output_path":"model.safetensors",
            "container_size":84,"container_sha256":"aa".repeat(32),"plaintext_size":32,"plaintext_sha256":"bb".repeat(32),
            "record_count":1,"first_global_record_counter":0}],"public_files":[],
        "runtime":{"minimum_runtime_version":"1.5.0","required_load_format":"safetensors","protection_profile":"encrypted-file-license"}});
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let envelope = |payload: &[u8]| SignatureEnvelope {
        algorithm: "Ed25519".into(),
        key_id: "package".into(),
        signature: BASE64.encode(signer.sign(payload).to_bytes()),
    };
    let signature = serde_json::to_vec(&envelope(&manifest_signature_payload(&bytes))).unwrap();
    private_file(&package.join("model.protection.json"), &bytes);
    private_file(&package.join("model.protection.sig"), &signature);
    private_file(&root.join("package.pub"), signer.verifying_key().as_bytes());
    private_file(&root.join("issuer-kek.bin"), &kek);
    let mut wrapped = [0; 40];
    KekAes256::from(kek)
        .wrap_with_padding(&dek, &mut wrapped)
        .unwrap();
    let record = IssuerDekRecord {
        format: ISSUER_RECORD_FORMAT.into(),
        format_version: ISSUER_RECORD_VERSION,
        artifact_id: manifest["artifact_id"].as_str().unwrap().into(),
        customer_scope_id: "test-scope".into(),
        model_id: "tiny".into(),
        model_version: "1".into(),
        manifest_sha256: hex(&dynamo_model_protection::manifest_digest(&bytes)),
        kek_key_id: "kek".into(),
        kek_key_version: "1".into(),
        wrapped_dek: BASE64.encode(wrapped),
    };
    let payload = serde_json::to_vec(&record).unwrap();
    private_file(
        &root.join("issuer-record.json"),
        &serde_json::to_vec(&SignedIssuerDekRecord {
            payload: BASE64.encode(&payload),
            signature: envelope(&issuer_record_signature_payload(&payload)),
        })
        .unwrap(),
    );
    let run = |command: &str, output: &str, extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_model-protection-issue"))
            .arg(command)
            .arg("--package")
            .arg(&package)
            .arg("--issuer-record")
            .arg(root.join("issuer-record.json"))
            .arg("--package-key-id")
            .arg("package")
            .arg("--package-public-key")
            .arg(root.join("package.pub"))
            .arg("--kek-key-id")
            .arg("kek")
            .arg("--kek-key-version")
            .arg("1")
            .arg("--kek-key-file")
            .arg(root.join("issuer-kek.bin"))
            .arg("--output")
            .arg(root.join(output))
            .args(extra)
            .current_dir(root)
            .output()
            .unwrap()
    };
    let result = run("export-file-key", "model-dek.bin", &[]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(fs::read(root.join("model-dek.bin")).unwrap(), dek);
    assert_eq!(
        fs::metadata(root.join("model-dek.bin"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert!(
        !run("export-file-key", "model-dek.bin", &[])
            .status
            .success()
    );
    let pem = license_signer
        .to_pkcs8_encrypted_pem(rsa::rand_core::OsRng, b"test-only", LineEnding::LF)
        .unwrap();
    private_file(&root.join("license.pk8"), pem.as_bytes());
    private_file(&root.join("license.pass"), b"test-only");
    let result = run(
        "issue-file-license",
        "license",
        &[
            "--license-signing-key",
            root.join("license.pk8").to_str().unwrap(),
            "--license-key-passphrase-file",
            root.join("license.pass").to_str().unwrap(),
            "--license-key-id",
            "license",
            "--license-id",
            "synthetic",
            "--generation",
            "1",
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let verified = verify_manifest(
        &bytes,
        &signature,
        "package",
        signer.verifying_key().as_bytes(),
    )
    .unwrap();
    let license = verify_file_license(
        &fs::read(root.join("license/model.protection.license.json")).unwrap(),
        &fs::read(root.join("license/model.protection.license.sig")).unwrap(),
        "license",
        license_signer.verifying_key().as_bytes(),
        &verified,
    )
    .unwrap();
    license.verify_key(&dek).unwrap();
    assert_eq!(license.entitlement.mode, "offline-perpetual");
    let mut altered = manifest;
    altered["runtime"]["protection_profile"] = "encrypted-tpm".into();
    let altered = serde_json::to_vec(&altered).unwrap();
    private_file(&package.join("model.protection.json"), &altered);
    private_file(
        &package.join("model.protection.sig"),
        &serde_json::to_vec(&envelope(&manifest_signature_payload(&altered))).unwrap(),
    );
    assert!(
        !run("export-file-key", "must-not-exist.bin", &[])
            .status
            .success()
    );
    assert!(!root.join("must-not-exist.bin").exists());
}
