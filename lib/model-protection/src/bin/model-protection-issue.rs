// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use dynamo_model_protection::{
    Entitlement, License, MAX_ISSUER_RECORD_BYTES, SignatureEnvelope, TPM_OAEP_LABEL,
    TPM_POLICY_REF_HEX, TPM_POLICY_SIGNATURE, TPM_PROFILE, TpmRecipient,
    approved_rsa_decrypt_policy, license_signature_payload, policy_authorize_auth_policy,
    rsa_decrypt_cp_hash, verify_issuer_record, verify_manifest,
};
use ring::signature;
use rustix::fs::{CWD, RenameFlags, renameat_with};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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

const CERTIFIED_DEVICE_SIGNATURE_DOMAIN: &[u8] =
    b"model-protection-certified-device-signature-v1\0";
const MAX_CONTROL_BYTES: u64 = 4 * 1024 * 1024;
const USAGE: &str = "Usage: model-protection-issue --package PATH --issuer-record PATH --certified-device PATH --certified-device-signature PATH --output PATH --package-key-id ID --package-public-key PATH --enrollment-key-id ID --enrollment-public-key PATH --license-signing-key PATH --license-key-passphrase-file PATH --license-key-id ID --policy-signing-key PATH --policy-key-passphrase-file PATH --policy-key-id ID --kek-key-file PATH --kek-key-id ID --kek-key-version VERSION --license-id ID --generation NUMBER";

type Result<T> = std::result::Result<T, IssueError>;

#[derive(Debug)]
struct IssueError(&'static str);

impl std::fmt::Display for IssueError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for IssueError {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CertifiedDevice {
    format: String,
    format_version: u16,
    certification_id: String,
    customer_scope_id: String,
    artifact_id: String,
    tpm_public: String,
    policy_authority_name: String,
}

#[derive(Serialize)]
struct LicenseEnvelope {
    algorithm: &'static str,
    key_id: String,
    signature: String,
}

fn main() {
    if env::args().len() == 2 && env::args().nth(1).as_deref() == Some("--help") {
        println!("{USAGE}");
        return;
    }
    if let Err(error) = run() {
        eprintln!("model_protection event=license_issue_failed code={error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    lock_process_memory().map_err(IssueError)?;
    let args = parse_args()?;
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
        || output.exists()
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

    let certified_bytes = read_bounded(&certified_device_path, MAX_CONTROL_BYTES)?;
    let certified_signature = read_bounded(&certified_device_signature_path, MAX_CONTROL_BYTES)?;
    verify_envelope(
        CERTIFIED_DEVICE_SIGNATURE_DOMAIN,
        &certified_bytes,
        &certified_signature,
        enrollment_key_id,
        &enrollment_key,
    )?;
    let certified: CertifiedDevice = serde_json::from_slice(&certified_bytes)
        .map_err(|_| IssueError("CERTIFIED_DEVICE_INVALID"))?;
    let manifest = verified.manifest();
    if certified.format != "model-protection-certified-device"
        || certified.format_version != 1
        || !valid_identifier(&certified.certification_id)
        || certified.customer_scope_id != manifest.customer_scope_id
        || certified.artifact_id != manifest.artifact_id
    {
        return Err(IssueError("CERTIFIED_DEVICE_INVALID"));
    }
    let expected_policy_authority_name = decode_hex::<34>(&certified.policy_authority_name)?;
    let (device_name, device_public_digest, modulus) =
        parse_tpm_public(&certified.tpm_public, &expected_policy_authority_name)?;

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
    .is_none_or(|name| name != expected_policy_authority_name)
    {
        return Err(IssueError("ISSUER_KEY_INVALID"));
    }
    let dek = unwrap_dek(&kek, &wrapped_issuer_dek).map_err(IssueError)?;
    let wrapped_dek = wrap_to_tpm(&modulus, &dek, TPM_OAEP_LABEL).map_err(IssueError)?;

    let operation = (|| {
        let cp_hash = rsa_decrypt_cp_hash(&device_name, &wrapped_dek);
        let approved_policy = approved_rsa_decrypt_policy(&cp_hash);
        let policy_ref = decode_hex::<32>(TPM_POLICY_REF_HEX)?;
        let mut policy_digest = Sha256::new();
        policy_digest.update(approved_policy);
        policy_digest.update(policy_ref);
        let policy_digest: [u8; 32] = policy_digest.finalize().into();
        let policy_signature =
            sign_p256_digest(&policy_signer, &policy_digest).map_err(IssueError)?;

        let generation = required(&args, "--generation")?
            .parse::<u64>()
            .map_err(|_| IssueError("ISSUER_CONFIG_INVALID"))?;
        if generation == 0 {
            return Err(IssueError("ISSUER_CONFIG_INVALID"));
        }
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
                device_key_name: lower_hex(&device_name),
                device_public_key_sha256: lower_hex(&device_public_digest),
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
    write_license_bundle(&output, &license, &signature)
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

    // Frozen TPMT_PUBLIC for the external P-256 ECDSA/SHA-256 policy authority.
    let mut public = Vec::with_capacity(86);
    public.extend_from_slice(&0x0023_u16.to_be_bytes()); // TPM_ALG_ECC
    public.extend_from_slice(&0x000b_u16.to_be_bytes()); // TPM_ALG_SHA256
    public.extend_from_slice(&0x0004_0040_u32.to_be_bytes()); // sign | userWithAuth
    public.extend_from_slice(&0_u16.to_be_bytes()); // empty authPolicy
    public.extend_from_slice(&0x0010_u16.to_be_bytes()); // symmetric: TPM_ALG_NULL
    public.extend_from_slice(&0x0018_u16.to_be_bytes()); // scheme: TPM_ALG_ECDSA
    public.extend_from_slice(&0x000b_u16.to_be_bytes()); // scheme hash: SHA256
    public.extend_from_slice(&0x0003_u16.to_be_bytes()); // curve: NIST P-256
    public.extend_from_slice(&0x0010_u16.to_be_bytes()); // KDF: TPM_ALG_NULL
    public.extend_from_slice(&32_u16.to_be_bytes());
    public.extend_from_slice(&point[1..33]);
    public.extend_from_slice(&32_u16.to_be_bytes());
    public.extend_from_slice(&point[33..65]);

    let mut name = [0_u8; 34];
    name[..2].copy_from_slice(&0x000b_u16.to_be_bytes());
    name[2..].copy_from_slice(&Sha256::digest(public));
    Some(name)
}

fn parse_tpm_public(
    encoded: &str,
    policy_authority_name: &[u8; 34],
) -> Result<([u8; 34], [u8; 32], [u8; 256])> {
    let public = BASE64
        .decode(encoded.as_bytes())
        .map_err(|_| IssueError("CERTIFIED_DEVICE_INVALID"))?;
    if public.len() != 310
        || public[0..2] != [0x00, 0x01]
        || public[2..4] != [0x00, 0x0b]
        || public[4..8] != [0x00, 0x02, 0x04, 0xb2]
        || public[8..10] != [0x00, 0x20]
        || public[42..44] != [0x00, 0x10]
        || public[44..46] != [0x00, 0x10]
        || public[46..48] != [0x08, 0x00]
        || public[48..52] != [0, 0, 0, 0]
        || public[52..54] != [0x01, 0x00]
    {
        return Err(IssueError("CERTIFIED_DEVICE_INVALID"));
    }
    let policy_ref = decode_hex::<32>(TPM_POLICY_REF_HEX)?;
    if policy_authority_name[..2] != [0, 0x0b] {
        return Err(IssueError("CERTIFIED_DEVICE_INVALID"));
    }
    let expected_policy = policy_authorize_auth_policy(policy_authority_name, &policy_ref);
    if public[10..42] != expected_policy {
        return Err(IssueError("CERTIFIED_DEVICE_INVALID"));
    }
    let digest: [u8; 32] = Sha256::digest(&public).into();
    let mut name = [0_u8; 34];
    name[..2].copy_from_slice(&[0, 0x0b]);
    name[2..].copy_from_slice(&digest);
    let modulus = public[54..]
        .try_into()
        .map_err(|_| IssueError("CERTIFIED_DEVICE_INVALID"))?;
    Ok((name, digest, modulus))
}

fn verify_envelope(
    domain: &[u8],
    payload: &[u8],
    envelope: &[u8],
    expected_key_id: &str,
    public_key: &[u8; 32],
) -> Result<()> {
    let envelope: SignatureEnvelope =
        serde_json::from_slice(envelope).map_err(|_| IssueError("CERTIFIED_DEVICE_INVALID"))?;
    if envelope.algorithm != "Ed25519" || envelope.key_id != expected_key_id {
        return Err(IssueError("CERTIFIED_DEVICE_INVALID"));
    }
    let signature_bytes = BASE64
        .decode(envelope.signature.as_bytes())
        .map_err(|_| IssueError("CERTIFIED_DEVICE_INVALID"))?;
    let signed = [domain, payload].concat();
    signature::UnparsedPublicKey::new(&signature::ED25519, public_key)
        .verify(&signed, &signature_bytes)
        .map_err(|_| IssueError("CERTIFIED_DEVICE_INVALID"))
}

fn write_license_bundle(output: &Path, license: &[u8], signature: &LicenseEnvelope) -> Result<()> {
    let temporary = output.with_extension(format!("partial-{}", uuid::Uuid::new_v4().simple()));
    fs::create_dir(&temporary).map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
    fs::set_permissions(&temporary, fs::Permissions::from_mode(0o700))
        .map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
    let result = (|| {
        write_new(&temporary.join("model.protection.license.json"), license)?;
        let signature =
            serde_json::to_vec_pretty(signature).map_err(|_| IssueError("LICENSE_INVALID"))?;
        write_new(&temporary.join("model.protection.license.sig"), &signature)?;
        sync_directory(&temporary)?;
        renameat_with(CWD, &temporary, CWD, output, RenameFlags::NOREPLACE)
            .map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
        sync_directory(output.parent().ok_or(IssueError("LICENSE_IO_ERROR"))?)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result
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

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(|_| IssueError("LICENSE_IO_ERROR"))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|_| IssueError("LICENSE_IO_ERROR"))
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|_| IssueError("LICENSE_IO_ERROR"))
}

fn lower_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
