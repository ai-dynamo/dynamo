// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use dynamo_model_protection::{
    SignatureEnvelope, certified_device_signature_payload,
    enrollment::{
        credential::make_challenge,
        format::{
            Challenge, MAX_BUNDLE_BYTES, ValidatedRequest, challenge_signature_payload, hex, parse,
            validate_request, verify_challenge, verify_response,
        },
        registry::Registry,
        trust::{VerifiedEk, verify_ek, verify_trust_policy},
    },
};
use rustix::fs::{CWD, Mode, OFlags, RenameFlags, ResolveFlags, openat2, renameat_with};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env,
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[path = "support/atomic_file.rs"]
mod atomic_file;
#[path = "support/secure_file.rs"]
mod secure_file;
#[path = "support/software_keys.rs"]
#[allow(dead_code)]
mod software_keys;

type Result<T> = std::result::Result<T, &'static str>;
const USAGE: &str = "Experimental offline enrollment authority (not a production release approval).\nCommands: init-registry, inspect-status, challenge, verify, export-certificate, disable-certificate, backup-registry.\nAll commands require --registry ABSOLUTE_PATH. inspect-status uses --challenge-id; export/disable use --certification-id.\nchallenge/verify require --request, --trust-policy, --trust-policy-signature, --trust-key-id, --trust-public-key, --minimum-trust-sequence.\nchallenge also requires --challenge-signing-key, --challenge-passphrase-file, --challenge-key-id, --state-kek-key-file, --state-output, --output, --ttl-seconds.\nverify also requires --challenge, --challenge-signature, --challenge-key-id, --challenge-public-key, --state, --state-kek-key-file, --response, --enrollment-signing-key, --enrollment-passphrase-file, --enrollment-key-id, --certification-id, --quota, --output.\nexport-certificate requires --enrollment-signing-key, --enrollment-passphrase-file, --enrollment-key-id, --output.\nbackup-registry requires --output ABSOLUTE_PATH in an existing owner-only directory.\nKeys/passphrases are owner-only files, never argv secrets. No TPM provisioning or remote certificate fetching.";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SealedState {
    format: String,
    format_version: u16,
    challenge_id: String,
    challenge_sha256: String,
    request_sha256: String,
    wrapped_activation_secret: String,
}

struct Args {
    command: String,
    values: BTreeMap<String, String>,
}
impl Args {
    fn value(&self, name: &str) -> Result<&str> {
        self.values
            .get(name)
            .map(String::as_str)
            .ok_or("ENROLLMENT_CONFIG_INVALID")
    }
    fn path(&self, name: &str) -> Result<PathBuf> {
        let path = PathBuf::from(self.value(name)?);
        if !path.is_absolute() {
            return Err("ENROLLMENT_CONFIG_INVALID");
        }
        Ok(path)
    }
    fn number(&self, name: &str) -> Result<u64> {
        self.value(name)?
            .parse()
            .map_err(|_| "ENROLLMENT_CONFIG_INVALID")
    }
    fn read(&self, name: &str, private: bool) -> Result<Vec<u8>> {
        secure_file::read_bounded(&self.path(name)?, MAX_BUNDLE_BYTES as u64, private)
            .map_err(|_| "ENROLLMENT_INPUT_INVALID")
    }
    fn public_key(&self, name: &str) -> Result<[u8; 32]> {
        secure_file::read_bounded(&self.path(name)?, 32, false)
            .map_err(|_| "ENROLLMENT_INPUT_INVALID")?
            .try_into()
            .map_err(|_| "ENROLLMENT_INPUT_INVALID")
    }
    fn signer(&self, key: &str, password: &str) -> Result<ed25519_dalek::SigningKey> {
        let passphrase = software_keys::read_passphrase(&self.path(password)?)?;
        software_keys::read_signing_key(&self.path(key)?, &passphrase)
    }
}

fn parse_args() -> Result<Args> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or("ENROLLMENT_CONFIG_INVALID")?;
    let allowed: &[&str] = match command.as_str() {
        "init-registry" => &["--registry"],
        "backup-registry" => &["--registry", "--output"],
        "inspect-status" => &["--registry", "--challenge-id"],
        "disable-certificate" => &["--registry", "--certification-id"],
        "export-certificate" => &[
            "--registry",
            "--certification-id",
            "--enrollment-signing-key",
            "--enrollment-passphrase-file",
            "--enrollment-key-id",
            "--output",
        ],
        "challenge" => &[
            "--registry",
            "--request",
            "--trust-policy",
            "--trust-policy-signature",
            "--trust-key-id",
            "--trust-public-key",
            "--minimum-trust-sequence",
            "--challenge-signing-key",
            "--challenge-passphrase-file",
            "--challenge-key-id",
            "--state-kek-key-file",
            "--state-output",
            "--output",
            "--ttl-seconds",
        ],
        "verify" => &[
            "--registry",
            "--request",
            "--trust-policy",
            "--trust-policy-signature",
            "--trust-key-id",
            "--trust-public-key",
            "--minimum-trust-sequence",
            "--challenge",
            "--challenge-signature",
            "--challenge-key-id",
            "--challenge-public-key",
            "--state",
            "--state-kek-key-file",
            "--response",
            "--enrollment-signing-key",
            "--enrollment-passphrase-file",
            "--enrollment-key-id",
            "--certification-id",
            "--quota",
            "--output",
        ],
        _ => return Err("ENROLLMENT_CONFIG_INVALID"),
    };
    let mut values = BTreeMap::new();
    while let Some(name) = args.next() {
        if !allowed.contains(&name.as_str()) || values.contains_key(&name) {
            return Err("ENROLLMENT_CONFIG_INVALID");
        }
        let value = args.next().ok_or("ENROLLMENT_CONFIG_INVALID")?;
        if value.is_empty() || value.starts_with("--") {
            return Err("ENROLLMENT_CONFIG_INVALID");
        }
        values.insert(name, value);
    }
    if values.len() != allowed.len() {
        return Err("ENROLLMENT_CONFIG_INVALID");
    }
    Ok(Args { command, values })
}

fn now() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|_| "ENROLLMENT_CLOCK_INVALID")
}

fn validated_request(
    args: &Args,
    time: u64,
    registry: &mut Registry,
) -> Result<(ValidatedRequest, VerifiedEk)> {
    let request = validate_request(
        parse(&args.read("--request", false)?).map_err(|_| "ENROLLMENT_INPUT_INVALID")?,
    )
    .map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
    let policy = verify_trust_policy(
        &args.read("--trust-policy", false)?,
        &args.read("--trust-policy-signature", false)?,
        args.value("--trust-key-id")?,
        &args.public_key("--trust-public-key")?,
        args.number("--minimum-trust-sequence")?,
        time,
    )
    .map_err(|_| "ENROLLMENT_EK_TRUST_INVALID")?;
    registry
        .accept_trust_policy(&policy)
        .map_err(|_| "ENROLLMENT_TRUST_POLICY_ROLLBACK")?;
    let ek = verify_ek(&request, &policy, time).map_err(|_| "ENROLLMENT_EK_TRUST_INVALID")?;
    Ok((request, ek))
}

fn envelope(key: &ed25519_dalek::SigningKey, key_id: &str, payload: &[u8]) -> Result<Vec<u8>> {
    if !dynamo_model_protection::enrollment::format::identifier(key_id) {
        return Err("ENROLLMENT_CONFIG_INVALID");
    }
    serde_json::to_vec(&SignatureEnvelope {
        algorithm: "Ed25519".into(),
        key_id: key_id.into(),
        signature: BASE64.encode(software_keys::sign_ed25519(key, payload)),
    })
    .map_err(|_| "ENROLLMENT_OUTPUT_FAILED")
}

fn validate_output_parent(output: &Path) -> Result<()> {
    if !output.is_absolute() || output.file_name().is_none() {
        return Err("ENROLLMENT_CONFIG_INVALID");
    }
    let fd = openat2(
        CWD,
        output.parent().ok_or("ENROLLMENT_CONFIG_INVALID")?,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
        Mode::empty(),
        ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
    )
    .map_err(|_| "ENROLLMENT_OUTPUT_FAILED")?;
    let stat = rustix::fs::fstat(&fd).map_err(|_| "ENROLLMENT_OUTPUT_FAILED")?;
    if stat.st_uid != rustix::process::geteuid().as_raw() || stat.st_mode & 0o077 != 0 {
        return Err("ENROLLMENT_OUTPUT_FAILED");
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|_| "ENROLLMENT_DURABILITY_UNCONFIRMED")
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)
        .map_err(|_| "ENROLLMENT_OUTPUT_FAILED")?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|_| "ENROLLMENT_OUTPUT_FAILED")
}

fn write_bundle(output: &Path, files: &[(&str, &[u8])]) -> Result<()> {
    validate_output_parent(output)?;
    let partial = output.with_extension(format!("partial-{}", uuid::Uuid::new_v4().simple()));
    fs::create_dir(&partial).map_err(|_| "ENROLLMENT_OUTPUT_FAILED")?;
    fs::set_permissions(&partial, fs::Permissions::from_mode(0o700))
        .map_err(|_| "ENROLLMENT_OUTPUT_FAILED")?;
    let result = (|| {
        for (name, bytes) in files {
            write_new(&partial.join(name), bytes)?;
        }
        sync_directory(&partial)?;
        renameat_with(CWD, &partial, CWD, output, RenameFlags::NOREPLACE)
            .map_err(|_| "ENROLLMENT_OUTPUT_EXISTS_OR_INVALID")?;
        sync_directory(output.parent().ok_or("ENROLLMENT_OUTPUT_FAILED")?)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&partial);
    }
    result
}

fn publish_certificate(args: &Args, registry: &Registry) -> Result<()> {
    let bytes = registry
        .certification_bytes(args.value("--certification-id")?)
        .map_err(|_| "ENROLLMENT_ADMISSION_DENIED")?;
    let key = args.signer("--enrollment-signing-key", "--enrollment-passphrase-file")?;
    let signature = envelope(
        &key,
        args.value("--enrollment-key-id")?,
        &certified_device_signature_payload(&bytes),
    )?;
    write_bundle(
        &args.path("--output")?,
        &[
            ("certified-device.json", &bytes),
            ("certified-device.sig", &signature),
        ],
    )
}

fn run() -> Result<()> {
    let args = parse_args()?;
    let time = now()?;
    let mut registry = Registry::open(&args.path("--registry")?, args.command == "init-registry")
        .map_err(|_| "ENROLLMENT_REGISTRY_INVALID")?;
    match args.command.as_str() {
        "init-registry" => {}
        "backup-registry" => registry
            .backup(&args.path("--output")?)
            .map_err(|_| "ENROLLMENT_BACKUP_FAILED")?,
        "inspect-status" => {
            let status = registry
                .challenge_status(args.value("--challenge-id")?)
                .map_err(|_| "ENROLLMENT_REGISTRY_INVALID")?;
            println!(
                "{}",
                serde_json::json!({"status":status,"production_ready":false})
            );
        }
        "disable-certificate" => registry
            .disable_certification(args.value("--certification-id")?, time)
            .map_err(|_| "ENROLLMENT_ADMISSION_DENIED")?,
        "export-certificate" => {
            software_keys::lock_process_memory()?;
            publish_certificate(&args, &registry)?;
        }
        "challenge" => {
            software_keys::lock_process_memory()?;
            let (request, ek) = validated_request(&args, time, &mut registry)?;
            let key = args.signer("--challenge-signing-key", "--challenge-passphrase-file")?;
            if key.verifying_key().to_bytes() == args.public_key("--trust-public-key")?
                || args.value("--challenge-key-id")? == args.value("--trust-key-id")?
            {
                return Err("ENROLLMENT_CONFIG_INVALID");
            }
            let activation = make_challenge(&request, &ek, time, args.number("--ttl-seconds")?)
                .map_err(|_| "ENROLLMENT_CHALLENGE_INVALID")?;
            let bytes = serde_json::to_vec(&activation.challenge)
                .map_err(|_| "ENROLLMENT_OUTPUT_FAILED")?;
            let signature = envelope(
                &key,
                args.value("--challenge-key-id")?,
                &challenge_signature_payload(&bytes),
            )?;
            let kek = software_keys::read_kek(&args.path("--state-kek-key-file")?)?;
            let state = SealedState {
                format: "model-protection-enrollment-sealed-state".into(),
                format_version: 1,
                challenge_id: activation.challenge.challenge_id.clone(),
                challenge_sha256: hex(&Sha256::digest(&bytes)),
                request_sha256: hex(request.digest()),
                wrapped_activation_secret: BASE64.encode(software_keys::wrap_dek(
                    &kek,
                    activation.activation_secret(),
                )?),
            };
            let state_path = args.path("--state-output")?;
            atomic_file::publish_new(
                &state_path,
                &serde_json::to_vec(&state).map_err(|_| "ENROLLMENT_OUTPUT_FAILED")?,
            )
            .map_err(|_| "ENROLLMENT_OUTPUT_EXISTS_OR_INVALID")?;
            registry
                .record_challenge(&request, &activation.challenge, time)
                .map_err(|_| "ENROLLMENT_REGISTRY_INVALID")?;
            write_bundle(
                &args.path("--output")?,
                &[("challenge.json", &bytes), ("challenge.sig", &signature)],
            )?;
        }
        "verify" => {
            software_keys::lock_process_memory()?;
            let (request, ek) = validated_request(&args, time, &mut registry)?;
            let challenge_bytes = args.read("--challenge", false)?;
            let challenge_key = args.public_key("--challenge-public-key")?;
            let enrollment_key =
                args.signer("--enrollment-signing-key", "--enrollment-passphrase-file")?;
            if challenge_key == enrollment_key.verifying_key().to_bytes()
                || challenge_key == args.public_key("--trust-public-key")?
                || enrollment_key.verifying_key().to_bytes()
                    == args.public_key("--trust-public-key")?
                || args.value("--challenge-key-id")? == args.value("--enrollment-key-id")?
                || args.value("--challenge-key-id")? == args.value("--trust-key-id")?
                || args.value("--enrollment-key-id")? == args.value("--trust-key-id")?
            {
                return Err("ENROLLMENT_CONFIG_INVALID");
            }
            let challenge: Challenge = verify_challenge(
                &challenge_bytes,
                &args.read("--challenge-signature", false)?,
                args.value("--challenge-key-id")?,
                &challenge_key,
                &request,
                time,
            )
            .map_err(|_| "ENROLLMENT_CHALLENGE_INVALID")?;
            let state: SealedState =
                parse(&args.read("--state", true)?).map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
            if state.format != "model-protection-enrollment-sealed-state"
                || state.format_version != 1
                || state.challenge_id != challenge.challenge_id
                || state.challenge_sha256 != hex(&Sha256::digest(&challenge_bytes))
                || state.request_sha256 != hex(request.digest())
            {
                return Err("ENROLLMENT_BINDING_INVALID");
            }
            let kek = software_keys::read_kek(&args.path("--state-kek-key-file")?)?;
            let wrapped = BASE64
                .decode(state.wrapped_activation_secret)
                .map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
            let secret = software_keys::unwrap_dek(&kek, &wrapped)?;
            let response = args.read("--response", false)?;
            let proof = verify_response(&request, &challenge, &response, &secret, time)
                .map_err(|_| "ENROLLMENT_PROOF_INVALID")?;
            let quota =
                u32::try_from(args.number("--quota")?).map_err(|_| "ENROLLMENT_CONFIG_INVALID")?;
            registry
                .certify(
                    &request,
                    &ek,
                    &proof,
                    args.value("--certification-id")?,
                    quota,
                    time,
                )
                .map_err(|_| "ENROLLMENT_ADMISSION_DENIED")?;
            publish_certificate(&args, &registry)?;
        }
        _ => return Err("ENROLLMENT_CONFIG_INVALID"),
    }
    Ok(())
}

fn main() {
    if env::args().len() == 2 && env::args().nth(1).as_deref() == Some("--help") {
        println!("{USAGE}");
        return;
    }
    if let Err(code) = run() {
        eprintln!("model_protection event=enrollment_authority_failed code={code}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_is_no_overwrite_and_rejects_symlink_parent() {
        let directory = tempfile::tempdir().unwrap();
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let output = directory.path().join("certificate");
        write_bundle(&output, &[("certified-device.json", b"first")]).unwrap();
        assert!(write_bundle(&output, &[("certified-device.json", b"second")]).is_err());
        assert_eq!(
            fs::read(output.join("certified-device.json")).unwrap(),
            b"first"
        );
        let link = directory.path().join("link");
        std::os::unix::fs::symlink(directory.path(), &link).unwrap();
        assert!(write_bundle(&link.join("other"), &[("certified-device.json", b"other")]).is_err());
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);
    }
}
