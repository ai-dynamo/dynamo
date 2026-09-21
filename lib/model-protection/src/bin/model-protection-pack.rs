// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use dynamo_model_protection::{
    Encryption, ISSUER_RECORD_FORMAT, ISSUER_RECORD_VERSION, IssuerDekRecord, Manifest,
    ModelIdentity, ProtectedFile, PublicFile, RuntimeRequirements, SignatureEnvelope,
    SignedIssuerDekRecord, encrypt_records, is_allowed_public_metadata,
    issuer_record_signature_payload, manifest_digest, manifest_signature_payload, parse_manifest,
    validate_safetensors_index,
};
use ring::rand::{SecureRandom, SystemRandom};
use rustix::fs::{CWD, RenameFlags, renameat_with};
use serde::Serialize;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

#[path = "support/secure_file.rs"]
mod secure_file;
#[path = "support/software_keys.rs"]
#[allow(dead_code)]
mod software_keys;
use secure_file::{open_regular, read_bounded};
use software_keys::{
    lock_process_memory, read_kek, read_passphrase, read_signing_key, sign_ed25519, wrap_dek,
};

const RECORD_BYTES: u32 = 16 * 1024 * 1024;
const USAGE: &str = "Usage: model-protection-pack --source PATH --output PATH --issuer-record PATH --customer-scope-id ID --model-id ID --model-version VERSION --minimum-runtime-version VERSION --package-signing-key PATH --package-key-passphrase-file PATH --package-key-id ID --kek-key-file PATH --kek-key-id ID --kek-key-version VERSION";

type Result<T> = std::result::Result<T, PackError>;

#[derive(Debug)]
struct PackError(&'static str);

struct IssuerRecordContext<'a> {
    manifest: &'a Manifest,
    manifest_bytes: &'a [u8],
    package_key_id: &'a str,
    kek_key_id: &'a str,
    kek_key_version: &'a str,
}

impl std::fmt::Display for PackError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for PackError {}

fn main() {
    if env::args().len() == 2 && env::args().nth(1).as_deref() == Some("--help") {
        println!("{USAGE}");
        return;
    }
    if let Err(error) = run() {
        eprintln!("model_protection event=package_failed code={error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    lock_process_memory().map_err(PackError)?;
    let args = parse_args()?;
    let source = required_path(&args, "--source")?;
    let output = required_path(&args, "--output")?;
    let issuer_record = required_path(&args, "--issuer-record")?;
    let package_signing_key_path = required_path(&args, "--package-signing-key")?;
    let package_key_passphrase =
        read_passphrase(&required_path(&args, "--package-key-passphrase-file")?)
            .map_err(PackError)?;
    let package_key_id = required(&args, "--package-key-id")?;
    let kek_key_path = required_path(&args, "--kek-key-file")?;
    let kek_key_id = required(&args, "--kek-key-id")?;
    let kek_key_version = required(&args, "--kek-key-version")?;
    require_identifier(kek_key_version)?;
    if package_key_id == kek_key_id {
        return Err(PackError("PACKAGE_CONFIG_INVALID"));
    }

    require_identifier(required(&args, "--customer-scope-id")?)?;
    require_identifier(required(&args, "--model-id")?)?;
    require_identifier(required(&args, "--model-version")?)?;
    if !source.is_absolute() || !output.is_absolute() || !issuer_record.is_absolute() {
        return Err(PackError("PACKAGE_CONFIG_INVALID"));
    }
    if !source.is_dir() || output.exists() || issuer_record.exists() {
        return Err(PackError("PACKAGE_PATH_INVALID"));
    }

    let random = SystemRandom::new();
    let mut artifact_id = [0_u8; 16];
    let mut nonce_prefix = [0_u8; 4];
    let mut dek = Zeroizing::new([0_u8; 32]);
    random
        .fill(&mut artifact_id)
        .and_then(|()| random.fill(&mut nonce_prefix))
        .and_then(|()| random.fill(&mut *dek))
        .map_err(|_| PackError("RANDOM_GENERATION_FAILED"))?;

    let temporary = output.with_extension(format!("partial-{}", uuid::Uuid::new_v4().simple()));
    fs::create_dir(&temporary).map_err(|_| PackError("PACKAGE_IO_ERROR"))?;
    fs::set_permissions(&temporary, fs::Permissions::from_mode(0o700))
        .map_err(|_| PackError("PACKAGE_IO_ERROR"))?;

    let mut issuer_record_created = false;
    let mut package_published = false;
    let result = (|| {
        let manifest = build_package(&source, &temporary, &args, &dek, artifact_id, nonce_prefix)?;
        let index_path = temporary.join("public/model.safetensors.index.json");
        let index = index_path
            .exists()
            .then(|| {
                read_bounded(
                    &index_path,
                    dynamo_model_protection::MAX_SAFETENSORS_INDEX_BYTES,
                    false,
                )
                .map_err(|_| PackError("SOURCE_IO_ERROR"))
            })
            .transpose()?;
        validate_safetensors_index(&manifest, index.as_deref())
            .map_err(|_| PackError("SOURCE_INVALID"))?;
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|_| PackError("MANIFEST_SERIALIZATION_FAILED"))?;
        parse_manifest(&manifest_bytes).map_err(|_| PackError("MANIFEST_INVALID"))?;

        let package_signing_key =
            read_signing_key(&package_signing_key_path, &package_key_passphrase)
                .map_err(PackError)?;
        let kek = read_kek(&kek_key_path).map_err(PackError)?;
        let (signature, issuer_record_bytes) = software_finalize(
            &package_signing_key,
            &kek,
            &dek,
            IssuerRecordContext {
                manifest: &manifest,
                manifest_bytes: &manifest_bytes,
                package_key_id,
                kek_key_id,
                kek_key_version,
            },
        )?;
        if signature.len() != 64 {
            return Err(PackError("ISSUER_OUTPUT_INVALID"));
        }
        write_new(&temporary.join("model.protection.json"), &manifest_bytes)?;
        write_json(
            &temporary.join("model.protection.sig"),
            &SignatureEnvelope {
                algorithm: "Ed25519".to_string(),
                key_id: package_key_id.to_string(),
                signature: BASE64.encode(signature),
            },
        )?;
        write_new(&issuer_record, &issuer_record_bytes)?;
        issuer_record_created = true;
        sync_parent(&issuer_record)?;
        sync_directory(&temporary.join("weights"))?;
        sync_directory(&temporary.join("public"))?;
        sync_directory(&temporary)?;
        publish_package(&temporary, &output, &mut package_published, sync_parent)?;
        Ok(())
    })();

    if result.is_err() {
        cleanup_failed_build(
            &temporary,
            &issuer_record,
            issuer_record_created,
            package_published,
        );
    }
    result
}

fn cleanup_failed_build(
    temporary: &Path,
    issuer_record: &Path,
    issuer_record_created: bool,
    package_published: bool,
) {
    let _ = fs::remove_dir_all(temporary);
    if issuer_record_created && !package_published {
        let _ = fs::remove_file(issuer_record);
    }
}

fn build_package(
    source: &Path,
    output: &Path,
    args: &BTreeMap<String, String>,
    dek: &[u8; 32],
    artifact_id: [u8; 16],
    nonce_prefix: [u8; 4],
) -> Result<Manifest> {
    fs::create_dir(output.join("weights")).map_err(|_| PackError("PACKAGE_IO_ERROR"))?;
    fs::create_dir(output.join("public")).map_err(|_| PackError("PACKAGE_IO_ERROR"))?;
    let mut names = source_names(source)?;
    names.sort();
    let mut protected_files = Vec::new();
    let mut public_files = Vec::new();
    let mut counter = 0_u64;

    for name in names {
        let input = source.join(&name);
        if name.ends_with(".safetensors") {
            let file_id = u32::try_from(protected_files.len() + 1)
                .map_err(|_| PackError("SOURCE_INVALID"))?;
            let container_path = format!("weights/{name}.protected");
            let plaintext = open_regular(&input, dynamo_model_protection::MAX_FILE_BYTES, false)
                .map_err(|_| PackError("SOURCE_INVALID"))?;
            let mut plaintext = plaintext.take(dynamo_model_protection::MAX_FILE_BYTES + 1);
            let container = create_new(&output.join(&container_path))?;
            let encrypted = encrypt_records(
                &mut plaintext,
                container,
                dek,
                &artifact_id,
                &nonce_prefix,
                file_id,
                counter,
                RECORD_BYTES,
            )
            .map_err(|_| PackError("ENCRYPTION_FAILED"))?;
            File::open(output.join(&container_path))
                .and_then(|file| file.sync_all())
                .map_err(|_| PackError("PACKAGE_IO_ERROR"))?;
            protected_files.push(ProtectedFile {
                file_id,
                container_path,
                output_path: name,
                container_size: encrypted.container_size,
                container_sha256: lower_hex(&encrypted.container_sha256),
                plaintext_size: encrypted.plaintext_size,
                plaintext_sha256: lower_hex(&encrypted.plaintext_sha256),
                record_count: encrypted.record_count,
                first_global_record_counter: counter,
            });
            counter = counter
                .checked_add(u64::from(encrypted.record_count))
                .ok_or(PackError("SOURCE_INVALID"))?;
        } else if is_allowed_public_metadata(&name) {
            let destination = output.join("public").join(&name);
            let mut input = open_regular(&input, dynamo_model_protection::MAX_FILE_BYTES, false)
                .map_err(|_| PackError("SOURCE_INVALID"))?;
            let (size, digest) = copy_hashed(&mut input, &destination)?;
            public_files.push(PublicFile {
                source_path: format!("public/{name}"),
                output_path: name,
                size,
                sha256: lower_hex(&digest),
                publish_to_model_card: true,
            });
        }
    }
    if protected_files.is_empty()
        || !public_files
            .iter()
            .any(|file| file.output_path == "config.json")
    {
        return Err(PackError("SOURCE_INVALID"));
    }

    Ok(Manifest {
        format: "secure-model-package".to_string(),
        format_version: 1,
        artifact_id: lower_hex(&artifact_id),
        customer_scope_id: required(args, "--customer-scope-id")?.to_string(),
        model: ModelIdentity {
            model_id: required(args, "--model-id")?.to_string(),
            model_version: required(args, "--model-version")?.to_string(),
            framework: "safetensors".to_string(),
        },
        encryption: Encryption {
            algorithm: "AES-256-GCM".to_string(),
            nonce_prefix: lower_hex(&nonce_prefix),
            tag_bits: 128,
            record_plaintext_limit: RECORD_BYTES,
        },
        protected_files,
        public_files,
        runtime: RuntimeRequirements {
            minimum_runtime_version: required(args, "--minimum-runtime-version")?.to_string(),
            required_load_format: "safetensors".to_string(),
        },
    })
}

fn software_finalize(
    signer: &ed25519_dalek::SigningKey,
    kek: &[u8; 32],
    dek: &[u8; 32],
    record: IssuerRecordContext<'_>,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let signature = sign_ed25519(signer, &manifest_signature_payload(record.manifest_bytes));
    let wrapped = wrap_dek(kek, dek).map_err(PackError)?;
    let issuer_record = IssuerDekRecord {
        format: ISSUER_RECORD_FORMAT.to_string(),
        format_version: ISSUER_RECORD_VERSION,
        artifact_id: record.manifest.artifact_id.clone(),
        customer_scope_id: record.manifest.customer_scope_id.clone(),
        model_id: record.manifest.model.model_id.clone(),
        model_version: record.manifest.model.model_version.clone(),
        manifest_sha256: lower_hex(&manifest_digest(record.manifest_bytes)),
        kek_key_id: record.kek_key_id.to_string(),
        kek_key_version: record.kek_key_version.to_string(),
        wrapped_dek: BASE64.encode(wrapped),
    };
    let payload =
        serde_json::to_vec(&issuer_record).map_err(|_| PackError("ISSUER_RECORD_INVALID"))?;
    let record_signature = sign_ed25519(signer, &issuer_record_signature_payload(&payload));
    let envelope = SignedIssuerDekRecord {
        payload: BASE64.encode(payload),
        signature: SignatureEnvelope {
            algorithm: "Ed25519".to_string(),
            key_id: record.package_key_id.to_string(),
            signature: BASE64.encode(record_signature),
        },
    };
    let envelope =
        serde_json::to_vec_pretty(&envelope).map_err(|_| PackError("ISSUER_RECORD_INVALID"))?;
    Ok((signature.to_vec(), envelope))
}

fn parse_args() -> Result<BTreeMap<String, String>> {
    let mut parsed = BTreeMap::new();
    let mut args = env::args().skip(1);
    while let Some(name) = args.next() {
        if !name.starts_with("--") || !allowed_argument(&name) || parsed.contains_key(&name) {
            return Err(PackError("PACKAGE_CONFIG_INVALID"));
        }
        let value = args.next().ok_or(PackError("PACKAGE_CONFIG_INVALID"))?;
        if value.is_empty() || value.starts_with("--") {
            return Err(PackError("PACKAGE_CONFIG_INVALID"));
        }
        parsed.insert(name, value);
    }
    Ok(parsed)
}

fn allowed_argument(name: &str) -> bool {
    matches!(
        name,
        "--source"
            | "--output"
            | "--issuer-record"
            | "--customer-scope-id"
            | "--model-id"
            | "--model-version"
            | "--minimum-runtime-version"
            | "--package-signing-key"
            | "--package-key-passphrase-file"
            | "--package-key-id"
            | "--kek-key-file"
            | "--kek-key-id"
            | "--kek-key-version"
    )
}

fn required<'a>(args: &'a BTreeMap<String, String>, name: &str) -> Result<&'a str> {
    args.get(name)
        .map(String::as_str)
        .ok_or(PackError("PACKAGE_CONFIG_INVALID"))
}

fn required_path(args: &BTreeMap<String, String>, name: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(required(args, name)?))
}

fn require_identifier(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 128 || value.chars().any(char::is_control) {
        return Err(PackError("PACKAGE_CONFIG_INVALID"));
    }
    Ok(())
}

fn source_names(source: &Path) -> Result<Vec<String>> {
    fs::read_dir(source)
        .map_err(|_| PackError("SOURCE_IO_ERROR"))?
        .map(|entry| {
            entry
                .map_err(|_| PackError("SOURCE_IO_ERROR"))?
                .file_name()
                .into_string()
                .map_err(|_| PackError("SOURCE_INVALID"))
        })
        .collect()
}

fn copy_hashed(input: &mut File, destination: &Path) -> Result<(u64, [u8; 32])> {
    let mut output = create_new(destination)?;
    let expected = input
        .metadata()
        .map_err(|_| PackError("SOURCE_IO_ERROR"))?
        .len();
    let mut input = input.take(dynamo_model_protection::MAX_FILE_BYTES + 1);
    let mut digest = Sha256::new();
    let mut size = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = input
            .read(&mut buffer)
            .map_err(|_| PackError("SOURCE_IO_ERROR"))?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .map_err(|_| PackError("PACKAGE_IO_ERROR"))?;
        digest.update(&buffer[..read]);
        size = size
            .checked_add(read as u64)
            .ok_or(PackError("SOURCE_INVALID"))?;
    }
    output
        .sync_all()
        .map_err(|_| PackError("PACKAGE_IO_ERROR"))?;
    if size != expected || size > dynamo_model_protection::MAX_FILE_BYTES {
        return Err(PackError("SOURCE_INVALID"));
    }
    Ok((size, digest.finalize().into()))
}

fn create_new(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(|_| PackError("PACKAGE_IO_ERROR"))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = create_new(path)?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|_| PackError("PACKAGE_IO_ERROR"))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|_| PackError("PACKAGE_IO_ERROR"))?;
    write_new(path, &bytes)
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|_| PackError("PACKAGE_IO_ERROR"))
}

fn sync_parent(path: &Path) -> Result<()> {
    sync_directory(path.parent().ok_or(PackError("PACKAGE_PATH_INVALID"))?)
}

fn publish_directory(source: &Path, destination: &Path) -> Result<()> {
    renameat_with(CWD, source, CWD, destination, RenameFlags::NOREPLACE)
        .map_err(|_| PackError("PACKAGE_IO_ERROR"))
}

fn publish_package<F>(
    source: &Path,
    destination: &Path,
    published: &mut bool,
    sync: F,
) -> Result<()>
where
    F: FnOnce(&Path) -> Result<()>,
{
    publish_directory(source, destination)?;
    *published = true;
    sync(destination)
}

fn lower_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_writer_never_removes_a_competing_artifact() {
        let directory = tempfile::tempdir().unwrap();
        let temporary = directory.path().join("partial");
        let winner = directory.path().join("package");
        let issuer_record = directory.path().join("issuer.json");
        fs::create_dir(&temporary).unwrap();
        fs::create_dir(&winner).unwrap();
        fs::write(winner.join("winner"), b"committed").unwrap();
        fs::write(&issuer_record, b"other writer").unwrap();

        assert!(publish_directory(&temporary, &winner).is_err());
        cleanup_failed_build(&temporary, &issuer_record, false, false);

        assert_eq!(fs::read(winner.join("winner")).unwrap(), b"committed");
        assert_eq!(fs::read(&issuer_record).unwrap(), b"other writer");
    }

    #[test]
    fn post_publish_failure_retains_package_and_issuer_record() {
        let directory = tempfile::tempdir().unwrap();
        let temporary = directory.path().join("partial");
        let package = directory.path().join("package");
        let issuer_record = directory.path().join("issuer.json");
        fs::create_dir(&temporary).unwrap();
        fs::write(temporary.join("weights"), b"encrypted").unwrap();
        fs::write(&issuer_record, b"issuer record").unwrap();
        let mut published = false;
        let result = publish_package(&temporary, &package, &mut published, |_| {
            Err(PackError("PACKAGE_IO_ERROR"))
        });

        assert!(result.is_err());
        assert!(published);
        cleanup_failed_build(&temporary, &issuer_record, true, published);

        assert_eq!(fs::read(package.join("weights")).unwrap(), b"encrypted");
        assert_eq!(fs::read(&issuer_record).unwrap(), b"issuer record");
    }

    #[test]
    fn pre_publish_failure_removes_owned_staging_and_issuer_record() {
        let directory = tempfile::tempdir().unwrap();
        let temporary = directory.path().join("partial");
        let issuer_record = directory.path().join("issuer.json");
        fs::create_dir(&temporary).unwrap();
        fs::write(temporary.join("weights"), b"encrypted").unwrap();
        fs::write(&issuer_record, b"issuer record").unwrap();

        cleanup_failed_build(&temporary, &issuer_record, true, false);

        assert!(!temporary.exists());
        assert!(!issuer_record.exists());
    }
}
