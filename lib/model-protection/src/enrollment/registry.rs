// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Issuer-owned transaction ledger. Evidence-only rows never authorize issuance.

use std::fs::{self, OpenOptions};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;
use std::time::Duration;

use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};
use sha2::{Digest, Sha256};

use super::format::{Binding, Challenge, ValidatedRequest, VerifiedTranscript, identifier};
use super::trust::{VerifiedEk, VerifiedTrustPolicy};
use super::{EnrollmentError as Error, Result};

const SCHEMA: &str = "
CREATE TABLE challenges (
 id TEXT PRIMARY KEY, request_digest BLOB NOT NULL CHECK(length(request_digest)=32),
 challenge_digest BLOB NOT NULL CHECK(length(challenge_digest)=32),
 expires_at INTEGER NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','evidence_checked','certified')),
 evidence_digest BLOB, checked_at INTEGER
) STRICT;
CREATE TABLE certifications (
 id TEXT PRIMARY KEY, customer_scope TEXT NOT NULL, artifact TEXT NOT NULL,
 manifest_digest TEXT NOT NULL, certificate_digest BLOB NOT NULL CHECK(length(certificate_digest)=32),
 status TEXT NOT NULL CHECK(status IN ('active','disabled')), quota INTEGER NOT NULL CHECK(quota>0),
 challenge_id TEXT UNIQUE REFERENCES challenges(id), certificate_bytes BLOB,
 policy_digest BLOB, evidence_digest BLOB
) STRICT;
CREATE TABLE issuances (
 license_id TEXT PRIMARY KEY, certification_id TEXT NOT NULL REFERENCES certifications(id),
 generation INTEGER NOT NULL CHECK(generation>0),
 intent_digest BLOB CHECK(intent_digest IS NULL OR length(intent_digest)=32),
 license_bytes BLOB, signature_bytes BLOB,
 CHECK((license_bytes IS NULL)=(signature_bytes IS NULL)),
 UNIQUE(certification_id,generation)
) STRICT;
CREATE TABLE audit (
 sequence INTEGER PRIMARY KEY, event TEXT NOT NULL, object_id TEXT NOT NULL, timestamp INTEGER NOT NULL
) STRICT;
CREATE TABLE trust_policies (
 id TEXT PRIMARY KEY, sequence INTEGER NOT NULL CHECK(sequence>0), digest BLOB NOT NULL CHECK(length(digest)=32)
) STRICT;
PRAGMA user_version=2;";

// Old reservations remain non-resumable: they have no authenticated issuance intent.
const MIGRATE_V1: &str = "
ALTER TABLE issuances ADD COLUMN intent_digest BLOB CHECK(intent_digest IS NULL OR length(intent_digest)=32);
ALTER TABLE issuances ADD COLUMN license_bytes BLOB;
ALTER TABLE issuances ADD COLUMN signature_bytes BLOB;
PRAGMA user_version=2;";

#[derive(Debug, PartialEq, Eq)]
pub struct IssuedBundle {
    pub license: Vec<u8>,
    pub signature: Vec<u8>,
}

/// Immutable public intent. Include source record and signing-key fingerprints in its digest.
pub struct IssuanceIntent<'a> {
    pub certification_id: &'a str,
    pub certificate_bytes: &'a [u8],
    pub binding: &'a Binding,
    pub license_id: &'a str,
    pub generation: u64,
    pub digest: &'a [u8; 32],
}

pub struct Registry {
    connection: Connection,
}

fn db_error(_: rusqlite::Error) -> Error {
    Error::Registry
}

#[cfg(target_os = "linux")]
fn validate_path(path: &Path) -> Result<()> {
    use rustix::fs::{CWD, Mode, OFlags, ResolveFlags, openat2};
    if !path.is_absolute() {
        return Err(Error::Registry);
    }
    let parent = path.parent().ok_or(Error::Registry)?;
    let fd = openat2(
        CWD,
        parent,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
        Mode::empty(),
        ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
    )
    .map_err(|_| Error::Registry)?;
    let stat = rustix::fs::fstat(&fd).map_err(|_| Error::Registry)?;
    if stat.st_uid != rustix::process::geteuid().as_raw() || stat.st_mode & 0o077 != 0 {
        return Err(Error::Registry);
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn validate_path(_: &Path) -> Result<()> {
    Err(Error::Registry)
}

impl Registry {
    /// Create a consistent owner-only snapshot; never overwrite an existing backup.
    pub fn backup(&self, output: &Path) -> Result<()> {
        validate_path(output)?;
        let partial = output.with_extension(format!("partial-{}", uuid::Uuid::new_v4().simple()));
        let result = (|| {
            let file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&partial)
                .map_err(|_| Error::Registry)?;
            let mut destination = Connection::open_with_flags(
                &partial,
                OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )
            .map_err(db_error)?;
            rusqlite::backup::Backup::new(&self.connection, &mut destination)
                .map_err(db_error)?
                .run_to_completion(128, Duration::from_millis(10), None)
                .map_err(db_error)?;
            drop(destination);
            file.sync_all().map_err(|_| Error::Registry)?;
            rustix::fs::renameat_with(
                rustix::fs::CWD,
                &partial,
                rustix::fs::CWD,
                output,
                rustix::fs::RenameFlags::NOREPLACE,
            )
            .map_err(|_| Error::Conflict)?;
            std::fs::File::open(output.parent().ok_or(Error::Registry)?)
                .and_then(|parent| parent.sync_all())
                .map_err(|_| Error::Registry)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&partial);
        }
        result
    }
    pub fn accept_trust_policy(&mut self, policy: &VerifiedTrustPolicy) -> Result<()> {
        self.accept_policy(policy.policy_id(), policy.sequence(), policy.digest())
    }

    fn accept_policy(&mut self, id: &str, sequence: u64, digest: &[u8; 32]) -> Result<()> {
        use rusqlite::OptionalExtension;
        let sequence = i64::try_from(sequence).map_err(|_| Error::Input)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let previous: Option<(i64, Vec<u8>)> = transaction
            .query_row(
                "SELECT sequence,digest FROM trust_policies WHERE id=?1",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(db_error)?;
        if previous.is_some_and(|(old, old_digest)| {
            sequence < old || (sequence == old && old_digest != digest)
        }) {
            return Err(Error::Conflict);
        }
        transaction.execute("INSERT INTO trust_policies VALUES (?1,?2,?3) ON CONFLICT(id) DO UPDATE SET sequence=excluded.sequence,digest=excluded.digest",params![id,sequence,digest.as_slice()]).map_err(db_error)?;
        transaction.commit().map_err(db_error)
    }
    /// The containing directory must be dedicated, owner-only and free of symlinks.
    pub fn open(path: &Path, create: bool) -> Result<Self> {
        validate_path(path)?;
        if create {
            let file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(path)
                .map_err(|_| Error::Conflict)?;
            file.sync_all().map_err(|_| Error::Registry)?;
        }
        let metadata = fs::symlink_metadata(path).map_err(|_| Error::Registry)?;
        if !metadata.is_file()
            || metadata.nlink() != 1
            || metadata.mode() & 0o077 != 0
            || metadata.uid() != rustix::process::geteuid().as_raw()
        {
            return Err(Error::Registry);
        }
        let connection = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(db_error)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(db_error)?;
        connection.execute_batch("PRAGMA foreign_keys=ON; PRAGMA trusted_schema=OFF; PRAGMA synchronous=FULL; PRAGMA journal_mode=DELETE;").map_err(db_error)?;
        let mut registry = Self { connection };
        let version: u32 = registry
            .connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .map_err(db_error)?;
        if create && version == 0 {
            let transaction = registry
                .connection
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(db_error)?;
            transaction.execute_batch(SCHEMA).map_err(db_error)?;
            transaction.commit().map_err(db_error)?;
            std::fs::File::open(path.parent().ok_or(Error::Registry)?)
                .and_then(|directory| directory.sync_all())
                .map_err(|_| Error::Registry)?;
        } else if version == 1 {
            let transaction = registry
                .connection
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(db_error)?;
            // Another opener may have migrated while this connection waited for the lock.
            let locked_version: u32 = transaction
                .query_row("PRAGMA user_version", [], |row| row.get(0))
                .map_err(db_error)?;
            if locked_version == 1 {
                transaction.execute_batch(MIGRATE_V1).map_err(db_error)?;
            } else if locked_version != 2 {
                return Err(Error::Registry);
            }
            transaction.commit().map_err(db_error)?;
        } else if version != 2 {
            return Err(Error::Registry);
        }
        Ok(registry)
    }

    pub fn record_challenge(
        &mut self,
        request: &ValidatedRequest,
        challenge: &Challenge,
        now: u64,
    ) -> Result<()> {
        challenge.validate(request, now)?;
        self.insert_challenge(
            &challenge.challenge_id,
            request.digest(),
            &Sha256::digest(serde_json::to_vec(challenge).map_err(|_| Error::Input)?),
            challenge.expires_at,
            now,
        )
    }

    fn insert_challenge(
        &mut self,
        id: &str,
        request: &[u8],
        challenge: &[u8],
        expires: u64,
        now: u64,
    ) -> Result<()> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        transaction
            .execute(
                "INSERT INTO challenges VALUES (?1,?2,?3,?4,'pending',NULL,NULL)",
                params![
                    id,
                    request,
                    challenge,
                    i64::try_from(expires).map_err(|_| Error::Input)?
                ],
            )
            .map_err(|_| Error::Conflict)?;
        transaction
            .execute(
                "INSERT INTO audit(event,object_id,timestamp) VALUES ('challenge_created',?1,?2)",
                params![id, i64::try_from(now).map_err(|_| Error::Input)?],
            )
            .map_err(db_error)?;
        transaction.commit().map_err(db_error)
    }

    /// Consume a cryptographically checked transcript, without creating a certification.
    pub fn record_transcript(&mut self, proof: &VerifiedTranscript, now: u64) -> Result<()> {
        self.consume(
            proof.challenge_id(),
            proof.request_digest(),
            proof.challenge_digest(),
            proof.evidence_digest(),
            now,
        )
    }

    fn consume(
        &mut self,
        id: &str,
        request: &[u8],
        challenge: &[u8],
        evidence: &[u8],
        now: u64,
    ) -> Result<()> {
        let now = i64::try_from(now).map_err(|_| Error::Input)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let changed = transaction.execute("UPDATE challenges SET status='evidence_checked',evidence_digest=?1,checked_at=?2 WHERE id=?3 AND request_digest=?4 AND challenge_digest=?5 AND status='pending' AND expires_at>?2",
            params![evidence,now,id,request,challenge]).map_err(db_error)?;
        if changed != 1 {
            return Err(Error::Conflict);
        }
        transaction
            .execute(
                "INSERT INTO audit(event,object_id,timestamp) VALUES ('transcript_checked',?1,?2)",
                params![id, now],
            )
            .map_err(db_error)?;
        transaction.commit().map_err(db_error)
    }

    /// Commit EK trust and AK/DUK proof together; only then expose bytes for the signer.
    pub fn certify(
        &mut self,
        request: &ValidatedRequest,
        ek: &VerifiedEk,
        proof: &VerifiedTranscript,
        certification_id: &str,
        quota: u32,
        now: u64,
    ) -> Result<Vec<u8>> {
        if !identifier(certification_id)
            || quota == 0
            || request.digest() != ek.request_digest()
            || request.digest() != proof.request_digest()
        {
            return Err(Error::Binding);
        }
        let certified = crate::CertifiedDevice {
            format: crate::CERTIFIED_DEVICE_FORMAT.into(),
            format_version: crate::CERTIFIED_DEVICE_VERSION,
            certification_id: certification_id.into(),
            customer_scope_id: request.request().binding.customer_scope_id.clone(),
            artifact_id: request.request().binding.artifact_id.clone(),
            tpm_public: request.request().duk_public.clone(),
            policy_authority_name: request.request().binding.policy_authority_name.clone(),
        };
        let bytes = serde_json::to_vec(&certified).map_err(|_| Error::Input)?;
        let digest = Sha256::digest(&bytes);
        let binding = &request.request().binding;
        let now = i64::try_from(now).map_err(|_| Error::Input)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        if transaction.execute("UPDATE challenges SET status='certified',evidence_digest=?1,checked_at=?2 WHERE id=?3 AND request_digest=?4 AND challenge_digest=?5 AND expires_at>?2 AND (status='pending' OR (status='evidence_checked' AND evidence_digest=?1))",
            params![proof.evidence_digest().as_slice(),now,proof.challenge_id(),proof.request_digest().as_slice(),proof.challenge_digest().as_slice()]).map_err(db_error)? !=1 {
            return Err(Error::Conflict);
        }
        transaction
            .execute(
                "INSERT INTO certifications VALUES (?1,?2,?3,?4,?5,'active',?6,?7,?8,?9,?10)",
                params![
                    certification_id,
                    binding.customer_scope_id,
                    binding.artifact_id,
                    binding.manifest_sha256,
                    digest.as_slice(),
                    quota,
                    proof.challenge_id(),
                    bytes,
                    ek.policy_digest().as_slice(),
                    proof.evidence_digest().as_slice()
                ],
            )
            .map_err(|_| Error::Conflict)?;
        transaction.execute("INSERT INTO audit(event,object_id,timestamp) VALUES ('certification_created',?1,?2)",params![certification_id,now]).map_err(db_error)?;
        transaction.commit().map_err(db_error)?;
        Ok(bytes)
    }

    /// Publication retry returns the committed bytes, never a newly assigned identity.
    pub fn certification_bytes(&self, id: &str) -> Result<Vec<u8>> {
        if !identifier(id) {
            return Err(Error::Input);
        }
        self.connection
            .query_row(
                "SELECT certificate_bytes FROM certifications WHERE id=?1 AND status='active'",
                [id],
                |row| row.get(0),
            )
            .map_err(db_error)
    }

    pub fn challenge_status(&self, id: &str) -> Result<String> {
        if !identifier(id) {
            return Err(Error::Input);
        }
        self.connection
            .query_row("SELECT status FROM challenges WHERE id=?1", [id], |row| {
                row.get(0)
            })
            .map_err(db_error)
    }

    /// Reserve an issuance before signing. Failed publication retains its quota reservation.
    pub fn reserve_issuance(
        &mut self,
        certification_id: &str,
        certificate_bytes: &[u8],
        binding: &Binding,
        license_id: &str,
        generation: u64,
    ) -> Result<()> {
        if !identifier(certification_id) || !identifier(license_id) || generation == 0 {
            return Err(Error::Input);
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|duration| i64::try_from(duration.as_secs()).ok())
            .ok_or(Error::Registry)?;
        let digest = Sha256::digest(certificate_bytes);
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let admissible: bool = transaction.query_row("SELECT EXISTS(SELECT 1 FROM certifications c WHERE id=?1 AND customer_scope=?2 AND artifact=?3 AND manifest_digest=?4 AND certificate_digest=?5 AND status='active' AND (SELECT count(*) FROM issuances WHERE certification_id=c.id)<quota)",
            params![certification_id,binding.customer_scope_id,binding.artifact_id,binding.manifest_sha256,digest.as_slice()],|row|row.get(0)).map_err(db_error)?;
        if !admissible {
            return Err(Error::Binding);
        }
        transaction
            .execute(
                "INSERT INTO issuances(license_id,certification_id,generation) VALUES (?1,?2,?3)",
                params![
                    license_id,
                    certification_id,
                    i64::try_from(generation).map_err(|_| Error::Input)?
                ],
            )
            .map_err(|_| Error::Conflict)?;
        transaction
            .execute(
                "INSERT INTO audit(event,object_id,timestamp) VALUES ('issuance_reserved',?1,?2)",
                params![license_id, now],
            )
            .map_err(db_error)?;
        transaction.commit().map_err(db_error)
    }

    /// Reserve once or resume exactly the same intent, including after process death.
    /// A retry of a finalized issuance returns its durable, byte-identical signed bundle.
    pub fn begin_issuance(&mut self, intent: &IssuanceIntent<'_>) -> Result<Option<IssuedBundle>> {
        validate_intent(intent)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        check_issuance_binding(&transaction, intent)?;
        let existing = transaction.query_row(
            "SELECT certification_id,generation,intent_digest,license_bytes,signature_bytes FROM issuances WHERE license_id=?1",
            [intent.license_id],
            |row| Ok((row.get::<_,String>(0)?,row.get::<_,i64>(1)?,row.get::<_,Option<Vec<u8>>>(2)?,row.get::<_,Option<Vec<u8>>>(3)?,row.get::<_,Option<Vec<u8>>>(4)?)),
        ).optional().map_err(db_error)?;
        if let Some((certification, generation, digest, license, signature)) = existing {
            if certification != intent.certification_id
                || generation != intent.generation as i64
                || digest.as_deref() != Some(intent.digest.as_slice())
            {
                return Err(Error::Conflict);
            }
            return decode_bundle(license, signature);
        }
        let available: bool = transaction.query_row(
            "SELECT (SELECT count(*) FROM issuances WHERE certification_id=?1)<quota FROM certifications WHERE id=?1",
            [intent.certification_id], |row| row.get(0),
        ).map_err(db_error)?;
        if !available {
            return Err(Error::Binding);
        }
        transaction.execute(
            "INSERT INTO issuances(license_id,certification_id,generation,intent_digest) VALUES (?1,?2,?3,?4)",
            params![intent.license_id,intent.certification_id,intent.generation as i64,intent.digest.as_slice()],
        ).map_err(|_| Error::Conflict)?;
        transaction
            .execute(
                "INSERT INTO audit(event,object_id,timestamp) VALUES ('issuance_reserved',?1,?2)",
                params![intent.license_id, current_timestamp()?],
            )
            .map_err(db_error)?;
        transaction.commit().map_err(db_error)?;
        Ok(None)
    }

    /// Commit before publication. Concurrent retries converge on the first committed bundle.
    /// No plaintext DEK or passphrase is stored in this ledger.
    pub fn finalize_issuance(
        &mut self,
        intent: &IssuanceIntent<'_>,
        candidate: IssuedBundle,
    ) -> Result<IssuedBundle> {
        validate_intent(intent)?;
        validate_bundle(&candidate)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        check_issuance_binding(&transaction, intent)?;
        let (license, signature): (Option<Vec<u8>>,Option<Vec<u8>>) = transaction.query_row(
            "SELECT license_bytes,signature_bytes FROM issuances WHERE license_id=?1 AND certification_id=?2 AND generation=?3 AND intent_digest=?4",
            params![intent.license_id,intent.certification_id,intent.generation as i64,intent.digest.as_slice()],
            |row| Ok((row.get(0)?,row.get(1)?)),
        ).map_err(|_| Error::Conflict)?;
        if let Some(bundle) = decode_bundle(license, signature)? {
            return Ok(bundle);
        }
        transaction
            .execute(
                "UPDATE issuances SET license_bytes=?1,signature_bytes=?2 WHERE license_id=?3",
                params![candidate.license, candidate.signature, intent.license_id],
            )
            .map_err(db_error)?;
        transaction
            .execute(
                "INSERT INTO audit(event,object_id,timestamp) VALUES ('issuance_finalized',?1,?2)",
                params![intent.license_id, current_timestamp()?],
            )
            .map_err(db_error)?;
        transaction.commit().map_err(db_error)?;
        Ok(candidate)
    }

    pub fn disable_certification(&mut self, id: &str, now: u64) -> Result<()> {
        if !identifier(id) {
            return Err(Error::Input);
        }
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        if transaction
            .execute(
                "UPDATE certifications SET status='disabled' WHERE id=?1 AND status='active'",
                [id],
            )
            .map_err(db_error)?
            != 1
        {
            return Err(Error::Conflict);
        }
        transaction.execute("INSERT INTO audit(event,object_id,timestamp) VALUES ('certification_disabled',?1,?2)",params![id,i64::try_from(now).map_err(|_| Error::Input)?]).map_err(db_error)?;
        transaction.commit().map_err(db_error)
    }
}

fn current_timestamp() -> Result<i64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_secs()).ok())
        .ok_or(Error::Registry)
}

fn validate_intent(intent: &IssuanceIntent<'_>) -> Result<()> {
    if !identifier(intent.certification_id)
        || !identifier(intent.license_id)
        || intent.generation == 0
        || i64::try_from(intent.generation).is_err()
    {
        return Err(Error::Input);
    }
    Ok(())
}

fn check_issuance_binding(
    transaction: &rusqlite::Transaction<'_>,
    intent: &IssuanceIntent<'_>,
) -> Result<()> {
    let valid: bool = transaction.query_row(
        "SELECT EXISTS(SELECT 1 FROM certifications WHERE id=?1 AND customer_scope=?2 AND artifact=?3 AND manifest_digest=?4 AND certificate_digest=?5 AND status='active')",
        params![intent.certification_id,intent.binding.customer_scope_id,intent.binding.artifact_id,intent.binding.manifest_sha256,Sha256::digest(intent.certificate_bytes).as_slice()],
        |row| row.get(0),
    ).map_err(db_error)?;
    if !valid {
        return Err(Error::Binding);
    }
    Ok(())
}

fn validate_bundle(bundle: &IssuedBundle) -> Result<()> {
    if bundle.license.is_empty()
        || bundle.signature.is_empty()
        || bundle.license.len() > super::format::MAX_BUNDLE_BYTES
        || bundle.signature.len() > super::format::MAX_BUNDLE_BYTES
    {
        return Err(Error::Input);
    }
    Ok(())
}

fn decode_bundle(
    license: Option<Vec<u8>>,
    signature: Option<Vec<u8>>,
) -> Result<Option<IssuedBundle>> {
    match (license, signature) {
        (None, None) => Ok(None),
        (Some(license), Some(signature)) => {
            let bundle = IssuedBundle { license, signature };
            validate_bundle(&bundle)?;
            Ok(Some(bundle))
        }
        _ => Err(Error::Registry),
    }
}

#[cfg(test)]
mod tests {
    use super::super::format::hex;
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn private_directory() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
        directory
    }

    fn insert_test_certificate(registry: &Registry) -> Binding {
        let binding = Binding {
            customer_scope_id: "customer".into(),
            artifact_id: hex(&[1; 16]),
            manifest_sha256: hex(&[2; 32]),
            policy_authority_name: hex(&[3; 34]),
        };
        registry.connection.execute("INSERT INTO certifications (id,customer_scope,artifact,manifest_digest,certificate_digest,status,quota) VALUES ('cert',?1,?2,?3,?4,'active',1)",params![binding.customer_scope_id,binding.artifact_id,binding.manifest_sha256,Sha256::digest(b"certificate").as_slice()]).unwrap();
        binding
    }

    fn test_intent(binding: &Binding) -> IssuanceIntent<'_> {
        IssuanceIntent {
            certification_id: "cert",
            certificate_bytes: b"certificate",
            binding,
            license_id: "license",
            generation: 1,
            digest: &[7; 32],
        }
    }

    fn test_bundle(value: u8) -> IssuedBundle {
        IssuedBundle {
            license: vec![value; 10],
            signature: vec![value; 5],
        }
    }

    #[test]
    fn issuance_recovers_both_crash_windows_and_rejects_changed_intent() {
        let directory = private_directory();
        let path = directory.path().join("registry.sqlite");
        let mut registry = Registry::open(&path, true).unwrap();
        let binding = insert_test_certificate(&registry);
        let intent = test_intent(&binding);
        assert!(registry.begin_issuance(&intent).unwrap().is_none());
        drop(registry); // Simulated death before signing/finalization.
        let mut registry = Registry::open(&path, false).unwrap();
        assert!(registry.begin_issuance(&intent).unwrap().is_none());
        let changed = IssuanceIntent {
            digest: &[8; 32],
            ..test_intent(&binding)
        };
        assert!(registry.begin_issuance(&changed).is_err());
        assert!(
            registry
                .finalize_issuance(&changed, test_bundle(9))
                .is_err()
        );
        let changed_generation = IssuanceIntent {
            generation: 2,
            ..test_intent(&binding)
        };
        assert!(registry.begin_issuance(&changed_generation).is_err());
        let other_license = IssuanceIntent {
            license_id: "other-license",
            ..test_intent(&binding)
        };
        assert!(registry.begin_issuance(&other_license).is_err());
        let committed = registry.finalize_issuance(&intent, test_bundle(1)).unwrap();
        drop(registry); // Simulated death after commit, before filesystem publication.
        let mut registry = Registry::open(&path, false).unwrap();
        assert_eq!(
            registry.begin_issuance(&intent).unwrap(),
            Some(test_bundle(1))
        );
        assert_eq!(
            registry.finalize_issuance(&intent, test_bundle(2)).unwrap(),
            committed
        );
        let count: i64 = registry
            .connection
            .query_row("SELECT count(*) FROM issuances", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
        registry.disable_certification("cert", 20).unwrap();
        assert!(registry.begin_issuance(&intent).is_err());
        assert!(registry.finalize_issuance(&intent, test_bundle(3)).is_err());
    }

    #[test]
    fn concurrent_issuance_retries_publish_one_committed_bundle() {
        let directory = private_directory();
        let path = directory.path().join("registry.sqlite");
        let registry = Registry::open(&path, true).unwrap();
        let binding = insert_test_certificate(&registry);
        drop(registry);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let threads: Vec<_> = (1..=2)
            .map(|value| {
                let path = path.clone();
                let binding = binding.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let mut registry = Registry::open(&path, false).unwrap();
                    let intent = test_intent(&binding);
                    assert!(registry.begin_issuance(&intent).unwrap().is_none());
                    barrier.wait();
                    registry
                        .finalize_issuance(&intent, test_bundle(value))
                        .unwrap()
                })
            })
            .collect();
        let bundles: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(bundles[0], bundles[1]);
        let registry = Registry::open(&path, false).unwrap();
        let count: i64 = registry
            .connection
            .query_row(
                "SELECT count(*) FROM audit WHERE event='issuance_finalized'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn schema_migration_preserves_old_non_resumable_reservations() {
        let directory = private_directory();
        let path = directory.path().join("registry.sqlite");
        let registry = Registry::open(&path, true).unwrap();
        let binding = insert_test_certificate(&registry);
        // Rebuild only this test table to represent the historical schema.
        registry.connection.execute_batch("DROP TABLE issuances; CREATE TABLE issuances(license_id TEXT PRIMARY KEY,certification_id TEXT NOT NULL REFERENCES certifications(id),generation INTEGER NOT NULL CHECK(generation>0),UNIQUE(certification_id,generation)) STRICT; INSERT INTO issuances VALUES ('license','cert',1); PRAGMA user_version=1;").unwrap();
        drop(registry);
        let mut registry = Registry::open(&path, false).unwrap();
        assert!(registry.begin_issuance(&test_intent(&binding)).is_err());
        let count: i64 = registry
            .connection
            .query_row("SELECT count(*) FROM issuances", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
        let version: u32 = registry
            .connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 2);
    }

    #[test]
    fn consumes_once_across_reopen_and_rejects_expired_and_mixed_binding() {
        let directory = private_directory();
        let path = directory.path().join("registry.sqlite");
        let mut registry = Registry::open(&path, true).unwrap();
        registry
            .insert_challenge("challenge", &[1; 32], &[2; 32], 100, 10)
            .unwrap();
        assert!(
            registry
                .consume("challenge", &[9; 32], &[2; 32], &[3; 32], 20)
                .is_err()
        );
        assert!(
            registry
                .consume("challenge", &[1; 32], &[9; 32], &[3; 32], 20)
                .is_err()
        );
        assert!(
            registry
                .consume("challenge", &[1; 32], &[2; 32], &[3; 32], 100)
                .is_err()
        );
        registry
            .consume("challenge", &[1; 32], &[2; 32], &[3; 32], 20)
            .unwrap();
        drop(registry);
        let mut restored = Registry::open(&path, false).unwrap();
        assert_eq!(
            restored.challenge_status("challenge").unwrap(),
            "evidence_checked"
        );
        assert!(
            restored
                .consume("challenge", &[1; 32], &[2; 32], &[3; 32], 21)
                .is_err()
        );
        restored.accept_policy("trust", 2, &[4; 32]).unwrap();
        assert!(restored.accept_policy("trust", 1, &[3; 32]).is_err());
        assert!(restored.accept_policy("trust", 2, &[3; 32]).is_err());
        restored.accept_policy("trust", 2, &[4; 32]).unwrap();
        let backup_path = directory.path().join("backup.sqlite");
        restored.backup(&backup_path).unwrap();
        assert!(restored.backup(&backup_path).is_err());
        let mut backup = Registry::open(&backup_path, false).unwrap();
        assert_eq!(
            backup.challenge_status("challenge").unwrap(),
            "evidence_checked"
        );
        assert!(
            backup
                .consume("challenge", &[1; 32], &[2; 32], &[3; 32], 21)
                .is_err()
        );
    }

    #[test]
    fn concurrent_consumers_have_exactly_one_winner() {
        let directory = private_directory();
        let path = directory.path().join("registry.sqlite");
        let mut registry = Registry::open(&path, true).unwrap();
        registry
            .insert_challenge("challenge", &[1; 32], &[2; 32], 100, 10)
            .unwrap();
        drop(registry);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let threads: Vec<_> = (0..2)
            .map(|_| {
                let path = path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let mut r = Registry::open(&path, false).unwrap();
                    barrier.wait();
                    r.consume("challenge", &[1; 32], &[2; 32], &[3; 32], 20)
                        .is_ok()
                })
            })
            .collect();
        let winners = threads
            .into_iter()
            .map(|t| usize::from(t.join().unwrap()))
            .sum::<usize>();
        assert_eq!(winners, 1);
    }

    #[test]
    fn admission_checks_exact_certificate_binding_quota_and_disable() {
        let directory = private_directory();
        let path = directory.path().join("registry.sqlite");
        let mut registry = Registry::open(&path, true).unwrap();
        let binding = Binding {
            customer_scope_id: "customer".into(),
            artifact_id: hex(&[1; 16]),
            manifest_sha256: hex(&[2; 32]),
            policy_authority_name: hex(&[3; 34]),
        };
        assert!(
            registry
                .reserve_issuance("cert", b"certificate", &binding, "license", 1)
                .is_err()
        );
        // Fixture bypass is test-only; real insertion requires EK and transcript wrappers.
        registry.connection.execute("INSERT INTO certifications (id,customer_scope,artifact,manifest_digest,certificate_digest,status,quota) VALUES ('cert',?1,?2,?3,?4,'active',2)",
            params![binding.customer_scope_id,binding.artifact_id,binding.manifest_sha256,Sha256::digest(b"certificate").as_slice()]).unwrap();
        assert!(
            registry
                .reserve_issuance("cert", b"changed", &binding, "license", 1)
                .is_err()
        );
        let mut changed = binding.clone();
        changed.artifact_id = hex(&[4; 16]);
        assert!(
            registry
                .reserve_issuance("cert", b"certificate", &changed, "license", 1)
                .is_err()
        );
        registry
            .reserve_issuance("cert", b"certificate", &binding, "license", 1)
            .unwrap();
        assert!(
            registry
                .reserve_issuance("cert", b"certificate", &binding, "license", 1)
                .is_err()
        );
        registry.disable_certification("cert", 20).unwrap();
        assert!(
            registry
                .reserve_issuance("cert", b"certificate", &binding, "license2", 2)
                .is_err()
        );
    }
}
