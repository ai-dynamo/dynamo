// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed protected-model package primitives.
//!
//! This module deliberately contains no vLLM or Python policy. It owns the
//! bounded wire format, cryptographic verification, authenticated records, and
//! safe filesystem operations used by backend adapters.

mod cancellation;
mod detector;
mod format;
mod issuer_record;
mod license;
#[cfg(target_os = "linux")]
mod materialize;
#[cfg(target_os = "linux")]
mod persistence;
mod records;
mod secret;
#[cfg(all(target_os = "linux", feature = "tpm2"))]
mod tpm;

pub use cancellation::CancellationToken;
pub use detector::{ModelProtectionKind, detect_model_protection};
pub use format::{
    Encryption, MANIFEST_SIGNATURE_DOMAIN, MAX_FILE_BYTES, MAX_SAFETENSORS_INDEX_BYTES, Manifest,
    ModelIdentity, ProtectedFile, PublicFile, RuntimeRequirements, SignatureEnvelope,
    VerifiedManifest, enforce_runtime_version, is_allowed_public_metadata, manifest_digest,
    manifest_signature_payload, parse_manifest, validate_safetensors_index, verify_manifest,
};
pub use issuer_record::{
    ISSUER_RECORD_FORMAT, ISSUER_RECORD_SIGNATURE_DOMAIN, ISSUER_RECORD_VERSION, IssuerDekRecord,
    MAX_ISSUER_RECORD_BYTES, SignedIssuerDekRecord, issuer_record_signature_payload,
    verify_issuer_record,
};
pub use license::{
    AuthorizedModel, Entitlement, LICENSE_SIGNATURE_DOMAIN, License, TPM_OAEP_LABEL,
    TPM_POLICY_REF_HEX, TPM_POLICY_SIGNATURE, TPM_PROFILE, TpmRecipient,
    approved_rsa_decrypt_policy, license_signature_payload, policy_authorize_auth_policy,
    rsa_decrypt_cp_hash,
};
#[cfg(target_os = "linux")]
pub use materialize::{
    SecureModelSession, load_authorized_model, load_verified_manifest, secure_model_root,
    validate_namespace,
};
#[cfg(target_os = "linux")]
pub use persistence::enforce_process_persistence_policy;
pub use records::{EncryptResult, encrypt_records};
pub use secret::SecretDek;
#[cfg(all(target_os = "linux", feature = "tpm2"))]
pub use tpm::{materialize_tpm_model, prepare_tpm_model, unwrap_tpm_dek};

use std::fmt;

use thiserror::Error;

/// Stable machine-readable failures for logs and the Python boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    SecurePackageInvalid,
    ManifestSignatureInvalid,
    PackageIntegrityInvalid,
    DecryptionFailed,
    TmpfsInvalid,
    CleanupFailed,
    SecureIoError,
    LicenseInvalid,
    LicenseSignatureInvalid,
    LicenseBindingMismatch,
    RuntimeUnsupported,
    TmpfsInsufficient,
    SessionConflict,
    TpmUnavailable,
    TpmAuthorizationFailed,
    SecretMemoryUnavailable,
    HostPolicyInvalid,
    IssuerRecordInvalid,
    MaterializationCancelled,
}

impl ErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SecurePackageInvalid => "SECURE_PACKAGE_INVALID",
            Self::ManifestSignatureInvalid => "MANIFEST_SIGNATURE_INVALID",
            Self::PackageIntegrityInvalid => "PACKAGE_INTEGRITY_INVALID",
            Self::DecryptionFailed => "DECRYPTION_FAILED",
            Self::TmpfsInvalid => "TMPFS_INVALID",
            Self::CleanupFailed => "CLEANUP_FAILED",
            Self::SecureIoError => "SECURE_IO_ERROR",
            Self::LicenseInvalid => "LICENSE_INVALID",
            Self::LicenseSignatureInvalid => "LICENSE_SIGNATURE_INVALID",
            Self::LicenseBindingMismatch => "LICENSE_BINDING_MISMATCH",
            Self::RuntimeUnsupported => "RUNTIME_UNSUPPORTED",
            Self::TmpfsInsufficient => "TMPFS_INSUFFICIENT",
            Self::SessionConflict => "SESSION_CONFLICT",
            Self::TpmUnavailable => "TPM_UNAVAILABLE",
            Self::TpmAuthorizationFailed => "TPM_AUTHORIZATION_FAILED",
            Self::SecretMemoryUnavailable => "SECRET_MEMORY_UNAVAILABLE",
            Self::HostPolicyInvalid => "HOST_POLICY_INVALID",
            Self::IssuerRecordInvalid => "ISSUER_RECORD_INVALID",
            Self::MaterializationCancelled => "MATERIALIZATION_CANCELLED",
        }
    }
}

/// Stable, sanitized failures exposed by the protection boundary.
#[derive(Error)]
pub enum ProtectionError {
    #[error("SECURE_PACKAGE_INVALID")]
    InvalidPackage(&'static str),
    #[error("MANIFEST_SIGNATURE_INVALID")]
    ManifestSignatureInvalid,
    #[error("PACKAGE_INTEGRITY_INVALID")]
    PackageIntegrityMismatch,
    #[error("DECRYPTION_FAILED")]
    DecryptionFailed,
    #[error("TMPFS_INVALID")]
    TmpfsInvalid,
    #[error("CLEANUP_FAILED")]
    CleanupFailed,
    #[error("SECURE_IO_ERROR")]
    Io,
    #[error("LICENSE_INVALID")]
    LicenseInvalid(&'static str),
    #[error("LICENSE_SIGNATURE_INVALID")]
    LicenseSignatureInvalid,
    #[error("LICENSE_BINDING_MISMATCH")]
    LicenseBindingMismatch,
    #[error("RUNTIME_UNSUPPORTED")]
    RuntimeUnsupported,
    #[error("TMPFS_INSUFFICIENT")]
    TmpfsInsufficient,
    #[error("SESSION_CONFLICT")]
    SessionConflict,
    #[error("TPM_UNAVAILABLE")]
    TpmUnavailable,
    #[error("TPM_AUTHORIZATION_FAILED")]
    TpmAuthorizationFailed,
    #[error("SECRET_MEMORY_UNAVAILABLE")]
    SecretMemoryUnavailable,
    #[error("HOST_POLICY_INVALID")]
    HostPolicyInvalid,
    #[error("ISSUER_RECORD_INVALID")]
    IssuerRecordInvalid,
    #[error("MATERIALIZATION_CANCELLED")]
    MaterializationCancelled,
}

impl fmt::Debug for ProtectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code().as_str())
    }
}

impl From<std::io::Error> for ProtectionError {
    fn from(_: std::io::Error) -> Self {
        Self::Io
    }
}

impl ProtectionError {
    pub const fn code(&self) -> ErrorCode {
        match self {
            Self::InvalidPackage(_) => ErrorCode::SecurePackageInvalid,
            Self::ManifestSignatureInvalid => ErrorCode::ManifestSignatureInvalid,
            Self::PackageIntegrityMismatch => ErrorCode::PackageIntegrityInvalid,
            Self::DecryptionFailed => ErrorCode::DecryptionFailed,
            Self::TmpfsInvalid => ErrorCode::TmpfsInvalid,
            Self::CleanupFailed => ErrorCode::CleanupFailed,
            Self::Io => ErrorCode::SecureIoError,
            Self::LicenseInvalid(_) => ErrorCode::LicenseInvalid,
            Self::LicenseSignatureInvalid => ErrorCode::LicenseSignatureInvalid,
            Self::LicenseBindingMismatch => ErrorCode::LicenseBindingMismatch,
            Self::RuntimeUnsupported => ErrorCode::RuntimeUnsupported,
            Self::TmpfsInsufficient => ErrorCode::TmpfsInsufficient,
            Self::SessionConflict => ErrorCode::SessionConflict,
            Self::TpmUnavailable => ErrorCode::TpmUnavailable,
            Self::TpmAuthorizationFailed => ErrorCode::TpmAuthorizationFailed,
            Self::SecretMemoryUnavailable => ErrorCode::SecretMemoryUnavailable,
            Self::HostPolicyInvalid => ErrorCode::HostPolicyInvalid,
            Self::IssuerRecordInvalid => ErrorCode::IssuerRecordInvalid,
            Self::MaterializationCancelled => ErrorCode::MaterializationCancelled,
        }
    }

    /// A bounded, non-secret reason suitable for internal structured logs.
    pub const fn sanitized_reason(&self) -> Option<&'static str> {
        match self {
            Self::InvalidPackage(reason) | Self::LicenseInvalid(reason) => Some(reason),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, ProtectionError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_expose_stable_codes_without_io_details() {
        let error = ProtectionError::from(std::io::Error::other("/secret/model/path"));
        assert_eq!(error.to_string(), "SECURE_IO_ERROR");
        assert_eq!(format!("{error:?}"), "SECURE_IO_ERROR");
        assert!(std::error::Error::source(&error).is_none());
        assert_eq!(error.code().as_str(), "SECURE_IO_ERROR");
        assert_eq!(error.sanitized_reason(), None);
    }
}
