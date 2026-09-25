// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Offline software issuer key handling.
//!
//! This module is intentionally small: the issuer owns the key bytes only for
//! the duration of one operation and callers receive signatures/wrapped bytes,
//! never a key export API.

use std::path::Path;

use aes_kw::KekAes256;
use ed25519_dalek::{Signer, SigningKey};
use p256::ecdsa::{SigningKey as P256SigningKey, signature::hazmat::PrehashSigner};
use pkcs8::DecodePrivateKey;
use rsa::{Oaep, RsaPublicKey};
use sha2::Sha256;
use zeroize::Zeroizing;

use super::secure_file::read_bounded;

const MAX_PRIVATE_KEY_BYTES: u64 = 16 * 1024;
const MAX_PASSPHRASE_BYTES: u64 = 4096;

/// Locks all issuer-process pages for this short-lived, dedicated process.
/// The issuer must run with a bounded, non-zero `RLIMIT_MEMLOCK`; a pageable
/// fallback would violate the key-custody contract.
#[cfg(target_os = "linux")]
pub fn lock_process_memory() -> Result<(), &'static str> {
    rustix::process::setrlimit(
        rustix::process::Resource::Core,
        rustix::process::Rlimit {
            current: Some(0),
            maximum: Some(0),
        },
    )
    .map_err(|_| "ISSUER_MEMORY_UNAVAILABLE")?;
    rustix::process::set_dumpable_behavior(rustix::process::DumpableBehavior::NotDumpable)
        .map_err(|_| "ISSUER_MEMORY_UNAVAILABLE")?;
    rustix::mm::mlockall(rustix::mm::MlockAllFlags::CURRENT | rustix::mm::MlockAllFlags::FUTURE)
        .map_err(|_| "ISSUER_MEMORY_UNAVAILABLE")
}

#[cfg(not(target_os = "linux"))]
pub fn lock_process_memory() -> Result<(), &'static str> {
    Err("ISSUER_MEMORY_UNAVAILABLE")
}

pub fn read_passphrase(path: &Path) -> Result<Zeroizing<Vec<u8>>, &'static str> {
    let mut bytes =
        read_bounded(path, MAX_PASSPHRASE_BYTES, true).map_err(|_| "ISSUER_PASSPHRASE_INVALID")?;
    while matches!(bytes.last(), Some(b'\n' | b'\r')) {
        bytes.pop();
    }
    if bytes.is_empty() {
        return Err("ISSUER_PASSPHRASE_INVALID");
    }
    Ok(Zeroizing::new(bytes))
}

pub fn read_signing_key(path: &Path, passphrase: &[u8]) -> Result<SigningKey, &'static str> {
    let bytes = Zeroizing::new(
        read_bounded(path, MAX_PRIVATE_KEY_BYTES, true).map_err(|_| "ISSUER_KEY_INVALID")?,
    );
    let key = if bytes.starts_with(b"-----BEGIN") {
        let pem = std::str::from_utf8(&bytes).map_err(|_| "ISSUER_KEY_INVALID")?;
        SigningKey::from_pkcs8_encrypted_pem(pem, passphrase)
    } else {
        SigningKey::from_pkcs8_encrypted_der(&bytes, passphrase)
    }
    .map_err(|_| "ISSUER_KEY_DECRYPT_FAILED")?;
    Ok(key)
}

pub fn read_policy_key(path: &Path, passphrase: &[u8]) -> Result<P256SigningKey, &'static str> {
    let bytes = Zeroizing::new(
        read_bounded(path, MAX_PRIVATE_KEY_BYTES, true).map_err(|_| "ISSUER_KEY_INVALID")?,
    );
    let key = if bytes.starts_with(b"-----BEGIN") {
        let pem = std::str::from_utf8(&bytes).map_err(|_| "ISSUER_KEY_INVALID")?;
        P256SigningKey::from_pkcs8_encrypted_pem(pem, passphrase)
    } else {
        P256SigningKey::from_pkcs8_encrypted_der(&bytes, passphrase)
    }
    .map_err(|_| "ISSUER_KEY_DECRYPT_FAILED")?;
    Ok(key)
}

pub fn read_kek(path: &Path) -> Result<Zeroizing<[u8; 32]>, &'static str> {
    Ok(Zeroizing::new(
        read_bounded(path, 32, true)
            .map_err(|_| "ISSUER_KEY_INVALID")?
            .try_into()
            .map_err(|_| "ISSUER_KEY_INVALID")?,
    ))
}

pub fn sign_ed25519(key: &SigningKey, message: &[u8]) -> [u8; 64] {
    key.sign(message).to_bytes()
}

pub fn sign_p256_digest(key: &P256SigningKey, digest: &[u8; 32]) -> Result<[u8; 64], &'static str> {
    let signature: p256::ecdsa::Signature = key
        .sign_prehash(digest)
        .map_err(|_| "ISSUER_CRYPTO_FAILED")?;
    Ok(signature.to_bytes().into())
}

pub fn wrap_dek(kek: &[u8; 32], dek: &[u8; 32]) -> Result<[u8; 40], &'static str> {
    let kek = KekAes256::from(*kek);
    let mut wrapped = [0_u8; 40];
    kek.wrap_with_padding(dek, &mut wrapped)
        .map_err(|_| "ISSUER_CRYPTO_FAILED")?;
    Ok(wrapped)
}

pub fn unwrap_dek(kek: &[u8; 32], wrapped: &[u8]) -> Result<Zeroizing<[u8; 32]>, &'static str> {
    if wrapped.len() != 40 {
        return Err("ISSUER_RECORD_INVALID");
    }
    let kek = KekAes256::from(*kek);
    let mut dek = [0_u8; 32];
    let plain = kek
        .unwrap_with_padding(wrapped, &mut dek)
        .map_err(|_| "ISSUER_DEK_UNWRAP_FAILED")?;
    if plain.len() != dek.len() {
        return Err("ISSUER_DEK_UNWRAP_FAILED");
    }
    Ok(Zeroizing::new(dek))
}

pub fn wrap_to_tpm(
    modulus: &[u8; 256],
    dek: &[u8; 32],
    label: &[u8],
) -> Result<[u8; 256], &'static str> {
    let public = RsaPublicKey::new(
        rsa::BigUint::from_bytes_be(modulus),
        rsa::BigUint::from(65_537_u32),
    )
    .map_err(|_| "CERTIFIED_DEVICE_INVALID")?;
    let label = std::str::from_utf8(label).map_err(|_| "ISSUER_CONFIG_INVALID")?;
    let wrapped = public
        .encrypt(
            &mut rsa::rand_core::OsRng,
            Oaep::new_with_label::<Sha256, _>(label),
            dek,
        )
        .map_err(|_| "ISSUER_RECIPIENT_WRAP_FAILED")?;
    wrapped.try_into().map_err(|_| "ISSUER_OUTPUT_INVALID")
}

#[cfg(test)]
mod tests {
    use super::*;
    use pkcs8::{EncodePrivateKey, LineEnding};
    use rsa::{RsaPrivateKey, traits::PublicKeyParts};
    use tempfile::tempdir;

    #[test]
    fn aes_kwp_roundtrip_preserves_dek() {
        let kek = [0x11; 32];
        let dek = [0x22; 32];
        let wrapped = wrap_dek(&kek, &dek).expect("wrap");
        assert_eq!(wrapped.len(), 40);
        assert_eq!(&*unwrap_dek(&kek, &wrapped).expect("unwrap"), &dek);
        assert!(unwrap_dek(&[0x33; 32], &wrapped).is_err());
    }

    #[test]
    fn encrypted_ed25519_pkcs8_is_loaded_only_from_private_file() {
        let directory = tempdir().unwrap();
        let key = SigningKey::from_bytes(&[0x44; 32]);
        let pem = key
            .to_pkcs8_encrypted_pem(rsa::rand_core::OsRng, b"pass", LineEnding::LF)
            .unwrap();
        let key_path = directory.path().join("package.pem");
        std::fs::write(&key_path, pem.as_bytes()).unwrap();
        std::fs::set_permissions(
            &key_path,
            std::os::unix::fs::PermissionsExt::from_mode(0o600),
        )
        .unwrap();
        assert!(read_signing_key(&key_path, b"wrong").is_err());
        let loaded = read_signing_key(&key_path, b"pass").unwrap();
        assert_eq!(loaded.verifying_key(), key.verifying_key());
    }

    #[test]
    fn rsa_wrap_uses_the_exact_tpm_label() {
        let private = RsaPrivateKey::new(&mut rsa::rand_core::OsRng, 2048).unwrap();
        let public = private.to_public_key();
        let modulus: [u8; 256] = public.n().to_bytes_be().try_into().unwrap();
        let dek = [0x55; 32];
        let wrapped = wrap_to_tpm(&modulus, &dek, b"model-protection-dek-v1\0").unwrap();
        let recovered = private
            .decrypt(
                Oaep::new_with_label::<Sha256, _>("model-protection-dek-v1\0"),
                &wrapped,
            )
            .unwrap();
        assert_eq!(recovered, dek);
        assert!(
            private
                .decrypt(Oaep::new_with_label::<Sha256, _>("wrong\0"), &wrapped,)
                .is_err()
        );
    }
}
