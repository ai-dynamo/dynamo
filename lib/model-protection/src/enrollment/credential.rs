// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use openssl::symm::{Cipher, Crypter, Mode};
use ring::{
    hmac,
    rand::{SecureRandom, SystemRandom},
};
use rsa::{Oaep, RsaPublicKey};
use sha2::Sha256;
use zeroize::Zeroizing;

use super::format::{Challenge, MAX_CHALLENGE_TTL_SECONDS, ValidatedRequest, hex};
use super::trust::VerifiedEk;
use super::{EnrollmentError as Error, Result};

pub struct ActivationChallenge {
    pub challenge: Challenge,
    secret: Zeroizing<[u8; 32]>,
}

impl ActivationChallenge {
    /// Activation secret only; not an issuer or model decryption key.
    pub fn activation_secret(&self) -> &[u8; 32] {
        &self.secret
    }
}

pub fn make_challenge(
    request: &ValidatedRequest,
    ek: &VerifiedEk,
    now: u64,
    ttl: u64,
) -> Result<ActivationChallenge> {
    if request.digest() != ek.request_digest() || ttl == 0 || ttl > MAX_CHALLENGE_TTL_SECONDS {
        return Err(Error::Binding);
    }
    let mut secret = Zeroizing::new([0; 32]);
    let mut seed = Zeroizing::new([0; 16]);
    let mut nonce = [0; 32];
    let random = SystemRandom::new();
    random.fill(&mut *secret).map_err(|_| Error::Proof)?;
    random.fill(&mut *seed).map_err(|_| Error::Proof)?;
    random.fill(&mut nonce).map_err(|_| Error::Proof)?;
    let public = BASE64
        .decode(&request.request().ek_public)
        .map_err(|_| Error::Input)?;
    let modulus: &[u8; 256] = public[58..].try_into().map_err(|_| Error::Input)?;
    let (credential, encrypted_seed) =
        protect_credential(modulus, request.ak_name(), &seed, &secret)?;
    Ok(ActivationChallenge {
        challenge: Challenge {
            format: "model-protection-enrollment-challenge".into(),
            format_version: 1,
            challenge_id: uuid::Uuid::new_v4().simple().to_string(),
            request_sha256: hex(request.digest()),
            nonce: hex(&nonce),
            issued_at: now,
            expires_at: now.checked_add(ttl).ok_or(Error::Input)?,
            credential_blob: BASE64.encode(credential),
            encrypted_secret: BASE64.encode(encrypted_seed),
        },
        secret,
    })
}

fn kdfa(seed: &[u8], label: &[u8], context: &[u8], bits: u32) -> Zeroizing<[u8; 32]> {
    let mut message = Vec::new();
    message.extend_from_slice(&1u32.to_be_bytes());
    message.extend_from_slice(label);
    message.push(0);
    message.extend_from_slice(context);
    message.extend_from_slice(&bits.to_be_bytes());
    let tag = hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, seed), &message);
    let mut key = Zeroizing::new([0; 32]);
    key.copy_from_slice(tag.as_ref());
    key
}

pub(super) fn protect_credential(
    modulus: &[u8; 256],
    name: &[u8; 34],
    seed: &[u8; 16],
    secret: &[u8; 32],
) -> Result<(Vec<u8>, Vec<u8>)> {
    let public = RsaPublicKey::new(
        rsa::BigUint::from_bytes_be(modulus),
        rsa::BigUint::from(65537u32),
    )
    .map_err(|_| Error::Input)?;
    let encrypted_seed = public
        .encrypt(
            &mut rsa::rand_core::OsRng,
            Oaep::new_with_label::<Sha256, _>("IDENTITY\0"),
            seed,
        )
        .map_err(|_| Error::Proof)?;
    let storage_key = kdfa(seed, b"STORAGE", name, 128);
    let integrity_key = kdfa(seed, b"INTEGRITY", &[], 256);
    let mut clear = Zeroizing::new([0; 34]);
    clear[..2].copy_from_slice(&32u16.to_be_bytes());
    clear[2..].copy_from_slice(secret);
    let mut cipher = Crypter::new(
        Cipher::aes_128_cfb128(),
        Mode::Encrypt,
        &storage_key[..16],
        Some(&[0; 16]),
    )
    .map_err(|_| Error::Proof)?;
    cipher.pad(false);
    let mut encrypted = vec![0; clear.len() + 16];
    let count = cipher
        .update(&*clear, &mut encrypted)
        .map_err(|_| Error::Proof)?;
    let final_count = cipher
        .finalize(&mut encrypted[count..])
        .map_err(|_| Error::Proof)?;
    encrypted.truncate(count + final_count);
    let mut message = encrypted.clone();
    message.extend_from_slice(name);
    let integrity = hmac::sign(
        &hmac::Key::new(hmac::HMAC_SHA256, &*integrity_key),
        &message,
    );
    // TPM2B_ID_OBJECT.buffer = sized integrity HMAC followed by encrypted identity.
    let mut credential = Vec::with_capacity(68);
    credential.extend_from_slice(&32u16.to_be_bytes());
    credential.extend_from_slice(integrity.as_ref());
    credential.extend_from_slice(&encrypted);
    Ok((credential, encrypted_seed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsa::{RsaPrivateKey, traits::PublicKeyParts};

    #[test]
    fn credential_protection_roundtrip_and_name_integrity() {
        let private = RsaPrivateKey::new(&mut rsa::rand_core::OsRng, 2048).unwrap();
        let modulus = private.n().to_bytes_be().try_into().unwrap();
        let name = [3; 34];
        let seed = [4; 16];
        let secret = [5; 32];
        let (credential, wrapped) = protect_credential(&modulus, &name, &seed, &secret).unwrap();
        assert_eq!(credential.len(), 68);
        assert_eq!(wrapped.len(), 256);
        assert_eq!(
            private
                .decrypt(Oaep::new_with_label::<Sha256, _>("IDENTITY\0"), &wrapped)
                .unwrap(),
            seed
        );
        assert!(
            private
                .decrypt(Oaep::new_with_label::<Sha256, _>("wrong\0"), &wrapped)
                .is_err()
        );
        let storage = kdfa(&seed, b"STORAGE", &name, 128);
        let integrity = kdfa(&seed, b"INTEGRITY", &[], 256);
        let plaintext = openssl::symm::decrypt(
            Cipher::aes_128_cfb128(),
            &storage[..16],
            Some(&[0; 16]),
            &credential[34..],
        )
        .unwrap();
        assert_eq!(&plaintext[..2], &[0, 32]);
        assert_eq!(&plaintext[2..], &secret);
        let mut message = credential[34..].to_vec();
        message.extend_from_slice(&name);
        assert!(
            hmac::verify(
                &hmac::Key::new(hmac::HMAC_SHA256, &*integrity),
                &message,
                &credential[2..34]
            )
            .is_ok()
        );
        message[0] ^= 1;
        assert!(
            hmac::verify(
                &hmac::Key::new(hmac::HMAC_SHA256, &*integrity),
                &message,
                &credential[2..34]
            )
            .is_err()
        );
    }
}
