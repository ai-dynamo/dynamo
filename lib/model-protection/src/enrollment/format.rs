// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use ring::{hmac, signature};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};

use super::{EnrollmentError as Error, Result};
use crate::format::{decode_lower_hex, has_valid_ed25519_signature, signature_payload};
use crate::validate_tpm_device_public;

pub const MAX_BUNDLE_BYTES: usize = 256 * 1024;
pub const MAX_CHALLENGE_TTL_SECONDS: u64 = 3600;
pub const CHALLENGE_DOMAIN: &[u8] = b"model-protection-enrollment-challenge-v1\0";
const REQUEST_DOMAIN: &[u8] = b"model-protection-enrollment-request-v1\0";
const TRANSCRIPT_DOMAIN: &[u8] = b"model-protection-enrollment-transcript-v1\0";
const ACTIVATION_DOMAIN: &[u8] = b"model-protection-enrollment-activation-v1\0";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub customer_scope_id: String,
    pub artifact_id: String,
    pub manifest_sha256: String,
    pub policy_authority_name: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub format: String,
    pub format_version: u16,
    pub request_id: String,
    pub binding: Binding,
    pub ek_public: String,
    pub ek_certificate_chain: Vec<String>,
    pub ak_public: String,
    pub ak_qualified_name: String,
    pub duk_public: String,
    pub duk_creation_hash: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Challenge {
    pub format: String,
    pub format_version: u16,
    pub challenge_id: String,
    pub request_sha256: String,
    pub nonce: String,
    pub issued_at: u64,
    pub expires_at: u64,
    pub credential_blob: String,
    pub encrypted_secret: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub format: String,
    pub format_version: u16,
    pub challenge_id: String,
    pub transcript_sha256: String,
    pub activation_proof: String,
    pub attestation: String,
    pub ak_signature: String,
}

pub struct ValidatedRequest {
    request: Request,
    digest: [u8; 32],
    ak_modulus: [u8; 256],
    duk_name: [u8; 34],
    ak_name: [u8; 34],
}

impl ValidatedRequest {
    pub fn request(&self) -> &Request {
        &self.request
    }
    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
    pub fn duk_name(&self) -> &[u8; 34] {
        &self.duk_name
    }
    pub fn ak_name(&self) -> &[u8; 34] {
        &self.ak_name
    }
}

/// Only transcript/activation/AK signature checks have passed, not EK chain policy.
pub struct VerifiedTranscript {
    challenge_id: String,
    request_digest: [u8; 32],
    evidence_digest: [u8; 32],
    challenge_digest: [u8; 32],
}

impl VerifiedTranscript {
    pub fn challenge_id(&self) -> &str {
        &self.challenge_id
    }
    pub fn request_digest(&self) -> &[u8; 32] {
        &self.request_digest
    }
    pub fn evidence_digest(&self) -> &[u8; 32] {
        &self.evidence_digest
    }
    pub fn challenge_digest(&self) -> &[u8; 32] {
        &self.challenge_digest
    }
}

pub fn parse<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    if bytes.is_empty() || bytes.len() > MAX_BUNDLE_BYTES {
        return Err(Error::Input);
    }
    serde_json::from_slice(bytes).map_err(|_| Error::Input)
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        && value != "."
        && value != ".."
}

fn fixed_hex<const N: usize>(value: &str) -> Result<[u8; N]> {
    decode_lower_hex(value, "enrollment hex").map_err(|_| Error::Input)
}

#[cfg(all(target_os = "linux", feature = "enrollment-client"))]
pub(super) fn decode_fixed_hex<const N: usize>(value: &str) -> Result<[u8; N]> {
    fixed_hex(value)
}

fn decode(value: &str, minimum: usize, maximum: usize) -> Result<Vec<u8>> {
    if value.len() > maximum.div_ceil(3) * 4 {
        return Err(Error::Input);
    }
    let decoded = BASE64.decode(value).map_err(|_| Error::Input)?;
    if decoded.len() < minimum || decoded.len() > maximum || BASE64.encode(&decoded) != value {
        return Err(Error::Input);
    }
    Ok(decoded)
}

pub fn validate_request(request: Request) -> Result<ValidatedRequest> {
    if request.format != "model-protection-enrollment-request"
        || request.format_version != 1
        || !identifier(&request.request_id)
        || !identifier(&request.binding.customer_scope_id)
        || request.ek_certificate_chain.is_empty()
        || request.ek_certificate_chain.len() > 5
    {
        return Err(Error::Input);
    }
    fixed_hex::<16>(&request.binding.artifact_id)?;
    fixed_hex::<32>(&request.binding.manifest_sha256)?;
    fixed_hex::<32>(&request.duk_creation_hash)?;
    let policy_name = fixed_hex::<34>(&request.binding.policy_authority_name)?;
    let qualified_name = fixed_hex::<34>(&request.ak_qualified_name)?;
    if qualified_name[..2] != [0, 0x0b] {
        return Err(Error::Input);
    }
    let duk_public = decode(&request.duk_public, 310, 310)?;
    let duk = validate_tpm_device_public(&duk_public, &policy_name).map_err(|_| Error::Input)?;
    // RSA-2048 restricted RSASSA/SHA256 AK, fixedTPM/fixedParent/sensitiveDataOrigin.
    let ak = decode(&request.ak_public, 280, 280)?;
    let ak_prefix: [u8; 24] = [
        0, 1, 0, 0x0b, 0, 5, 0, 0x72, 0, 0, 0, 0x10, 0, 0x14, 0, 0x0b, 8, 0, 0, 0, 0, 0, 1, 0,
    ];
    if ak[..24] != ak_prefix || ak[24] & 0x80 == 0 || ak[279] & 1 == 0 {
        return Err(Error::Input);
    }
    let ak_modulus = ak[24..].try_into().map_err(|_| Error::Input)?;
    let mut ak_name = [0; 34];
    ak_name[..2].copy_from_slice(&[0, 0x0b]);
    ak_name[2..].copy_from_slice(&Sha256::digest(&ak));
    // Low-range TCG RSA EK template with AES-128-CFB and standard endorsement policy.
    let ek = decode(&request.ek_public, 314, 314)?;
    let ek_policy =
        fixed_hex::<32>("837197674484b3f81a90cc8d46a5d724fd52d76e06520b64f2a1da1b331469aa")?;
    if ek[..10] != [0, 1, 0, 0x0b, 0, 3, 0, 0xb2, 0, 0x20]
        || ek[10..42] != ek_policy
        || ek[42..58] != [0, 6, 0, 0x80, 0, 0x43, 0, 0x10, 8, 0, 0, 0, 0, 0, 1, 0]
        || ek[58] & 0x80 == 0
        || ek[313] & 1 == 0
    {
        return Err(Error::Input);
    }
    for certificate in &request.ek_certificate_chain {
        decode(certificate, 1, 16 * 1024)?;
    }
    let bytes = serde_json::to_vec(&request).map_err(|_| Error::Input)?;
    if bytes.len() > MAX_BUNDLE_BYTES {
        return Err(Error::Input);
    }
    let digest = Sha256::digest(signature_payload(REQUEST_DOMAIN, &bytes)).into();
    Ok(ValidatedRequest {
        request,
        digest,
        ak_modulus,
        duk_name: *duk.name(),
        ak_name,
    })
}

impl Challenge {
    pub fn validate(&self, request: &ValidatedRequest, now: u64) -> Result<()> {
        if self.format != "model-protection-enrollment-challenge"
            || self.format_version != 1
            || !identifier(&self.challenge_id)
            || fixed_hex::<32>(&self.request_sha256)? != request.digest
            || self.expires_at <= self.issued_at
            || self.expires_at - self.issued_at > MAX_CHALLENGE_TTL_SECONDS
        {
            return Err(Error::Binding);
        }
        fixed_hex::<32>(&self.nonce)?;
        decode(&self.credential_blob, 1, 512)?;
        decode(&self.encrypted_secret, 256, 256)?;
        if now < self.issued_at || now >= self.expires_at {
            return Err(Error::Expired);
        }
        Ok(())
    }
}

pub fn challenge_signature_payload(bytes: &[u8]) -> Vec<u8> {
    signature_payload(CHALLENGE_DOMAIN, bytes)
}

pub fn verify_challenge(
    bytes: &[u8],
    signature: &[u8],
    key_id: &str,
    key: &[u8; 32],
    request: &ValidatedRequest,
    now: u64,
) -> Result<Challenge> {
    if !has_valid_ed25519_signature(CHALLENGE_DOMAIN, bytes, signature, key_id, key) {
        return Err(Error::Signature);
    }
    let challenge: Challenge = parse(bytes)?;
    challenge.validate(request, now)?;
    Ok(challenge)
}

pub fn transcript_digest(request: &ValidatedRequest, challenge: &Challenge) -> Result<[u8; 32]> {
    let bytes = serde_json::to_vec(challenge).map_err(|_| Error::Input)?;
    let mut hash = Sha256::new();
    hash.update(TRANSCRIPT_DOMAIN);
    hash.update(request.digest);
    hash.update((bytes.len() as u32).to_be_bytes());
    hash.update(bytes);
    Ok(hash.finalize().into())
}

pub fn activation_proof(secret: &[u8; 32], transcript: &[u8; 32]) -> [u8; 32] {
    let message = [ACTIVATION_DOMAIN, transcript.as_slice()].concat();
    let tag = hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, secret), &message);
    let mut proof = [0; 32];
    proof.copy_from_slice(tag.as_ref());
    proof
}

pub fn verify_response(
    request: &ValidatedRequest,
    challenge: &Challenge,
    response_bytes: &[u8],
    secret: &[u8; 32],
    now: u64,
) -> Result<VerifiedTranscript> {
    challenge.validate(request, now)?;
    let response: Response = parse(response_bytes)?;
    let transcript = transcript_digest(request, challenge)?;
    if response.format != "model-protection-enrollment-response"
        || response.format_version != 1
        || response.challenge_id != challenge.challenge_id
        || fixed_hex::<32>(&response.transcript_sha256)? != transcript
    {
        return Err(Error::Binding);
    }
    let proof = fixed_hex::<32>(&response.activation_proof)?;
    hmac::verify(
        &hmac::Key::new(hmac::HMAC_SHA256, secret),
        &[ACTIVATION_DOMAIN, transcript.as_slice()].concat(),
        &proof,
    )
    .map_err(|_| Error::Proof)?;
    let attestation = decode(&response.attestation, 1, 2048)?;
    let signature = decode(&response.ak_signature, 256, 256)?;
    signature::RsaPublicKeyComponents {
        n: request.ak_modulus.as_slice(),
        e: &[1, 0, 1],
    }
    .verify(
        &signature::RSA_PKCS1_2048_8192_SHA256,
        &attestation,
        &signature,
    )
    .map_err(|_| Error::Signature)?;
    verify_creation_attestation(
        &attestation,
        &request.request.ak_qualified_name,
        &request.duk_name,
        &fixed_hex::<32>(&request.request.duk_creation_hash)?,
        &transcript,
    )?;
    Ok(VerifiedTranscript {
        challenge_id: challenge.challenge_id.clone(),
        request_digest: request.digest,
        evidence_digest: Sha256::digest(response_bytes).into(),
        challenge_digest: Sha256::digest(serde_json::to_vec(challenge).map_err(|_| Error::Input)?)
            .into(),
    })
}

fn verify_creation_attestation(
    bytes: &[u8],
    signer: &str,
    duk: &[u8; 34],
    creation_hash: &[u8; 32],
    transcript: &[u8; 32],
) -> Result<()> {
    struct Cursor<'a>(&'a [u8]);
    impl<'a> Cursor<'a> {
        fn take(&mut self, n: usize) -> Result<&'a [u8]> {
            if self.0.len() < n {
                return Err(Error::Input);
            }
            let (value, rest) = self.0.split_at(n);
            self.0 = rest;
            Ok(value)
        }
        fn sized(&mut self) -> Result<&'a [u8]> {
            let length = self.take(2)?;
            self.take(u16::from_be_bytes([length[0], length[1]]) as usize)
        }
    }
    let mut cursor = Cursor(bytes);
    if cursor.take(6)? != [0xff, 0x54, 0x43, 0x47, 0x80, 0x1a]
        || cursor.sized()? != fixed_hex::<34>(signer)?
        || cursor.sized()? != transcript
    {
        return Err(Error::Binding);
    }
    // TPMS_CLOCK_INFO.safe is a byte boolean. No monotonic-boot claim is made here.
    let clock_info = cursor.take(17)?;
    if clock_info[16] > 1 {
        return Err(Error::Input);
    }
    cursor.take(8)?;
    if cursor.sized()? != duk || cursor.sized()? != creation_hash || !cursor.0.is_empty() {
        return Err(Error::Binding);
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn test_request() -> Request {
    let mut policy_name = [0x22; 34];
    policy_name[..2].copy_from_slice(&[0, 0x0b]);
    let duk = crate::device::test_device_public(&policy_name);
    let mut ak = vec![
        0, 1, 0, 0x0b, 0, 5, 0, 0x72, 0, 0, 0, 0x10, 0, 0x14, 0, 0x0b, 8, 0, 0, 0, 0, 0, 1, 0,
    ];
    ak.extend_from_slice(&[0x81; 256]);
    let mut ek = vec![0, 1, 0, 0x0b, 0, 3, 0, 0xb2, 0, 0x20];
    ek.extend_from_slice(
        &fixed_hex::<32>("837197674484b3f81a90cc8d46a5d724fd52d76e06520b64f2a1da1b331469aa")
            .unwrap(),
    );
    ek.extend_from_slice(&[0, 6, 0, 0x80, 0, 0x43, 0, 0x10, 8, 0, 0, 0, 0, 0, 1, 0]);
    ek.extend_from_slice(&[0x81; 256]);
    Request {
        format: "model-protection-enrollment-request".into(),
        format_version: 1,
        request_id: "request-1".into(),
        binding: Binding {
            customer_scope_id: "customer-1".into(),
            artifact_id: hex(&[0x11; 16]),
            manifest_sha256: hex(&[0x33; 32]),
            policy_authority_name: hex(&policy_name),
        },
        ek_public: BASE64.encode(ek),
        ek_certificate_chain: vec![BASE64.encode(b"unverified test certificate")],
        ak_public: BASE64.encode(ak),
        ak_qualified_name: hex(&policy_name),
        duk_public: BASE64.encode(duk),
        duk_creation_hash: hex(&[0x44; 32]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_parsing_bounds_and_identifiers() {
        assert!(parse::<Binding>(br#"{"customer_scope_id":"c","customer_scope_id":"d"}"#).is_err());
        assert!(parse::<Binding>(&vec![b' '; MAX_BUNDLE_BYTES + 1]).is_err());
        assert!(parse::<Binding>(br#"{"customer_scope_id":"c","artifact_id":"a","manifest_sha256":"b","policy_authority_name":"c","extra":0}"#).is_err());
        for bad in ["", "..", "a/b", "a\n", "é"] {
            assert!(!identifier(bad));
        }
        assert!(identifier("customer-01_v1"));
        assert!(decode("YQ", 1, 1).is_err());
    }

    #[test]
    fn activation_is_bound_to_secret_and_transcript() {
        let proof = activation_proof(&[1; 32], &[2; 32]);
        assert_ne!(proof, activation_proof(&[3; 32], &[2; 32]));
        assert_ne!(proof, activation_proof(&[1; 32], &[4; 32]));
        assert_eq!(
            hex(&proof),
            "6e60a886da24ec7c4dcdd9dc4c1166a9c881249ff053f91c808d19c6a5de8cac"
        );
    }

    #[test]
    fn creation_attestation_rejects_nonce_object_and_trailing_data() {
        let mut signer = [5; 34];
        signer[..2].copy_from_slice(&[0, 0x0b]);
        let duk = [6; 34];
        let hash = [7; 32];
        let nonce = [8; 32];
        let mut bytes = vec![0xff, 0x54, 0x43, 0x47, 0x80, 0x1a];
        for value in [signer.as_slice(), nonce.as_slice()] {
            bytes.extend_from_slice(&(value.len() as u16).to_be_bytes());
            bytes.extend_from_slice(value);
        }
        bytes.extend_from_slice(&[0; 25]);
        for value in [duk.as_slice(), hash.as_slice()] {
            bytes.extend_from_slice(&(value.len() as u16).to_be_bytes());
            bytes.extend_from_slice(value);
        }
        assert!(verify_creation_attestation(&bytes, &hex(&signer), &duk, &hash, &nonce).is_ok());
        assert!(verify_creation_attestation(&bytes, &hex(&signer), &duk, &hash, &[9; 32]).is_err());
        for n in 0..bytes.len() {
            assert!(
                verify_creation_attestation(&bytes[..n], &hex(&signer), &duk, &hash, &nonce)
                    .is_err()
            );
        }
        bytes.push(0);
        assert!(verify_creation_attestation(&bytes, &hex(&signer), &duk, &hash, &nonce).is_err());
    }

    #[test]
    fn profiles_transcripts_and_signed_challenges_are_bound() {
        use ring::signature::KeyPair;
        let request = validate_request(test_request()).unwrap();
        let challenge = Challenge {
            format: "model-protection-enrollment-challenge".into(),
            format_version: 1,
            challenge_id: "challenge-1".into(),
            request_sha256: hex(request.digest()),
            nonce: hex(&[1; 32]),
            issued_at: 10,
            expires_at: 100,
            credential_blob: BASE64.encode([2; 68]),
            encrypted_secret: BASE64.encode([3; 256]),
        };
        let key = signature::Ed25519KeyPair::from_seed_unchecked(&[4; 32]).unwrap();
        let key_bytes = key.public_key().as_ref().try_into().unwrap();
        let bytes = serde_json::to_vec(&challenge).unwrap();
        let envelope = serde_json::to_vec(&crate::SignatureEnvelope {
            algorithm: "Ed25519".into(),
            key_id: "challenge-key".into(),
            signature: BASE64.encode(key.sign(&challenge_signature_payload(&bytes)).as_ref()),
        })
        .unwrap();
        assert!(
            verify_challenge(&bytes, &envelope, "challenge-key", &key_bytes, &request, 10).is_ok()
        );
        assert!(
            verify_challenge(
                &bytes,
                &envelope,
                "challenge-key",
                &key_bytes,
                &request,
                100
            )
            .is_err()
        );
        assert!(
            verify_challenge(&bytes, &envelope, "other-key", &key_bytes, &request, 10).is_err()
        );
        let mut other = test_request();
        other.binding.artifact_id = hex(&[9; 16]);
        let other = validate_request(other).unwrap();
        assert!(
            verify_challenge(&bytes, &envelope, "challenge-key", &key_bytes, &other, 10).is_err()
        );
        assert_ne!(
            transcript_digest(&request, &challenge).unwrap(),
            transcript_digest(&other, &challenge).unwrap()
        );
        let mut invalid = test_request();
        let mut public = BASE64.decode(&invalid.ak_public).unwrap();
        public[7] ^= 0x10;
        invalid.ak_public = BASE64.encode(public);
        assert!(validate_request(invalid).is_err());
    }
}
