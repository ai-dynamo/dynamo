// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sha2::{Digest, Sha256};

use crate::format::decode_lower_hex;
use crate::{ProtectionError, Result, TPM_POLICY_REF_HEX, policy_authorize_auth_policy};

pub const TPM_DEVICE_PUBLIC_BYTES: usize = 310;

pub fn validate_tpm_policy_authority_public(public: &[u8]) -> Result<[u8; 34]> {
    if public.len() != 88
        || public[..22]
            != [
                0, 0x23, 0, 0x0b, 0, 4, 0, 0x40, 0, 0, 0, 0x10, 0, 0x18, 0, 0x0b, 0, 3, 0, 0x10, 0,
                0x20,
            ]
        || public[54..56] != [0, 0x20]
    {
        return Err(ProtectionError::CertifiedDeviceInvalid);
    }
    let mut name = [0; 34];
    name[..2].copy_from_slice(&[0, 0x0b]);
    name[2..].copy_from_slice(&Sha256::digest(public));
    Ok(name)
}

/// Frozen TPMT_PUBLIC for the public-only P-256 policy signer.
pub fn tpm_policy_authority_public(point: &[u8]) -> Result<Vec<u8>> {
    if point.len() != 65 || point[0] != 4 {
        return Err(ProtectionError::CertifiedDeviceInvalid);
    }
    let mut public = vec![
        0, 0x23, 0, 0x0b, 0, 4, 0, 0x40, 0, 0, 0, 0x10, 0, 0x18, 0, 0x0b, 0, 3, 0, 0x10, 0, 0x20,
    ];
    public.extend_from_slice(&point[1..33]);
    public.extend_from_slice(&[0, 0x20]);
    public.extend_from_slice(&point[33..]);
    Ok(public)
}

/// Public identity only; validating this profile does not attest hardware origin.
pub struct TpmDevicePublic {
    name: [u8; 34],
    digest: [u8; 32],
    modulus: [u8; 256],
}

impl TpmDevicePublic {
    pub fn name(&self) -> &[u8; 34] {
        &self.name
    }

    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn modulus(&self) -> &[u8; 256] {
        &self.modulus
    }
}

/// Validate the frozen V1 TPMT_PUBLIC, without a TPM2B size prefix.
pub fn validate_tpm_device_public(
    public: &[u8],
    policy_authority_name: &[u8; 34],
) -> Result<TpmDevicePublic> {
    if public.len() != TPM_DEVICE_PUBLIC_BYTES
        || public[0..2] != [0x00, 0x01]
        || public[2..4] != [0x00, 0x0b]
        || public[4..8] != [0x00, 0x02, 0x04, 0xb2]
        || public[8..10] != [0x00, 0x20]
        || public[42..44] != [0x00, 0x10]
        || public[44..46] != [0x00, 0x10]
        || public[46..48] != [0x08, 0x00]
        || public[48..52] != [0, 0, 0, 0]
        || public[52..54] != [0x01, 0x00]
        || policy_authority_name[..2] != [0, 0x0b]
    {
        return Err(ProtectionError::CertifiedDeviceInvalid);
    }
    let policy_ref = decode_lower_hex::<32>(TPM_POLICY_REF_HEX, "TPM policy ref")
        .map_err(|_| ProtectionError::CertifiedDeviceInvalid)?;
    if public[10..42] != policy_authorize_auth_policy(policy_authority_name, &policy_ref) {
        return Err(ProtectionError::CertifiedDeviceInvalid);
    }
    let digest: [u8; 32] = Sha256::digest(public).into();
    let mut name = [0_u8; 34];
    name[..2].copy_from_slice(&[0, 0x0b]);
    name[2..].copy_from_slice(&digest);
    let modulus = public[54..]
        .try_into()
        .map_err(|_| ProtectionError::CertifiedDeviceInvalid)?;
    Ok(TpmDevicePublic {
        name,
        digest,
        modulus,
    })
}

#[cfg(test)]
pub(crate) fn test_device_public(policy_name: &[u8; 34]) -> [u8; TPM_DEVICE_PUBLIC_BYTES] {
    let mut public = [0_u8; TPM_DEVICE_PUBLIC_BYTES];
    public[..10].copy_from_slice(&[0, 1, 0, 0x0b, 0, 2, 4, 0xb2, 0, 0x20]);
    let policy_ref = decode_lower_hex(TPM_POLICY_REF_HEX, "test policy ref").unwrap();
    public[10..42].copy_from_slice(&policy_authorize_auth_policy(policy_name, &policy_ref));
    public[42..54].copy_from_slice(&[0, 0x10, 0, 0x10, 8, 0, 0, 0, 0, 0, 1, 0]);
    public[54..].fill(0x81);
    public
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_signer_profile_has_exact_name_and_no_framing_or_attribute_drift() {
        let mut point = [0x22; 65];
        point[0] = 4;
        let public = tpm_policy_authority_public(&point).unwrap();
        let name = validate_tpm_policy_authority_public(&public).unwrap();
        assert_eq!(public.len(), 88);
        assert_eq!(&name[2..], Sha256::digest(&public).as_slice());
        for offset in [0, 2, 4, 8, 10, 12, 14, 16, 18, 20, 54] {
            let mut wrong = public.clone();
            wrong[offset] ^= 1;
            assert!(validate_tpm_policy_authority_public(&wrong).is_err());
        }
        assert!(validate_tpm_policy_authority_public(&public[..87]).is_err());
        assert!(validate_tpm_policy_authority_public(&[public.as_slice(), &[0]].concat()).is_err());
        point[0] = 2;
        assert!(tpm_policy_authority_public(&point).is_err());
    }

    fn policy_name() -> [u8; 34] {
        let mut name = [0x22; 34];
        name[..2].copy_from_slice(&[0, 0x0b]);
        name
    }

    #[test]
    fn derives_identity_from_exact_tpm_public_bytes() {
        let public = test_device_public(&policy_name());
        let identity = validate_tpm_device_public(&public, &policy_name()).unwrap();
        let expected_digest = decode_lower_hex::<32>(
            "2721cca858932d80e34e66ef6ab439129d5655a03a692a58ff5daab6b3d7796a",
            "test public digest",
        )
        .unwrap();
        assert_eq!(identity.digest(), &expected_digest);
        assert_eq!(&identity.name()[..2], &[0, 0x0b]);
        assert_eq!(&identity.name()[2..], identity.digest());
        assert_eq!(identity.modulus(), &[0x81; 256]);
        let mut different = public;
        different[55] ^= 1;
        let other = validate_tpm_device_public(&different, &policy_name()).unwrap();
        assert_ne!(identity.name(), other.name());
    }

    #[test]
    fn rejects_profile_policy_and_framing_changes() {
        let public = test_device_public(&policy_name());
        for offset in [0, 2, 4, 7, 8, 10, 42, 44, 46, 48, 52] {
            let mut invalid = public;
            invalid[offset] ^= 1;
            assert!(validate_tpm_device_public(&invalid, &policy_name()).is_err());
        }
        assert!(validate_tpm_device_public(&public[..309], &policy_name()).is_err());
        let mut trailing = public.to_vec();
        trailing.push(0);
        assert!(validate_tpm_device_public(&trailing, &policy_name()).is_err());
        let mut wrong_authority = policy_name();
        wrong_authority[2] ^= 1;
        assert!(validate_tpm_device_public(&public, &wrong_authority).is_err());
    }

    #[cfg(feature = "tpm2")]
    #[test]
    fn agrees_with_esapi_public_marshalling_without_hardware() {
        use tss_esapi::structures::Public;
        use tss_esapi::traits::{Marshall, UnMarshall};

        let public = test_device_public(&policy_name());
        let esapi = Public::unmarshall(&public).unwrap();
        let encoded = esapi.marshall().unwrap();
        assert_eq!(encoded, public);
        assert!(validate_tpm_device_public(&encoded, &policy_name()).is_ok());
    }
}
