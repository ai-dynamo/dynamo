// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::convert::TryFrom;
use std::path::Path;
use std::str::FromStr;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use sha2::{Digest as _, Sha256};
use tss_esapi::Context;
use tss_esapi::TctiNameConf;
use tss_esapi::attributes::SessionAttributesBuilder;
use tss_esapi::constants::{CommandCode, SessionType};
use tss_esapi::handles::{KeyHandle, PersistentTpmHandle, SessionHandle, TpmHandle};
use tss_esapi::interface_types::algorithm::{HashingAlgorithm, RsaDecryptAlgorithm};
use tss_esapi::interface_types::key_bits::RsaKeyBits;
use tss_esapi::interface_types::session_handles::{AuthSession, PolicySession};
use tss_esapi::structures::{
    Data, Digest, EccParameter, EccSignature, Name, Nonce, Public, PublicKeyRsa,
    RsaDecryptionScheme, RsaScheme, Signature, SymmetricDefinition, SymmetricDefinitionObject,
    VerifiedTicket,
};
use tss_esapi::traits::Marshall;

use crate::license::{TPM_OAEP_LABEL, decode_license_hex, policy_authorize_auth_policy};
use crate::{
    AuthorizedModel, CancellationToken, ProtectionError, Result, SecretDek, SecureModelSession,
    load_authorized_model,
};

/// Verify, authorize, unwrap, and materialize one protected model.
///
/// The TPM handles must refer to persistent, passwordless objects provisioned
/// according to the V1 profile. The DEK never crosses this Rust boundary.
#[allow(clippy::too_many_arguments)]
pub fn prepare_tpm_model(
    package_root: &Path,
    license_root: &Path,
    namespace: &str,
    package_key_id: &str,
    package_verifying_key: &[u8; 32],
    license_key_id: &str,
    license_verifying_key: &[u8; 32],
    policy_authority_key_id: &str,
    tcti: &str,
    device_key_handle: u32,
    policy_authority_key_handle: u32,
    process_memory_margin: u64,
) -> Result<SecureModelSession> {
    let authorized = load_authorized_model(
        package_root,
        license_root,
        package_key_id,
        package_verifying_key,
        license_key_id,
        license_verifying_key,
    )?;
    // Reserve and validate the private tmpfs before releasing key material.
    let mut session = SecureModelSession::prepare(namespace, &authorized, process_memory_margin)?;
    session.stage_public_metadata(package_root, &authorized)?;
    materialize_tpm_model(
        &mut session,
        package_root,
        &authorized,
        policy_authority_key_id,
        tcti,
        device_key_handle,
        policy_authority_key_handle,
        &CancellationToken::default(),
    )?;
    Ok(session)
}

/// Release the TPM-bound DEK and decrypt weights into a pre-staged session.
#[allow(clippy::too_many_arguments)]
pub fn materialize_tpm_model(
    session: &mut SecureModelSession,
    package_root: &Path,
    authorized: &AuthorizedModel,
    policy_authority_key_id: &str,
    tcti: &str,
    device_key_handle: u32,
    policy_authority_key_handle: u32,
    cancellation: &CancellationToken,
) -> Result<()> {
    if authorized.license().recipient.policy_authority_key_id != policy_authority_key_id {
        return Err(ProtectionError::LicenseBindingMismatch);
    }
    let tcti = TctiNameConf::from_str(tcti).map_err(|_| ProtectionError::TpmUnavailable)?;
    let mut context = Context::new(tcti).map_err(|_| ProtectionError::TpmUnavailable)?;
    let device_key = persistent_key(&mut context, device_key_handle)?;
    let policy_authority_key = persistent_key(&mut context, policy_authority_key_handle)?;
    let key = unwrap_tpm_dek(&mut context, device_key, policy_authority_key, authorized)?;
    session.materialize_protected_cancellable(package_root, authorized, key, cancellation)
}

fn persistent_key(context: &mut Context, raw_handle: u32) -> Result<KeyHandle> {
    let handle = PersistentTpmHandle::new(raw_handle)
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)?;
    context
        .tr_from_tpm_public(TpmHandle::Persistent(handle))
        .map(KeyHandle::from)
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)
}

/// Authorize the exact signed RSA-OAEP command and unwrap a model DEK in TPM 2.0.
///
/// Provisioning remains outside this operation. The caller must resolve both
/// handles from trusted activation state and the configured policy-authority
/// key ID.
pub fn unwrap_tpm_dek(
    context: &mut Context,
    device_key: KeyHandle,
    policy_authority_key: KeyHandle,
    authorized: &AuthorizedModel,
) -> Result<SecretDek> {
    let recipient = &authorized.license().recipient;
    let (device_public, device_name, _) = context
        .read_public(device_key)
        .map_err(|_| ProtectionError::TpmUnavailable)?;
    let expected_name =
        decode_license_hex::<34>(&recipient.device_key_name, "license device key name")?;
    let expected_public_digest = decode_license_hex::<32>(
        &recipient.device_public_key_sha256,
        "license device key digest",
    )?;
    let policy_authority_name = context
        .tr_get_name(policy_authority_key.into())
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)?;
    validate_device_key_profile(&device_public, &policy_authority_name)?;
    let public_bytes = device_public
        .marshall()
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)?;
    let public_digest: [u8; 32] = Sha256::digest(public_bytes).into();
    if device_name.value() != expected_name || public_digest != expected_public_digest {
        return Err(ProtectionError::LicenseBindingMismatch);
    }

    let approved_policy = decode_license_hex::<32>(
        &recipient.approved_policy_digest,
        "license approved policy digest",
    )?;
    let command_parameters_hash = decode_license_hex::<32>(
        &recipient.command_parameters_hash,
        "license command parameters hash",
    )?;
    let policy_ref = decode_license_hex::<32>(&recipient.policy_ref, "license policy ref")?;
    let signature: [u8; 64] = BASE64
        .decode(recipient.policy_signature.as_bytes())
        .map_err(|_| ProtectionError::LicenseInvalid("policy signature"))?
        .try_into()
        .map_err(|_| ProtectionError::LicenseInvalid("policy signature"))?;
    let wrapped_dek = BASE64
        .decode(authorized.license().wrapped_dek.as_bytes())
        .map_err(|_| ProtectionError::LicenseInvalid("wrapped dek"))?;

    let mut signed_policy = Sha256::new();
    signed_policy.update(approved_policy);
    signed_policy.update(policy_ref);
    let signed_policy = Digest::try_from(signed_policy.finalize().to_vec())
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)?;
    let signature = Signature::EcDsa(
        EccSignature::create(
            HashingAlgorithm::Sha256,
            EccParameter::try_from(signature[..32].to_vec())
                .map_err(|_| ProtectionError::TpmAuthorizationFailed)?,
            EccParameter::try_from(signature[32..].to_vec())
                .map_err(|_| ProtectionError::TpmAuthorizationFailed)?,
        )
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)?,
    );
    let ticket = context
        .verify_signature(policy_authority_key, signed_policy, signature)
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)?;

    let auth_session = context
        .start_auth_session(
            None,
            None,
            None,
            SessionType::Policy,
            SymmetricDefinition::AES_256_CFB,
            HashingAlgorithm::Sha256,
        )
        .map_err(|_| ProtectionError::TpmUnavailable)?
        .ok_or(ProtectionError::TpmUnavailable)?;
    let (attributes, mask) = SessionAttributesBuilder::new()
        .with_continue_session(true)
        .build();
    context
        .tr_sess_set_attributes(auth_session, attributes, mask)
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)?;
    let policy_session = PolicySession::try_from(auth_session)
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)?;

    let result = authorize_and_unwrap(
        context,
        auth_session,
        policy_session,
        device_key,
        &policy_authority_name,
        approved_policy,
        command_parameters_hash,
        policy_ref,
        ticket,
        wrapped_dek,
    );
    context.set_sessions((None, None, None));
    let flush_result = context.flush_context(SessionHandle::from(auth_session).into());
    if flush_result.is_err() && result.is_ok() {
        return Err(ProtectionError::TpmAuthorizationFailed);
    }
    result
}

fn validate_device_key_profile(device_public: &Public, policy_authority_name: &Name) -> Result<()> {
    const REQUIRED_ATTRIBUTES: u32 = 0x0002_04b2;
    let Public::Rsa {
        object_attributes,
        name_hashing_algorithm,
        auth_policy,
        parameters,
        unique,
    } = device_public
    else {
        return Err(ProtectionError::TpmAuthorizationFailed);
    };
    let attributes: u32 = (*object_attributes).into();
    let policy_ref =
        decode_license_hex::<32>(crate::license::TPM_POLICY_REF_HEX, "license policy ref")?;
    let expected_policy = policy_authorize_auth_policy(policy_authority_name.value(), &policy_ref);
    if attributes != REQUIRED_ATTRIBUTES
        || *name_hashing_algorithm != HashingAlgorithm::Sha256
        || auth_policy.value() != expected_policy
        || parameters.symmetric_definition_object() != SymmetricDefinitionObject::Null
        || parameters.rsa_scheme() != RsaScheme::Null
        || parameters.key_bits() != RsaKeyBits::Rsa2048
        || parameters.exponent().value() != 0
        || unique.value().len() != 256
    {
        return Err(ProtectionError::TpmAuthorizationFailed);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn authorize_and_unwrap(
    context: &mut Context,
    auth_session: AuthSession,
    policy_session: PolicySession,
    device_key: KeyHandle,
    policy_authority_name: &Name,
    approved_policy: [u8; 32],
    command_parameters_hash: [u8; 32],
    policy_ref: [u8; 32],
    ticket: VerifiedTicket,
    wrapped_dek: Vec<u8>,
) -> Result<SecretDek> {
    context
        .policy_command_code(policy_session, CommandCode::RsaDecrypt)
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)?;
    context
        .policy_cp_hash(
            policy_session,
            Digest::try_from(command_parameters_hash.to_vec())
                .map_err(|_| ProtectionError::TpmAuthorizationFailed)?,
        )
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)?;
    context
        .policy_authorize(
            policy_session,
            Digest::try_from(approved_policy.to_vec())
                .map_err(|_| ProtectionError::TpmAuthorizationFailed)?,
            Nonce::try_from(policy_ref.to_vec())
                .map_err(|_| ProtectionError::TpmAuthorizationFailed)?,
            policy_authority_name,
            ticket,
        )
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)?;
    context.set_sessions((Some(auth_session), None, None));
    let plaintext = context
        .rsa_decrypt(
            device_key,
            PublicKeyRsa::try_from(wrapped_dek)
                .map_err(|_| ProtectionError::TpmAuthorizationFailed)?,
            RsaDecryptionScheme::create(RsaDecryptAlgorithm::Oaep, Some(HashingAlgorithm::Sha256))
                .map_err(|_| ProtectionError::TpmAuthorizationFailed)?,
            Data::try_from(TPM_OAEP_LABEL.to_vec())
                .map_err(|_| ProtectionError::TpmAuthorizationFailed)?,
        )
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)?;
    let key: [u8; 32] = plaintext
        .value()
        .try_into()
        .map_err(|_| ProtectionError::TpmAuthorizationFailed)?;
    SecretDek::from_tpm(key)
}
