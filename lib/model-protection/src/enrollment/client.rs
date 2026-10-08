// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Activation and creation proof; no key export or enrollment trust bypass.

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use tss_esapi::Context;
use tss_esapi::attributes::SessionAttributesBuilder;
use tss_esapi::constants::SessionType;
use tss_esapi::handles::{AuthHandle, KeyHandle, PersistentTpmHandle, TpmHandle};
use tss_esapi::interface_types::algorithm::HashingAlgorithm;
use tss_esapi::interface_types::session_handles::PolicySession;
use tss_esapi::structures::{EncryptedSecret, IdObject, SymmetricDefinition};
use tss_esapi::traits::Marshall;
use zeroize::Zeroizing;

use super::format::{
    Challenge, Response, ValidatedRequest, activation_proof, hex, transcript_digest,
    verify_response,
};
use super::{EnrollmentError as Error, Result, certify_creation};

pub struct ActivationHandles {
    pub ek: u32,
    pub ak: u32,
    pub duk: u32,
}

fn load(context: &mut Context, handle: u32) -> Result<KeyHandle> {
    let persistent = PersistentTpmHandle::new(handle).map_err(|_| Error::Input)?;
    context
        .tr_from_tpm_public(TpmHandle::Persistent(persistent))
        .map(KeyHandle::from)
        .map_err(|_| Error::Proof)
}

/// Only call after authenticating the authority challenge. Locks the short-lived
/// process before TPM activation so TSS/ring heap copies cannot be swapped/dumped.
pub fn respond(
    context: &mut Context,
    device: &str,
    handles: &ActivationHandles,
    request: &ValidatedRequest,
    challenge: &Challenge,
    creation_ticket: &[u8],
    now: u64,
) -> Result<Vec<u8>> {
    challenge.validate(request, now)?;
    rustix::process::setrlimit(
        rustix::process::Resource::Core,
        rustix::process::Rlimit {
            current: Some(0),
            maximum: Some(0),
        },
    )
    .map_err(|_| Error::Proof)?;
    rustix::process::set_dumpable_behavior(rustix::process::DumpableBehavior::NotDumpable)
        .map_err(|_| Error::Proof)?;
    rustix::mm::mlockall(rustix::mm::MlockAllFlags::CURRENT | rustix::mm::MlockAllFlags::FUTURE)
        .map_err(|_| Error::Proof)?;
    respond_with(
        context,
        handles,
        request,
        challenge,
        creation_ticket,
        now,
        |hash, transcript, ticket| {
            certify_creation::certify(
                device,
                (handles.ak, request.ak_name()),
                (handles.duk, request.duk_name()),
                hash,
                transcript,
                ticket,
            )
        },
    )
}

fn respond_with(
    context: &mut Context,
    handles: &ActivationHandles,
    request: &ValidatedRequest,
    challenge: &Challenge,
    creation_ticket: &[u8],
    now: u64,
    certify: impl FnOnce(&[u8; 32], &[u8; 32], &[u8]) -> Result<certify_creation::CreationProof>,
) -> Result<Vec<u8>> {
    challenge.validate(request, now)?;
    let ek = load(context, handles.ek)?;
    let ak = load(context, handles.ak)?;
    let duk = load(context, handles.duk)?;
    for (handle, expected) in [
        (ek, &request.request().ek_public),
        (ak, &request.request().ak_public),
        (duk, &request.request().duk_public),
    ] {
        let (public, _, qualified) = context.read_public(handle).map_err(|_| Error::Proof)?;
        if BASE64.encode(public.marshall().map_err(|_| Error::Proof)?) != *expected
            || (handle == ak && hex(qualified.value()) != request.request().ak_qualified_name)
        {
            return Err(Error::Binding);
        }
    }
    let credential = IdObject::try_from(
        BASE64
            .decode(&challenge.credential_blob)
            .map_err(|_| Error::Input)?,
    )
    .map_err(|_| Error::Input)?;
    let encrypted = EncryptedSecret::try_from(
        BASE64
            .decode(&challenge.encrypted_secret)
            .map_err(|_| Error::Input)?,
    )
    .map_err(|_| Error::Input)?;
    // Salt to the validated EK and encrypt the returned activation secret.
    let session = context
        .start_auth_session(
            Some(ek),
            None,
            None,
            SessionType::Policy,
            SymmetricDefinition::AES_128_CFB,
            HashingAlgorithm::Sha256,
        )
        .map_err(|_| Error::Proof)?
        .ok_or(Error::Proof)?;
    let (attributes, mask) = SessionAttributesBuilder::new().with_encrypt(true).build();
    context
        .tr_sess_set_attributes(session, attributes, mask)
        .map_err(|_| Error::Proof)?;
    let activated = context
        .execute_with_temporary_object(
            tss_esapi::handles::SessionHandle::from(session).into(),
            |context, _| {
                context.execute_with_nullauth_session(|context| {
                    context.policy_secret(
                        PolicySession::try_from(session)?,
                        AuthHandle::Endorsement,
                        Default::default(),
                        Default::default(),
                        Default::default(),
                        None,
                    )
                })?;
                context.execute_with_sessions(
                    (
                        Some(tss_esapi::interface_types::session_handles::AuthSession::Password),
                        Some(session),
                        None,
                    ),
                    |context| context.activate_credential(ak, ek, credential, encrypted),
                )
            },
        )
        .map_err(|_| Error::Proof)?;
    let secret = Zeroizing::new(<[u8; 32]>::try_from(activated.value()).map_err(|_| Error::Proof)?);
    let transcript = transcript_digest(request, challenge)?;
    let hash: [u8; 32] = super::format::decode_fixed_hex(&request.request().duk_creation_hash)?;
    let proof = certify(&hash, &transcript, creation_ticket)?;
    let response = Response {
        format: "model-protection-enrollment-response".into(),
        format_version: 1,
        challenge_id: challenge.challenge_id.clone(),
        transcript_sha256: hex(&transcript),
        activation_proof: hex(&activation_proof(&secret, &transcript)),
        attestation: BASE64.encode(proof.attestation),
        ak_signature: BASE64.encode(proof.signature),
    };
    let bytes = serde_json::to_vec(&response).map_err(|_| Error::Input)?;
    verify_response(request, challenge, &bytes, &secret, now)?;
    Ok(bytes)
}

#[cfg(all(test, feature = "enrollment-authority"))]
mod tests {
    use super::super::format::{Binding, Request, validate_request};
    use super::*;
    use std::{
        net::TcpListener,
        process::{Child, Command},
        str::FromStr,
    };
    use tss_esapi::TctiNameConf;

    struct Simulator {
        child: Child,
        control_port: u16,
    }
    impl Drop for Simulator {
        fn drop(&mut self) {
            let _ = Command::new("swtpm_ioctl")
                .args(["--tcp", &format!(":{}", self.control_port), "-s"])
                .status();
            for _ in 0..50 {
                if self.child.try_wait().ok().flatten().is_some() {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            // Do not block indefinitely if process signals are restricted by the harness.
            if self.child.kill().is_ok() {
                let _ = self.child.wait();
            }
        }
    }

    #[test]
    #[ignore = "starts isolated swtpm; no physical TPM, production keys or OEM trust"]
    fn swtpm_activates_software_credential_and_certifies_policy_only_duk() {
        let directory = tempfile::tempdir().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let control_port = port.checked_add(1).unwrap();
        let control = TcpListener::bind(("127.0.0.1", control_port)).unwrap();
        drop(listener);
        drop(control);
        let mut simulator = Simulator {
            control_port,
            child: Command::new("swtpm")
                .args([
                    "socket",
                    "--tpm2",
                    "--tpmstate",
                    &format!("dir={}", directory.path().display()),
                    "--server",
                    &format!("type=tcp,port={port},bindaddr=127.0.0.1"),
                    "--ctrl",
                    &format!("type=tcp,port={control_port},bindaddr=127.0.0.1"),
                    "--flags",
                    "not-need-init,startup-clear",
                ])
                .spawn()
                .unwrap(),
        };
        let tcti = format!("swtpm:host=127.0.0.1,port={port}");
        let mut context = None;
        for _ in 0..50 {
            assert!(
                simulator.child.try_wait().unwrap().is_none(),
                "simulator exited"
            );
            if let Ok(connection) = Context::new(TctiNameConf::from_str(&tcti).unwrap()) {
                context = Some(connection);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let mut context = context.expect("isolated simulator must become ready");
        use super::super::provision;
        let targets = provision::Handles {
            ek: 0x8101_2001,
            ak: 0x8101_2002,
            duk: 0x8101_2003,
        };
        let signer = p256::ecdsa::SigningKey::from_slice(&[0x22; 32]).unwrap();
        let policy_public = crate::tpm_policy_authority_public(
            signer.verifying_key().to_encoded_point(false).as_bytes(),
        )
        .unwrap();
        let prepared = provision::prepare(&mut context, targets, &policy_public).unwrap();
        let journal = serde_json::to_vec(&prepared).unwrap();
        let prepared: provision::PreparedEnrollment =
            super::super::format::parse(&journal).unwrap();
        prepared.validate_inputs(&targets, &policy_public).unwrap();
        // Simulated death after EK/AK persistence; no re-generated AK/DUK on retry.
        assert!(
            provision::commit_with_checkpoint(&mut context, &prepared, |step| if step == 2 {
                Err(Error::Proof)
            } else {
                Ok(())
            })
            .is_err()
        );
        drop(context);
        let mut context = Context::new(TctiNameConf::from_str(&tcti).unwrap()).unwrap();
        provision::commit(&mut context, &prepared).unwrap();
        provision::commit(&mut context, &prepared).unwrap();
        assert!(provision::prepare(&mut context, targets, &policy_public).is_err());
        let handles = ActivationHandles {
            ek: targets.ek,
            ak: targets.ak,
            duk: targets.duk,
        };
        let policy_name: [u8; 34] =
            super::super::format::decode_fixed_hex(&prepared.state().policy_authority_name)
                .unwrap();
        let creation_hash = prepared.state().duk_creation_hash.clone();
        let ticket = BASE64
            .decode(prepared.state().duk_creation_ticket.as_ref().unwrap())
            .unwrap();
        let ek = load(&mut context, handles.ek).unwrap();
        let ak = load(&mut context, handles.ak).unwrap();
        let duk = load(&mut context, handles.duk).unwrap();
        let (ek_public, _, _) = context.read_public(ek).unwrap();
        let (ak_public, _, ak_qualified) = context.read_public(ak).unwrap();
        let (duk_public, _, _) = context.read_public(duk).unwrap();
        let request = validate_request(Request {
            format: "model-protection-enrollment-request".into(),
            format_version: 1,
            request_id: "simulator".into(),
            binding: Binding {
                customer_scope_id: "test".into(),
                artifact_id: hex(&[1; 16]),
                manifest_sha256: hex(&[2; 32]),
                policy_authority_name: hex(&policy_name),
            },
            ek_public: BASE64.encode(ek_public.marshall().unwrap()),
            ek_certificate_chain: vec![BASE64.encode(b"not an OEM certificate")],
            ak_public: BASE64.encode(ak_public.marshall().unwrap()),
            ak_qualified_name: hex(ak_qualified.value()),
            duk_public: BASE64.encode(duk_public.marshall().unwrap()),
            duk_creation_hash: creation_hash,
        })
        .unwrap();
        let ek_wire = ek_public.marshall().unwrap();
        let secret = [5; 32];
        let (credential, encrypted) = super::super::credential::protect_credential(
            ek_wire[58..].try_into().unwrap(),
            request.ak_name(),
            &[4; 16],
            &secret,
        )
        .unwrap();
        let challenge = Challenge {
            format: "model-protection-enrollment-challenge".into(),
            format_version: 1,
            challenge_id: "challenge".into(),
            request_sha256: hex(request.digest()),
            nonce: hex(&[3; 32]),
            issued_at: 10,
            expires_at: 100,
            credential_blob: BASE64.encode(credential),
            encrypted_secret: BASE64.encode(encrypted),
        };
        let produce = |context: &mut Context, ticket: &[u8]| {
            respond_with(
                context,
                &handles,
                &request,
                &challenge,
                ticket,
                20,
                |hash, transcript, ticket| {
                    certify_creation::certify_test_tcti(
                        &tcti,
                        (handles.ak, request.ak_name()),
                        (handles.duk, request.duk_name()),
                        hash,
                        transcript,
                        ticket,
                    )
                },
            )
        };
        // Test-only simulator fixture skips process-wide mlockall; production has no bypass.
        let bytes = produce(&mut context, &ticket).unwrap();
        eprintln!("simulator: activation and creation proof returned");
        verify_response(&request, &challenge, &bytes, &secret, 20).unwrap();
        let mut wrong_ticket = ticket.clone();
        *wrong_ticket.last_mut().unwrap() ^= 1;
        assert!(produce(&mut context, &wrong_ticket).is_err());
        eprintln!("simulator: wrong creation ticket rejected");
        let mut wrong: Response = super::super::format::parse(&bytes).unwrap();
        wrong.transcript_sha256 = hex(&[9; 32]);
        assert!(
            verify_response(
                &request,
                &challenge,
                &serde_json::to_vec(&wrong).unwrap(),
                &secret,
                20
            )
            .is_err()
        );
        eprintln!("simulator: transcript mismatch rejected");
        assert_policy_unwrap(&mut context, duk, &policy_public, &signer);
    }

    fn assert_policy_unwrap(
        context: &mut Context,
        duk: KeyHandle,
        policy_public: &[u8],
        signer: &p256::ecdsa::SigningKey,
    ) {
        use crate::{
            Entitlement, License, TPM_OAEP_LABEL, TPM_POLICY_REF_HEX, TPM_POLICY_SIGNATURE,
            TPM_PROFILE, TpmRecipient,
        };
        use p256::ecdsa::signature::hazmat::PrehashSigner;
        use ring::signature::{Ed25519KeyPair, KeyPair};
        use sha2::Digest as _;
        use tss_esapi::{
            interface_types::resource_handles::Hierarchy, structures::Public, traits::UnMarshall,
        };
        let (public, name, _) = context.read_public(duk).unwrap();
        let public = public.marshall().unwrap();
        let identity = crate::validate_tpm_device_public(
            &public,
            &crate::validate_tpm_policy_authority_public(policy_public).unwrap(),
        )
        .unwrap();
        let dek = [0x55; 32];
        let rsa = rsa::RsaPublicKey::new(
            rsa::BigUint::from_bytes_be(identity.modulus()),
            rsa::BigUint::from(65537u32),
        )
        .unwrap();
        let wrapped: [u8; 256] = rsa
            .encrypt(
                &mut rsa::rand_core::OsRng,
                rsa::Oaep::new_with_label::<sha2::Sha256, _>(
                    std::str::from_utf8(TPM_OAEP_LABEL).unwrap(),
                ),
                &dek,
            )
            .unwrap()
            .try_into()
            .unwrap();
        let cp_hash = crate::rsa_decrypt_cp_hash(name.value().try_into().unwrap(), &wrapped);
        let approved = crate::approved_rsa_decrypt_policy(&cp_hash);
        let policy_ref: [u8; 32] =
            super::super::format::decode_fixed_hex(TPM_POLICY_REF_HEX).unwrap();
        let digest: [u8; 32] =
            sha2::Sha256::digest([approved.as_slice(), policy_ref.as_slice()].concat()).into();
        let signature: p256::ecdsa::Signature = signer.sign_prehash(&digest).unwrap();
        let manifest = br#"{"format":"secure-model-package","format_version":1,"artifact_id":"00112233445566778899aabbccddeeff","customer_scope_id":"customer-a","model":{"model_id":"tiny","model_version":"1","framework":"safetensors"},"encryption":{"algorithm":"AES-256-GCM","nonce_prefix":"01020304","tag_bits":128,"record_plaintext_limit":1024},"protected_files":[{"file_id":1,"container_path":"weights/model.safetensors.protected","output_path":"model.safetensors","container_size":84,"container_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","plaintext_size":32,"plaintext_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","record_count":1,"first_global_record_counter":0}],"public_files":[],"runtime":{"minimum_runtime_version":"1.5.0","required_load_format":"safetensors"}}"#;
        let package_key = Ed25519KeyPair::from_seed_unchecked(&[1; 32]).unwrap();
        let envelope = serde_json::json!({"algorithm":"Ed25519","key_id":"package-test","signature":BASE64.encode(package_key.sign(&crate::manifest_signature_payload(manifest)).as_ref())});
        let manifest = crate::verify_manifest(
            manifest,
            &serde_json::to_vec(&envelope).unwrap(),
            "package-test",
            package_key.public_key().as_ref().try_into().unwrap(),
        )
        .unwrap();
        let license = License {
            format: "model-protection-license".into(),
            format_version: 1,
            license_id: "test".into(),
            customer_scope_id: "customer-a".into(),
            artifact_id: "00112233445566778899aabbccddeeff".into(),
            manifest_sha256: hex(manifest.digest()),
            model_id: "tiny".into(),
            model_version: "1".into(),
            entitlement: Entitlement {
                mode: "offline-perpetual".into(),
                generation: 1,
            },
            recipient: TpmRecipient {
                kind: "tpm2".into(),
                profile: TPM_PROFILE.into(),
                device_key_name: hex(identity.name()),
                device_public_key_sha256: hex(identity.digest()),
                command_parameters_hash: hex(&cp_hash),
                approved_policy_digest: hex(&approved),
                policy_ref: TPM_POLICY_REF_HEX.into(),
                policy_authority_key_id: "policy-test".into(),
                policy_signature_algorithm: TPM_POLICY_SIGNATURE.into(),
                policy_signature: BASE64.encode(signature.to_bytes()),
            },
            wrapped_dek: BASE64.encode(wrapped),
        };
        let license_key = Ed25519KeyPair::from_seed_unchecked(&[2; 32]).unwrap();
        let license_bytes = serde_json::to_vec(&license).unwrap();
        let envelope = serde_json::json!({"algorithm":"Ed25519","key_id":"license-test","signature":BASE64.encode(license_key.sign(&crate::license_signature_payload(&license_bytes)).as_ref())});
        let authorized = crate::license::verify_license(
            &license_bytes,
            &serde_json::to_vec(&envelope).unwrap(),
            "license-test",
            license_key.public_key().as_ref().try_into().unwrap(),
            manifest,
        )
        .unwrap();
        let policy = context
            .load_external_public(Public::unmarshall(policy_public).unwrap(), Hierarchy::Owner)
            .unwrap();
        let unwrapped = crate::unwrap_tpm_dek(context, duk, policy, &authorized).unwrap();
        assert_eq!(unwrapped.as_bytes(), &dek);
        let mut wrong = authorized.license().clone();
        wrong.recipient.device_key_name = hex(&[0x11; 34]);
        assert!(
            crate::unwrap_tpm_dek(
                context,
                duk,
                policy,
                &crate::AuthorizedModel::new(wrong, authorized.manifest().clone())
            )
            .is_err()
        );
        let mut wrong = authorized.license().clone();
        let mut ciphertext = wrapped;
        ciphertext[0] ^= 1;
        wrong.wrapped_dek = BASE64.encode(ciphertext);
        assert!(
            crate::unwrap_tpm_dek(
                context,
                duk,
                policy,
                &crate::AuthorizedModel::new(wrong, authorized.manifest().clone())
            )
            .is_err()
        );
        let mut wrong = authorized.license().clone();
        let wrong_ref = [0x33; 32];
        wrong.recipient.policy_ref = hex(&wrong_ref);
        let digest: [u8; 32] =
            sha2::Sha256::digest([approved.as_slice(), wrong_ref.as_slice()].concat()).into();
        let signature: p256::ecdsa::Signature = signer.sign_prehash(&digest).unwrap();
        wrong.recipient.policy_signature = BASE64.encode(signature.to_bytes());
        assert!(
            crate::unwrap_tpm_dek(
                context,
                duk,
                policy,
                &crate::AuthorizedModel::new(wrong, authorized.manifest().clone())
            )
            .is_err()
        );
        // Same ciphertext through password authorization must not bypass PolicyAuthorize.
        assert!(
            context
                .execute_with_nullauth_session(|ctx| ctx.rsa_decrypt(
                    duk,
                    tss_esapi::structures::PublicKeyRsa::try_from(wrapped.to_vec())?,
                    tss_esapi::structures::RsaDecryptionScheme::Oaep(
                        tss_esapi::structures::HashScheme::new(HashingAlgorithm::Sha256)
                    ),
                    tss_esapi::structures::Data::try_from(TPM_OAEP_LABEL.to_vec())?
                ))
                .is_err()
        );
        context.flush_context(policy.into()).unwrap();
        eprintln!(
            "TPM: signed-license policy-only DEK unwrap passed; wrong Name/ciphertext/policyRef and password bypass rejected"
        );
    }

    #[test]
    #[ignore = "explicit opt-in; reads already approved test handles on physical TPM, never provisions"]
    fn physical_tpm_activates_certifies_and_unwraps_existing_test_duk() {
        use super::super::format::{CHALLENGE_DOMAIN, verify_challenge};
        use p256::pkcs8::DecodePrivateKey;
        use ring::signature::{Ed25519KeyPair, KeyPair};
        use std::os::unix::fs::MetadataExt;
        let directory = std::path::PathBuf::from(
            std::env::var("DYNAMO_TPM_HARDWARE_TEST_DIR")
                .expect("set explicit approved test directory"),
        );
        assert!(directory.is_absolute());
        let read = |filename: &str| {
            let path = directory.join(filename);
            let metadata = std::fs::symlink_metadata(&path).unwrap();
            assert!(
                metadata.is_file()
                    && metadata.nlink() == 1
                    && metadata.uid() == rustix::process::geteuid().as_raw()
                    && metadata.mode() & 0o077 == 0
                    && metadata.len() <= 256 * 1024
            );
            std::fs::read(path).unwrap()
        };
        let state: super::super::provision::ActivationState =
            super::super::format::parse(&read("enrollment-state.json")).unwrap();
        state.validate().unwrap();
        assert_eq!(
            [state.ek_handle, state.ak_handle, state.duk_handle],
            [0x8101_2001, 0x8101_2002, 0x8101_2003],
            "only explicitly approved test handles"
        );
        let policy_public = read("test-policy.tpmt-public");
        assert_eq!(
            hex(&crate::validate_tpm_policy_authority_public(&policy_public).unwrap()),
            state.policy_authority_name
        );
        let pem = zeroize::Zeroizing::new(read("test-policy.pem"));
        let signer =
            p256::ecdsa::SigningKey::from_pkcs8_pem(std::str::from_utf8(&pem).unwrap()).unwrap();
        assert_eq!(
            crate::tpm_policy_authority_public(
                signer.verifying_key().to_encoded_point(false).as_bytes()
            )
            .unwrap(),
            policy_public
        );
        let mut context =
            Context::new(TctiNameConf::from_str("device:/dev/tpmrm0").unwrap()).unwrap();
        let handles = ActivationHandles {
            ek: state.ek_handle,
            ak: state.ak_handle,
            duk: state.duk_handle,
        };
        let mut publics = Vec::new();
        let mut qualified = Vec::new();
        for (handle, expected) in [handles.ek, handles.ak, handles.duk].into_iter().zip([
            &state.ek_name,
            &state.ak_name,
            &state.duk_name,
        ]) {
            let key = load(&mut context, handle).unwrap();
            let (public, name, qn) = context.read_public(key).unwrap();
            assert_eq!(hex(name.value()), *expected);
            publics.push(public.marshall().unwrap());
            qualified.push(hex(qn.value()));
        }
        let request = validate_request(Request {
            format: "model-protection-enrollment-request".into(),
            format_version: 1,
            request_id: "physical-test".into(),
            binding: Binding {
                customer_scope_id: "test-only".into(),
                artifact_id: hex(&[1; 16]),
                manifest_sha256: hex(&[2; 32]),
                policy_authority_name: state.policy_authority_name,
            },
            ek_public: BASE64.encode(&publics[0]),
            ek_certificate_chain: vec![BASE64.encode(b"test-only, not OEM acceptance")],
            ak_public: BASE64.encode(&publics[1]),
            ak_qualified_name: qualified[1].clone(),
            duk_public: BASE64.encode(&publics[2]),
            duk_creation_hash: state.duk_creation_hash,
        })
        .unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let secret = zeroize::Zeroizing::new([0x55; 32]);
        let (credential, encrypted) = super::super::credential::protect_credential(
            publics[0][58..].try_into().unwrap(),
            request.ak_name(),
            &[0x22; 16],
            &secret,
        )
        .unwrap();
        let challenge = Challenge {
            format: "model-protection-enrollment-challenge".into(),
            format_version: 1,
            challenge_id: "physical-test-challenge".into(),
            request_sha256: hex(request.digest()),
            nonce: hex(&[0x33; 32]),
            issued_at: now,
            expires_at: now + 300,
            credential_blob: BASE64.encode(credential),
            encrypted_secret: BASE64.encode(encrypted),
        };
        let challenge_bytes = serde_json::to_vec(&challenge).unwrap();
        let challenge_key = Ed25519KeyPair::from_seed_unchecked(&[0x44; 32]).unwrap();
        let envelope = serde_json::json!({"algorithm":"Ed25519","key_id":"test-only","signature":BASE64.encode(challenge_key.sign(&crate::format::signature_payload(CHALLENGE_DOMAIN,&challenge_bytes)).as_ref())});
        let challenge = verify_challenge(
            &challenge_bytes,
            &serde_json::to_vec(&envelope).unwrap(),
            "test-only",
            challenge_key.public_key().as_ref().try_into().unwrap(),
            &request,
            now,
        )
        .unwrap();
        let ticket = BASE64.decode(state.duk_creation_ticket.unwrap()).unwrap();
        let response = respond(
            &mut context,
            "/dev/tpmrm0",
            &handles,
            &request,
            &challenge,
            &ticket,
            now,
        )
        .unwrap();
        verify_response(&request, &challenge, &response, &secret, now).unwrap();
        if let Ok(binary) = std::env::var("DYNAMO_TPM_TEST_COLLECTOR_BIN") {
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            assert!(std::path::Path::new(&binary).is_absolute());
            let prefix = uuid::Uuid::new_v4().simple().to_string();
            let publish = |suffix: &str, bytes: &[u8]| {
                let path = directory.join(format!("{prefix}-{suffix}"));
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(&path)
                    .unwrap();
                file.write_all(bytes).unwrap();
                file.sync_all().unwrap();
                path
            };
            let binding_file = publish(
                "binding.json",
                &serde_json::to_vec(&request.request().binding).unwrap(),
            );
            let chain_file = publish(
                "chain.json",
                &serde_json::to_vec(&request.request().ek_certificate_chain).unwrap(),
            );
            let request_file = directory.join(format!("{prefix}-request.json"));
            let output = Command::new(&binary)
                .arg("request")
                .args(["--device", "/dev/tpmrm0", "--state"])
                .arg(directory.join("enrollment-state.json"))
                .arg("--binding")
                .arg(binding_file)
                .arg("--ek-certificate-chain")
                .arg(chain_file)
                .arg("--output")
                .arg(&request_file)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "collector request failed: {}",
                String::from_utf8_lossy(&output.stdout)
            );
            let cli_request = validate_request(
                super::super::format::parse(&std::fs::read(&request_file).unwrap()).unwrap(),
            )
            .unwrap();
            assert_eq!(cli_request.duk_name(), request.duk_name());
            // A new request ID requires its own signed challenge/transcript.
            let cli_challenge = Challenge {
                request_sha256: hex(cli_request.digest()),
                ..challenge.clone()
            };
            let challenge_bytes = serde_json::to_vec(&cli_challenge).unwrap();
            let envelope = serde_json::json!({"algorithm":"Ed25519","key_id":"test-only","signature":BASE64.encode(challenge_key.sign(&crate::format::signature_payload(CHALLENGE_DOMAIN,&challenge_bytes)).as_ref())});
            let challenge_file = publish("challenge.json", &challenge_bytes);
            let signature_file = publish("challenge.sig", &serde_json::to_vec(&envelope).unwrap());
            let key_file = publish("challenge.pub", challenge_key.public_key().as_ref());
            let response_file = directory.join(format!("{prefix}-response.json"));
            let output = Command::new(&binary)
                .arg("respond")
                .args(["--device", "/dev/tpmrm0", "--state"])
                .arg(directory.join("enrollment-state.json"))
                .arg("--request")
                .arg(request_file)
                .arg("--challenge")
                .arg(challenge_file)
                .arg("--challenge-signature")
                .arg(signature_file)
                .args(["--challenge-key-id", "test-only", "--challenge-public-key"])
                .arg(key_file)
                .arg("--output")
                .arg(&response_file)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "collector respond failed: {}",
                String::from_utf8_lossy(&output.stdout)
            );
            verify_response(
                &cli_request,
                &cli_challenge,
                &std::fs::read(response_file).unwrap(),
                &secret,
                now,
            )
            .unwrap();
            eprintln!(
                "physical TPM: collector request/respond CLI exchange verified (test-only, no OEM certification)"
            );
        }
        eprintln!(
            "physical TPM: signed challenge activation and CertifyCreation verified; OEM chain deliberately not accepted"
        );
        let duk = load(&mut context, handles.duk).unwrap();
        assert_policy_unwrap(&mut context, duk, &policy_public, &signer);
    }
}
