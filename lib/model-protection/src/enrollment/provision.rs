// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Two-phase enrollment provisioning. Persist the returned journal before commit.
//! The journal contains TPM-wrapped child blobs, not exportable private keys.

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use tss_esapi::{
    Context,
    abstraction::{ak, ek},
    constants::CapabilityType,
    handles::{KeyHandle, PersistentTpmHandle, TpmHandle},
    interface_types::{
        algorithm::{AsymmetricAlgorithm, HashingAlgorithm, SignatureSchemeAlgorithm},
        dynamic_handles::Persistent,
        key_bits::RsaKeyBits,
        resource_handles::{Hierarchy, Provision},
    },
    structures::{CapabilityData, Private, Public, RsaExponent, SymmetricDefinitionObject},
    traits::{Marshall, UnMarshall},
};

use super::{
    EnrollmentError as Error, Result,
    format::{decode_fixed_hex, hex},
};

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Handles {
    pub ek: u32,
    pub ak: u32,
    pub duk: u32,
}

impl Handles {
    pub fn validate(&self) -> Result<()> {
        let values = self.values();
        if values
            .iter()
            .enumerate()
            .any(|(i, h)| !(0x8100_0000..=0x817f_ffff).contains(h) || values[..i].contains(h))
        {
            return Err(Error::Input);
        }
        Ok(())
    }
    fn values(&self) -> [u32; 3] {
        [self.ek, self.ak, self.duk]
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationState {
    pub format: String,
    pub format_version: u16,
    pub ek_handle: u32,
    pub ak_handle: u32,
    pub duk_handle: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_authority_handle: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_authority_public: Option<String>,
    pub ek_name: String,
    pub ak_name: String,
    pub duk_name: String,
    pub policy_authority_name: String,
    pub duk_creation_hash: String,
    #[serde(default)]
    pub duk_creation_ticket: Option<String>,
}

impl ActivationState {
    pub fn validate(&self) -> Result<()> {
        if self.format != "model-protection-enrollment-state"
            || !matches!(self.format_version, 1 | 2)
            || (self.format_version == 1 && self.duk_creation_ticket.is_some())
            || (self.format_version == 2 && self.duk_creation_ticket.is_none())
        {
            return Err(Error::Input);
        }
        Handles {
            ek: self.ek_handle,
            ak: self.ak_handle,
            duk: self.duk_handle,
        }
        .validate()?;
        match (
            self.format_version,
            self.policy_authority_handle,
            &self.policy_authority_public,
        ) {
            (1, Some(handle), None)
                if (0x8100_0000..=0x817f_ffff).contains(&handle)
                    && ![self.ek_handle, self.ak_handle, self.duk_handle].contains(&handle) => {}
            (2, None, Some(encoded)) => {
                let bytes = decode(encoded, 88)?;
                policy_public(&bytes)?;
                if name(&bytes) != self.policy_authority_name {
                    return Err(Error::Binding);
                }
            }
            _ => return Err(Error::Input),
        }
        for name in [
            &self.ek_name,
            &self.ak_name,
            &self.duk_name,
            &self.policy_authority_name,
        ] {
            if decode_fixed_hex::<34>(name)?[..2] != [0, 0x0b] {
                return Err(Error::Input);
            }
        }
        decode_fixed_hex::<32>(&self.duk_creation_hash)?;
        if let Some(ticket) = &self.duk_creation_ticket {
            let bytes = decode(ticket, 72)?;
            if bytes.len() < 9
                || bytes[..6] != [0x80, 0x21, 0x40, 0, 0, 1]
                || bytes.len() != 8 + usize::from(u16::from_be_bytes([bytes[6], bytes[7]]))
            {
                return Err(Error::Input);
            }
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedEnrollment {
    format: String,
    format_version: u16,
    handles: Handles,
    policy_public: String,
    ek_public: String,
    ak_public: String,
    ak_private: String,
    duk_public: String,
    duk_private: String,
    storage_name: String,
    state: ActivationState,
}

fn decode(value: &str, max: usize) -> Result<Vec<u8>> {
    if value.len() > max.div_ceil(3) * 4 {
        return Err(Error::Input);
    }
    let bytes = BASE64.decode(value).map_err(|_| Error::Input)?;
    if bytes.is_empty() || bytes.len() > max || BASE64.encode(&bytes) != value {
        return Err(Error::Input);
    }
    Ok(bytes)
}

fn name(public: &[u8]) -> String {
    hex(&[&[0, 0x0b][..], Sha256::digest(public).as_slice()].concat())
}

fn public(bytes: &[u8]) -> Result<Public> {
    let public = Public::unmarshall(bytes).map_err(|_| Error::Input)?;
    if public.marshall().map_err(|_| Error::Input)? != bytes {
        return Err(Error::Input);
    }
    Ok(public)
}

fn policy_public(bytes: &[u8]) -> Result<Public> {
    crate::validate_tpm_policy_authority_public(bytes).map_err(|_| Error::Input)?;
    public(bytes)
}

fn occupied(context: &mut Context) -> Result<Vec<u32>> {
    let (data, more) = context
        .get_capability(CapabilityType::Handles, 0x8100_0000, 64)
        .map_err(|_| Error::Proof)?;
    let CapabilityData::Handles(handles) = data else {
        return Err(Error::Proof);
    };
    if more {
        return Err(Error::Conflict);
    }
    Ok(handles.into_inner().into_iter().map(u32::from).collect())
}

fn read_key(context: &mut Context, handle: u32) -> Result<KeyHandle> {
    context
        .tr_from_tpm_public(TpmHandle::Persistent(
            PersistentTpmHandle::new(handle).map_err(|_| Error::Input)?,
        ))
        .map(KeyHandle::from)
        .map_err(|_| Error::Proof)
}

fn storage(context: &mut Context) -> Result<KeyHandle> {
    let template = tss_esapi::utils::create_restricted_decryption_rsa_public(
        SymmetricDefinitionObject::AES_128_CFB,
        RsaKeyBits::Rsa2048,
        RsaExponent::default(),
    )
    .map_err(|_| Error::Input)?;
    context
        .execute_with_nullauth_session(|context| {
            context.create_primary(Hierarchy::Owner, template, None, None, None, None)
        })
        .map(|key| key.key_handle)
        .map_err(|_| Error::Proof)
}

/// Generate all objects transiently, without modifying persistent handles.
pub fn prepare(
    context: &mut Context,
    handles: Handles,
    policy: &[u8],
) -> Result<PreparedEnrollment> {
    handles.validate()?;
    let policy_object = context
        .load_external_public(policy_public(policy)?, Hierarchy::Owner)
        .map_err(|_| Error::Input)?;
    context
        .flush_context(policy_object.into())
        .map_err(|_| Error::Proof)?;
    let present = occupied(context)?;
    if handles
        .values()
        .iter()
        .any(|handle| present.contains(handle))
    {
        return Err(Error::Conflict);
    }
    let ek =
        ek::create_ek_object(context, AsymmetricAlgorithm::Rsa, None).map_err(|_| Error::Proof)?;
    let (ek_public, ek_name, _) = context.read_public(ek).map_err(|_| Error::Proof)?;
    let ak_result = ak::create_ak(
        context,
        ek,
        HashingAlgorithm::Sha256,
        SignatureSchemeAlgorithm::RsaSsa,
        None,
        None,
    )
    .map_err(|_| Error::Proof)?;
    // Verify the wrapped AK can be loaded before committing its journal.
    let ak_public = BASE64.encode(ak_result.out_public.marshall().map_err(|_| Error::Input)?);
    let ak_private = BASE64.encode(ak_result.out_private.value());
    let ak = ak::load_ak(
        context,
        ek,
        None,
        ak_result.out_private,
        ak_result.out_public,
    )
    .map_err(|_| Error::Proof)?;
    let (_, ak_name, _) = context.read_public(ak).map_err(|_| Error::Proof)?;
    context.flush_context(ak.into()).map_err(|_| Error::Proof)?;
    context.flush_context(ek.into()).map_err(|_| Error::Proof)?;
    let storage = storage(context)?;
    let (_, storage_name, _) = context.read_public(storage).map_err(|_| Error::Proof)?;
    let policy_name: [u8; 34] = decode_fixed_hex(&name(policy))?;
    let policy_ref: [u8; 32] = decode_fixed_hex(crate::TPM_POLICY_REF_HEX)?;
    let mut duk_template = vec![0, 1, 0, 0x0b, 0, 2, 4, 0xb2, 0, 0x20];
    duk_template.extend_from_slice(&crate::policy_authorize_auth_policy(
        &policy_name,
        &policy_ref,
    ));
    duk_template.extend_from_slice(&[0, 0x10, 0, 0x10, 8, 0, 0, 0, 0, 0, 0, 0]);
    let duk_result = context
        .execute_with_nullauth_session(|context| {
            context.create(
                storage,
                Public::unmarshall(&duk_template)?,
                None,
                None,
                None,
                None,
            )
        })
        .map_err(|_| Error::Proof)?;
    let duk_public = duk_result.out_public.marshall().map_err(|_| Error::Input)?;
    crate::validate_tpm_device_public(&duk_public, &policy_name).map_err(|_| Error::Input)?;
    context
        .flush_context(storage.into())
        .map_err(|_| Error::Proof)?;
    let ticket: tss_esapi::tss2_esys::TPMT_TK_CREATION = duk_result
        .creation_ticket
        .try_into()
        .map_err(|_| Error::Input)?;
    let ticket_bytes = [
        ticket.tag.to_be_bytes().as_slice(),
        ticket.hierarchy.to_be_bytes().as_slice(),
        ticket.digest.size.to_be_bytes().as_slice(),
        &ticket.digest.buffer[..usize::from(ticket.digest.size)],
    ]
    .concat();
    let state = ActivationState {
        format: "model-protection-enrollment-state".into(),
        format_version: 2,
        ek_handle: handles.ek,
        ak_handle: handles.ak,
        duk_handle: handles.duk,
        policy_authority_handle: None,
        policy_authority_public: Some(BASE64.encode(policy)),
        ek_name: hex(ek_name.value()),
        ak_name: hex(ak_name.value()),
        duk_name: name(&duk_public),
        policy_authority_name: hex(&policy_name),
        duk_creation_hash: hex(duk_result.creation_hash.value()),
        duk_creation_ticket: Some(BASE64.encode(ticket_bytes)),
    };
    state.validate()?;
    Ok(PreparedEnrollment {
        format: "model-protection-provision-journal".into(),
        format_version: 1,
        handles,
        policy_public: BASE64.encode(policy),
        ek_public: BASE64.encode(ek_public.marshall().map_err(|_| Error::Input)?),
        ak_public,
        ak_private,
        duk_public: BASE64.encode(duk_public),
        duk_private: BASE64.encode(duk_result.out_private.value()),
        storage_name: hex(storage_name.value()),
        state,
    })
}

impl PreparedEnrollment {
    pub fn validate_inputs(&self, handles: &Handles, policy: &[u8]) -> Result<()> {
        self.handles.validate()?;
        self.state.validate()?;
        if self.format != "model-protection-provision-journal"
            || self.format_version != 1
            || &self.handles != handles
            || decode(&self.policy_public, 88)? != policy
        {
            return Err(Error::Binding);
        }
        policy_public(policy)?;
        // Reject malformed journal blobs before any persistent-handle mutation.
        Private::try_from(decode(&self.ak_private, 2048)?).map_err(|_| Error::Input)?;
        Private::try_from(decode(&self.duk_private, 2048)?).map_err(|_| Error::Input)?;
        let names = [
            &self.state.ek_name,
            &self.state.ak_name,
            &self.state.duk_name,
            &self.state.policy_authority_name,
        ];
        for (bytes, expected) in self.publics()?.iter().zip(names) {
            public(bytes)?;
            if name(bytes) != *expected {
                return Err(Error::Binding);
            }
        }
        if handles.values()
            != [
                self.state.ek_handle,
                self.state.ak_handle,
                self.state.duk_handle,
            ]
        {
            return Err(Error::Binding);
        }
        crate::validate_tpm_device_public(
            &decode(&self.duk_public, 310)?,
            &decode_fixed_hex(&self.state.policy_authority_name)?,
        )
        .map_err(|_| Error::Input)?;
        decode_fixed_hex::<34>(&self.storage_name)?;
        Ok(())
    }
    fn publics(&self) -> Result<[Vec<u8>; 4]> {
        Ok([
            decode(&self.ek_public, 314)?,
            decode(&self.ak_public, 280)?,
            decode(&self.duk_public, 310)?,
            decode(&self.policy_public, 88)?,
        ])
    }
    pub fn state(&self) -> &ActivationState {
        &self.state
    }
}

fn persist(
    context: &mut Context,
    target: u32,
    expected: &[u8],
    create: impl FnOnce(&mut Context) -> Result<KeyHandle>,
) -> Result<KeyHandle> {
    if occupied(context)?.contains(&target) {
        let key = read_key(context, target)?;
        if context
            .read_public(key)
            .map_err(|_| Error::Proof)?
            .0
            .marshall()
            .map_err(|_| Error::Input)?
            != expected
        {
            return Err(Error::Conflict);
        }
        return Ok(key);
    }
    let transient = create(context)?;
    let checked = context
        .read_public(transient)
        .map_err(|_| Error::Proof)?
        .0
        .marshall()
        .map_err(|_| Error::Input)?;
    if checked != expected {
        return Err(Error::Binding);
    }
    // EvictControl is only ever passed a freshly loaded transient object. If an
    // unrelated key occupies target in a race, TPM rejects it; it is not evicted.
    context
        .execute_with_temporary_object(transient.into(), |context, object| {
            context.execute_with_nullauth_session(|context| {
                context.evict_control(
                    Provision::Owner,
                    object,
                    Persistent::Persistent(PersistentTpmHandle::new(target)?),
                )
            })
        })
        .map(KeyHandle::from)
        .map_err(|_| Error::Proof)
}

/// Resume only the journal's exact objects, never replace occupied handles.
pub fn commit(context: &mut Context, prepared: &PreparedEnrollment) -> Result<()> {
    commit_with_checkpoint(context, prepared, |_| Ok(()))
}

pub(super) fn commit_with_checkpoint(
    context: &mut Context,
    prepared: &PreparedEnrollment,
    mut checkpoint: impl FnMut(usize) -> Result<()>,
) -> Result<()> {
    let publics = prepared.publics()?;
    prepared.validate_inputs(&prepared.handles, &publics[3])?;
    let targets = prepared.handles.values();
    // Preflight every existing target before adding any persistent object.
    let present = occupied(context)?;
    for (target, expected) in targets.into_iter().zip(&publics) {
        if present.contains(&target) {
            let key = read_key(context, target)?;
            if context
                .read_public(key)
                .map_err(|_| Error::Proof)?
                .0
                .marshall()
                .map_err(|_| Error::Input)?
                != *expected
            {
                return Err(Error::Conflict);
            }
        }
    }
    let ek = persist(context, targets[0], &publics[0], |context| {
        ek::create_ek_object(context, AsymmetricAlgorithm::Rsa, None).map_err(|_| Error::Proof)
    })?;
    checkpoint(1)?;
    persist(context, targets[1], &publics[1], |context| {
        ak::load_ak(
            context,
            ek,
            None,
            Private::try_from(decode(&prepared.ak_private, 2048)?).map_err(|_| Error::Input)?,
            public(&publics[1])?,
        )
        .map_err(|_| Error::Proof)
    })?;
    checkpoint(2)?;
    persist(context, targets[2], &publics[2], |context| {
        let storage = storage(context)?;
        if hex(context
            .read_public(storage)
            .map_err(|_| Error::Proof)?
            .1
            .value())
            != prepared.storage_name
        {
            return Err(Error::Binding);
        }
        let private =
            Private::try_from(decode(&prepared.duk_private, 2048)?).map_err(|_| Error::Input)?;
        let public = public(&publics[2])?;
        let key = context
            .execute_with_nullauth_session(|context| context.load(storage, private, public))
            .map_err(|_| Error::Proof)?;
        context
            .flush_context(storage.into())
            .map_err(|_| Error::Proof)?;
        Ok(key)
    })?;
    checkpoint(3)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_are_distinct_owner_persistent_targets() {
        let handles = Handles {
            ek: 0x8101_2001,
            ak: 0x8101_2002,
            duk: 0x8101_2003,
        };
        handles.validate().unwrap();
        assert!(
            Handles {
                ak: handles.ek,
                ..handles
            }
            .validate()
            .is_err()
        );
        for ek in [0x8000_0000, 0x8180_0000, 0x4000_0001] {
            assert!(Handles { ek, ..handles }.validate().is_err());
        }
    }

    #[test]
    fn v2_state_binds_public_policy_and_original_owner_creation_ticket() {
        let mut point = [0x22; 65];
        point[0] = 4;
        let public = crate::tpm_policy_authority_public(&point).unwrap();
        let mut state = ActivationState {
            format: "model-protection-enrollment-state".into(),
            format_version: 2,
            ek_handle: 0x8101_2001,
            ak_handle: 0x8101_2002,
            duk_handle: 0x8101_2003,
            policy_authority_handle: None,
            policy_authority_public: Some(BASE64.encode(&public)),
            ek_name: hex(&[&[0, 0x0b][..], &[1; 32]].concat()),
            ak_name: hex(&[&[0, 0x0b][..], &[2; 32]].concat()),
            duk_name: hex(&[&[0, 0x0b][..], &[3; 32]].concat()),
            policy_authority_name: name(&public),
            duk_creation_hash: hex(&[4; 32]),
            duk_creation_ticket: Some(BASE64.encode([0x80, 0x21, 0x40, 0, 0, 1, 0, 1, 5])),
        };
        state.validate().unwrap();
        state.policy_authority_handle = Some(0x8101_2004);
        assert!(state.validate().is_err());
        state.policy_authority_handle = None;
        state.duk_creation_ticket = Some(BASE64.encode([0x80, 0x21, 0x40, 0, 0, 1, 0, 2, 5]));
        assert!(state.validate().is_err());
        state.duk_creation_ticket = None;
        assert!(state.validate().is_err());
    }
}
