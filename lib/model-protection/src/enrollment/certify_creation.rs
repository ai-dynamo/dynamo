// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Narrow RAII bridge for the command not exposed by the pinned tss-esapi API.
//! Never borrows or transmutes the private layout of tss-esapi::Context.

use std::ffi::CString;
use std::ptr;

use tss_esapi::tss2_esys as ffi;

use super::{EnrollmentError as Error, Result};

struct NativeContext {
    esys: *mut ffi::ESYS_CONTEXT,
    tcti: *mut ffi::TSS2_TCTI_CONTEXT,
}

impl Drop for NativeContext {
    fn drop(&mut self) {
        // SAFETY: these pointers are owned by this wrapper, initialized by TSS,
        // and finalized once, ESYS before its transport. TSS accepts null pointers.
        unsafe {
            ffi::Esys_Finalize(&mut self.esys);
            ffi::Tss2_TctiLdr_Finalize(&mut self.tcti);
        }
    }
}

struct Allocation<T>(*mut T);

impl<T> Default for Allocation<T> {
    fn default() -> Self {
        Self(ptr::null_mut())
    }
}

impl<T> Allocation<T> {
    fn get(&self) -> Result<&T> {
        // SAFETY: a non-null pointer is assigned only by the matching ESYS
        // output parameter; its allocation lives until this wrapper is dropped.
        unsafe { self.0.as_ref() }.ok_or(Error::Proof)
    }
}

impl<T> Drop for Allocation<T> {
    fn drop(&mut self) {
        // SAFETY: outputs are ESYS-owned allocations (or null), freed once.
        unsafe { ffi::Esys_Free(self.0.cast()) };
    }
}

impl NativeContext {
    fn connect(device: &str) -> Result<Self> {
        let number = device
            .strip_prefix("/dev/tpmrm")
            .filter(|value| {
                !value.is_empty()
                    && value.len() <= 3
                    && value.bytes().all(|byte| byte.is_ascii_digit())
            })
            .ok_or(Error::Input)?;
        number.parse::<u8>().map_err(|_| Error::Input)?;
        let configuration = CString::new(format!("device:{device}")).map_err(|_| Error::Input)?;
        Self::initialize(&configuration)
    }

    fn initialize(configuration: &CString) -> Result<Self> {
        let mut context = Self {
            esys: ptr::null_mut(),
            tcti: ptr::null_mut(),
        };
        // SAFETY: configuration is a live C string; output slots belong to this
        // owner. No environment/default TCTI or simulator fallback is selected.
        let transport =
            unsafe { ffi::Tss2_TctiLdr_Initialize(configuration.as_ptr(), &mut context.tcti) };
        if transport != 0 {
            return Err(Error::Proof);
        }
        // SAFETY: initialized transport remains owned until after ESYS finalization.
        let initialized =
            unsafe { ffi::Esys_Initialize(&mut context.esys, context.tcti, ptr::null_mut()) };
        if initialized != 0 {
            return Err(Error::Proof);
        }
        // SAFETY: context has been initialized. Bound transport waits on a failed TPM.
        if unsafe { ffi::Esys_SetTimeout(context.esys, 5_000) } != 0 {
            return Err(Error::Proof);
        }
        Ok(context)
    }

    fn persistent(&mut self, handle: u32, expected_name: &[u8; 34]) -> Result<ffi::ESYS_TR> {
        if !(0x8100_0000..=0x81ff_ffff).contains(&handle) {
            return Err(Error::Input);
        }
        let mut object = ffi::ESYS_TR_NONE;
        // SAFETY: context is initialized; the object output slot is writable.
        let rc = unsafe {
            ffi::Esys_TR_FromTPMPublic(
                self.esys,
                handle,
                ffi::ESYS_TR_NONE,
                ffi::ESYS_TR_NONE,
                ffi::ESYS_TR_NONE,
                &mut object,
            )
        };
        if rc != 0 {
            return Err(Error::Proof);
        }
        let mut name = Allocation::<ffi::TPM2B_NAME>::default();
        // SAFETY: ESYS owns this resource; allocation is immediately RAII-managed.
        let rc = unsafe { ffi::Esys_TR_GetName(self.esys, object, &mut name.0) };
        if rc != 0 {
            return Err(Error::Proof);
        }
        let name = name.get()?;
        if name.size != 34 || name.name[..34] != *expected_name {
            return Err(Error::Binding);
        }
        Ok(object)
    }
}

fn creation_ticket(bytes: &[u8]) -> Result<ffi::TPMT_TK_CREATION> {
    if bytes.len() < 8 || bytes[..6] != [0x80, 0x21, 0x40, 0, 0, 1] {
        return Err(Error::Input);
    }
    let size = u16::from_be_bytes([bytes[6], bytes[7]]);
    if size == 0 || size > 64 || bytes.len() != 8 + usize::from(size) {
        return Err(Error::Input);
    }
    let mut ticket = ffi::TPMT_TK_CREATION {
        tag: 0x8021,
        hierarchy: 0x4000_0001,
        ..Default::default()
    };
    ticket.digest.size = size;
    ticket.digest.buffer[..usize::from(size)].copy_from_slice(&bytes[8..]);
    Ok(ticket)
}

pub(super) struct CreationProof {
    pub attestation: Vec<u8>,
    pub signature: Vec<u8>,
}

/// CertifyCreation authorizes the AK only, not a password path on the policy-only DUK.
pub(super) fn certify(
    device: &str,
    ak: (u32, &[u8; 34]),
    duk: (u32, &[u8; 34]),
    hash: &[u8; 32],
    transcript: &[u8; 32],
    ticket: &[u8],
) -> Result<CreationProof> {
    certify_in_context(
        NativeContext::connect(device)?,
        ak,
        duk,
        hash,
        transcript,
        ticket,
    )
}

#[cfg(all(test, feature = "enrollment-authority"))]
pub(super) fn certify_test_tcti(
    configuration: &str,
    ak: (u32, &[u8; 34]),
    duk: (u32, &[u8; 34]),
    hash: &[u8; 32],
    transcript: &[u8; 32],
    ticket: &[u8],
) -> Result<CreationProof> {
    let configuration = CString::new(configuration).map_err(|_| Error::Input)?;
    certify_in_context(
        NativeContext::initialize(&configuration)?,
        ak,
        duk,
        hash,
        transcript,
        ticket,
    )
}

fn certify_in_context(
    mut context: NativeContext,
    ak: (u32, &[u8; 34]),
    duk: (u32, &[u8; 34]),
    hash: &[u8; 32],
    transcript: &[u8; 32],
    ticket: &[u8],
) -> Result<CreationProof> {
    let ticket = creation_ticket(ticket)?;
    let ak = context.persistent(ak.0, ak.1)?;
    let duk = context.persistent(duk.0, duk.1)?;
    let mut qualifying = ffi::TPM2B_DATA {
        size: 32,
        ..Default::default()
    };
    qualifying.buffer[..32].copy_from_slice(transcript);
    let mut creation_hash = ffi::TPM2B_DIGEST {
        size: 32,
        ..Default::default()
    };
    creation_hash.buffer[..32].copy_from_slice(hash);
    let scheme = ffi::TPMT_SIG_SCHEME {
        scheme: 0x0010,
        ..Default::default()
    };
    let mut attestation = Allocation::<ffi::TPM2B_ATTEST>::default();
    let mut signature = Allocation::<ffi::TPMT_SIGNATURE>::default();
    // SAFETY: all inputs are initialized bounded TSS structures, handles belong
    // to this live context, and both ESYS allocations are freed on every path.
    let rc = unsafe {
        ffi::Esys_CertifyCreation(
            context.esys,
            ak,
            duk,
            ffi::ESYS_TR_PASSWORD,
            ffi::ESYS_TR_NONE,
            ffi::ESYS_TR_NONE,
            &qualifying,
            &creation_hash,
            &scheme,
            &ticket,
            &mut attestation.0,
            &mut signature.0,
        )
    };
    if rc != 0 {
        return Err(Error::Proof);
    }
    let attestation = attestation.get()?;
    let signature = signature.get()?;
    if attestation.size == 0 || attestation.size > 2048 || signature.sigAlg != 0x0014 {
        return Err(Error::Proof);
    }
    // SAFETY: the discriminant above selects the RSASSA union member.
    let rsa = unsafe { &signature.signature.rsassa };
    if rsa.hash != 0x000b || rsa.sig.size != 256 {
        return Err(Error::Proof);
    }
    Ok(CreationProof {
        attestation: attestation.attestationData[..usize::from(attestation.size)].to_vec(),
        signature: rsa.sig.buffer[..256].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticket_parser_rejects_null_wrong_hierarchy_and_trailing_data() {
        let valid = [vec![0x80, 0x21, 0x40, 0, 0, 1, 0, 32], vec![1; 32]].concat();
        assert!(creation_ticket(&valid).is_ok());
        for offset in [0, 1, 2, 5, 6, 7] {
            let mut wrong = valid.clone();
            wrong[offset] ^= 1;
            assert!(creation_ticket(&wrong).is_err());
        }
        assert!(creation_ticket(&valid[..39]).is_err());
        assert!(creation_ticket(&[valid, b"x".to_vec()].concat()).is_err());
    }
}
