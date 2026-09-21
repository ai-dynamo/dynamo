// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(all(target_os = "linux", feature = "tpm2"))]
use zeroize::Zeroize;
use zeroize::Zeroizing;

#[cfg(all(target_os = "linux", feature = "tpm2"))]
use rustix::mm::{Advice, MapFlags, ProtFlags, madvise, mlock, mmap_anonymous, munlock, munmap};
#[cfg(all(target_os = "linux", feature = "tpm2"))]
use std::ffi::c_void;
#[cfg(all(target_os = "linux", feature = "tpm2"))]
use std::ptr::NonNull;

/// An owned model data-encryption key. Its bytes are never exposed by the public API.
pub struct SecretDek(SecretStorage);

enum SecretStorage {
    #[allow(dead_code)]
    Test(Zeroizing<[u8; 32]>),
    #[cfg(all(target_os = "linux", feature = "tpm2"))]
    Locked(LockedSecret),
}

#[cfg(all(target_os = "linux", feature = "tpm2"))]
struct LockedSecret {
    page: NonNull<c_void>,
    page_size: usize,
}

#[cfg(all(target_os = "linux", feature = "tpm2"))]
// SAFETY: the mapping is uniquely owned, has no interior aliases, and is only
// read through `SecretDek::as_bytes` while its owner is alive.
unsafe impl Send for LockedSecret {}

#[cfg(all(target_os = "linux", feature = "tpm2"))]
impl LockedSecret {
    fn new(bytes: &[u8; 32]) -> crate::Result<Self> {
        let page_size = rustix::param::page_size();
        // SAFETY: a null hint requests a fresh private anonymous mapping.
        let page = unsafe {
            mmap_anonymous(
                std::ptr::null_mut(),
                page_size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::PRIVATE,
            )
        }
        .map_err(|_| crate::ProtectionError::SecretMemoryUnavailable)?;
        let page = match NonNull::new(page) {
            Some(page) => page,
            None => {
                // SAFETY: even an unexpected address-zero mapping must not leak.
                let _ = unsafe { munmap(page, page_size) };
                return Err(crate::ProtectionError::SecretMemoryUnavailable);
            }
        };
        // SAFETY: `page` owns one writable mapping of `page_size` bytes.
        if unsafe { mlock(page.as_ptr(), page_size) }.is_err()
            || unsafe { madvise(page.as_ptr(), page_size, Advice::LinuxDontDump) }.is_err()
        {
            // SAFETY: the mapping was created above and has no Rust references.
            let _ = unsafe { munlock(page.as_ptr(), page_size) };
            let _ = unsafe { munmap(page.as_ptr(), page_size) };
            return Err(crate::ProtectionError::SecretMemoryUnavailable);
        }
        // SAFETY: the mapping is writable and at least 32 bytes long.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), page.as_ptr().cast(), 32) };
        Ok(Self { page, page_size })
    }

    fn bytes(&self) -> &[u8; 32] {
        // SAFETY: the first 32 bytes are initialized in `new` and remain mapped.
        unsafe { &*self.page.as_ptr().cast::<[u8; 32]>() }
    }
}

#[cfg(all(target_os = "linux", feature = "tpm2"))]
impl Drop for LockedSecret {
    fn drop(&mut self) {
        // SAFETY: this owner has exclusive access to the live mapping.
        unsafe { std::slice::from_raw_parts_mut(self.page.as_ptr().cast::<u8>(), 32) }.zeroize();
        // SAFETY: no references survive this drop and the mapping was created by `new`.
        let _ = unsafe { munlock(self.page.as_ptr(), self.page_size) };
        let _ = unsafe { munmap(self.page.as_ptr(), self.page_size) };
    }
}

impl SecretDek {
    #[cfg(all(target_os = "linux", feature = "tpm2"))]
    pub(crate) fn from_tpm(mut bytes: [u8; 32]) -> crate::Result<Self> {
        let secret = LockedSecret::new(&bytes).map(|secret| Self(SecretStorage::Locked(secret)));
        bytes.zeroize();
        secret
    }

    #[cfg(test)]
    pub(crate) fn new(bytes: [u8; 32]) -> Self {
        Self(SecretStorage::Test(Zeroizing::new(bytes)))
    }

    pub(crate) fn as_bytes(&self) -> &[u8; 32] {
        match &self.0 {
            SecretStorage::Test(bytes) => bytes,
            #[cfg(all(target_os = "linux", feature = "tpm2"))]
            SecretStorage::Locked(secret) => secret.bytes(),
        }
    }
}

impl std::fmt::Debug for SecretDek {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SecretDek([REDACTED])")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_never_exposes_key_bytes() {
        let key = SecretDek::new([0x5a; 32]);
        assert_eq!(format!("{key:?}"), "SecretDek([REDACTED])");
    }

    #[cfg(all(target_os = "linux", feature = "tpm2"))]
    #[test]
    fn tpm_key_requires_locked_nondump_memory() {
        let key = SecretDek::from_tpm([0x5a; 32]).expect("lock one secret page");
        assert_eq!(key.as_bytes(), &[0x5a; 32]);
    }
}
