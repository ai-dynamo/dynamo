// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Debug)]
pub struct Allocation<S: Storage> {
    storage: S,
    handles: RegistrationHandles,
}

impl<S: Storage> Allocation<S> {
    pub fn new(storage: S) -> Self {
        Self {
            storage,
            handles: RegistrationHandles::new(),
        }
    }

    pub fn is_registered(&self, key: &str) -> bool {
        self.handles.is_registered(key)
    }

    pub fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.registration_handle(key)
    }
}

impl<S: Storage> Allocation<S> {
    pub fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        if !self.storage.registerable() {
            return Err(StorageError::StorageIsNotRegisterable);
        }

        self.handles.register(key, handle)
    }
}

impl<S: Storage> MaybeRegisterable for Allocation<S> {
    fn supports_registration(&self) -> bool {
        self.storage.supports_registration()
    }
}

impl<S: Storage> Drop for Allocation<S> {
    fn drop(&mut self) {
        self.handles.release();
    }
}

impl<S: Storage> Storage for Allocation<S> {
    fn storage_type(&self) -> StorageType {
        self.storage.storage_type()
    }

    fn addr(&self) -> u64 {
        self.storage.addr()
    }

    fn size(&self) -> usize {
        self.storage.size()
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.storage.as_ptr()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.storage.as_mut_ptr()
    }
}
