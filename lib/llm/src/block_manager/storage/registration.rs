// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

// /// Optional registration trait for storage types that may support registration
// ///
// /// This trait provides runtime checking for registration capabilities.
// /// Use this when you need to conditionally register storage based on its type.
// /// All storage types should implement this trait, even if they don't support registration.
// pub trait MaybeRegisterable: Storage {
//     /// Attempt to register a handle with a key
//     /// Returns `Ok(())` if registration succeeded
//     /// Returns `Err(StorageError::NotAccessible)` if this storage type doesn't support registration
//     /// Returns other errors for registration-specific failures
//     fn try_register(
//         &mut self,
//         _key: &str,
//         _handle: Box<dyn RegistationHandle>,
//     ) -> Result<(), StorageError> {
//         Err(StorageError::StorageIsNotRegisterable)
//     }

//     /// Check if this storage type supports registration
//     fn supports_registration(&self) -> bool {
//         false || self.registerable()
//     }

//     /// Check if a handle is registered with a key
//     /// Returns `false` if registration is not supported
//     fn is_registered(&self, _key: &str) -> bool {
//         false
//     }

//     /// Get a reference to the registration handle for a key
//     /// Returns `None` if registration is not supported or key not found
//     fn registration_handle(&self, _key: &str) -> Option<&dyn RegistationHandle> {
//         None
//     }
// }

// /// Registerable storage is a [Storage] that can be associated with one or more
// /// [RegistationHandle]s.
// ///
// /// The core concept here is that the storage might be registered with a library
// /// like NIXL or some other custom library which might make some system calls on
// /// virtual addresses of the storage.
// ///
// /// Before the [Storage] is dropped, the [RegistationHandle]s should be released.
// ///
// /// The behavior is enforced via the [Drop] implementation for [RegistrationHandles].
// ///
// /// This trait provides compile-time guarantees that the storage supports registration.
// /// Use this when you need to ensure at compile time that storage can be registered.
// /// Only storage types that actually support registration should implement this trait.
// pub trait RegisterableStorage: Storage + MaybeRegisterable {
//     fn register(
//         &mut self,
//         key: &str,
//         handle: Box<dyn RegistationHandle>,
//     ) -> Result<(), StorageError>;

//     /// Check if a handle is registered with a key
//     /// Returns `false` if registration is not supported
//     fn required_is_registered(&self, _key: &str) -> bool {
//         false
//     }

//     /// Get a reference to the registration handle for a key
//     /// Returns `None` if registration is not supported or key not found
//     fn required_registration_handle(&self, _key: &str) -> Option<&dyn RegistationHandle> {
//         None
//     }

//     /// Check if this storage type supports registration
//     /// Returns `true` if registration is supported
//     fn required_supports_registration(&self) -> bool {
//         true
//     }
// }

// impl<T> MaybeRegisterable for T
// where
//     T: RegisterableStorage,
// {
//     fn try_register(
//         &mut self,
//         key: &str,
//         handle: Box<dyn RegistationHandle>,
//     ) -> Result<(), StorageError> {
//         self.register(key, handle)
//     }

//     fn supports_registration(&self) -> bool {
//         self.required_supports_registration()
//     }

//     fn is_registered(&self, key: &str) -> bool {
//         self.required_is_registered(key)
//     }

//     fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
//         self.required_registration_handle(key)
//     }
// }

/// Designed to be implemented by any type that can be used as a handle to a
/// [RegisterableStorage].
///
/// See [RegisterableStorage] for more details.
pub trait RegistationHandle: std::any::Any + Send + Sync + 'static {
    /// Release the [RegistationHandle].
    /// This should be called when the external registration of this storage
    /// is no longer needed.
    ///
    /// Note: All [RegistrationHandle]s should be explicitly released before
    /// the [Storage] is dropped.
    fn release(&mut self);
}

/// A collection of [RegistrationHandle]s for a [RegisterableStorage].
///
/// This is used to ensure that all [RegistrationHandle]s are explicitly released
/// before the [RegisterableStorage] is dropped.
#[derive(Default)]
pub struct RegistrationHandles {
    handles: HashMap<String, Box<dyn RegistationHandle>>,
}

impl RegistrationHandles {
    /// Create a new [RegistrationHandles] instance
    pub fn new() -> Self {
        Self {
            handles: HashMap::new(),
        }
    }

    /// Register a handle with a key
    /// If a handle with the same key already exists, an error is returned
    pub fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        let key = key.to_string();
        if self.handles.contains_key(&key) {
            return Err(StorageError::RegistrationKeyExists(key));
        }
        self.handles.insert(key, handle);
        Ok(())
    }

    /// Release all handles
    pub(crate) fn release(&mut self) {
        for handle in self.handles.values_mut() {
            handle.release();
        }
        self.handles.clear();
    }

    /// Check if a handle is registered with a key
    pub(crate) fn is_registered(&self, key: &str) -> bool {
        self.handles.contains_key(key)
    }

    /// Get a reference to the registration handle for a key
    pub(crate) fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.get(key).map(|h| h.as_ref())
    }
}

impl std::fmt::Debug for RegistrationHandles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RegistrationHandles {{ count: {:?} }}",
            self.handles.len()
        )
    }
}

impl Drop for RegistrationHandles {
    fn drop(&mut self) {
        if !self.handles.is_empty() {
            panic!("RegistrationHandles dropped with {} handles remaining; RegistrationHandles::release() needs to be explicitly called", self.handles.len());
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[derive(Debug)]
    pub struct MockStorage {
        size: usize,
        registerable: bool,
    }

    impl MockStorage {
        pub fn new(size: usize, registerable: bool) -> Self {
            Self { size, registerable }
        }
    }

    impl Storage for MockStorage {
        fn storage_type(&self) -> StorageType {
            StorageType::Null
        }

        fn addr(&self) -> u64 {
            0
        }

        fn size(&self) -> usize {
            self.size
        }

        unsafe fn as_ptr(&self) -> *const u8 {
            std::ptr::null()
        }

        unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
            std::ptr::null_mut()
        }
    }

    impl MaybeRegisterable for MockStorage {
        fn supports_registration(&self) -> bool {
            self.registerable
        }
    }

    #[derive(Debug, Default)]
    struct NullHandle;

    impl RegistationHandle for NullHandle {
        fn release(&mut self) {}
    }

    #[test]
    fn test_maybe_registerable() {
        let storage = Allocation::<MockStorage>::new(MockStorage::new(1024, false));
        assert!(!storage.supports_registration());
        assert!(!storage.is_registered("test"));
        assert!(storage.registration_handle("test").is_none());
    }

    #[test]
    fn test_registerable_storage() {
        let mut storage = Allocation::<MockStorage>::new(MockStorage::new(1024, true));
        assert!(storage.supports_registration());
        assert!(!storage.is_registered("test"));
        assert!(storage.registration_handle("test").is_none());

        let handle = Box::new(NullHandle);
        storage.register("test", handle).unwrap();
        assert!(storage.is_registered("test"));
        assert!(storage.registration_handle("test").is_some());
    }
}
