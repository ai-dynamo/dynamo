//! # Storage Registration Examples
//!
//! This module demonstrates how to use the different registration traits
//! for both runtime and compile-time scenarios.

use super::storage::*;

/// Example of runtime registration checking
/// Use this pattern when you have mixed storage types and need to conditionally register
pub fn runtime_registration_example() {
    let system_allocator = SystemAllocator::default();
    let mut system_storage = system_allocator.allocate(1024).unwrap();

    let null_allocator = tests::NullDeviceAllocator;
    let mut null_storage = null_allocator.allocate(1024).unwrap();

    // Runtime checking - works for any storage type
    register_if_supported(&mut system_storage);
    register_if_supported(&mut null_storage);
}

/// Generic function that works with any storage type
/// Uses runtime checking to conditionally register
fn register_if_supported<S: MaybeRegisterable>(storage: &mut S) {
    if storage.supports_registration() {
        println!("Storage supports registration, attempting to register...");
        // In real code, you'd create an actual handle here
        // storage.try_register("my_key", handle).unwrap();
    } else {
        println!("Storage does not support registration, skipping...");
    }
}

/// Example of compile-time registration guarantees
/// Use this pattern when you need to ensure at compile time that storage can be registered
pub fn compile_time_registration_example() {
    let system_allocator = SystemAllocator::default();
    let system_storage = system_allocator.allocate(1024).unwrap();

    // This function only accepts storage types that implement RegisterableStorage
    register_guaranteed(system_storage);

    // This would fail to compile:
    // let null_allocator = tests::NullDeviceAllocator;
    // let null_storage = null_allocator.allocate(1024).unwrap();
    // register_guaranteed(null_storage); // Compile error!
}

/// Function that requires compile-time registration guarantees
/// Only storage types that implement RegisterableStorage can be passed here
fn register_guaranteed<S: RegisterableStorage>(mut storage: S) {
    println!("Storage is guaranteed to support registration");
    // In real code, you'd create an actual handle here
    // storage.register("my_key", handle).unwrap();
}

/// Example of NIXL-specific compile-time guarantees
/// Use this pattern when you specifically need NIXL registration
pub fn nixl_registration_example() {
    // This would be used with storage types that implement NixlRegisterable
    // For example, PinnedStorage or other NIXL-compatible storage types

    // fn register_with_nixl<S: NixlRegisterable>(mut storage: S, agent: &nixl_sys::Agent) {
    //     storage.nixl_register(agent, None).unwrap();
    // }
}

/// Example showing how to handle mixed storage types in a collection
pub fn mixed_storage_collection_example() {
    // Vector of different storage types, all implementing MaybeRegisterable
    let mut storages: Vec<Box<dyn MaybeRegisterable>> = vec![
        Box::new(SystemAllocator::default().allocate(1024).unwrap()),
        Box::new(tests::NullDeviceAllocator.allocate(1024).unwrap()),
        Box::new(tests::NullHostAllocator.allocate(1024).unwrap()),
    ];

    // Process each storage, registering only those that support it
    for (i, storage) in storages.iter_mut().enumerate() {
        if storage.supports_registration() {
            println!("Storage {} supports registration", i);
            // storage.try_register(&format!("key_{}", i), handle).unwrap();
        } else {
            println!("Storage {} does not support registration", i);
        }
    }
}

/// Example of a generic function that works with different registration levels
pub fn flexible_registration_function<S>(mut storage: S)
where
    S: MaybeRegisterable + 'static,
{
    // Always works - runtime check
    if storage.supports_registration() {
        println!("Attempting runtime registration...");
        // storage.try_register("runtime_key", handle).unwrap();
    }

    // Only works if S also implements RegisterableStorage
    if let Ok(registerable) = try_as_registerable(storage) {
        println!("Storage has compile-time registration guarantees");
        // registerable.register("compile_time_key", handle).unwrap();
    }
}

/// Helper function to check if storage implements RegisterableStorage
fn try_as_registerable<S: MaybeRegisterable + 'static>(storage: S) -> Result<S, S>
where
    S: RegisterableStorage,
{
    Ok(storage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_registration() {
        runtime_registration_example();
    }

    #[test]
    fn test_compile_time_registration() {
        compile_time_registration_example();
    }

    #[test]
    fn test_mixed_storage_collection() {
        mixed_storage_collection_example();
    }
}
