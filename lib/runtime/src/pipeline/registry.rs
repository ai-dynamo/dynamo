// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// Registry struct that manages both shared and unique objects.
///
/// # Examples
///
/// ```
/// use dynamo_runtime::pipeline::registry::Registry;
///
/// let mut registry = Registry::new();
///
/// // Insert and retrieve shared objects
/// registry.insert_shared("shared1", 42);
/// assert_eq!(*registry.get_shared::<i32>("shared1").unwrap(), 42);
///
/// // Insert and take unique objects
/// registry.insert_unique("unique1", "Hello".to_string());
/// assert_eq!(registry.take_unique::<String>("unique1").unwrap(), "Hello");
///
/// // Taking the same unique again should fail since it's not cloneable
/// assert!(registry.take_unique::<String>("unique1").is_err());
///
/// // Insert and clone unique objects
/// registry.insert_unique("unique2", "World".to_string());
/// assert_eq!(registry.clone_unique::<String>("unique2").unwrap(), "World");
///
/// // Taking the same cloned unique should is ok
/// assert!(registry.take_unique::<String>("unique2").is_ok());
///
/// ```
#[derive(Debug, Default)]
pub struct Registry {
    shared_storage: HashMap<String, Arc<dyn Any + Send + Sync>>, // Shared objects
    unique_storage: HashMap<String, Box<dyn Any + Send + Sync>>, // Takable objects
}

impl Registry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Registry {
            shared_storage: HashMap::new(),
            unique_storage: HashMap::new(),
        }
    }

    /// Check if a shared object exists in the registry by key.
    pub fn contains_shared(&self, key: &str) -> bool {
        self.shared_storage.contains_key(key)
    }

    /// Insert a shared object into the registry with a specific key.
    pub fn insert_shared<K: ToString, U: Send + Sync + 'static>(&mut self, key: K, value: U) {
        self.shared_storage.insert(
            key.to_string(),
            Arc::new(value) as Arc<dyn Any + Send + Sync>,
        );
    }

    /// Retrieve a shared object from the registry by key and type.
    pub fn get_shared<V: Send + Sync + 'static>(&self, key: &str) -> Result<Arc<V>, String> {
        match self.shared_storage.get(key) {
            Some(boxed) => boxed.clone().downcast::<V>().map_err(|_| {
                format!(
                    "Failed to downcast to the requested type for shared key: {}",
                    key
                )
            }),
            None => Err(format!("Shared key not found: {}", key)),
        }
    }

    /// Check if a unique object exists in the registry by key.
    pub fn contains_unique(&self, key: &str) -> bool {
        self.unique_storage.contains_key(key)
    }

    /// Insert a unique object into the registry with a specific key.
    pub fn insert_unique<K: ToString, U: Send + Sync + 'static>(&mut self, key: K, value: U) {
        self.unique_storage.insert(
            key.to_string(),
            Box::new(value) as Box<dyn Any + Send + Sync>,
        );
    }

    /// Take a unique object from the registry by key and type, removing it from the registry.
    pub fn take_unique<V: Send + Sync + 'static>(&mut self, key: &str) -> Result<V, String> {
        match self.unique_storage.remove(key) {
            Some(boxed) => boxed.downcast::<V>().map(|b| *b).map_err(|_| {
                format!(
                    "Failed to downcast to the requested type for unique key: {}",
                    key
                )
            }),
            None => Err(format!("Takable key not found: {}", key)),
        }
    }

    /// Clone a unique object from the registry if it implements `Clone`.
    pub fn clone_unique<V: Clone + Send + Sync + 'static>(&self, key: &str) -> Result<V, String> {
        match self.unique_storage.get(key) {
            Some(boxed) => boxed.downcast_ref::<V>().cloned().ok_or_else(|| {
                format!(
                    "Failed to downcast to the requested type for unique key: {}",
                    key
                )
            }),
            None => Err(format!("Takable key not found: {}", key)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get_shared() {
        let mut registry = Registry::new();
        registry.insert_shared("shared1", 42);
        assert_eq!(*registry.get_shared::<i32>("shared1").unwrap(), 42);
        assert!(registry.get_shared::<f64>("shared1").is_err()); // Testing a downcast failure
    }

    #[test]
    fn test_insert_and_take_unique() {
        let mut registry = Registry::new();
        registry.insert_unique("unique1", "Hello".to_string());
        assert_eq!(registry.take_unique::<String>("unique1").unwrap(), "Hello");
        assert!(registry.take_unique::<String>("unique1").is_err()); // Key is now missing
    }

    #[test]
    fn test_insert_and_clone_then_take_unique() {
        let mut registry = Registry::new();

        registry.insert_unique("unique2", "World".to_string());

        assert_eq!(registry.clone_unique::<String>("unique2").unwrap(), "World");

        // When cloned, the object should still be available for taking
        assert!(registry.take_unique::<String>("unique2").is_ok());
    }

    #[test]
    fn test_failed_take_after_cloning() {
        let mut registry = Registry::new();

        registry.insert_unique("unique3", "Another".to_string());
        assert_eq!(
            registry.clone_unique::<String>("unique3").unwrap(),
            "Another"
        );

        // Cloned, then Take is OK
        assert_eq!(
            registry.take_unique::<String>("unique3").unwrap(),
            "Another"
        );

        // Take, then Take again should fail
        assert!(registry.take_unique::<String>("unique3").is_err());
    }
}
