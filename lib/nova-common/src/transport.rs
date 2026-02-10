// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport key type for type-safe transport identification.

use std::fmt;
use std::sync::Arc;

/// A type-safe wrapper around transport keys for WorkerAddress.
///
/// This provides a zero-cost abstraction over `Arc<str>` with type safety
/// to prevent accidentally mixing transport keys with other string types.
///
/// # Examples
///
/// ```
/// use dynamo_nova_common::TransportKey;
///
/// let key = TransportKey::new("tcp");
/// assert_eq!(key.as_str(), "tcp");
///
/// // Ergonomic conversions
/// let key2: TransportKey = "rdma".into();
/// let key3 = TransportKey::from("udp");
///
/// // Use in collections
/// use std::collections::HashMap;
/// let mut transports = HashMap::new();
/// transports.insert(TransportKey::from("tcp"), "tcp://127.0.0.1:5555");
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TransportKey(Arc<str>);

impl TransportKey {
    /// Create a new TransportKey from any type that can be converted into Arc<str>.
    pub fn new(key: impl Into<Arc<str>>) -> Self {
        Self(key.into())
    }

    /// Get the key as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// Deref to str for ergonomic usage
impl std::ops::Deref for TransportKey {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// AsRef for flexible parameter types
impl AsRef<str> for TransportKey {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// From conversions for ergonomic construction
impl From<&str> for TransportKey {
    fn from(s: &str) -> Self {
        Self(Arc::from(s))
    }
}

impl From<String> for TransportKey {
    fn from(s: String) -> Self {
        Self(Arc::from(s))
    }
}

impl From<Arc<str>> for TransportKey {
    fn from(s: Arc<str>) -> Self {
        Self(s)
    }
}

impl From<&String> for TransportKey {
    fn from(s: &String) -> Self {
        Self(Arc::from(s.as_str()))
    }
}

impl From<TransportKey> for String {
    fn from(val: TransportKey) -> Self {
        val.0.to_string()
    }
}

// Borrow trait for HashMap lookups with &str
impl std::borrow::Borrow<str> for TransportKey {
    fn borrow(&self) -> &str {
        &self.0
    }
}

// Display for printing
impl fmt::Display for TransportKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_transport_key_creation() {
        // Test new() method
        let key1 = TransportKey::new("tcp");
        assert_eq!(key1.as_str(), "tcp");

        // Test From<&str>
        let key2: TransportKey = "rdma".into();
        assert_eq!(key2.as_str(), "rdma");

        // Test From<String>
        let key3 = TransportKey::from(String::from("udp"));
        assert_eq!(key3.as_str(), "udp");

        // Test From<&String>
        let s = String::from("grpc");
        let key4 = TransportKey::from(&s);
        assert_eq!(key4.as_str(), "grpc");

        // Test From<Arc<str>>
        let arc_str: Arc<str> = Arc::from("http");
        let key5 = TransportKey::from(arc_str);
        assert_eq!(key5.as_str(), "http");
    }

    #[test]
    fn test_transport_key_deref() {
        let key = TransportKey::from("tcp");

        // Deref to str methods should work
        assert_eq!(key.len(), 3);
        assert_eq!(key.chars().count(), 3);
        assert!(key.starts_with("tc"));
        assert!(key.ends_with("cp"));

        // Can use str slicing through Deref
        assert_eq!(&key[0..2], "tc");
    }

    #[test]
    fn test_transport_key_as_ref() {
        let key = TransportKey::from("tcp");

        // AsRef<str> allows passing to functions expecting &str
        fn takes_str_ref(s: &str) -> usize {
            s.len()
        }

        assert_eq!(takes_str_ref(&key), 3);
        assert_eq!(takes_str_ref(key.as_ref()), 3);
    }

    #[test]
    fn test_transport_key_display() {
        let key = TransportKey::from("tcp");
        assert_eq!(format!("{}", key), "tcp");
        assert_eq!(key.to_string(), "tcp");
    }

    #[test]
    fn test_transport_key_debug() {
        let key = TransportKey::from("tcp");
        let debug_str = format!("{:?}", key);
        assert!(debug_str.contains("TransportKey"));
        assert!(debug_str.contains("tcp"));
    }

    #[test]
    fn test_transport_key_equality() {
        let key1 = TransportKey::from("tcp");
        let key2 = TransportKey::from("tcp");
        let key3 = TransportKey::from("rdma");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);

        // Test with different source types
        let key4: TransportKey = String::from("tcp").into();
        assert_eq!(key1, key4);
    }

    #[test]
    fn test_transport_key_ordering() {
        let mut keys = [
            TransportKey::from("udp"),
            TransportKey::from("tcp"),
            TransportKey::from("rdma"),
            TransportKey::from("grpc"),
        ];

        keys.sort();

        assert_eq!(keys[0], TransportKey::from("grpc"));
        assert_eq!(keys[1], TransportKey::from("rdma"));
        assert_eq!(keys[2], TransportKey::from("tcp"));
        assert_eq!(keys[3], TransportKey::from("udp"));
    }

    #[test]
    fn test_transport_key_hash() {
        let mut set = HashSet::new();
        set.insert(TransportKey::from("tcp"));
        set.insert(TransportKey::from("rdma"));
        set.insert(TransportKey::from("tcp")); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&TransportKey::from("tcp")));
        assert!(set.contains(&TransportKey::from("rdma")));
        assert!(!set.contains(&TransportKey::from("udp")));
    }

    #[test]
    fn test_transport_key_in_hashmap() {
        let mut map = HashMap::new();
        map.insert(TransportKey::from("tcp"), "tcp://127.0.0.1:5555");
        map.insert(TransportKey::from("rdma"), "rdma://10.0.0.1:6666");

        // Can lookup with TransportKey
        assert_eq!(
            map.get(&TransportKey::from("tcp")),
            Some(&"tcp://127.0.0.1:5555")
        );

        // Can lookup with &str via Borrow trait
        assert_eq!(map.get("tcp"), Some(&"tcp://127.0.0.1:5555"));
        assert_eq!(map.get("rdma"), Some(&"rdma://10.0.0.1:6666"));
        assert_eq!(map.get("udp"), None);
    }

    #[test]
    fn test_transport_key_clone() {
        let key1 = TransportKey::from("tcp");
        let key2 = key1.clone();

        assert_eq!(key1, key2);
        assert_eq!(key1.as_str(), key2.as_str());

        // Verify Arc is shared (same pointer)
        let ptr1 = key1.as_str().as_ptr();
        let ptr2 = key2.as_str().as_ptr();
        assert_eq!(ptr1, ptr2);
    }
}
