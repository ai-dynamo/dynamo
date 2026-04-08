// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core traits and types for the pluggable registry architecture.

pub mod error;
pub mod events;
pub mod eviction;
pub mod key;
pub mod lease;
pub mod metadata;
pub mod storage;
pub mod transport;
pub mod value;

// Error types
pub use error::{RegistryError, RegistryResult};

// Lease management
pub use lease::{LeaseInfo, LeaseManager};

// Storage & Eviction
pub use eviction::{Eviction, NoEviction, PositionalEviction, TailEviction};
pub use storage::{FlatStorage, HashMapStorage, PositionalStorageKey, RadixStorage, Storage};

// Key, Value, Metadata
pub use key::{CompositeKey, Key128, PositionalKey, RegistryKey};
pub use metadata::{NoMetadata, PositionMetadata, RegistryMetadata, TimestampMetadata};
pub use value::{RegistryValue, StorageBackend, StorageLocation};

// Transport
pub use transport::{InProcessHub, InProcessTransport, RegistryTransport};

// Event Bus
pub use events::{
    EventBus, EventBusConfig, EventHandler, EventReceiver, EventTopic, EvictionReason,
    InProcessEventBus, RegistryEvent, StorageTier, StorageType,
};
