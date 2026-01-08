// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core traits and types for the pluggable registry architecture.

pub mod builder;
pub mod codec;
pub mod error;
pub mod eviction;
pub mod hub;
pub mod hub_transport;
pub mod key;
pub mod lease;
pub mod metadata;
pub mod registry;
pub mod storage;
pub mod transport;
pub mod value;
pub mod zmq_hub;
pub mod zmq_transport;

#[cfg(test)]
mod tests;

// Codec
pub use codec::{
    BinaryCodec, OffloadStatus, PROTOCOL_VERSION, QueryType, RegistryCodec, ResponseType,
};

// Error types
pub use error::{RegistryError, RegistryResult};

// Lease management
pub use lease::{LeaseInfo, LeaseManager};

// Storage & Eviction
pub use eviction::{Eviction, NoEviction, PositionalEviction, TailEviction};
pub use storage::{HashMapStorage, PositionalStorageKey, RadixStorage, Storage};

// Key, Value, Metadata
pub use key::{CompositeKey, Key128, PositionalKey, RegistryKey};
pub use metadata::{NoMetadata, PositionMetadata, RegistryMetadata, TimestampMetadata};
pub use value::{RegistryValue, StorageBackend, StorageLocation};

// Client
pub use registry::{OffloadResult, Registry, RegistryClient};
pub use transport::{InProcessHub, InProcessTransport, RegistryTransport};
pub use zmq_transport::{ZmqTransport, ZmqTransportConfig};

// Hub (Server)
pub use hub::{HubStats, RegistryHub};
pub use hub_transport::{
    ClientId, HubMessage, HubTransport, InProcessClientHandle, InProcessHubTransport, ZmqHubConfig,
    ZmqHubTransport,
};

// Builder
pub use builder::{ClientBuilder, HubBuilder, client, hub};

// ZMQ Hub
pub use zmq_hub::{ZmqHub, ZmqHubConfig as ZmqHubServerConfig};
