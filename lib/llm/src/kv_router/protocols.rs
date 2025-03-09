// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ForwardPassMetrics {
    pub request_active_slots: u64,
    pub request_total_slots: u64,
    pub kv_active_blocks: u64,
    pub kv_total_blocks: u64,
}

/// A [`BlockHash`] is a hash computed from the tokens_ids, extra_token_ids and the optional
/// lora_id of a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct LocalBlockHash(pub u64);

/// A sequence aware hash of a block where the hash is computed from the tokens_ids, extra_token_ids
/// and the optional lora_id of a block, PLUS the hash of the parent block.
///
/// In this case, the hashing function is external and unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ExternalSequenceBlockHash(pub u64);

/// Represents a collection of cache events and a shutdown flag.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KvCacheEvents {
    /// A list of cache events.
    pub events: Vec<KvCacheEvent>,
    /// A flag indicating whether the cache is shutting down.
    pub shutdown: bool,
}

/// Represents a single cache event with an ID and associated data.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KvCacheEvent {
    /// The unique identifier of the event.
    pub event_id: u64,
    /// The data associated with the event.
    pub data: KvCacheEventData,
}

/// Represents the data associated with a cache event.
///
/// Data is either stored or removed.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheEventData {
    /// Data for a stored cache event.
    Stored(KvCacheStoreData),
    /// Data for a removed cache event.
    Removed(KvCacheRemoveData),
}

/// Represents the data associated with a stored cache event.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KvCacheStoreData {
    /// The optional hash of the parent block.
    pub parent_hash: Option<ExternalSequenceBlockHash>,
    /// A list of stored blocked data.
    pub blocks: Vec<KvCacheStoredBlockData>,
}

/// Represents data for a stored block.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KvCacheStoredBlockData {
    /// The hash of the block.
    pub block_hash: ExternalSequenceBlockHash,
    /// The hash of the tokens in the block.
    pub tokens_hash: LocalBlockHash,
}

/// Represents the data associated with a removed cache event.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KvCacheRemoveData {
    /// A list of block hashes to remove.
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
}

impl Serialize for LocalBlockHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for LocalBlockHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Ok(LocalBlockHash(value))
    }
}

impl Serialize for ExternalSequenceBlockHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for ExternalSequenceBlockHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Ok(ExternalSequenceBlockHash(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_local_block_hash_serialization() {
        let hash = LocalBlockHash(12345);
        let serialized = serde_json::to_string(&hash).unwrap();
        assert_eq!(serialized, "12345");

        let deserialized: LocalBlockHash = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, hash);
    }

    #[test]
    fn test_external_sequence_block_hash_serialization() {
        let hash = ExternalSequenceBlockHash(67890);
        let serialized = serde_json::to_string(&hash).unwrap();
        assert_eq!(serialized, "67890");

        let deserialized: ExternalSequenceBlockHash = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, hash);
    }

    #[test]
    fn test_kv_cache_events_serialization() {
        let event_data = KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: Some(ExternalSequenceBlockHash(1)),
            blocks: vec![KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(2),
                tokens_hash: LocalBlockHash(3),
            }],
        });

        let event = KvCacheEvent {
            event_id: 1,
            data: event_data,
        };

        let events = KvCacheEvents {
            events: vec![event],
            shutdown: false,
        };

        let serialized = serde_json::to_string(&events).unwrap();
        let deserialized: KvCacheEvents = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.events.len(), 1);
        assert_eq!(deserialized.events[0].event_id, 1);
        if let KvCacheEventData::Stored(store_data) = &deserialized.events[0].data {
            assert_eq!(store_data.parent_hash.unwrap().0, 1);
            assert_eq!(store_data.blocks.len(), 1);
            assert_eq!(store_data.blocks[0].block_hash.0, 2);
            assert_eq!(store_data.blocks[0].tokens_hash.0, 3);
        } else {
            panic!("Expected KvCacheEventData::Stored variant");
        }
        assert!(!deserialized.shutdown);
    }

    #[test]
    fn test_kv_cache_remove_data_serialization() {
        let remove_data = KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(4), ExternalSequenceBlockHash(5)],
        };

        let serialized = serde_json::to_string(&remove_data).unwrap();
        let deserialized: KvCacheRemoveData = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.block_hashes.len(), 2);
        assert_eq!(deserialized.block_hashes[0].0, 4);
        assert_eq!(deserialized.block_hashes[1].0, 5);
    }
}
