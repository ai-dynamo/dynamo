// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sequence hashes for request tracking: the public xxh3 chain, or a keyed variant whose
//! seeds derive from a provider key, the partition, and the block size, so hashes from
//! different scopes or key epochs never collide by construction. The kernel receives the
//! resolved [`TrackingHashConfig`]; reading the flat configuration and the key file is the
//! composed router's job.

use std::fmt;
use std::str::FromStr;

use dynamo_tokens::SequenceHash;
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use crate::identity::RoutingPartitionRef;
use crate::protocols::{
    BlockHashOptions, LocalBlockHash, complete_block_count, compute_block_hash_for_seq,
    compute_seq_hash_for_block, compute_seq_hash_for_tokens_with_seeds,
};

pub(crate) const KEY_SIZE: usize = 32;

const KEYED_XXH3_V1_DOMAIN: &[u8] = b"dynamo.router.tracking-hash/keyed-xxh3-v1\0";

/// Hash algorithm used only for router-derived active-sequence tracking state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrackingHashAlgorithm {
    /// Existing public XXH3 block and sequence hash construction.
    #[default]
    PublicXxh3V1,
    /// Provider-keyed scope derivation with XXH3 block and chain hashing.
    KeyedXxh3V1,
}

impl fmt::Display for TrackingHashAlgorithm {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::PublicXxh3V1 => "public-xxh3-v1",
            Self::KeyedXxh3V1 => "keyed-xxh3-v1",
        })
    }
}

impl FromStr for TrackingHashAlgorithm {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "public-xxh3-v1" => Ok(Self::PublicXxh3V1),
            "keyed-xxh3-v1" => Ok(Self::KeyedXxh3V1),
            _ => Err(format!(
                "router_tracking_hash must be public-xxh3-v1 or keyed-xxh3-v1, got {value:?}"
            )),
        }
    }
}

/// Stable, trusted scope shared by router instances that should produce the
/// same derived tracking identities.
///
/// TODO(#11971): Evaluate deriving the static tracking-domain identity from
/// `IndexerDomainId`. `PoolId` and `ClusterId` represent placement and must remain
/// excluded from tracking identity.
#[derive(Clone, Copy, Debug)]
pub struct TrackingHashScope<'a> {
    pub partition: RoutingPartitionRef<'a>,
    pub block_size: u32,
}

/// What the tracking hash needs, resolved: the algorithm, the key epoch it reports, and
/// the key material itself. The composed router reads these from its flat configuration
/// and key file; the kernel never opens a file.
#[derive(Default)]
pub struct TrackingHashConfig {
    pub algorithm: TrackingHashAlgorithm,
    pub key_id: Option<Box<str>>,
    pub provider_key: Option<Zeroizing<[u8; KEY_SIZE]>>,
}

impl Clone for TrackingHashConfig {
    fn clone(&self) -> Self {
        Self {
            algorithm: self.algorithm,
            key_id: self.key_id.clone(),
            provider_key: self.provider_key.as_ref().map(|key| Zeroizing::new(**key)),
        }
    }
}

/// Runtime tracking-hash state. Secret bytes are intentionally excluded from
/// serialization and debug output.
pub struct TrackingHashContext {
    algorithm: TrackingHashAlgorithm,
    key_id: Option<Box<str>>,
    provider_key: Option<Zeroizing<[u8; KEY_SIZE]>>,
}

impl fmt::Debug for TrackingHashContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TrackingHashContext")
            .field("algorithm", &self.algorithm)
            .field("key_id", &self.key_id)
            .field(
                "provider_key",
                &self.provider_key.as_ref().map(|_| "[REDACTED]"),
            )
            .finish()
    }
}

impl TrackingHashContext {
    /// Build the context from a resolved configuration (algorithm, key id, key material).
    pub fn new(config: TrackingHashConfig) -> Self {
        Self {
            algorithm: config.algorithm,
            key_id: config.key_id,
            provider_key: config.provider_key,
        }
    }

    pub fn algorithm(&self) -> TrackingHashAlgorithm {
        self.algorithm
    }

    /// Return the provider-managed key epoch without exposing secret material.
    pub fn key_id(&self) -> Option<&str> {
        self.key_id.as_deref()
    }

    /// Compute sequence identities for active tracking. Public mode may reuse
    /// already-computed public block hashes; keyed mode must hash canonical
    /// block bytes under its derived block seed.
    pub(crate) fn compute_sequence_hashes(
        &self,
        scope: TrackingHashScope<'_>,
        tokens: &[u32],
        options: BlockHashOptions<'_>,
        precomputed_public_block_hashes: Option<&[LocalBlockHash]>,
    ) -> Vec<SequenceHash> {
        match self.algorithm {
            TrackingHashAlgorithm::PublicXxh3V1 => {
                if let Some(block_hashes) = precomputed_public_block_hashes {
                    compute_seq_hash_for_block(block_hashes)
                } else {
                    compute_seq_hash_for_block(&compute_block_hash_for_seq(
                        tokens,
                        scope.block_size,
                        options,
                    ))
                }
            }
            TrackingHashAlgorithm::KeyedXxh3V1 => {
                let (block_seed, chain_seed) = self.derive_seeds(scope, options);
                compute_seq_hash_for_tokens_with_seeds(
                    tokens,
                    scope.block_size,
                    options,
                    block_seed,
                    chain_seed,
                )
            }
        }
    }

    /// Compute active-tracking identities using the configured reuse policy.
    ///
    /// Callers that start from raw tokens should use this method rather than
    /// choosing between keyed/public and random identities themselves.
    pub fn compute_sequence_hashes_for_tracking(
        &self,
        scope: TrackingHashScope<'_>,
        tokens: &[u32],
        options: BlockHashOptions<'_>,
        assume_kv_reuse: bool,
        precomputed_public_block_hashes: Option<&[LocalBlockHash]>,
    ) -> Vec<SequenceHash> {
        let num_blocks = complete_block_count(
            tokens.len(),
            scope.block_size,
            options.is_eagle.unwrap_or(false),
        );
        if num_blocks == 0 {
            return Vec::new();
        }

        if assume_kv_reuse {
            self.compute_sequence_hashes(scope, tokens, options, precomputed_public_block_hashes)
        } else {
            (0..num_blocks).map(|_| fastrand::u64(..)).collect()
        }
    }

    fn derive_seeds(
        &self,
        scope: TrackingHashScope<'_>,
        options: BlockHashOptions<'_>,
    ) -> (u64, u64) {
        let key = self
            .provider_key
            .as_ref()
            .expect("keyed tracking hash context must contain a provider key");
        let mut hasher = blake3::Hasher::new_keyed(key);
        hasher.update(KEYED_XXH3_V1_DOMAIN);
        frame_string(&mut hasher, 1, self.key_id.as_deref().unwrap_or_default());
        frame_string(&mut hasher, 2, scope.partition.model_name);
        frame_string(&mut hasher, 3, scope.partition.routing_group);
        frame_fixed(&mut hasher, 4, &scope.block_size.to_le_bytes());
        frame_optional_string(&mut hasher, 5, normalize_optional(options.cache_namespace));
        frame_optional_string(&mut hasher, 6, normalize_optional(options.lora_name));
        frame_fixed(
            &mut hasher,
            7,
            &[u8::from(options.is_eagle.unwrap_or(false))],
        );

        let digest = hasher.finalize();
        let bytes = digest.as_bytes();
        let block_seed = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let chain_seed = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        (block_seed, chain_seed)
    }
}

fn normalize_optional(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.is_empty())
}

fn frame_string(hasher: &mut blake3::Hasher, tag: u8, value: &str) {
    frame_fixed(hasher, tag, value.as_bytes());
}

fn frame_optional_string(hasher: &mut blake3::Hasher, tag: u8, value: Option<&str>) {
    hasher.update(&[tag, u8::from(value.is_some())]);
    if let Some(value) = value {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
}

fn frame_fixed(hasher: &mut blake3::Hasher, tag: u8, value: &[u8]) {
    hasher.update(&[tag]);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

#[cfg(test)]
mod tests {
    use zeroize::Zeroize;

    use super::*;
    use crate::protocols::{BlockExtraInfo, BlockMmObjectInfo};

    /// A keyed configuration from raw key bytes, no file involved; the first element keeps the
    /// `(_guard, config)` destructuring the vector tests share with the flat-config tests.
    fn keyed_config(key_id: &str, key_bytes: &[u8]) -> ((), TrackingHashConfig) {
        let mut provider_key = Zeroizing::new([0_u8; KEY_SIZE]);
        provider_key.copy_from_slice(key_bytes);
        (
            (),
            TrackingHashConfig {
                algorithm: TrackingHashAlgorithm::KeyedXxh3V1,
                key_id: Some(key_id.into()),
                provider_key: Some(provider_key),
            },
        )
    }

    fn scope<'a>(model_name: &'a str, routing_group: &'a str) -> TrackingHashScope<'a> {
        TrackingHashScope {
            partition: RoutingPartitionRef::new(model_name, routing_group),
            block_size: 4,
        }
    }

    #[test]
    fn public_mode_is_bit_compatible() {
        let context = TrackingHashContext::new(TrackingHashConfig::default());
        let tokens: Vec<u32> = (0..8).collect();
        let options = BlockHashOptions {
            cache_namespace: Some("tenant-a"),
            lora_name: Some("adapter-a"),
            ..Default::default()
        };
        let public_blocks = compute_block_hash_for_seq(&tokens, 4, options);

        assert_eq!(
            context.compute_sequence_hashes(
                scope("model", "default"),
                &tokens,
                options,
                Some(&public_blocks),
            ),
            compute_seq_hash_for_block(&public_blocks)
        );
    }

    #[test]
    fn keyed_hash_vector_pins_scope_and_chain_framing() {
        let (_key_file, config) = keyed_config("2026-01", &[0x5a; KEY_SIZE]);
        let context = TrackingHashContext::new(config.clone());
        let tokens: Vec<u32> = (0..12).collect();

        let hashes = context.compute_sequence_hashes(
            scope("model-a", "tenant-a"),
            &tokens,
            BlockHashOptions {
                cache_namespace: Some("cache-a"),
                lora_name: Some("adapter-a"),
                ..Default::default()
            },
            None,
        );

        assert_eq!(
            hashes,
            vec![
                4_363_769_719_052_127_296,
                14_998_523_962_162_619_427,
                13_920_914_207_884_994_756,
            ]
        );
    }

    #[test]
    fn keyed_contexts_are_stable_and_scope_sensitive() {
        let (_key_file, config) = keyed_config("2026-01", &[0x23; KEY_SIZE]);
        let first = TrackingHashContext::new(config.clone());
        let second = TrackingHashContext::new(config.clone());
        let (_next_key_file, next_config) = keyed_config("2026-02", &[0x23; KEY_SIZE]);
        let next_epoch = TrackingHashContext::new(next_config.clone());
        let tokens: Vec<u32> = (0..12).collect();
        let base_options = BlockHashOptions::default();
        let base =
            first.compute_sequence_hashes(scope("model-a", "group-a"), &tokens, base_options, None);

        assert_eq!(
            base,
            second.compute_sequence_hashes(
                scope("model-a", "group-a"),
                &tokens,
                base_options,
                None,
            )
        );
        assert_ne!(
            base,
            first
                .compute_sequence_hashes(scope("model-b", "group-a"), &tokens, base_options, None,)
        );
        assert_ne!(
            base,
            first
                .compute_sequence_hashes(scope("model-a", "group-b"), &tokens, base_options, None,)
        );
        assert_ne!(
            base,
            first.compute_sequence_hashes(
                TrackingHashScope {
                    partition: RoutingPartitionRef::new("model-a", "group-a"),
                    block_size: 3,
                },
                &tokens,
                base_options,
                None,
            )
        );
        assert_ne!(
            base,
            next_epoch.compute_sequence_hashes(
                scope("model-a", "group-a"),
                &tokens,
                base_options,
                None,
            )
        );
        assert_ne!(
            base,
            first.compute_sequence_hashes(
                scope("model-a", "group-a"),
                &tokens,
                BlockHashOptions {
                    cache_namespace: Some("cache-a"),
                    ..Default::default()
                },
                None,
            )
        );
        assert_ne!(
            base,
            first.compute_sequence_hashes(
                scope("model-a", "group-a"),
                &tokens,
                BlockHashOptions {
                    lora_name: Some("adapter-a"),
                    ..Default::default()
                },
                None,
            )
        );
        assert_ne!(
            base,
            first.compute_sequence_hashes(
                scope("model-a", "group-a"),
                &tokens,
                BlockHashOptions {
                    is_eagle: Some(true),
                    ..Default::default()
                },
                None,
            )
        );
    }

    #[test]
    fn keyed_vectors_pin_multimodal_eagle_and_partial_block_rules() {
        let (_key_file, config) = keyed_config("2026-01", &[0x41; KEY_SIZE]);
        let context = TrackingHashContext::new(config.clone());
        let tokens: Vec<u32> = (0..10).collect();
        let mm_infos = vec![
            Some(BlockExtraInfo {
                mm_objects: vec![BlockMmObjectInfo {
                    mm_hash: 42,
                    offsets: vec![(0, 2)],
                }],
            }),
            None,
        ];

        let without_mm = context.compute_sequence_hashes(
            scope("model", "default"),
            &tokens,
            BlockHashOptions::default(),
            None,
        );
        let with_mm = context.compute_sequence_hashes(
            scope("model", "default"),
            &tokens,
            BlockHashOptions {
                block_mm_infos: Some(&mm_infos),
                ..Default::default()
            },
            None,
        );
        let with_eagle = context.compute_sequence_hashes(
            scope("model", "default"),
            &tokens,
            BlockHashOptions {
                is_eagle: Some(true),
                ..Default::default()
            },
            None,
        );

        assert_eq!(without_mm.len(), 2);
        assert_eq!(
            with_mm,
            vec![13_077_030_603_177_067_515, 12_131_634_976_806_651_614]
        );
        assert_eq!(
            with_eagle,
            vec![18_351_479_723_295_049_348, 8_577_555_336_206_814_019]
        );
        assert_ne!(without_mm, with_mm);
    }

    #[test]
    fn retained_key_is_zeroizable_and_only_epoch_is_exposed() {
        let (_key_file, config) = keyed_config("2026-01", &[0xab; KEY_SIZE]);
        let mut context = TrackingHashContext::new(config.clone());

        assert_eq!(context.key_id(), Some("2026-01"));
        let provider_key = context.provider_key.as_mut().unwrap();
        provider_key.zeroize();
        assert_eq!(provider_key.as_ref(), &[0_u8; KEY_SIZE]);

        let public = TrackingHashContext::new(TrackingHashConfig::default());
        assert_eq!(public.key_id(), None);
    }
}
