// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-owned hashing for derived active-sequence tracking state.

use std::fmt;
use std::fs;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use dynamo_tokens::SequenceHash;
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use crate::config::KvRouterConfig;
use crate::identity::RoutingPartitionRef;
use crate::protocols::{
    BlockHashOptions, LocalBlockHash, complete_block_count, compute_block_hash_for_seq,
    compute_seq_hash_for_block, compute_seq_hash_for_tokens_with_seeds,
};

const KEY_SIZE: usize = 32;
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

pub(crate) fn validate_tracking_hash_options(
    algorithm: TrackingHashAlgorithm,
    has_key_file: bool,
    key_id: Option<&str>,
) -> std::result::Result<(), String> {
    match algorithm {
        TrackingHashAlgorithm::PublicXxh3V1 if has_key_file || key_id.is_some() => Err(
            "router tracking key options require router_tracking_hash=keyed-xxh3-v1".to_string(),
        ),
        TrackingHashAlgorithm::KeyedXxh3V1 if !has_key_file => {
            Err("keyed-xxh3-v1 requires router_tracking_key_file".to_string())
        }
        TrackingHashAlgorithm::KeyedXxh3V1
            if !key_id.is_some_and(|value| !value.is_empty() && value.trim() == value) =>
        {
            Err("keyed-xxh3-v1 requires a nonempty router_tracking_key_id".to_string())
        }
        _ => Ok(()),
    }
}

/// Stable, trusted scope shared by router instances that should produce the
/// same derived tracking identities.
///
/// TODO(#11971): Evaluate deriving the static tracking-domain identity from
/// `IndexerDomainId`. `PoolId` and `DcId` represent placement and must remain
/// excluded from tracking identity.
#[derive(Clone, Copy, Debug)]
pub struct TrackingHashScope<'a> {
    pub partition: RoutingPartitionRef<'a>,
    pub block_size: u32,
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
    /// Load and validate the runtime context from router configuration.
    pub fn from_config(config: &KvRouterConfig) -> Result<Self> {
        validate_tracking_hash_options(
            config.router_tracking_hash,
            config.router_tracking_key_file.is_some(),
            config.router_tracking_key_id.as_deref(),
        )
        .map_err(anyhow::Error::msg)?;

        match config.router_tracking_hash {
            TrackingHashAlgorithm::PublicXxh3V1 => Ok(Self {
                algorithm: TrackingHashAlgorithm::PublicXxh3V1,
                key_id: None,
                provider_key: None,
            }),
            TrackingHashAlgorithm::KeyedXxh3V1 => {
                let key_id = config
                    .router_tracking_key_id
                    .as_deref()
                    .expect("validated keyed tracking config must have a key ID");
                let key_path = config
                    .router_tracking_key_file
                    .as_ref()
                    .expect("validated keyed tracking config must have a key file");
                let key_bytes = Zeroizing::new(
                    fs::read(key_path).context("failed to read router tracking key file")?,
                );
                let actual_size = key_bytes.len();
                if actual_size != KEY_SIZE {
                    bail!(
                        "router tracking key file must contain exactly {KEY_SIZE} raw bytes; found {actual_size}"
                    );
                }
                let mut provider_key = Zeroizing::new([0_u8; KEY_SIZE]);
                provider_key.copy_from_slice(&key_bytes);
                Ok(Self {
                    algorithm: TrackingHashAlgorithm::KeyedXxh3V1,
                    key_id: Some(key_id.into()),
                    provider_key: Some(provider_key),
                })
            }
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
    use std::io::Write;

    use tempfile::NamedTempFile;
    use zeroize::Zeroize;

    use super::*;
    use crate::protocols::{BlockExtraInfo, BlockMmObjectInfo};

    fn keyed_config(key_id: &str, key_bytes: &[u8]) -> (NamedTempFile, KvRouterConfig) {
        let mut key_file = NamedTempFile::new().unwrap();
        key_file.write_all(key_bytes).unwrap();
        let config = KvRouterConfig {
            router_tracking_hash: TrackingHashAlgorithm::KeyedXxh3V1,
            router_tracking_key_file: Some(key_file.path().to_path_buf()),
            router_tracking_key_id: Some(key_id.to_string()),
            ..Default::default()
        };
        (key_file, config)
    }

    fn scope<'a>(model_name: &'a str, routing_group: &'a str) -> TrackingHashScope<'a> {
        TrackingHashScope {
            partition: RoutingPartitionRef::new(model_name, routing_group),
            block_size: 4,
        }
    }

    #[test]
    fn public_mode_is_bit_compatible() {
        let context = TrackingHashContext::from_config(&KvRouterConfig::default()).unwrap();
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
        let context = TrackingHashContext::from_config(&config).unwrap();
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
        let first = TrackingHashContext::from_config(&config).unwrap();
        let second = TrackingHashContext::from_config(&config).unwrap();
        let (_next_key_file, next_config) = keyed_config("2026-02", &[0x23; KEY_SIZE]);
        let next_epoch = TrackingHashContext::from_config(&next_config).unwrap();
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
        let context = TrackingHashContext::from_config(&config).unwrap();
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
    fn key_loading_rejects_missing_and_malformed_files_without_exposing_bytes() {
        let (_short_file, short_config) = keyed_config("2026-01", &[7; KEY_SIZE - 1]);
        assert!(
            TrackingHashContext::from_config(&short_config)
                .unwrap_err()
                .to_string()
                .contains("exactly 32 raw bytes")
        );

        let (_long_file, long_config) = keyed_config("2026-01", &[8; KEY_SIZE + 1]);
        assert!(
            TrackingHashContext::from_config(&long_config)
                .unwrap_err()
                .to_string()
                .contains("exactly 32 raw bytes")
        );

        let missing_config = KvRouterConfig {
            router_tracking_hash: TrackingHashAlgorithm::KeyedXxh3V1,
            router_tracking_key_file: Some("/definitely/missing/tracking-key".into()),
            router_tracking_key_id: Some("2026-01".to_string()),
            ..Default::default()
        };
        let missing_error = TrackingHashContext::from_config(&missing_config)
            .unwrap_err()
            .to_string();
        assert!(missing_error.contains("failed to read router tracking key file"));
        assert!(!missing_error.contains("/definitely/missing/tracking-key"));

        let unreadable_dir = tempfile::tempdir().unwrap();
        let unreadable_config = KvRouterConfig {
            router_tracking_hash: TrackingHashAlgorithm::KeyedXxh3V1,
            router_tracking_key_file: Some(unreadable_dir.path().to_path_buf()),
            router_tracking_key_id: Some("2026-01".to_string()),
            ..Default::default()
        };
        let unreadable_error = TrackingHashContext::from_config(&unreadable_config)
            .unwrap_err()
            .to_string();
        assert!(unreadable_error.contains("failed to read router tracking key file"));
        assert!(!unreadable_error.contains(&unreadable_dir.path().display().to_string()));

        let (_key_file, valid_config) = keyed_config("2026-01", &[0xab; KEY_SIZE]);
        let debug = format!(
            "{:?}",
            TrackingHashContext::from_config(&valid_config).unwrap()
        );
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("171"));
    }

    #[test]
    fn retained_key_is_zeroizable_and_only_epoch_is_exposed() {
        let (_key_file, config) = keyed_config("2026-01", &[0xab; KEY_SIZE]);
        let mut context = TrackingHashContext::from_config(&config).unwrap();

        assert_eq!(context.key_id(), Some("2026-01"));
        let provider_key = context.provider_key.as_mut().unwrap();
        provider_key.zeroize();
        assert_eq!(provider_key.as_ref(), &[0_u8; KEY_SIZE]);

        let public = TrackingHashContext::from_config(&KvRouterConfig::default()).unwrap();
        assert_eq!(public.key_id(), None);
    }

    #[test]
    fn config_validation_enforces_mode_specific_options() {
        let assert_rejected_by_both = |config: &KvRouterConfig, expected: &str| {
            assert_eq!(config.validate_config().unwrap_err(), expected);
            assert_eq!(
                TrackingHashContext::from_config(config)
                    .unwrap_err()
                    .to_string(),
                expected
            );
        };

        let public_with_key = KvRouterConfig {
            router_tracking_key_file: Some("key".into()),
            ..Default::default()
        };
        assert_rejected_by_both(
            &public_with_key,
            "router tracking key options require router_tracking_hash=keyed-xxh3-v1",
        );

        let keyed_without_options = KvRouterConfig {
            router_tracking_hash: TrackingHashAlgorithm::KeyedXxh3V1,
            ..Default::default()
        };
        assert_rejected_by_both(
            &keyed_without_options,
            "keyed-xxh3-v1 requires router_tracking_key_file",
        );

        let (_key_file, keyed_with_whitespace_id) = keyed_config(" 2026-01", &[1; KEY_SIZE]);
        assert_rejected_by_both(
            &keyed_with_whitespace_id,
            "keyed-xxh3-v1 requires a nonempty router_tracking_key_id",
        );

        let (_key_file, keyed) = keyed_config("2026-01", &[1; KEY_SIZE]);
        assert!(keyed.validate_config().is_ok());
        assert!(TrackingHashContext::from_config(&keyed).is_ok());
    }
}
