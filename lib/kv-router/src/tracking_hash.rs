// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Router-owned hashing for derived active-sequence tracking state.

use std::fs;

use anyhow::{Context, Result, bail};
use zeroize::Zeroizing;

use crate::config::KvRouterConfig;
use crate::scheduling::tracking_hash::KEY_SIZE;
pub use crate::scheduling::tracking_hash::{
    TrackingHashAlgorithm, TrackingHashConfig, TrackingHashContext, TrackingHashScope,
};

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

/// Resolve the tracking-hash slice of the flat configuration: validate the option
/// combination, then read and size-check the key file for the keyed algorithm.
pub fn tracking_hash_config(config: &KvRouterConfig) -> Result<TrackingHashConfig> {
    validate_tracking_hash_options(
        config.router_tracking_hash,
        config.router_tracking_key_file.is_some(),
        config.router_tracking_key_id.as_deref(),
    )
    .map_err(anyhow::Error::msg)?;

    match config.router_tracking_hash {
        TrackingHashAlgorithm::PublicXxh3V1 => Ok(TrackingHashConfig {
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
            Ok(TrackingHashConfig {
                algorithm: TrackingHashAlgorithm::KeyedXxh3V1,
                key_id: Some(key_id.into()),
                provider_key: Some(provider_key),
            })
        }
    }
}

impl TrackingHashContext {
    /// The context for a flat configuration; see [`tracking_hash_config`].
    pub fn from_config(config: &KvRouterConfig) -> Result<Self> {
        Ok(Self::new(tracking_hash_config(config)?))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

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
