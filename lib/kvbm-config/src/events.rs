// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event publishing configuration for KV cache coordination.
//!
//! This module defines the configuration for the event publishing pipeline
//! that broadcasts block registration/removal events to distributed consumers
//! (e.g., KvbmHub for radix tree maintenance).

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Configuration for event publishing.
///
/// Events are broadcast when blocks are registered or removed from the cache.
/// The pipeline batches events for efficient wire transmission.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct EventsConfig {
    /// Whether event publishing is enabled.
    ///
    /// When disabled, no events are emitted and no publisher is started.
    /// Default: false
    #[serde(default)]
    pub enabled: bool,

    /// Batching configuration for the event pipeline.
    #[serde(default)]
    #[validate(nested)]
    pub batching: BatchingConfig,

    /// Broadcast channel capacity for the EventsManager.
    ///
    /// This determines how many events can be buffered before slow
    /// subscribers start lagging. Default: 1024
    #[serde(default = "default_channel_capacity")]
    #[validate(range(min = 16, max = 65536))]
    pub channel_capacity: usize,

    /// Subject/topic pattern for publishing events.
    ///
    /// This is the NATS/messaging subject where events are published.
    /// Default: "kvbm.events"
    #[serde(default = "default_subject")]
    pub subject: String,

    /// Event emission policy.
    ///
    /// Determines which blocks trigger events:
    /// - `power_of_two`: Only emit for blocks at power-of-2 positions (default)
    /// - `all`: Emit for all blocks (testing/debugging)
    #[serde(default)]
    pub policy: EventPolicyConfig,
}

impl Default for EventsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            batching: BatchingConfig::default(),
            channel_capacity: default_channel_capacity(),
            subject: default_subject(),
            policy: EventPolicyConfig::default(),
        }
    }
}

fn default_channel_capacity() -> usize {
    1024
}

fn default_subject() -> String {
    "kvbm.events".to_string()
}

/// Batching configuration for the event pipeline.
///
/// Events are batched before publishing to reduce wire traffic.
/// Batches are flushed when:
/// - The window duration expires
/// - The max batch size is reached
/// - The event type switches (Create -> Remove or vice versa)
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct BatchingConfig {
    /// Maximum time to wait before flushing a batch (in milliseconds).
    ///
    /// Default: 10ms
    #[serde(default = "default_window_duration_ms")]
    #[validate(range(min = 1, max = 10000))]
    pub window_duration_ms: u64,

    /// Maximum number of events in a batch before flushing.
    ///
    /// Default: 1024
    #[serde(default = "default_max_batch_size")]
    #[validate(range(min = 1, max = 65536))]
    pub max_batch_size: usize,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            window_duration_ms: default_window_duration_ms(),
            max_batch_size: default_max_batch_size(),
        }
    }
}

fn default_window_duration_ms() -> u64 {
    10
}

fn default_max_batch_size() -> usize {
    1024
}

/// Event emission policy configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EventPolicyConfig {
    /// Emit events only for blocks at power-of-2 positions (default).
    ///
    /// This creates sparse sampling at positions 16, 32, 64, ..., 65536
    /// for efficient radix tree construction without tracking every block.
    #[default]
    PowerOfTwo,

    /// Emit events for all blocks.
    ///
    /// Useful for testing or when complete block tracking is needed.
    All,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EventsConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.batching.window_duration_ms, 10);
        assert_eq!(config.batching.max_batch_size, 1024);
        assert_eq!(config.channel_capacity, 1024);
        assert_eq!(config.subject, "kvbm.events");
        assert_eq!(config.policy, EventPolicyConfig::PowerOfTwo);
    }

    #[test]
    fn test_serde_roundtrip() {
        let json = r#"{
            "enabled": true,
            "batching": {
                "window_duration_ms": 50,
                "max_batch_size": 512
            },
            "channel_capacity": 2048,
            "subject": "my.events",
            "policy": "all"
        }"#;

        let config: EventsConfig = serde_json::from_str(json).unwrap();
        assert!(config.enabled);
        assert_eq!(config.batching.window_duration_ms, 50);
        assert_eq!(config.batching.max_batch_size, 512);
        assert_eq!(config.channel_capacity, 2048);
        assert_eq!(config.subject, "my.events");
        assert_eq!(config.policy, EventPolicyConfig::All);

        // Roundtrip
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: EventsConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.enabled, config.enabled);
        assert_eq!(deserialized.policy, config.policy);
    }

    #[test]
    fn test_empty_json_uses_defaults() {
        let json = r#"{}"#;
        let config: EventsConfig = serde_json::from_str(json).unwrap();
        assert!(!config.enabled);
        assert_eq!(config.batching.window_duration_ms, 10);
    }

    #[test]
    fn test_partial_config() {
        // Only override enabled, everything else uses defaults
        let json = r#"{"enabled": true}"#;
        let config: EventsConfig = serde_json::from_str(json).unwrap();
        assert!(config.enabled);
        assert_eq!(config.batching.window_duration_ms, 10);
        assert_eq!(config.channel_capacity, 1024);
    }

    #[test]
    fn test_validation() {
        let config = EventsConfig {
            enabled: true,
            batching: BatchingConfig {
                window_duration_ms: 10,
                max_batch_size: 1024,
            },
            channel_capacity: 1024,
            subject: "test".to_string(),
            policy: EventPolicyConfig::PowerOfTwo,
        };
        assert!(config.validate().is_ok());
    }
}
