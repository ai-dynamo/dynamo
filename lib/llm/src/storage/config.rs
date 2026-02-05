//! Configuration for stateful responses storage
//!
//! Provides environment-based configuration for storage backends,
//! TTL settings, and operational parameters.

use std::time::Duration;

/// Configuration for response storage behavior
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Default TTL for stored responses
    pub default_ttl: Duration,
    /// Maximum TTL allowed (to prevent indefinite storage)
    pub max_ttl: Duration,
    /// Whether to store responses by default when `store` field is absent
    pub store_by_default: bool,
    /// Maximum responses per session (0 = unlimited)
    pub max_responses_per_session: usize,
    /// Storage backend type
    pub backend: StorageBackend,
}

/// Available storage backends
#[derive(Debug, Clone, PartialEq)]
pub enum StorageBackend {
    /// In-memory storage (default, for testing/single-instance)
    InMemory,
    /// Redis storage (for production/multi-instance)
    Redis { url: String },
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(24 * 60 * 60), // 24 hours
            max_ttl: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
            store_by_default: false,
            max_responses_per_session: 1000,
            backend: StorageBackend::InMemory,
        }
    }
}

impl StorageConfig {
    /// Create config from environment variables
    ///
    /// # Environment Variables
    ///
    /// - `DYNAMO_RESPONSES_DEFAULT_TTL_SECS`: Default TTL in seconds (default: 86400 = 24h)
    /// - `DYNAMO_RESPONSES_MAX_TTL_SECS`: Maximum TTL in seconds (default: 604800 = 7d)
    /// - `DYNAMO_RESPONSES_STORE_BY_DEFAULT`: Store by default (default: false)
    /// - `DYNAMO_RESPONSES_MAX_PER_SESSION`: Max responses per session (default: 1000)
    /// - `DYNAMO_RESPONSES_BACKEND`: Backend type: "memory" or "redis" (default: memory)
    /// - `DYNAMO_RESPONSES_REDIS_URL`: Redis URL if using Redis backend
    pub fn from_env() -> Self {
        let default_ttl = std::env::var("DYNAMO_RESPONSES_DEFAULT_TTL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(24 * 60 * 60));

        let max_ttl = std::env::var("DYNAMO_RESPONSES_MAX_TTL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(7 * 24 * 60 * 60));

        let store_by_default = std::env::var("DYNAMO_RESPONSES_STORE_BY_DEFAULT")
            .ok()
            .map(|s| s.to_lowercase() == "true" || s == "1")
            .unwrap_or(false);

        let max_responses_per_session = std::env::var("DYNAMO_RESPONSES_MAX_PER_SESSION")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);

        let backend = match std::env::var("DYNAMO_RESPONSES_BACKEND")
            .unwrap_or_else(|_| "memory".to_string())
            .to_lowercase()
            .as_str()
        {
            "redis" => {
                let url = std::env::var("DYNAMO_RESPONSES_REDIS_URL")
                    .unwrap_or_else(|_| "redis://localhost:6379".to_string());
                StorageBackend::Redis { url }
            }
            _ => StorageBackend::InMemory,
        };

        Self {
            default_ttl,
            max_ttl,
            store_by_default,
            max_responses_per_session,
            backend,
        }
    }

    /// Validate and clamp TTL to allowed range
    pub fn validate_ttl(&self, requested: Option<Duration>) -> Duration {
        match requested {
            Some(ttl) => ttl.min(self.max_ttl),
            None => self.default_ttl.min(self.max_ttl),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = StorageConfig::default();
        assert_eq!(config.default_ttl, Duration::from_secs(24 * 60 * 60));
        assert!(!config.store_by_default);
        assert_eq!(config.backend, StorageBackend::InMemory);
    }

    #[test]
    fn test_validate_ttl() {
        let config = StorageConfig {
            default_ttl: Duration::from_secs(3600),
            max_ttl: Duration::from_secs(86400),
            ..Default::default()
        };

        // None returns default
        assert_eq!(config.validate_ttl(None), Duration::from_secs(3600));

        // Within range returns as-is
        assert_eq!(
            config.validate_ttl(Some(Duration::from_secs(7200))),
            Duration::from_secs(7200)
        );

        // Exceeds max returns max
        assert_eq!(
            config.validate_ttl(Some(Duration::from_secs(100000))),
            Duration::from_secs(86400)
        );
    }
}
