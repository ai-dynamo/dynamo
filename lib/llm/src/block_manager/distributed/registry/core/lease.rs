// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lease management for preventing duplicate storage.
//!
//! When a client requests to offload keys, they receive leases that grant
//! exclusive rights to store those keys. Other clients see `Leased` status
//! until the lease expires or the key is registered.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tracing::debug;

/// Information about an active lease.
#[derive(Debug, Clone)]
pub struct LeaseInfo {
    /// Client that holds the lease.
    pub client_id: u64,
    /// When the lease was granted.
    pub granted_at: Instant,
    /// When the lease expires.
    pub expires_at: Instant,
}

impl LeaseInfo {
    /// Check if the lease has expired.
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    /// Get remaining time until expiration.
    pub fn remaining(&self) -> Duration {
        self.expires_at.saturating_duration_since(Instant::now())
    }
}

/// Manages leases for registry keys.
///
/// Leases prevent race conditions where multiple workers try to store
/// the same block simultaneously. When a client calls `can_offload`,
/// they receive a lease granting exclusive rights to store the key.
///
/// # Lease Lifecycle
///
/// 1. Client calls `can_offload([k1, k2, k3])`
/// 2. Hub grants leases for keys not already stored/leased
/// 3. Client stores the blocks to object storage
/// 4. Client calls `register([k1, k2, k3])` to convert leases to entries
/// 5. If client crashes, leases expire after TTL
///
/// # Thread Safety
///
/// The lease manager uses `RwLock` for concurrent access from multiple
/// hub tasks.
pub struct LeaseManager<K>
where
    K: Eq + Hash + Clone,
{
    /// Active leases: key -> lease info
    leases: RwLock<HashMap<K, LeaseInfo>>,
    /// Default lease duration
    lease_ttl: Duration,
    /// Counter for generating unique client IDs
    client_counter: AtomicU64,
    /// Statistics
    leases_granted: AtomicU64,
    leases_expired: AtomicU64,
    leases_converted: AtomicU64,
}

impl<K> LeaseManager<K>
where
    K: Eq + Hash + Clone,
{
    /// Create a new lease manager with the given TTL.
    pub fn new(lease_ttl: Duration) -> Self {
        Self {
            leases: RwLock::new(HashMap::new()),
            lease_ttl,
            client_counter: AtomicU64::new(1),
            leases_granted: AtomicU64::new(0),
            leases_expired: AtomicU64::new(0),
            leases_converted: AtomicU64::new(0),
        }
    }

    /// Create with default TTL of 30 seconds.
    pub fn with_default_ttl() -> Self {
        Self::new(Duration::from_secs(30))
    }

    /// Generate a unique client ID.
    pub fn next_client_id(&self) -> u64 {
        self.client_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Try to acquire a lease for a key.
    ///
    /// Returns `Some(lease_info)` if the lease was granted,
    /// `None` if the key is already leased by another client.
    pub fn try_acquire(&self, key: K, client_id: u64) -> Option<LeaseInfo> {
        let now = Instant::now();
        let mut leases = self.leases.write();

        // Check if there's an existing lease
        if let Some(existing) = leases.get(&key) {
            if !existing.is_expired() && existing.client_id != client_id {
                // Another client holds an active lease
                return None;
            }
            // Lease expired or same client - we can take it
        }

        let info = LeaseInfo {
            client_id,
            granted_at: now,
            expires_at: now + self.lease_ttl,
        };

        leases.insert(key, info.clone());
        self.leases_granted.fetch_add(1, Ordering::Relaxed);
        Some(info)
    }

    /// Check if a key is currently leased (and not expired).
    pub fn is_leased(&self, key: &K) -> bool {
        let leases = self.leases.read();
        leases.get(key).map(|l| !l.is_expired()).unwrap_or(false)
    }

    /// Check if a key is leased by a specific client.
    pub fn is_leased_by(&self, key: &K, client_id: u64) -> bool {
        let leases = self.leases.read();
        leases
            .get(key)
            .map(|l| !l.is_expired() && l.client_id == client_id)
            .unwrap_or(false)
    }

    /// Get lease info for a key.
    pub fn get_lease(&self, key: &K) -> Option<LeaseInfo> {
        let leases = self.leases.read();
        leases.get(key).filter(|l| !l.is_expired()).cloned()
    }

    /// Release a lease (called when registration completes).
    ///
    /// Returns true if the lease was released.
    pub fn release(&self, key: &K) -> bool {
        let mut leases = self.leases.write();
        if leases.remove(key).is_some() {
            self.leases_converted.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Release multiple leases.
    pub fn release_many(&self, keys: &[K]) {
        let mut leases = self.leases.write();
        for key in keys {
            if leases.remove(key).is_some() {
                self.leases_converted.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Clean up expired leases.
    ///
    /// Returns the number of leases that were expired.
    pub fn cleanup_expired(&self) -> usize {
        let mut leases = self.leases.write();
        let before = leases.len();
        leases.retain(|_, info| !info.is_expired());
        let expired = before - leases.len();
        self.leases_expired
            .fetch_add(expired as u64, Ordering::Relaxed);
        expired
    }

    /// Get the number of active leases.
    pub fn active_count(&self) -> usize {
        let leases = self.leases.read();
        leases.values().filter(|l| !l.is_expired()).count()
    }

    /// Get the total number of leases (including potentially expired).
    pub fn total_count(&self) -> usize {
        self.leases.read().len()
    }

    /// Get lease statistics.
    pub fn stats(&self) -> LeaseStats {
        LeaseStats {
            active: self.active_count(),
            granted: self.leases_granted.load(Ordering::Relaxed),
            expired: self.leases_expired.load(Ordering::Relaxed),
            converted: self.leases_converted.load(Ordering::Relaxed),
        }
    }

    /// Get the lease TTL.
    pub fn ttl(&self) -> Duration {
        self.lease_ttl
    }
}

/// Statistics about lease activity.
#[derive(Debug, Clone, Default)]
pub struct LeaseStats {
    /// Currently active leases.
    pub active: usize,
    /// Total leases granted.
    pub granted: u64,
    /// Leases that expired without registration.
    pub expired: u64,
    /// Leases converted to stored entries.
    pub converted: u64,
}

/// Background task that periodically cleans up expired leases.
pub async fn lease_cleanup_task<K>(
    manager: std::sync::Arc<LeaseManager<K>>,
    interval: Duration,
    cancel: tokio_util::sync::CancellationToken,
) where
    K: Eq + Hash + Clone + Send + Sync + 'static,
{
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                debug!("Lease cleanup task shutting down");
                break;
            }
            _ = ticker.tick() => {
                let expired = manager.cleanup_expired();
                if expired > 0 {
                    debug!(expired, active = manager.active_count(), "Cleaned up expired leases");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acquire_lease() {
        let manager = LeaseManager::<u64>::new(Duration::from_secs(30));

        // First client can acquire
        let lease = manager.try_acquire(1, 100);
        assert!(lease.is_some());
        assert!(manager.is_leased(&1));
        assert!(manager.is_leased_by(&1, 100));

        // Same client can re-acquire (refresh)
        let lease2 = manager.try_acquire(1, 100);
        assert!(lease2.is_some());

        // Different client cannot acquire
        let lease3 = manager.try_acquire(1, 200);
        assert!(lease3.is_none());
    }

    #[test]
    fn test_lease_expiration() {
        let manager = LeaseManager::<u64>::new(Duration::from_millis(10));

        // Acquire lease
        let _lease = manager.try_acquire(1, 100);
        assert!(manager.is_leased(&1));

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(20));

        // Lease should be expired
        assert!(!manager.is_leased(&1));

        // Another client can now acquire
        let lease2 = manager.try_acquire(1, 200);
        assert!(lease2.is_some());
    }

    #[test]
    fn test_release_lease() {
        let manager = LeaseManager::<u64>::new(Duration::from_secs(30));

        manager.try_acquire(1, 100);
        assert!(manager.is_leased(&1));

        manager.release(&1);
        assert!(!manager.is_leased(&1));
    }

    #[test]
    fn test_cleanup_expired() {
        let manager = LeaseManager::<u64>::new(Duration::from_millis(10));

        manager.try_acquire(1, 100);
        manager.try_acquire(2, 100);
        manager.try_acquire(3, 100);

        assert_eq!(manager.total_count(), 3);

        std::thread::sleep(Duration::from_millis(20));

        let expired = manager.cleanup_expired();
        assert_eq!(expired, 3);
        assert_eq!(manager.total_count(), 0);
    }

    #[test]
    fn test_stats() {
        let manager = LeaseManager::<u64>::new(Duration::from_secs(30));

        manager.try_acquire(1, 100);
        manager.try_acquire(2, 100);
        manager.release(&1);

        let stats = manager.stats();
        assert_eq!(stats.granted, 2);
        assert_eq!(stats.converted, 1);
        assert_eq!(stats.active, 1);
    }
}

