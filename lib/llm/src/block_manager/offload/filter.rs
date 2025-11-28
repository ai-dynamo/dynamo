// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, MutexGuard};

use tokio::runtime::Handle;
use tokio::sync::Notify;
use tokio::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::tokens::SequenceHash;

use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;

pub trait OffloadFilter: Send + Sync + Debug {
    fn should_offload(&self, sequence_hash: SequenceHash) -> bool;
}

/// A filter that offloads blocks based on their frequency of use.
///
/// The frequency of use is tracked in a map, and the filter will offload blocks that have been used more than the minimum offload frequency.
///
/// The map is pruned periodically, and will be notified if the map is too large.
///
/// The overall strategy is to double the count on increment and decrement by 1 on each decay step.
#[derive(Debug, Clone)]
pub struct FrequencyFilter {
    min_offload_frequency: i64,
    frequency_map: Arc<Mutex<HashMap<SequenceHash, i64>>>,
    max_num_entries: usize,
    oversize_notify: Arc<Notify>,
}

impl FrequencyFilter {
    pub fn new(
        min_offload_frequency: i64,
        flush_interval: Duration,
        max_num_entries: usize,
        cancel_token: CancellationToken,
        runtime: Handle,
    ) -> anyhow::Result<Self> {
        let frequency_map = Arc::new(Mutex::new(HashMap::new()));
        let frequency_map_clone = frequency_map.clone();

        let oversize_notify = Arc::new(Notify::new());
        let oversize_notify_clone = oversize_notify.clone();

        CriticalTaskExecutionHandle::new_with_runtime(
            move |cancel_token| async move {
                let mut interval = tokio::time::interval(flush_interval);
                loop {
                    tokio::select! {
                        // Observe cancellation and exit the loop.
                        _ = cancel_token.cancelled() => {
                            break;
                        }

                        // Prune the frequency map upon the flush interval.
                        _ = interval.tick() => {
                            let mut frequency_map = frequency_map_clone.lock().unwrap();
                            Self::decrement_and_prune(&mut frequency_map);
                        }

                        // Trigger a prune if we're notified that the frequency map is too large.
                        _ = oversize_notify_clone.notified() => {
                            let mut frequency_map = frequency_map_clone.lock().unwrap();

                            // It may take multiple rounds of pruning to sufficiently reduce the size.
                            while frequency_map.len() > max_num_entries {
                                Self::decrement_and_prune(&mut frequency_map);
                            }

                            // Reset our flush interval.
                            interval.reset();
                        }
                    }
                }
                Ok(())
            },
            cancel_token,
            "Frequency Decay Handler",
            &runtime,
        )?
        .detach();

        Ok(Self {
            min_offload_frequency,
            frequency_map,
            max_num_entries,
            oversize_notify,
        })
    }

    fn decrement_and_prune(frequency_map: &mut MutexGuard<HashMap<SequenceHash, i64>>) {
        // Decrement all values and prune the entries with value 0.
        frequency_map.retain(|_, count| {
            *count -= 1;
            *count > 0
        });
    }
}

impl OffloadFilter for FrequencyFilter {
    fn should_offload(&self, sequence_hash: SequenceHash) -> bool {
        let mut frequency_map = self.frequency_map.lock().unwrap();

        // Double the value of the entry, or initialize it to 1.
        let entry = frequency_map
            .entry(sequence_hash)
            .and_modify(|count| {
                *count = count.saturating_mul(2);
            })
            .or_insert(1);

        let should_offload = *entry >= self.min_offload_frequency;
        // Notify the offload manager that the frequency map is too large.
        if frequency_map.len() > self.max_num_entries {
            self.oversize_notify.notify_one();
        }

        should_offload
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_filter(min_offload_frequency: i64, max_num_entries: usize) -> FrequencyFilter {
        let cancel_token = CancellationToken::new();
        let runtime = Handle::current();
        FrequencyFilter::new(
            min_offload_frequency,
            Duration::from_secs(3600),
            max_num_entries,
            cancel_token,
            runtime,
        )
        .unwrap()
    }

    fn hash(x: u32) -> SequenceHash {
        SequenceHash::from(x)
    }

    #[tokio::test]
    async fn test_basic_frequency_filter() {
        let filter = make_filter(2, 100);

        assert!(!filter.should_offload(hash(0)));
        assert!(filter.should_offload(hash(0)));
        assert!(!filter.should_offload(hash(1)));
        assert!(!filter.should_offload(hash(2)));
    }

    #[tokio::test]
    async fn test_decay() {
        let filter = make_filter(4, 2);

        // Add the first hashes, and bump it up to 2.
        assert!(!filter.should_offload(hash(0)));
        assert!(!filter.should_offload(hash(0)));

        // Add the second hash
        assert!(!filter.should_offload(hash(1)));
        assert!(!filter.should_offload(hash(1)));

        // Now, the value of the first hash is 4, so we should offload it.
        assert!(filter.should_offload(hash(0)));

        // This will cause a decay.
        assert!(!filter.should_offload(hash(2)));

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Now, the priority of 1
        assert!(!filter.should_offload(hash(1)));
    }

    #[tokio::test]
    async fn test_time_based_decay() {
        let cancel_token = CancellationToken::new();
        let runtime = Handle::current();
        let filter = FrequencyFilter::new(
            4,
            Duration::from_millis(250),
            100,
            cancel_token.clone(),
            runtime,
        )
        .unwrap();

        assert!(!filter.should_offload(hash(0)));
        assert!(!filter.should_offload(hash(0)));
        assert!(filter.should_offload(hash(0)));

        tokio::time::sleep(Duration::from_millis(300)).await;

        // The count should have decayed from 4 to 2.
        {
            let frequency_map = filter.frequency_map.lock().unwrap();
            assert_eq!(*frequency_map.get(&hash(0)).unwrap(), 2);
        }

        tokio::time::sleep(Duration::from_millis(250)).await;

        // The count should have decayed from 2 to 1, and should be pruned.
        {
            let frequency_map = filter.frequency_map.lock().unwrap();
            assert_eq!(*frequency_map.get(&hash(0)).unwrap(), 1);
        }

        tokio::time::sleep(Duration::from_millis(250)).await;

        // The count should have decayed from 1 to 0, and should be pruned.
        {
            let frequency_map = filter.frequency_map.lock().unwrap();
            assert!(frequency_map.get(&hash(0)).is_none());
        }
    }

    #[tokio::test]
    async fn test_multi_prune_decay() {
        let filter = make_filter(10, 2);

        assert!(!filter.should_offload(hash(0)));
        assert!(!filter.should_offload(hash(1)));

        assert_eq!(filter.frequency_map.lock().unwrap().len(), 2);

        assert!(!filter.should_offload(hash(2)));

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(filter.frequency_map.lock().unwrap().is_empty());

        assert!(!filter.should_offload(hash(3)));

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(filter.frequency_map.lock().unwrap().len(), 1);
    }
}
