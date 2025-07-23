// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#[derive(Debug, Clone)]
pub struct FrequencyFilter {
    frequency_threshold: i64,
    frequency_map: Arc<Mutex<HashMap<SequenceHash, i64>>>,
    max_num_entries: usize,
    oversize_notify: Arc<Notify>,
}

impl FrequencyFilter {
    pub fn new(
        frequency_threshold: i64,
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
            move |_cancel_token| async move {
                let mut interval = tokio::time::interval(flush_interval);
                loop {
                    tokio::select! {
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
            },
            cancel_token,
            "Frequency Decay Handler",
            &runtime,
        )?
        .detach();

        Ok(Self {
            frequency_threshold,
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
                *count *= 2;
            })
            .or_insert(1);

        let should_offload = *entry >= self.frequency_threshold;

        // Notify the offload manager that the frequency map is too large.
        if frequency_map.len() > self.max_num_entries {
            self.oversize_notify.notify_one();
        }

        should_offload
    }
}
