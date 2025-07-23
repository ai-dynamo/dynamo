// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::runtime::Handle;
use tokio::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::tokens::SequenceHash;

use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;

pub trait OffloadFilter: Send + Sync {
    fn should_offload(&self, sequence_hash: SequenceHash) -> bool;
}

pub struct FrequencyFilter {
    frequency_threshold: i64,
    frequency_map: Arc<Mutex<HashMap<SequenceHash, i64>>>,
}

impl FrequencyFilter {
    pub fn new(
        frequency_threshold: u64,
        cancel_token: CancellationToken,
        runtime: Handle,
    ) -> anyhow::Result<Self> {
        let frequency_map = Arc::new(Mutex::new(HashMap::new()));
        let frequency_map_clone = frequency_map.clone();

        CriticalTaskExecutionHandle::new_with_runtime(
            move |_cancel_token| async move {
                let mut interval = tokio::time::interval(Duration::from_secs(300));
                loop {
                    interval.tick().await;

                    let mut frequency_map = frequency_map_clone.lock().unwrap();
                    frequency_map.retain(|_, count| {
                        *count -= 1;
                        *count > 0
                    });
                }
            },
            cancel_token,
            "Frequency Decay",
            &runtime,
        )?
        .detach();

        Ok(Self {
            frequency_threshold: frequency_threshold as i64,
            frequency_map,
        })
    }
}

impl OffloadFilter for FrequencyFilter {
    fn should_offload(&self, sequence_hash: SequenceHash) -> bool {
        let mut frequency_map = self.frequency_map.lock().unwrap();
        let entry = frequency_map.entry(sequence_hash).or_insert(0);

        *entry += 1;

        *entry >= self.frequency_threshold
    }
}
