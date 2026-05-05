// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use tokio::sync::Barrier;
use tokio::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct WorkerTimelines<T> {
    entries: Vec<Vec<T>>,
}

impl<T> WorkerTimelines<T> {
    pub fn new(entries: Vec<Vec<T>>) -> Self {
        Self { entries }
    }

    pub fn from_rescaled<S, GetTimestamp, WithTimestamp>(
        entries: &[Vec<S>],
        benchmark_duration_ms: u64,
        timestamp_of: GetTimestamp,
        with_timestamp: WithTimestamp,
    ) -> Self
    where
        GetTimestamp: Fn(&S) -> u64 + Copy,
        WithTimestamp: Fn(&S, u64) -> T + Copy,
    {
        let target_us = u128::from(benchmark_duration_ms) * 1000;

        let entries = entries
            .iter()
            .map(|worker_trace| {
                if worker_trace.is_empty() {
                    return Vec::new();
                }

                let max_timestamp_us = worker_trace.last().map(timestamp_of).unwrap_or(1).max(1);

                worker_trace
                    .iter()
                    .map(|entry| {
                        let scaled_timestamp = u128::from(timestamp_of(entry)) * target_us
                            / u128::from(max_timestamp_us);
                        with_timestamp(entry, scaled_timestamp.min(u128::from(u64::MAX)) as u64)
                    })
                    .collect()
            })
            .collect();

        Self { entries }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Vec<T>> {
        self.entries.iter()
    }

    pub fn into_inner(self) -> Vec<Vec<T>> {
        self.entries
    }

    pub fn rescale<GetTimestamp, WithTimestamp>(
        &self,
        benchmark_duration_ms: u64,
        timestamp_of: GetTimestamp,
        with_timestamp: WithTimestamp,
    ) -> Self
    where
        GetTimestamp: Fn(&T) -> u64 + Copy,
        WithTimestamp: Fn(&T, u64) -> T + Copy,
    {
        Self::from_rescaled(
            &self.entries,
            benchmark_duration_ms,
            timestamp_of,
            with_timestamp,
        )
    }
}

pub struct TimelineSource<'a, T> {
    worker_idx: usize,
    source: &'static str,
    entries: &'a [T],
}

impl<'a, T> TimelineSource<'a, T> {
    pub fn new(worker_idx: usize, source: &'static str, entries: &'a [T]) -> Self {
        Self {
            worker_idx,
            source,
            entries,
        }
    }

    pub fn validate_order(&self, timestamp_of: impl Fn(&T) -> u64) -> anyhow::Result<()> {
        for i in 1..self.entries.len() {
            let prev = timestamp_of(&self.entries[i - 1]);
            let curr = timestamp_of(&self.entries[i]);
            if curr < prev {
                anyhow::bail!(
                    "worker {} {} timestamps are not ordered at index {}: prev={}, curr={}",
                    self.worker_idx,
                    self.source,
                    i,
                    prev,
                    curr
                );
            }
        }

        Ok(())
    }
}

pub struct OrderedMerge<'a, L, R> {
    left: TimelineSource<'a, L>,
    right: TimelineSource<'a, R>,
}

impl<'a, L, R> OrderedMerge<'a, L, R> {
    pub fn left_first(
        worker_idx: usize,
        left_source: &'static str,
        left: &'a [L],
        right_source: &'static str,
        right: &'a [R],
    ) -> Self {
        Self {
            left: TimelineSource::new(worker_idx, left_source, left),
            right: TimelineSource::new(worker_idx, right_source, right),
        }
    }

    pub fn merge<T, LeftTimestamp, RightTimestamp, MapLeft, MapRight>(
        &self,
        left_timestamp: LeftTimestamp,
        right_timestamp: RightTimestamp,
        map_left: MapLeft,
        map_right: MapRight,
    ) -> anyhow::Result<Vec<T>>
    where
        LeftTimestamp: Fn(&L) -> u64 + Copy,
        RightTimestamp: Fn(&R) -> u64 + Copy,
        MapLeft: Fn(&L) -> T,
        MapRight: Fn(&R) -> T,
    {
        self.left.validate_order(left_timestamp)?;
        self.right.validate_order(right_timestamp)?;

        let mut left_idx = 0;
        let mut right_idx = 0;
        let mut merged = Vec::with_capacity(self.left.entries.len() + self.right.entries.len());

        while left_idx < self.left.entries.len() || right_idx < self.right.entries.len() {
            let take_left = right_idx >= self.right.entries.len()
                || left_idx < self.left.entries.len()
                    && left_timestamp(&self.left.entries[left_idx])
                        <= right_timestamp(&self.right.entries[right_idx]);

            if take_left {
                merged.push(map_left(&self.left.entries[left_idx]));
                left_idx += 1;
                continue;
            }

            merged.push(map_right(&self.right.entries[right_idx]));
            right_idx += 1;
        }

        Ok(merged)
    }
}

#[derive(Clone)]
pub struct ReplayStartGate {
    ready_barrier: Arc<Barrier>,
    start_barrier: Arc<Barrier>,
}

impl ReplayStartGate {
    pub fn new(num_workers: usize) -> Self {
        Self {
            ready_barrier: Arc::new(Barrier::new(num_workers + 1)),
            start_barrier: Arc::new(Barrier::new(num_workers + 1)),
        }
    }

    pub async fn wait_for_start(&self) {
        self.ready_barrier.wait().await;
        self.start_barrier.wait().await;
    }

    pub async fn start(&self) -> Instant {
        self.ready_barrier.wait().await;
        let started_at = Instant::now();
        self.start_barrier.wait().await;
        started_at
    }
}

#[cfg(test)]
mod tests {
    use super::{OrderedMerge, WorkerTimelines};

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Entry {
        timestamp_us: u64,
        label: &'static str,
    }

    fn ts(entry: &Entry) -> u64 {
        entry.timestamp_us
    }

    #[test]
    fn ordered_merge_is_left_first_on_equal_timestamps() {
        let reads = [
            Entry {
                timestamp_us: 10,
                label: "r10",
            },
            Entry {
                timestamp_us: 20,
                label: "r20",
            },
        ];
        let writes = [
            Entry {
                timestamp_us: 10,
                label: "w10",
            },
            Entry {
                timestamp_us: 30,
                label: "w30",
            },
        ];

        let merged = OrderedMerge::left_first(0, "reads", &reads, "writes", &writes)
            .merge(ts, ts, |entry| entry.label, |entry| entry.label)
            .unwrap();

        assert_eq!(merged, ["r10", "w10", "r20", "w30"]);
    }

    #[test]
    fn ordered_merge_rejects_unordered_sources_with_context() {
        let reads = [
            Entry {
                timestamp_us: 20,
                label: "late",
            },
            Entry {
                timestamp_us: 10,
                label: "early",
            },
        ];
        let writes = [Entry {
            timestamp_us: 30,
            label: "write",
        }];

        let err = OrderedMerge::left_first(7, "reads", &reads, "writes", &writes)
            .merge(ts, ts, |entry| entry.label, |entry| entry.label)
            .unwrap_err()
            .to_string();

        assert!(err.contains("worker 7 reads timestamps are not ordered at index 1"));
        assert!(err.contains("prev=20, curr=10"));
    }

    #[test]
    fn worker_timelines_rescale_empty_and_nonzero_start_traces() {
        let timelines = WorkerTimelines::new(vec![
            Vec::new(),
            vec![
                Entry {
                    timestamp_us: 10,
                    label: "a",
                },
                Entry {
                    timestamp_us: 20,
                    label: "b",
                },
            ],
        ]);

        let scaled = timelines.rescale(1_000, ts, |entry, timestamp_us| Entry {
            timestamp_us,
            label: entry.label,
        });
        let scaled = scaled.into_inner();

        assert!(scaled[0].is_empty());
        assert_eq!(scaled[1][0].timestamp_us, 500_000);
        assert_eq!(scaled[1][1].timestamp_us, 1_000_000);
    }
}
