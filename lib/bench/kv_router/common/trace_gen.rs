// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[derive(Clone, Debug, Default)]
pub struct WorkerTimelines<T> {
    entries: Vec<Vec<T>>,
}

impl<T> WorkerTimelines<T> {
    pub fn new(entries: Vec<Vec<T>>) -> Self {
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

    pub fn into_rescaled_from_first<GetTimestamp, WithTimestamp>(
        self,
        benchmark_duration_ms: u64,
        timestamp_of: GetTimestamp,
        with_timestamp: WithTimestamp,
    ) -> Self
    where
        GetTimestamp: Fn(&T) -> u64 + Copy,
        WithTimestamp: Fn(T, u64) -> T + Copy,
    {
        let target_us = u128::from(benchmark_duration_ms) * 1000;
        let entries = self
            .entries
            .into_iter()
            .map(|worker_trace| {
                let Some(first_timestamp_us) = worker_trace.first().map(timestamp_of) else {
                    return Vec::new();
                };
                let span_us = worker_trace
                    .last()
                    .map(timestamp_of)
                    .unwrap_or(first_timestamp_us)
                    .saturating_sub(first_timestamp_us)
                    .max(1);

                worker_trace
                    .into_iter()
                    .map(|entry| {
                        let relative_us = timestamp_of(&entry).saturating_sub(first_timestamp_us);
                        let scaled_timestamp =
                            u128::from(relative_us) * target_us / u128::from(span_us);
                        with_timestamp(entry, scaled_timestamp.min(u128::from(u64::MAX)) as u64)
                    })
                    .collect()
            })
            .collect();

        Self { entries }
    }

    /// Rescale every worker with one shared origin and span: the earliest and latest
    /// timestamps across all workers map to 0 and `benchmark_duration_ms`. Unlike
    /// [`Self::into_rescaled_from_first`], relative offsets between workers survive.
    pub fn into_rescaled_global<GetTimestamp, WithTimestamp>(
        self,
        benchmark_duration_ms: u64,
        timestamp_of: GetTimestamp,
        with_timestamp: WithTimestamp,
    ) -> Self
    where
        GetTimestamp: Fn(&T) -> u64 + Copy,
        WithTimestamp: Fn(T, u64) -> T + Copy,
    {
        let target_us = u128::from(benchmark_duration_ms) * 1000;
        let first_us = self
            .entries
            .iter()
            .filter_map(|worker| worker.first().map(timestamp_of))
            .min()
            .unwrap_or(0);
        let last_us = self
            .entries
            .iter()
            .filter_map(|worker| worker.last().map(timestamp_of))
            .max()
            .unwrap_or(first_us);
        let span_us = u128::from(last_us.saturating_sub(first_us).max(1));
        let entries = self
            .entries
            .into_iter()
            .map(|worker_trace| {
                worker_trace
                    .into_iter()
                    .map(|entry| {
                        let relative_us = u128::from(timestamp_of(&entry).saturating_sub(first_us));
                        let scaled = relative_us * target_us / span_us;
                        with_timestamp(entry, scaled.min(u128::from(u64::MAX)) as u64)
                    })
                    .collect()
            })
            .collect();
        Self { entries }
    }
}

#[cfg(test)]
mod tests {
    use super::WorkerTimelines;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Entry {
        timestamp_us: u64,
        label: &'static str,
    }

    fn ts(entry: &Entry) -> u64 {
        entry.timestamp_us
    }

    #[test]
    fn worker_timelines_rescale_from_each_workers_first_entry() {
        let timelines = WorkerTimelines::new(vec![
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
            vec![
                Entry {
                    timestamp_us: 1_000,
                    label: "c",
                },
                Entry {
                    timestamp_us: 1_010,
                    label: "d",
                },
            ],
        ]);

        let scaled = timelines.into_rescaled_from_first(1_000, ts, |entry, timestamp_us| Entry {
            timestamp_us,
            label: entry.label,
        });
        let scaled = scaled.into_inner();

        assert_eq!(scaled[0][0].timestamp_us, 0);
        assert_eq!(scaled[0][1].timestamp_us, 1_000_000);
        assert_eq!(scaled[1][0].timestamp_us, 0);
        assert_eq!(scaled[1][1].timestamp_us, 1_000_000);
    }

    #[test]
    fn worker_timelines_rescale_globally_keeps_worker_offsets() {
        let entry = |timestamp_us| Entry {
            timestamp_us,
            label: "x",
        };
        let timelines = WorkerTimelines::new(vec![
            vec![entry(100), entry(300)],
            vec![entry(200), entry(500)],
            Vec::new(),
        ]);
        let scaled = timelines
            .into_rescaled_global(4, ts, |entry, timestamp_us| Entry {
                timestamp_us,
                label: entry.label,
            })
            .into_inner();
        let stamps = scaled
            .iter()
            .map(|worker| worker.iter().map(ts).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(stamps, vec![vec![0, 2_000], vec![1_000, 4_000], vec![]]);
    }
}
