# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dev.observability.dynamo_local_resource_monitor import (
    MetricsCollector,
    ProcessTracker,
)


def _name(pid: int) -> str:
    return f"process-{pid}"


def test_ranked_ids_use_cached_recency_and_rolling_total():
    tracker = ProcessTracker(maxlen=2, prune=False)

    assert tracker.record({1: 1.0, 2: 1.0}, _name)
    tracker.record({1: 0.0, 2: 2.0}, _name)
    tracker.record({1: 0.0, 2: 0.0}, _name)

    assert list(tracker.series[2]) == [2.0, 0.0]
    assert tracker.ranked_ids(2) == [2]


def test_delta_other_only_aggregates_requested_samples():
    tracker = ProcessTracker(maxlen=5, prune=False)

    tracker.record({1: 1.0, 2: 2.0, 3: 3.0}, _name)
    tracker.record({1: 4.0, 2: 5.0, 3: 6.0}, _name)

    series = tracker.series_for_ids([1], slice(1, None))

    assert [(pid, values) for pid, _, _, values in series] == [
        (1, [4.0]),
        (-1, [11.0]),
    ]


def test_gpu_other_aggregate_is_restored_from_cached_series():
    tracker = ProcessTracker(maxlen=5, prune=False)
    tracker.record({1: 1.0, 2: 2.0}, _name)
    tracker.record({1: 4.0, 2: 5.0}, _name)
    restored = ProcessTracker(maxlen=5, prune=False)
    restored.load_dict(tracker.to_dict())

    series = restored.series_for_ids([1])

    assert [(pid, values) for pid, _, _, values in series] == [
        (1, [1.0, 4.0]),
        (-1, [2.0, 5.0]),
    ]


def test_pruning_uses_cached_last_active_index():
    tracker = ProcessTracker(maxlen=300, prune=True)

    tracker.record({1: 1.0}, _name)
    for _ in range(200):
        tracker.record({}, _name)

    assert tracker.series == {}


def test_top_selection_refreshes_on_interval_or_membership_change():
    gpu_tracker = ProcessTracker(maxlen=10, prune=False)
    gpu_tracker.record({1: 1.0}, _name)
    collector = MetricsCollector.__new__(MetricsCollector)
    collector.proc_cpu = ProcessTracker(maxlen=10, prune=True)
    collector.proc_gpu_mem = [gpu_tracker]
    collector._cpu_top_n = 5
    collector.top_n = 1
    collector._top_refresh_samples = 5
    collector._top_cache_counter = -5
    collector._top_cache_dirty = True
    collector._cpu_top_ids = []
    collector._gpu_top_ids = [[]]
    collector.counter_main = 1

    collector._refresh_top_ids()
    assert collector._gpu_top_ids == [[1]]

    gpu_tracker.record({1: 0.0, 2: 1.0}, _name)
    collector.counter_main = 2
    collector._refresh_top_ids()
    assert collector._gpu_top_ids == [[1]]

    collector.counter_main = 6
    collector._refresh_top_ids()
    assert collector._gpu_top_ids == [[2]]
