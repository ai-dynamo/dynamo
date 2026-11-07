# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("pynvml")

from dev.observability import dynamo_local_resource_monitor as monitor  # noqa: E402
from dev.observability.dynamo_local_resource_monitor import (  # noqa: E402
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


def test_ranked_ids_exclude_floating_point_residuals():
    tracker = ProcessTracker(maxlen=2, prune=False)

    tracker.record({1: 1.0}, _name)
    tracker._series_total[1] = 1e-12

    assert tracker.ranked_ids() == []
    assert tracker.series_for_ids([]) == []


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


def test_restored_other_matches_the_retained_series_length():
    tracker = ProcessTracker(maxlen=2, prune=False)
    tracker.record({1: 1.0, 2: 2.0}, _name)
    for _ in range(7):
        tracker.record({}, _name)
    tracker.record({1: 4.0, 2: 5.0}, _name)
    tracker.record({1: 7.0, 2: 8.0}, _name)

    restored = ProcessTracker(maxlen=5, prune=False)
    restored.load_dict(tracker.to_dict())

    assert [(pid, values) for pid, _, _, values in restored.series_for_ids([1])] == [
        (1, [4.0, 7.0]),
        (-1, [5.0, 8.0]),
    ]


def test_pruning_uses_cached_last_active_index():
    tracker = ProcessTracker(maxlen=300, prune=True)

    tracker.record({1: 1.0}, _name)
    for _ in range(200):
        tracker.record({}, _name)

    assert tracker.series == {}


def test_pruning_drops_inactive_series_after_the_retained_window():
    tracker = ProcessTracker(maxlen=2, prune=True)

    tracker.record({1: 1.0}, _name)
    tracker.record({}, _name)
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


@pytest.fixture
def nvml_gpu(monkeypatch):
    monkeypatch.setattr(monitor, "HAS_NVML", True)
    monkeypatch.setattr(monitor.pynvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(monitor.pynvml, "nvmlDeviceGetHandleByIndex", lambda _: 123)
    monkeypatch.setattr(monitor.pynvml, "nvmlDeviceGetName", lambda _: b"NVIDIA GB10")
    monkeypatch.setattr(
        monitor.pynvml, "nvmlDeviceGetPcieThroughput", Mock(return_value=0)
    )
    memory_info = Mock()
    monkeypatch.setattr(monitor.pynvml, "nvmlDeviceGetMemoryInfo", memory_info)
    return memory_info


@pytest.mark.parametrize("supported", [False, True])
def test_collector_starts_without_supported_total_memory(nvml_gpu, supported):
    nvml_gpu.return_value = SimpleNamespace(total=8 * 1024**3)
    if not supported:
        nvml_gpu.side_effect = monitor.pynvml.NVMLError_NotSupported()

    collector = MetricsCollector(window_sec=60)

    assert collector.gpu_names == ["NVIDIA GB10"]
    assert collector.gpu_mem_total_gib == [8.0 if supported else 0.0]
    assert len(collector.proc_gpu_mem) == 1
    assert len(collector.gpu_util) == 1
    assert len(collector.gpu_temp) == 1


@pytest.mark.parametrize("supported", [False, True])
def test_main_passes_gpu_banner_without_supported_total_memory(
    monkeypatch, capsys, nvml_gpu, supported
):
    nvml_gpu.return_value = SimpleNamespace(total=8 * 1024**3)
    if not supported:
        nvml_gpu.side_effect = monitor.pynvml.NVMLError_NotSupported()
    monkeypatch.setattr(sys, "argv", ["monitor"])
    monkeypatch.setattr(monitor, "_missing_dashboard_packages", lambda: ["flask"])
    gauge = Mock()
    monkeypatch.setattr(monitor, "GPU_MEM_TOTAL_GIB", gauge)

    class StartupReached(Exception):
        pass

    monkeypatch.setattr(monitor, "PrometheusUpdater", Mock(side_effect=StartupReached))
    with pytest.raises(StartupReached):
        monitor.main()

    output = capsys.readouterr().out
    if supported:
        assert "GPU 0: NVIDIA GB10 (8.0 GiB)" in output
        gauge.labels.assert_called_once_with(gpu="0")
        gauge.labels.return_value.set.assert_called_once_with(8.0)
    else:
        assert "GPU 0: NVIDIA GB10 (memory unavailable)" in output
        gauge.labels.assert_not_called()


@pytest.mark.parametrize("entrypoint", ["collector", "main"])
def test_startup_propagates_other_nvml_errors(monkeypatch, nvml_gpu, entrypoint):
    nvml_gpu.side_effect = monitor.pynvml.NVMLError_Unknown()
    monkeypatch.setattr(sys, "argv", ["monitor"])
    monkeypatch.setattr(monitor, "_missing_dashboard_packages", lambda: ["flask"])

    with pytest.raises(monitor.pynvml.NVMLError_Unknown):
        if entrypoint == "collector":
            MetricsCollector(window_sec=60)
        else:
            monitor.main()
