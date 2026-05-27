# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-test GPU resource snapshot in tests/conftest.py.

Covers the two pure helpers (`_read_visible_vram_mb_sum`,
`_scrape_kv_cache_metrics`) plus the marker-gate constant
(`_GPU_MARKERS_REQUIRING_SNAPSHOT`). The autouse `_gpu_resource_snapshot`
fixture itself is intentionally not exercised here — it's mostly try/except
plumbing that gates on real pynvml/allure presence; testing it end-to-end
would require pytester + heavy sys.modules patching and isn't worth the
maintenance cost.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.conftest import (
    _GPU_MARKERS_REQUIRING_SNAPSHOT,
    _read_visible_vram_mb_sum,
    _scrape_kv_cache_metrics,
)


# ── Marker-gate sanity ────────────────────────────────────────────────────────


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_marker_gate_excludes_cpu_only_tests():
    """`gpu_0` must NOT be in the gate — CPU-only tests should never invoke
    pynvml. The fixture's marker check short-circuits before any GPU work."""
    assert "gpu_0" not in _GPU_MARKERS_REQUIRING_SNAPSHOT


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_marker_gate_includes_all_registered_gpu_markers():
    """Every `gpu_<N>` (N>=1) marker registered in pyproject.toml should
    trigger the snapshot. Drifts if a new gpu_16 marker lands without an
    update here."""
    assert set(_GPU_MARKERS_REQUIRING_SNAPSHOT) == {"gpu_1", "gpu_2", "gpu_4", "gpu_8"}


# ── _read_visible_vram_mb_sum ────────────────────────────────────────────────


def _fake_pynvml(per_device_used_bytes):
    """Build a SimpleNamespace masquerading as pynvml for the helper.

    per_device_used_bytes: list of ints — bytes used on each visible GPU.
    Use ``"raise"`` instead of an int to make a specific device error;
    the helper must skip that one and keep summing the rest.
    """
    def _count():
        return len(per_device_used_bytes)

    def _handle(i):
        return i  # handle is just an index for the fake

    def _mem(handle):
        v = per_device_used_bytes[handle]
        if v == "raise":
            raise RuntimeError("NVML error on device %d" % handle)
        return SimpleNamespace(used=v)

    return SimpleNamespace(
        nvmlDeviceGetCount=_count,
        nvmlDeviceGetHandleByIndex=_handle,
        nvmlDeviceGetMemoryInfo=_mem,
    )


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_read_visible_vram_sums_across_devices():
    """Two GPUs each at 1 GiB used → returns 2048.0 MiB."""
    one_gib = 1024 * 1024 * 1024
    fake = _fake_pynvml([one_gib, one_gib])
    assert _read_visible_vram_mb_sum(fake) == pytest.approx(2048.0)


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_read_visible_vram_returns_zero_for_no_devices():
    """Zero visible GPUs → 0.0 (not None — that's reserved for NVML errors)."""
    fake = _fake_pynvml([])
    assert _read_visible_vram_mb_sum(fake) == 0.0


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_read_visible_vram_returns_none_on_count_error():
    """Total NVML failure (count raises) → None so caller skips the snapshot."""
    class _Bad:
        def nvmlDeviceGetCount(self):
            raise RuntimeError("driver not loaded")
    assert _read_visible_vram_mb_sum(_Bad()) is None


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_read_visible_vram_skips_per_device_errors():
    """One bad device must NOT poison the whole reading — sum the rest."""
    one_gib = 1024 * 1024 * 1024
    fake = _fake_pynvml([one_gib, "raise", one_gib])
    # 2 good * 1024 MiB = 2048
    assert _read_visible_vram_mb_sum(fake) == pytest.approx(2048.0)


# ── _scrape_kv_cache_metrics ─────────────────────────────────────────────────


class _FakeUrlOpenContext:
    """Minimal context-manager mock for urllib.request.urlopen."""

    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _patch_urlopen(body: str):
    return patch(
        "urllib.request.urlopen",
        return_value=_FakeUrlOpenContext(body),
    )


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_scrape_extracts_dynamo_prefixed_metrics():
    """Standard `dynamo_kv_cache_*` metric names land both tokens + hit-rate."""
    body = (
        "# HELP dynamo_kv_cache_tokens_total tokens cached\n"
        "# TYPE dynamo_kv_cache_tokens_total gauge\n"
        "dynamo_kv_cache_tokens_total 5120\n"
        "# HELP dynamo_kv_cache_hit_rate cache hit rate\n"
        "# TYPE dynamo_kv_cache_hit_rate gauge\n"
        "dynamo_kv_cache_hit_rate 0.8423\n"
    )
    with _patch_urlopen(body):
        out = _scrape_kv_cache_metrics("http://localhost:9090/metrics")
    assert out == {"tokens": 5120, "hit_rate": pytest.approx(0.8423)}


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_scrape_matches_alternate_prefixes():
    """Loose name matching — `vllm_kv_cache_tokens_used` must also resolve."""
    body = (
        "vllm_kv_cache_tokens_used 2048\n"
        "sglang_kv_cache_hit_rate{model=\"llama\"} 0.50\n"
    )
    with _patch_urlopen(body):
        out = _scrape_kv_cache_metrics("http://x/metrics")
    assert out == {"tokens": 2048, "hit_rate": pytest.approx(0.50)}


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_scrape_returns_empty_when_no_kv_metrics_present():
    """Endpoint exists but exposes no KV-cache metrics → empty dict, no
    Allure labels emitted by the caller."""
    body = (
        "process_cpu_seconds_total 12.5\n"
        "go_memstats_alloc_bytes 1234\n"
    )
    with _patch_urlopen(body):
        out = _scrape_kv_cache_metrics("http://x/metrics")
    assert out == {}


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_scrape_returns_empty_on_network_error():
    """A stuck dynamo server / network error → empty dict, never raises."""
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        out = _scrape_kv_cache_metrics("http://nope/metrics")
    assert out == {}


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_scrape_skips_malformed_values_per_field():
    """One field is parseable and the other is garbage — only the good one
    lands. Matches the caster behavior on the uploader side."""
    body = (
        "dynamo_kv_cache_tokens_total 1024\n"
        "dynamo_kv_cache_hit_rate notanum\n"
    )
    with _patch_urlopen(body):
        out = _scrape_kv_cache_metrics("http://x/metrics")
    # `tokens` parses; `hit_rate` regex requires numeric chars only, so it
    # simply won't match — the field stays absent.
    assert out == {"tokens": 1024}


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_scrape_ignores_comment_lines_with_matching_substring():
    """`# HELP` / `# TYPE` lines mention the metric name — must not be
    misparsed as a value row."""
    body = (
        "# HELP dynamo_kv_cache_tokens_total tokens cached\n"
        "# TYPE dynamo_kv_cache_tokens_total gauge\n"
        # No actual value row.
    )
    with _patch_urlopen(body):
        out = _scrape_kv_cache_metrics("http://x/metrics")
    assert out == {}
