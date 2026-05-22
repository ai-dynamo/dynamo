# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for topology domain utilities.

Tests the read_topology_config() function that reads topology files from a
deployment-provided directory and KV transfer policy from env vars at worker
startup.

These tests import topology.py directly (bypassing the dynamo package hierarchy)
so they work without GPU, CUDA, or any backend installed.
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = (pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge)

# ---------------------------------------------------------------------------
# Module loading: import topology without triggering the full dynamo package
# (which requires dynamo.llm, CUDA, etc.)
# ---------------------------------------------------------------------------
_TOPOLOGY_PY = Path(__file__).resolve().parents[2] / "utils" / "topology.py"


def _load_topology_module():
    """Load topology.py as a standalone module."""
    spec = importlib.util.spec_from_file_location("topology", _TOPOLOGY_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


topology = _load_topology_module()
read_topology_config = topology.read_topology_config
apply_topology_config = topology.apply_topology_config


class FakeRuntimeConfig:
    def __init__(self):
        self.topology_domains = {}
        self.kv_transfer_domain = None
        self.kv_transfer_enforcement = None
        self.kv_transfer_preferred_weight = None
        self.taints = set()


def _enable_topology(
    monkeypatch,
    topology_dir: Path,
    transfer_domain: str = "zone",
    enforcement: str = "required",
    preferred_weight: float | None = None,
):
    monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
    monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(topology_dir))
    monkeypatch.setenv("DYN_KV_TRANSFER_DOMAIN", transfer_domain)
    monkeypatch.setenv("DYN_KV_TRANSFER_ENFORCEMENT", enforcement)
    if preferred_weight is not None:
        monkeypatch.setenv("DYN_KV_TRANSFER_PREFERRED_WEIGHT", str(preferred_weight))
    else:
        monkeypatch.delenv("DYN_KV_TRANSFER_PREFERRED_WEIGHT", raising=False)


def _mock_topology_clock(monkeypatch, on_sleep=None):
    """Replace topology's clock so polling tests do not use wall time."""
    now = 0.0
    sleep_calls = []

    def monotonic():
        return now

    def sleep(seconds):
        nonlocal now
        sleep_calls.append(seconds)
        if on_sleep is not None:
            on_sleep(len(sleep_calls))
        now += seconds

    monkeypatch.setattr(topology.time, "monotonic", monotonic)
    monkeypatch.setattr(topology.time, "sleep", sleep)
    return sleep_calls


class TestReadTopologyConfig:
    """Tests for read_topology_config()."""

    def test_returns_empty_when_not_enabled(self, monkeypatch):
        """When DYN_TOPOLOGY_ENABLED is not set, returns empty config."""
        monkeypatch.delenv("DYN_TOPOLOGY_ENABLED", raising=False)
        config = read_topology_config()
        assert not config.enabled
        assert config.topology_domains == {}
        assert config.kv_transfer_domain is None
        assert config.kv_transfer_enforcement is None
        assert config.kv_transfer_preferred_weight is None

    def test_returns_empty_when_enabled_false(self, monkeypatch):
        """When DYN_TOPOLOGY_ENABLED=false, returns empty config."""
        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "false")
        config = read_topology_config()
        assert not config.enabled

    def test_returns_empty_when_enabled_empty_string(self, monkeypatch):
        """When DYN_TOPOLOGY_ENABLED='', returns empty config."""
        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "")
        config = read_topology_config()
        assert not config.enabled

    def test_reads_all_non_hidden_topology_files(self, monkeypatch, tmp_path):
        """Reads every visible, non-empty file under the topology mount."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")
        (topology_dir / "rack").write_text("rack-22")
        (topology_dir / "empty").write_text("")
        (topology_dir / ".hidden").write_text("ignored")
        (topology_dir / "..data").mkdir()
        (topology_dir / "nested").mkdir()

        _enable_topology(monkeypatch, topology_dir)

        config = read_topology_config()

        assert config.enabled
        assert config.topology_domains == {
            "rack": "rack-22",
            "zone": "us-east-1a",
        }
        assert config.kv_transfer_domain == "zone"
        assert config.kv_transfer_enforcement == "required"
        assert config.kv_transfer_preferred_weight is None

    def test_domain_keys_preserve_file_names(self, monkeypatch, tmp_path):
        """Domain keys match the visible topology file names exactly."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "ZONE").write_text("us-east-1a")
        (topology_dir / "Rack").write_text("rack-22")

        _enable_topology(monkeypatch, topology_dir, transfer_domain="ZONE")

        config = read_topology_config()
        assert config.topology_domains == {
            "Rack": "rack-22",
            "ZONE": "us-east-1a",
        }
        assert config.kv_transfer_domain == "ZONE"

    def test_hard_exit_when_transfer_domain_env_not_set(self, monkeypatch, tmp_path):
        """Exits when enabled but DYN_KV_TRANSFER_DOMAIN is not set."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
        monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(topology_dir))
        monkeypatch.delenv("DYN_KV_TRANSFER_DOMAIN", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            read_topology_config()
        assert exc_info.value.code == 1

    @pytest.mark.timeout(5)
    def test_hard_exit_after_timeout_transfer_domain_file_missing(
        self, monkeypatch, tmp_path
    ):
        """Exits when the transfer-domain file never appears within timeout."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "rack").write_text("rack-22")

        _enable_topology(monkeypatch, topology_dir)
        sleep_calls = _mock_topology_clock(monkeypatch)

        with pytest.raises(SystemExit) as exc_info:
            read_topology_config(poll_interval=0.05, poll_timeout=0.15)
        assert exc_info.value.code == 1
        assert sleep_calls == pytest.approx([0.05, 0.05, 0.05])

    @pytest.mark.timeout(5)
    def test_retry_succeeds_when_transfer_domain_file_appears(
        self, monkeypatch, tmp_path
    ):
        """Transfer-domain file appears after a delay; retry loop picks it up."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "rack").write_text("rack-22")

        _enable_topology(monkeypatch, topology_dir)

        def publish_transfer_domain_on_retry(_sleep_count):
            (topology_dir / "zone").write_text("us-west-2a")

        sleep_calls = _mock_topology_clock(
            monkeypatch, on_sleep=publish_transfer_domain_on_retry
        )
        config = read_topology_config(poll_interval=0.05, poll_timeout=2.0)
        assert config.topology_domains == {
            "rack": "rack-22",
            "zone": "us-west-2a",
        }
        assert sleep_calls == [0.05]

    def test_reads_preferred_transfer_policy_env_vars(self, monkeypatch, tmp_path):
        """Reads preferred KV transfer policy and weight from env vars."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        _enable_topology(
            monkeypatch, topology_dir, enforcement="preferred", preferred_weight=0.85
        )
        monkeypatch.setenv("DYN_KV_TRANSFER_ENFORCEMENT", "preferred")
        monkeypatch.setenv("DYN_KV_TRANSFER_PREFERRED_WEIGHT", "0.85")

        config = read_topology_config()
        assert config.kv_transfer_domain == "zone"
        assert config.kv_transfer_enforcement == "preferred"
        assert config.kv_transfer_preferred_weight == 0.85

    def test_non_float_preferred_weight_raises_value_error(self, monkeypatch, tmp_path):
        """Python still parses string env vars before calling the float setter."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        _enable_topology(monkeypatch, topology_dir, enforcement="preferred")
        monkeypatch.setenv("DYN_KV_TRANSFER_PREFERRED_WEIGHT", "heavy")

        with pytest.raises(ValueError, match="could not convert string to float"):
            read_topology_config()
