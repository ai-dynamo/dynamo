# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for topology domain utilities.

Tests the read_topology_config() function that reads topology files from a
Downward API volume and KV transfer policy from env vars at worker startup.

These tests import topology.py directly (bypassing the dynamo package hierarchy)
so they work without GPU, CUDA, or any backend installed.
"""

import importlib.util
import threading
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

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


def _enable_topology(monkeypatch, topology_dir: Path, transfer_domain: str = "zone"):
    monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
    monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(topology_dir))
    monkeypatch.setenv("DYN_KV_TRANSFER_DOMAIN", transfer_domain)
    monkeypatch.delenv("DYN_KV_TRANSFER_ENFORCEMENT", raising=False)
    monkeypatch.delenv("DYN_KV_TRANSFER_PREFERRED_WEIGHT", raising=False)


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

    def test_strips_whitespace_from_file_values(self, monkeypatch, tmp_path):
        """Strips whitespace/newlines from topology file values."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("  us-east-1a\n")
        (topology_dir / "rack").write_text("\t rack-22  \n")

        _enable_topology(monkeypatch, topology_dir)

        config = read_topology_config()
        assert config.topology_domains == {
            "rack": "rack-22",
            "zone": "us-east-1a",
        }

    def test_domain_keys_are_lowercased_from_file_names(self, monkeypatch, tmp_path):
        """Domain keys are lowercased even when file names are mixed case."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "ZONE").write_text("us-east-1a")
        (topology_dir / "Rack").write_text("rack-22")

        _enable_topology(monkeypatch, topology_dir, transfer_domain="ZONE")

        config = read_topology_config()
        assert config.topology_domains == {
            "rack": "rack-22",
            "zone": "us-east-1a",
        }
        assert config.kv_transfer_domain == "zone"

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

    def test_hard_exit_after_timeout_transfer_domain_file_missing(
        self, monkeypatch, tmp_path
    ):
        """Exits when the transfer-domain file never appears within timeout."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "rack").write_text("rack-22")

        _enable_topology(monkeypatch, topology_dir)

        with pytest.raises(SystemExit) as exc_info:
            read_topology_config(poll_interval=0.05, poll_timeout=0.15)
        assert exc_info.value.code == 1

    def test_hard_exit_after_timeout_transfer_domain_file_empty(
        self, monkeypatch, tmp_path
    ):
        """Exits when the transfer-domain file exists but stays empty."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("")
        (topology_dir / "rack").write_text("rack-22")

        _enable_topology(monkeypatch, topology_dir)

        with pytest.raises(SystemExit) as exc_info:
            read_topology_config(poll_interval=0.05, poll_timeout=0.15)
        assert exc_info.value.code == 1

    def test_retry_succeeds_when_transfer_domain_file_appears(
        self, monkeypatch, tmp_path
    ):
        """Transfer-domain file appears after a delay; retry loop picks it up."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "rack").write_text("rack-22")

        _enable_topology(monkeypatch, topology_dir)

        def write_after_delay():
            time.sleep(0.1)
            (topology_dir / "zone").write_text("us-west-2a")

        writer = threading.Thread(target=write_after_delay)
        writer.start()

        config = read_topology_config(poll_interval=0.05, poll_timeout=2.0)
        writer.join()
        assert config.topology_domains == {
            "rack": "rack-22",
            "zone": "us-west-2a",
        }

    def test_retry_succeeds_when_empty_transfer_domain_file_gets_content(
        self, monkeypatch, tmp_path
    ):
        """Empty transfer-domain file gets content after a delay."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        zone_file = topology_dir / "zone"
        zone_file.write_text("")

        _enable_topology(monkeypatch, topology_dir)

        def write_after_delay():
            time.sleep(0.1)
            zone_file.write_text("eu-central-1a")

        writer = threading.Thread(target=write_after_delay)
        writer.start()

        config = read_topology_config(poll_interval=0.05, poll_timeout=2.0)
        writer.join()
        assert config.topology_domains == {"zone": "eu-central-1a"}

    def test_reads_required_transfer_policy_env_vars(self, monkeypatch, tmp_path):
        """Reads required KV transfer policy from env vars."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        _enable_topology(monkeypatch, topology_dir)
        monkeypatch.setenv("DYN_KV_TRANSFER_ENFORCEMENT", "required")

        config = read_topology_config()
        assert config.kv_transfer_domain == "zone"
        assert config.kv_transfer_enforcement == "required"
        assert config.kv_transfer_preferred_weight is None

    def test_reads_preferred_transfer_policy_env_vars(self, monkeypatch, tmp_path):
        """Reads preferred KV transfer policy and weight from env vars."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        _enable_topology(monkeypatch, topology_dir)
        monkeypatch.setenv("DYN_KV_TRANSFER_ENFORCEMENT", "preferred")
        monkeypatch.setenv("DYN_KV_TRANSFER_PREFERRED_WEIGHT", "0.85")

        config = read_topology_config()
        assert config.kv_transfer_domain == "zone"
        assert config.kv_transfer_enforcement == "preferred"
        assert config.kv_transfer_preferred_weight == 0.85

    def test_required_is_default_enforcement_when_domain_set(
        self, monkeypatch, tmp_path
    ):
        """Missing enforcement defaults to required when a transfer domain is set."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        _enable_topology(monkeypatch, topology_dir)

        config = read_topology_config()
        assert config.kv_transfer_enforcement == "required"

    def test_transfer_policy_lowercased(self, monkeypatch, tmp_path):
        """Transfer policy env var values are lowercased."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        _enable_topology(monkeypatch, topology_dir, transfer_domain="ZONE")
        monkeypatch.setenv("DYN_KV_TRANSFER_ENFORCEMENT", "Preferred")
        monkeypatch.setenv("DYN_KV_TRANSFER_PREFERRED_WEIGHT", "0.5")

        config = read_topology_config()
        assert config.kv_transfer_domain == "zone"
        assert config.kv_transfer_enforcement == "preferred"
        assert config.kv_transfer_preferred_weight == 0.5

    def test_preferred_weight_required_when_preferred(self, monkeypatch, tmp_path):
        """Missing preferred weight exits for preferred enforcement."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        _enable_topology(monkeypatch, topology_dir)
        monkeypatch.setenv("DYN_KV_TRANSFER_ENFORCEMENT", "preferred")
        monkeypatch.delenv("DYN_KV_TRANSFER_PREFERRED_WEIGHT", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            read_topology_config()
        assert exc_info.value.code == 1

    def test_invalid_transfer_enforcement_exits(self, monkeypatch, tmp_path):
        """Invalid enforcement mode exits during worker startup."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        _enable_topology(monkeypatch, topology_dir)
        monkeypatch.setenv("DYN_KV_TRANSFER_ENFORCEMENT", "fallback")

        with pytest.raises(SystemExit) as exc_info:
            read_topology_config()
        assert exc_info.value.code == 1

    @pytest.mark.parametrize("raw_weight", ["1.1", "-0.1", "nan", "inf", "heavy"])
    def test_invalid_preferred_weight_exits(self, monkeypatch, tmp_path, raw_weight):
        """Preferred weight must be finite and within [0.0, 1.0]."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        _enable_topology(monkeypatch, topology_dir)
        monkeypatch.setenv("DYN_KV_TRANSFER_ENFORCEMENT", "preferred")
        monkeypatch.setenv("DYN_KV_TRANSFER_PREFERRED_WEIGHT", raw_weight)

        with pytest.raises(SystemExit) as exc_info:
            read_topology_config()
        assert exc_info.value.code == 1

    def test_preferred_weight_with_required_enforcement_exits(
        self, monkeypatch, tmp_path
    ):
        """Preferred weight is invalid unless enforcement is preferred."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        _enable_topology(monkeypatch, topology_dir)
        monkeypatch.setenv("DYN_KV_TRANSFER_ENFORCEMENT", "required")
        monkeypatch.setenv("DYN_KV_TRANSFER_PREFERRED_WEIGHT", "0.85")

        with pytest.raises(SystemExit) as exc_info:
            read_topology_config()
        assert exc_info.value.code == 1

    def test_apply_topology_config_reads_env_and_leaves_existing_taints(
        self, monkeypatch, tmp_path
    ):
        """Python publishes topology domains; Rust derives topology taints."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")
        (topology_dir / "rack").write_text("rack-22")

        _enable_topology(monkeypatch, topology_dir)
        monkeypatch.setenv("DYN_KV_TRANSFER_ENFORCEMENT", "preferred")
        monkeypatch.setenv("DYN_KV_TRANSFER_PREFERRED_WEIGHT", "0.85")

        runtime_config = FakeRuntimeConfig()
        runtime_config.taints = {"user.taint/example"}

        returned = apply_topology_config(runtime_config)

        expected_topology_domains = {
            "rack": "rack-22",
            "zone": "us-east-1a",
        }
        assert returned.topology_domains == expected_topology_domains
        assert runtime_config.topology_domains == expected_topology_domains
        assert runtime_config.kv_transfer_domain == "zone"
        assert runtime_config.kv_transfer_enforcement == "preferred"
        assert runtime_config.kv_transfer_preferred_weight == 0.85
        assert runtime_config.taints == {"user.taint/example"}
