# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Topology domain utilities for topology-aware KV transfer routing.

Workers read their topology placement (e.g. zone, rack) from files provided
by the deployment environment, and read KV transfer policy from environment
variables. Each visible, non-empty file name is treated as a topology domain
and its contents as this worker's value for that domain. The Rust runtime
derives canonical topology taints from the published topology domains.

Environment variables:
    DYN_TOPOLOGY_ENABLED: Set to "true" to enable topology reading.
    DYN_TOPOLOGY_MOUNT_PATH: Directory containing topology domain files
        (default: /etc/dynamo/topology).
    DYN_KV_TRANSFER_DOMAIN: Which topology domain the router should enforce
        for KV transfer constraints (e.g. "zone"). The file at
        {mount_path}/{domain} must contain the transfer-domain topology value.
    DYN_KV_TRANSFER_ENFORCEMENT: KV transfer enforcement mode, either
        "required" or "preferred" (default: "required" when a domain is set).
    DYN_KV_TRANSFER_PREFERRED_WEIGHT: Preferred-taint weight used when
        enforcement is "preferred".
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_TOPOLOGY_ENABLED_VAR = "DYN_TOPOLOGY_ENABLED"
_TOPOLOGY_MOUNT_PATH_VAR = "DYN_TOPOLOGY_MOUNT_PATH"
_KV_TRANSFER_DOMAIN_VAR = "DYN_KV_TRANSFER_DOMAIN"
_KV_TRANSFER_ENFORCEMENT_VAR = "DYN_KV_TRANSFER_ENFORCEMENT"
_KV_TRANSFER_PREFERRED_WEIGHT_VAR = "DYN_KV_TRANSFER_PREFERRED_WEIGHT"
_DEFAULT_MOUNT_PATH = "/etc/dynamo/topology"
_POLL_INTERVAL_SECS = 1.0
_POLL_TIMEOUT_SECS = 30.0

logger = logging.getLogger(__name__)


@dataclass
class TopologyConfig:
    """Topology and KV transfer policy configuration read at worker startup."""

    topology_domains: dict[str, str] = field(default_factory=dict)
    kv_transfer_domain: str | None = None
    kv_transfer_enforcement: str | None = None
    kv_transfer_preferred_weight: float | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.topology_domains)


def _read_topology_domains(mount_path: Path) -> dict[str, str]:
    """Read all non-hidden, non-empty topology files from the mount path."""
    try:
        topology_files = sorted(mount_path.iterdir(), key=lambda path: path.name)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("Unable to list topology mount path %s: %s", mount_path, exc)
        return {}

    topology_domains: dict[str, str] = {}
    for topology_file in topology_files:
        file_name = topology_file.name
        if file_name.startswith("."):
            continue

        try:
            if not topology_file.is_file():
                continue
            value = topology_file.read_text().strip()
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Unable to read topology file %s: %s", topology_file, exc)
            continue

        if value:
            topology_domains[file_name] = value

    return topology_domains


def _read_kv_transfer_policy() -> tuple[str, str | None, float | None]:
    kv_transfer_domain = os.environ.get(_KV_TRANSFER_DOMAIN_VAR, None)
    kv_transfer_domain = (
        kv_transfer_domain.strip() if kv_transfer_domain is not None else None
    )
    kv_transfer_enforcement = os.environ.get(_KV_TRANSFER_ENFORCEMENT_VAR, "required")
    raw_preferred_weight = os.environ.get(_KV_TRANSFER_PREFERRED_WEIGHT_VAR, None)

    if not kv_transfer_domain:
        logger.error(
            "DYN_TOPOLOGY_ENABLED=true but %s is not set. The deployment "
            "environment must set the KV transfer domain when topology is "
            "enabled. Exiting.",
            _KV_TRANSFER_DOMAIN_VAR,
        )
        sys.exit(1)

    preferred_weight = float(raw_preferred_weight) if raw_preferred_weight else None
    return kv_transfer_domain, kv_transfer_enforcement, preferred_weight


def read_topology_config(
    poll_interval: float = _POLL_INTERVAL_SECS,
    poll_timeout: float = _POLL_TIMEOUT_SECS,
) -> TopologyConfig:
    """Read topology config from a file-backed source and env vars.

    The deployment environment injects env vars for topology location and
    transfer policy:
      - DYN_TOPOLOGY_ENABLED=true
      - DYN_TOPOLOGY_MOUNT_PATH=/etc/dynamo/topology
      - DYN_KV_TRANSFER_DOMAIN=zone
      - DYN_KV_TRANSFER_ENFORCEMENT=required
      - DYN_KV_TRANSFER_PREFERRED_WEIGHT=0.85

    Topology values are read by listing non-hidden files under the mount path.
    The topology source may populate files asynchronously, so this function
    polls until the transfer-domain file has content, sleeping between retries.

    Args:
        poll_interval: Seconds between retries (default: 1.0).
        poll_timeout: Maximum seconds to wait (default: 30.0).

    Returns:
        TopologyConfig with topology domains and transfer policy.
        Empty config if topology is not enabled.

    Raises:
        SystemExit: If DYN_TOPOLOGY_ENABLED=true but DYN_KV_TRANSFER_DOMAIN is
            not set, the transfer-domain topology file is still missing or
            empty after the timeout.
    """
    enabled = os.environ.get(_TOPOLOGY_ENABLED_VAR, "").strip().lower()
    if enabled != "true":
        return TopologyConfig()

    (
        kv_transfer_domain,
        kv_transfer_enforcement,
        kv_transfer_preferred_weight,
    ) = _read_kv_transfer_policy()

    mount_path = os.environ.get(_TOPOLOGY_MOUNT_PATH_VAR, _DEFAULT_MOUNT_PATH)
    topology_mount_path = Path(mount_path)
    transfer_domain_file = topology_mount_path / kv_transfer_domain

    # Poll until the transfer-domain file has content or timeout expires.
    # This supports topology sources that populate files asynchronously.
    deadline = time.monotonic() + poll_timeout
    topology_domains = _read_topology_domains(topology_mount_path)
    while kv_transfer_domain not in topology_domains and time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        logger.info(
            "Waiting for topology domain %s in %s (%.0fs remaining)...",
            kv_transfer_domain,
            topology_mount_path,
            remaining,
        )
        time.sleep(min(poll_interval, max(remaining, 0)))
        topology_domains = _read_topology_domains(topology_mount_path)

    if kv_transfer_domain not in topology_domains:
        logger.error(
            "DYN_TOPOLOGY_ENABLED=true but topology file %s was not populated "
            "within %.0fs. This indicates the configured topology source did "
            "not publish the selected transfer domain. Exiting.",
            transfer_domain_file,
            poll_timeout,
        )
        sys.exit(1)

    config = TopologyConfig(
        topology_domains=topology_domains,
        kv_transfer_domain=kv_transfer_domain,
        kv_transfer_enforcement=kv_transfer_enforcement,
        kv_transfer_preferred_weight=kv_transfer_preferred_weight,
    )
    logger.info("Topology config: %s (from %s)", config, topology_mount_path)
    return config


def apply_topology_config(runtime_config) -> TopologyConfig:
    """Apply topology config to a ModelRuntimeConfig-like object."""
    topology_config = read_topology_config()
    if not topology_config.enabled:
        return topology_config

    runtime_config.topology_domains = topology_config.topology_domains
    runtime_config.kv_transfer_domain = topology_config.kv_transfer_domain
    runtime_config.kv_transfer_enforcement = topology_config.kv_transfer_enforcement
    runtime_config.kv_transfer_preferred_weight = (
        topology_config.kv_transfer_preferred_weight
    )
    return topology_config
