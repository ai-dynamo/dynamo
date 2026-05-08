# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MtlsConfig (PR 2 sub-task 2-4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dynamo.planner.plugins.transport import MtlsConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


_VALID_PEM = b"""-----BEGIN CERTIFICATE-----
TElDRU5TRUQgVU5ERVIgQVBBQ0hFLTIuMA==
-----END CERTIFICATE-----
"""


# ----- from_files -----


def test_from_files_happy_path(tmp_path: Path):
    ca = tmp_path / "ca.crt"
    cert = tmp_path / "tls.crt"
    key = tmp_path / "tls.key"
    for p in (ca, cert, key):
        p.write_bytes(_VALID_PEM)
    m = MtlsConfig.from_files(ca, cert, key)
    assert m.ca_bundle == _VALID_PEM
    assert m.client_cert == _VALID_PEM
    assert m.client_key == _VALID_PEM


def test_from_files_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        MtlsConfig.from_files(
            tmp_path / "noexist-ca.crt",
            tmp_path / "tls.crt",
            tmp_path / "tls.key",
        )


def test_from_files_empty_pem_rejected(tmp_path: Path):
    ca = tmp_path / "ca.crt"
    ca.write_bytes(b"")
    cert = tmp_path / "tls.crt"
    cert.write_bytes(_VALID_PEM)
    key = tmp_path / "tls.key"
    key.write_bytes(_VALID_PEM)
    with pytest.raises(ValueError, match="empty"):
        MtlsConfig.from_files(ca, cert, key)


def test_from_files_non_pem_rejected(tmp_path: Path):
    ca = tmp_path / "ca.crt"
    ca.write_bytes(b"this is not PEM at all, just garbage data without the marker")
    cert = tmp_path / "tls.crt"
    cert.write_bytes(_VALID_PEM)
    key = tmp_path / "tls.key"
    key.write_bytes(_VALID_PEM)
    with pytest.raises(ValueError, match="does not look like PEM"):
        MtlsConfig.from_files(ca, cert, key)


# ----- from_k8s_secret_mount -----


def test_from_k8s_secret_mount_uses_three_key_convention(tmp_path: Path):
    """v11 C-3: cert-manager / certificateSecret convention is
    ``ca.crt`` / ``tls.crt`` / ``tls.key``."""
    (tmp_path / "ca.crt").write_bytes(_VALID_PEM)
    (tmp_path / "tls.crt").write_bytes(_VALID_PEM)
    (tmp_path / "tls.key").write_bytes(_VALID_PEM)
    m = MtlsConfig.from_k8s_secret_mount(tmp_path)
    assert m.ca_bundle == _VALID_PEM


def test_from_k8s_secret_mount_missing_keys(tmp_path: Path):
    """If any of the three required files missing, FileNotFoundError."""
    # Only ca.crt present
    (tmp_path / "ca.crt").write_bytes(_VALID_PEM)
    with pytest.raises(FileNotFoundError):
        MtlsConfig.from_k8s_secret_mount(tmp_path)
