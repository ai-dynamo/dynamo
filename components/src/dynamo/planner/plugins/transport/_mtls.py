# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mTLS credential loading utilities.

**v11 C-3 decision**: reuse dynamo platform cert-manager / certificateSecret
convention — three-key K8s Secret mount with ``tls.crt`` / ``tls.key`` /
``ca.crt``. Production grpc transport MUST mount the secret; in-line cert
config in YAML is forbidden (prevents leak + reuses platform auto-rotation).

Test-only ``from_files`` API exists for local self-signed cert fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MtlsConfig:
    """mTLS credential bundle (PEM format, raw bytes).

    Fields match the gRPC ``grpc.ssl_channel_credentials`` API:
    - ``ca_bundle``: trusted CA certs to verify the server
    - ``client_cert``: this client's certificate chain (presented to server)
    - ``client_key``: this client's private key
    """

    ca_bundle: bytes
    client_cert: bytes
    client_key: bytes

    @classmethod
    def from_files(
        cls,
        ca_path: str | Path,
        cert_path: str | Path,
        key_path: str | Path,
    ) -> "MtlsConfig":
        """Load PEM credentials from individual file paths.

        Used primarily by tests with self-signed cert fixtures. Production
        should use ``from_k8s_secret_mount`` to align with cert-manager.

        Raises:
            FileNotFoundError: any path does not exist
            ValueError: file content does not look like PEM
        """
        ca = Path(ca_path).read_bytes()
        cert = Path(cert_path).read_bytes()
        key = Path(key_path).read_bytes()
        cls._validate_pem(ca, "ca_bundle", ca_path)
        cls._validate_pem(cert, "client_cert", cert_path)
        cls._validate_pem(key, "client_key", key_path)
        return cls(ca_bundle=ca, client_cert=cert, client_key=key)

    @classmethod
    def from_k8s_secret_mount(cls, mount_dir: str | Path) -> "MtlsConfig":
        """Load credentials from a K8s Secret mount with the standard
        cert-manager three-key convention (v11 C-3).

        Expected files under ``mount_dir``:
        - ``ca.crt``
        - ``tls.crt``
        - ``tls.key``

        These match dynamo platform's ``certificateSecret`` convention
        (see ``deploy/helm/charts/platform/values.yaml``). cert-manager
        manages rotation by triggering a Pod restart; v1 does not implement
        in-process hot reload.
        """
        mount = Path(mount_dir)
        return cls.from_files(
            ca_path=mount / "ca.crt",
            cert_path=mount / "tls.crt",
            key_path=mount / "tls.key",
        )

    @staticmethod
    def _validate_pem(data: bytes, name: str, source: str | Path) -> None:
        """Lightweight PEM sanity check — looks for ``-----BEGIN`` marker."""
        if not data:
            raise ValueError(f"mTLS {name} from {source}: file is empty")
        if b"-----BEGIN" not in data[:100]:
            raise ValueError(
                f"mTLS {name} from {source}: does not look like PEM "
                f"(missing '-----BEGIN' marker in first 100 bytes)"
            )


__all__ = ["MtlsConfig"]
